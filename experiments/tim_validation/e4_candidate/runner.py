"""Run mechanically nested candidate-universe sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import input_columns
from prism_benchmark.portable_checkpoints import INFERENCE_ONLY_ENV
from prism_benchmark.representative_prism_checkpoints import (
    fit_prism_checkpoint_for_view,
    predict_prism_checkpoint_for_view,
    verify_prism_checkpoint_reload,
)
from prism_benchmark.tim_e4_universe import (
    ENVIRONMENT_VARIABLE,
    UNIVERSES,
    history_grid_for_universe,
    universe_manifest,
)
from prism_benchmark.v2_views import development_dynamic_views, development_input_views
from prism_benchmark.v211_a import run_a_view
from prism_benchmark.v211_c import run_c_view
from prism_benchmark.v211_config import PUBLIC_ALL_PROTOCOL, load_v211_configs
from prism_benchmark.v211_k import run_k_channel
from prism_benchmark.v211_public_all_config import PublicAllPaths
from prism_benchmark.v211_w import run_w_view

from experiments.tim_validation.common.structure_signature import StructureSignature
from experiments.tim_validation.e3_multiscale.runner import (
    FULL_MODEL,
    _candidate_config_count,
    _candidate_is_neutral,
    _freeze_common_support,
    _scale_classes,
)


TASKS: dict[str, dict[str, Any]] = {
    "TEP_G12": {
        "dataset": "tep", "head": "TEP_G_NOWCAST_H0__H0__W1",
        "H": 0, "W": 1, "proxy_policy": "proxy_excluded",
        "history_steps": [128, 256], "rod": None,
    },
    "DEB_C4": {
        "dataset": "debutanizer", "head": "DEB_C4__H5__W1",
        "H": 5, "W": 1, "proxy_policy": "primary", "rod": None,
    },
    "METRO_OIL20": {
        "dataset": "metropt", "head": "METRO_OIL20__H120__W12",
        "H": 120, "W": 12, "proxy_policy": "primary", "rod": None,
    },
    "CZ_R1_TO_R2": {
        "dataset": "cz_czochralski", "head": "CZ_DIAM_RAW2S_CURRENT_L256_H4",
        "H": 4, "W": 1, "proxy_policy": "primary",
        "history_steps": [256], "rod": "Rod_1_to_Rod_2",
    },
    "CZ_R2_TO_R1": {
        "dataset": "cz_czochralski", "head": "CZ_DIAM_RAW2S_CURRENT_L256_H4",
        "H": 4, "W": 1, "proxy_policy": "primary",
        "history_steps": [256], "rod": "Rod_2_to_Rod_1",
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _views(shared: Path, task: str) -> tuple[Any, Any]:
    record = TASKS[task]
    input_matches = [
        view for view in development_input_views(shared)
        if view.head.head_id == record["head"]
        and view.proxy_policy == record["proxy_policy"]
    ]
    dynamic_matches = [
        view for view in development_dynamic_views(shared)
        if view.head.head_id == record["head"]
        and view.proxy_policy == record["proxy_policy"]
        and view.availability_scenario == "record_time"
    ]
    if len(input_matches) != 1 or len(dynamic_matches) != 1:
        raise RuntimeError(
            f"E4 view resolution failed for {task}: "
            f"input={len(input_matches)}, dynamic={len(dynamic_matches)}"
        )
    for view in (input_matches[0], dynamic_matches[0]):
        if view.head.h_steps != record["H"] or view.head.w_steps != record["W"]:
            raise RuntimeError(f"E4 H/W drift for {task}: {view.head}")
    return input_matches[0], dynamic_matches[0]


def prepare_manifest(project: Path, output: Path) -> dict[str, Any]:
    if os.environ.get(ENVIRONMENT_VARIABLE) is not None:
        raise RuntimeError(f"unset {ENVIRONMENT_VARIABLE} before freezing E4 manifest")
    v211, v21, v2 = load_v211_configs(project.resolve(), PUBLIC_ALL_PROTOCOL)
    result = universe_manifest(v211, v21, v2)
    result["task_history_steps"] = {
        task: {
            universe: history_grid_for_universe(record["history_steps"], universe)
            for universe in UNIVERSES
        }
        for task, record in TASKS.items()
        if "history_steps" in record
    }
    result["tasks"] = TASKS
    destination = output.resolve() / "universe_manifest.json"
    if destination.exists():
        previous = _read(destination)
        if previous != result:
            raise RuntimeError("refusing to replace a different frozen E4 universe manifest")
    else:
        _write_json(destination, result)
    shutil.copy2(Path(__file__), output.resolve() / "universe_generator.py")
    return result


def _k_job(job: tuple[Any, Any, str, list[int] | None]) -> dict[str, Any]:
    paths, view, channel, histories = job
    return run_k_channel(
        paths.shared, paths.project, paths.output, view, channel,
        PUBLIC_ALL_PROTOCOL, registered_history_steps=histories,
    )


def _valid_k_result(path: Path, universe: str, histories: list[int] | None) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _read(path)
    expected = None if histories is None else sorted(histories)
    if (
        value.get("status") == "PASS"
        and value.get("test_accessed") is False
        and value.get("registered_history_steps") == expected
        and value.get("tim_e4_candidate_universe") == universe
    ):
        return value
    raise RuntimeError(f"invalid E4 K resume artifact: {path}")


def _run_k(
    paths: PublicAllPaths, view: Any, task: str, universe: str, workers: int
) -> list[dict[str, Any]]:
    base_histories = TASKS[task].get("history_steps")
    histories = (
        None if base_histories is None
        else history_grid_for_universe(base_histories, universe)
    )
    channels = input_columns(paths.shared, view.head.task_id, view.proxy_policy)
    existing: dict[str, dict[str, Any]] = {}
    jobs = []
    for channel in channels:
        path = (
            paths.output / "DEVELOPMENT" / "K" / view.head.head_id
            / view.proxy_policy / channel / "RESULT.json"
        )
        value = _valid_k_result(path, universe, histories)
        if value is None:
            jobs.append((paths, view, channel, histories))
        else:
            existing[channel] = value
    if workers <= 1:
        produced = map(_k_job, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        produced = executor.map(_k_job, jobs)
    try:
        for value in produced:
            existing[str(value["channel"])] = value
    finally:
        if workers > 1:
            executor.shutdown(wait=True)
    return [existing[channel] for channel in channels]


def _tag_stage_result(path: Path, universe: str) -> dict[str, Any]:
    result = _read(path)
    result["tim_e4_candidate_universe"] = universe
    _write_json(path, result)
    return result


def _stage_path(paths: PublicAllPaths, view: Any, stage: str) -> Path:
    path = paths.output / "DEVELOPMENT" / stage / view.head.head_id
    if stage == "A":
        path = path / view.availability_scenario / view.proxy_policy
    else:
        path = path / view.proxy_policy
    return path / "RESULT.json"


def _stage_path_from_record(
    paths: PublicAllPaths, record: dict[str, Any], stage: str
) -> Path:
    path = paths.output / "DEVELOPMENT" / stage / record["target_head"]
    if stage == "A":
        path = path / record["availability_scenario"] / record["proxy_policy"]
    else:
        path = path / record["proxy_policy"]
    return path / "RESULT.json"


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    manifest_path = args.run_root.resolve() / "universe_manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("status") != "FROZEN_BEFORE_E4_TEST_ACCESS":
        raise RuntimeError("E4 universe manifest must be frozen before development")
    os.environ[ENVIRONMENT_VARIABLE] = args.universe
    os.environ["PRISM_V211_K_INNER_WORKERS"] = "1"
    project = args.project.resolve()
    shared = args.shared.resolve()
    task_root = args.run_root.resolve() / args.task / args.universe
    if task_root.exists() and any(task_root.iterdir()) and not args.resume:
        raise RuntimeError(f"refusing nonempty E4 task root: {task_root}")
    paths = PublicAllPaths(project, shared, task_root)
    input_view, dynamic_view = _views(shared, args.task)
    k_results = _run_k(paths, input_view, args.task, args.universe, args.workers)
    for result in k_results:
        path = (
            paths.output / "DEVELOPMENT" / "K" / input_view.head.head_id
            / input_view.proxy_policy / str(result["channel"]) / "RESULT.json"
        )
        _tag_stage_result(path, args.universe)
    c_result = run_c_view(shared, project, paths.output, input_view, PUBLIC_ALL_PROTOCOL)
    c_result = _tag_stage_result(_stage_path(paths, input_view, "C"), args.universe)
    w_result = run_w_view(shared, project, paths.output, input_view, PUBLIC_ALL_PROTOCOL)
    w_result = _tag_stage_result(_stage_path(paths, input_view, "W"), args.universe)
    a_result = run_a_view(shared, project, paths.output, dynamic_view, PUBLIC_ALL_PROTOCOL)
    a_result = _tag_stage_result(_stage_path(paths, dynamic_view, "A"), args.universe)
    statuses = {
        "K": [value.get("status") for value in k_results],
        "C": c_result.get("status"), "W": w_result.get("status"),
        "A": a_result.get("status"),
    }
    if not all(value == "PASS" for value in statuses["K"]) or any(
        statuses[stage] != "PASS" for stage in ("C", "W", "A")
    ):
        raise RuntimeError(f"E4 native stage failure: {statuses}")
    _freeze_common_support(paths, dynamic_view)
    selection_seal = {
        "status": "SELECTION_FROZEN", "task": args.task,
        "candidate_universe": args.universe,
        "universe_manifest_sha256": _sha256(manifest_path),
        "candidate_counts": _candidate_config_count(paths.output, dynamic_view),
        "test_accessed": False,
    }
    _write_json(paths.freeze / "E4_SELECTION_FREEZE.json", selection_seal)
    checkpoint = fit_prism_checkpoint_for_view(paths, dynamic_view, task_root / "checkpoints")
    _write_json(
        paths.freeze / "CHECKPOINTS_SEALED.json",
        {"status": "CHECKPOINTS_SEALED", "checkpoint": checkpoint, "test_accessed": False},
    )
    result = {
        "status": "READY_FOR_ISOLATED_TEST_INFERENCE", "task": args.task,
        "candidate_universe": args.universe, "stage_statuses": statuses,
        "wall_seconds": time.time() - started, "python": platform.python_version(),
        "test_accessed": False,
    }
    _write_json(task_root / "DEVELOPMENT_COMPLETE.json", result)
    return result


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError(f"{INFERENCE_ONLY_ENV}=1 is required")
    os.environ[ENVIRONMENT_VARIABLE] = args.universe
    task_root = args.run_root.resolve() / args.task / args.universe
    paths = PublicAllPaths(args.project.resolve(), args.shared.resolve(), task_root)
    _, view = _views(paths.shared, args.task)
    checkpoint_root = task_root / "checkpoints"
    checkpoint_dir = next((checkpoint_root / "prism").iterdir())
    records = predict_prism_checkpoint_for_view(
        paths, view, checkpoint_root, split="test"
    )
    result = {
        "status": "PASS", "task": args.task, "candidate_universe": args.universe,
        "reload_audit": verify_prism_checkpoint_reload(checkpoint_dir),
        "records": records, "fit_called_in_inference": False, "test_accessed": True,
    }
    _write_json(task_root / "inference" / "INFERENCE_COMPLETE.json", result)
    return result


def _inference_subprocess(args: argparse.Namespace) -> None:
    environment = dict(os.environ)
    environment[INFERENCE_ONLY_ENV] = "1"
    environment[ENVIRONMENT_VARIABLE] = args.universe
    command = [
        sys.executable, "-m", "experiments.tim_validation.e4_candidate.runner", "infer",
        "--project", str(args.project.resolve()), "--shared", str(args.shared.resolve()),
        "--run-root", str(args.run_root.resolve()), "--task", args.task,
        "--universe", args.universe,
    ]
    subprocess.run(command, cwd=args.project.resolve().parent, env=environment, check=True)


def _full_record(task_root: Path) -> dict[str, Any]:
    records = _read(task_root / "inference" / "INFERENCE_COMPLETE.json")["records"]
    matches = [
        value for value in records
        if value.get("model") == FULL_MODEL and value.get("status") == "PASS"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing E4 full PRISM inference record: {len(matches)}")
    return matches[0]


def _write_structure_signature(
    args: argparse.Namespace, task_root: Path, record: dict[str, Any]
) -> dict[str, Any]:
    paths = PublicAllPaths(args.project.resolve(), args.shared.resolve(), task_root)
    k_root = (
        paths.output / "DEVELOPMENT" / "K" / record["target_head"]
        / record["proxy_policy"]
    )
    k_results = [_read(path) for path in sorted(k_root.glob("*/RESULT.json"))]
    admitted = [str(value["channel"]) for value in k_results if value.get("active") is True]
    rejected = [str(value["channel"]) for value in k_results if value.get("active") is not True]
    c_result = _read(_stage_path_from_record(paths, record, "C"))
    w_result = _read(_stage_path_from_record(paths, record, "W"))
    a_result = _read(_stage_path_from_record(paths, record, "A"))
    selected_c = c_result.get("final_selected_candidate", c_result.get("selected_candidate"))
    selected_w = w_result.get("final_selected_candidate", w_result.get("selected_candidate"))
    selected_a = a_result.get("final_selected_candidate", a_result.get("selected_candidate"))
    flags = [bool(admitted), not _candidate_is_neutral(selected_c),
             not _candidate_is_neutral(selected_w), not _candidate_is_neutral(selected_a)]
    task = TASKS[args.task]
    signature = StructureSignature(
        task=args.task, head=str(record["target_head"]), H=int(task["H"]), W=int(task["W"]),
        outer_fold="registered_outer_test", run=f"E4_{args.universe}", rod=task["rod"], seed=0,
        candidate_universe=args.universe, model_variant=FULL_MODEL,
        admitted_channels=admitted, rejected_channels=rejected,
        selected_profile_by_channel={str(value["channel"]): value.get("selected_profile") for value in k_results},
        selected_history_by_channel={str(value["channel"]): int(value["selected_profile_history_steps"]) for value in k_results},
        selected_scale_class_by_channel=_scale_classes(k_results),
        K_admitted=flags[0], C_admitted=flags[1], W_admitted=flags[2], A_admitted=flags[3],
        selected_K_candidate={str(value["channel"]): value.get("final_selected_candidate") for value in k_results},
        selected_C_candidate=selected_c, selected_W_candidate=selected_w, selected_A_candidate=selected_a,
        validation_gain_C=c_result.get("validation_relative_gain"),
        validation_gain_W=w_result.get("validation_relative_gain"),
        validation_gain_A=a_result.get("active_near_zero_audit", {}).get("validation_relative_gain"),
        parameter_count=record.get("parameter_count"), active_channel_count=len(admitted),
        active_stage_count=sum(flags), RMSE=float(record["rmse"]), MAE=float(record["mae"]),
        R2=float(record.get("r2_level_reconstructed", record["r2_delta"])),
        support_id=str(record["scoring_support_hash"]),
        config_hash=_sha256(args.run_root.resolve() / "universe_manifest.json"),
    )
    signature.write(task_root / "STRUCTURE_SIGNATURE.json")
    return signature.to_dict()


def finalize_task(args: argparse.Namespace) -> dict[str, Any]:
    task_root = args.run_root.resolve() / args.task / args.universe
    record = _full_record(task_root)
    signature = _write_structure_signature(args, task_root, record)
    result = {
        "status": "COMPLETED", "task": args.task,
        "candidate_universe": args.universe, "record": record,
        "structure_signature_hash": signature["signature_hash"],
        "test_accessed_only_after_checkpoint_seal": True,
    }
    _write_json(task_root / "TASK_RESULT.json", result)
    return result


def _jaccard(left: set[str], right: set[str]) -> tuple[float, bool]:
    if not left and not right:
        return 1.0, True
    return len(left & right) / len(left | right), False


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _common_support_metrics(
    run_root: Path, task: str, results: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    for universe in UNIVERSES:
        record = results[universe]["record"]
        path = run_root / task / universe / str(record["prediction_path"])
        frame = pd.read_parquet(path, columns=["base_origin_id", "y_true", "y_pred"])
        if frame["base_origin_id"].duplicated().any():
            raise RuntimeError(f"duplicate E4 base_origin_id for {task}/{universe}")
        frames[universe] = frame.set_index("base_origin_id", drop=False)
    common = set.intersection(
        *(set(frame.index.astype(str)) for frame in frames.values())
    )
    ordered_ids = [
        str(value) for value in frames["standard"].index.astype(str)
        if str(value) in common
    ]
    if not ordered_ids:
        raise RuntimeError(f"empty E4 common support for {task}")
    support_id = hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()
    reference = frames["standard"].loc[ordered_ids, "y_true"].to_numpy(dtype=np.float64)
    denominator = float(np.sum(np.square(reference - np.mean(reference))))
    metrics: dict[str, dict[str, Any]] = {}
    for universe, frame in frames.items():
        y_true = frame.loc[ordered_ids, "y_true"].to_numpy(dtype=np.float64)
        if not np.array_equal(y_true, reference):
            raise RuntimeError(f"E4 y_true drift on common support for {task}/{universe}")
        prediction = frame.loc[ordered_ids, "y_pred"].to_numpy(dtype=np.float64)
        error = prediction - reference
        mse = float(np.mean(np.square(error), dtype=np.float64))
        metrics[universe] = {
            "rows": len(ordered_ids), "RMSE": float(np.sqrt(mse)),
            "MAE": float(np.mean(np.abs(error), dtype=np.float64)),
            "R2": float("nan") if denominator == 0.0 else 1.0 - float(np.sum(np.square(error))) / denominator,
            "support_id": support_id,
        }
    audit = {
        "status": "PASS", "task": task, "rows": len(ordered_ids),
        "support_id": support_id, "identical_target_rows": True,
        "raw_support_rows": {u: len(frame) for u, frame in frames.items()},
        "raw_support_hashes": {
            u: results[u]["record"]["scoring_support_hash"] for u in UNIVERSES
        },
        "intersection_used_for_selection": False,
        "test_accessed_after_all_universe_checkpoint_seals": True,
    }
    return metrics, audit


def build_report(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    task_names = [
        path.name for path in sorted(run_root.iterdir())
        if path.is_dir() and all((path / universe / "TASK_RESULT.json").is_file() for universe in UNIVERSES)
    ]
    aggregate: list[dict[str, Any]] = []
    prediction: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    scales: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    complexity: list[dict[str, Any]] = []
    per_fold: list[dict[str, Any]] = []
    for task in task_names:
        results = {u: _read(run_root / task / u / "TASK_RESULT.json") for u in UNIVERSES}
        signatures = {u: _read(run_root / task / u / "STRUCTURE_SIGNATURE.json") for u in UNIVERSES}
        records = {u: results[u]["record"] for u in UNIVERSES}
        common_metrics, support_audit = _common_support_metrics(run_root, task, results)
        _write_json(run_root / task / "E4_COMMON_SUPPORT_AUDIT.json", support_audit)
        standard_rmse = float(common_metrics["standard"]["RMSE"])
        for universe in UNIVERSES:
            record = records[universe]
            signature = signatures[universe]
            metrics = common_metrics[universe]
            common_signature = {
                key: value for key, value in signature.items() if key != "signature_hash"
            }
            common_signature.update(
                {"RMSE": metrics["RMSE"], "MAE": metrics["MAE"],
                 "R2": metrics["R2"], "support_id": metrics["support_id"]}
            )
            common_signature["signature_hash"] = hashlib.sha256(
                json.dumps(common_signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            _write_json(
                run_root / task / universe / "STRUCTURE_SIGNATURE_COMMON_SUPPORT.json",
                common_signature,
            )
            aggregate.append({
                "task": task, "universe": universe, "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"], "R2": metrics["R2"],
                "support_id": metrics["support_id"], "rows": metrics["rows"],
                "raw_RMSE": record["rmse"], "raw_rows": record["rows"],
            })
            prediction.append({
                "task": task, "universe": universe,
                "D_pred": (float(metrics["RMSE"]) - standard_rmse) / standard_rmse,
            })
            complexity.append({
                "task": task, "universe": universe,
                "parameter_count": signature.get("parameter_count"),
                "active_channel_count": signature["active_channel_count"],
                "active_stage_count": signature["active_stage_count"],
            })
        for other in ("coarse", "expanded"):
            left = set(signatures["standard"]["admitted_channels"])
            right = set(signatures[other]["admitted_channels"])
            score, empty_empty = _jaccard(left, right)
            channels.append({"task": task, "comparison": f"standard_vs_{other}", "jaccard": score, "empty_empty": empty_empty})
            common = sorted(left & right)
            same = sum(
                signatures["standard"]["selected_scale_class_by_channel"].get(channel)
                == signatures[other]["selected_scale_class_by_channel"].get(channel)
                for channel in common
            )
            scales.append({
                "task": task, "comparison": f"standard_vs_{other}",
                "common_admitted_channels": len(common),
                "scale_agreement": None if not common else same / len(common),
            })
            names = ("K_admitted", "C_admitted", "W_admitted", "A_admitted")
            vector_standard = [bool(signatures["standard"][name]) for name in names]
            vector_other = [bool(signatures[other][name]) for name in names]
            stages.append({
                "task": task, "comparison": f"standard_vs_{other}",
                "stage_vector_standard": json.dumps(vector_standard),
                "stage_vector_other": json.dumps(vector_other),
                "stage_agreement": sum(a == b for a, b in zip(vector_standard, vector_other)) / 4,
                "conclusion_reversal": vector_standard != vector_other,
            })
        for universe in UNIVERSES:
            root = run_root / task / universe / "results" / "DEVELOPMENT"
            for path in root.rglob("RESULT.json"):
                value = _read(path)
                for candidate, losses in value.get("candidate_fold_losses", {}).items():
                    for fold, loss in enumerate(losses):
                        per_fold.append({"task": task, "universe": universe, "stage": value.get("stage"), "candidate": candidate, "fold": fold, "loss": loss})
    for name, rows in (
        ("aggregate.csv", aggregate), ("prediction_sensitivity.csv", prediction),
        ("channel_agreement.csv", channels), ("scale_agreement.csv", scales),
        ("stage_agreement.csv", stages), ("complexity.csv", complexity),
        ("per_fold.csv", per_fold),
    ):
        _write_csv(run_root / name, rows)
    reversals = sum(bool(row["conclusion_reversal"]) for row in stages)
    text = (
        "# E4 Candidate-Universe Sensitivity\n\n"
        f"Status: `{'COMPLETED' if task_names else 'PARTIAL'}`.\n\n"
        f"Completed tasks: {', '.join(task_names) or 'none'}.\n\n"
        f"Stage-admission conclusion reversals: {reversals}/{len(stages)} comparisons.\n\n"
        "Prediction, channel, conditional-scale, stage, and complexity sensitivity are reported separately.\n"
    )
    (run_root / "E4_INTERPRETATION.md").write_text(text, encoding="utf-8")
    (run_root / "README.md").write_text(text, encoding="utf-8")
    result = {"status": "COMPLETED" if task_names else "PARTIAL", "tasks": task_names, "stage_conclusion_reversals": reversals}
    _write_json(run_root / "E4_REPORT_MANIFEST.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--project", type=Path, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    for name in ("all", "develop", "infer", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--project", type=Path, required=True)
        command.add_argument("--shared", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--task", choices=tuple(TASKS), required=True)
        command.add_argument("--universe", choices=UNIVERSES, required=True)
        if name in {"all", "develop"}:
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--resume", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--run-root", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare_manifest(args.project, args.run_root)
    elif args.command == "infer":
        run_inference(args)
    elif args.command == "finalize":
        finalize_task(args)
    elif args.command == "report":
        build_report(args.run_root)
    elif args.command == "develop":
        run_development(args)
    else:
        run_development(args)
        _inference_subprocess(args)
        finalize_task(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
