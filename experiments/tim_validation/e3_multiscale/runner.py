"""Run E3 with a shared-history arm and native per-channel PRISM histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from prism_benchmark.cpu_data import input_columns
from prism_benchmark.portable_checkpoints import INFERENCE_ONLY_ENV
from prism_benchmark.representative_prism_checkpoints import (
    fit_prism_checkpoint_for_view,
    predict_prism_checkpoint_for_view,
    verify_prism_checkpoint_reload,
)
from prism_benchmark.v211_a import run_a_view
from prism_benchmark.v211_c import run_c_view
from prism_benchmark.v211_config import PUBLIC_ALL_PROTOCOL, load_v211_configs
from prism_benchmark.v211_k import run_k_channel
from prism_benchmark.v211_public_all_baselines import apply_common_requirements
from prism_benchmark.v211_public_all_closure import (
    _support_frame,
    view_support_requirements,
)
from prism_benchmark.v211_public_all_config import PublicAllPaths
from prism_benchmark.v211_public_all_views import (
    public_all_dynamic_views,
    public_all_input_views,
)
from prism_benchmark.v211_support import SUPPORT_CONTRACT, support_id_hash
from prism_benchmark.v211_w import run_w_view

from .selection import parse_profile_losses, select_uniform_history


TASKS = {
    "TEP_G12": {"dataset": "tep", "proxy_policy": "proxy_excluded"},
    "DEB_C4": {"dataset": "debutanizer", "proxy_policy": "primary"},
    "PMSM_PM5": {"dataset": "pmsm", "proxy_policy": "proxy_excluded"},
}
FULL_MODEL = "PRISM_V2_1_1_PHYSICS_FIRST"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _views(shared: Path, task: str) -> tuple[Any, Any]:
    if task not in TASKS:
        raise ValueError(f"unregistered E3 task: {task}")
    proxy = TASKS[task]["proxy_policy"]
    input_matches = [
        view
        for view in public_all_input_views(shared)
        if view.head.task_id == task and view.proxy_policy == proxy
    ]
    dynamic_matches = [
        view
        for view in public_all_dynamic_views(shared)
        if view.head.task_id == task
        and view.proxy_policy == proxy
        and view.availability_scenario == "record_time"
    ]
    if len(input_matches) != 1 or len(dynamic_matches) != 1:
        raise RuntimeError(
            f"E3 view resolution failed for {task}: "
            f"input={len(input_matches)}, dynamic={len(dynamic_matches)}"
        )
    return input_matches[0], dynamic_matches[0]


def _source_k_results(
    source_paths: PublicAllPaths, input_view: Any
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = (
        source_paths.output
        / "DEVELOPMENT"
        / "K"
        / input_view.head.head_id
        / input_view.proxy_policy
    )
    results = []
    provenance = []
    for channel in input_columns(
        source_paths.shared, input_view.head.task_id, input_view.proxy_policy
    ):
        path = root / channel / "RESULT.json"
        value = _read(path)
        if value.get("status") != "PASS" or value.get("test_accessed") is not False:
            raise RuntimeError(f"unsealed scale-aware K source: {path}")
        results.append(value)
        provenance.append({"path": str(path), "sha256": _sha256(path)})
    return results, provenance


def _copy_tree_hardlinked(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"refusing existing E3 shadow path: {destination}")
    try:
        shutil.copytree(source, destination, copy_function=os.link)
    except OSError:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def _clone_multiscale_task(
    source_paths: PublicAllPaths,
    shadow_paths: PublicAllPaths,
    input_view: Any,
    dynamic_view: Any,
) -> None:
    mappings = [
        (
            source_paths.output
            / "DEVELOPMENT"
            / "K"
            / input_view.head.head_id
            / input_view.proxy_policy,
            shadow_paths.output
            / "DEVELOPMENT"
            / "K"
            / input_view.head.head_id
            / input_view.proxy_policy,
        ),
    ]
    for stage in ("C", "W"):
        mappings.append(
            (
                source_paths.output
                / "DEVELOPMENT"
                / stage
                / input_view.head.head_id
                / input_view.proxy_policy,
                shadow_paths.output
                / "DEVELOPMENT"
                / stage
                / input_view.head.head_id
                / input_view.proxy_policy,
            )
        )
    mappings.append(
        (
            source_paths.output
            / "DEVELOPMENT"
            / "A"
            / dynamic_view.head.head_id
            / dynamic_view.availability_scenario
            / dynamic_view.proxy_policy,
            shadow_paths.output
            / "DEVELOPMENT"
            / "A"
            / dynamic_view.head.head_id
            / dynamic_view.availability_scenario
            / dynamic_view.proxy_policy,
        )
    )
    for source, destination in mappings:
        _copy_tree_hardlinked(source, destination)
    shadow_paths.freeze.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_paths.leaderboard_support_path,
        shadow_paths.leaderboard_support_path,
    )


def _freeze_common_support(paths: PublicAllPaths, dynamic_view: Any) -> dict[str, Any]:
    requirements = view_support_requirements(paths, dynamic_view)
    splits: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        source = paths.shared / "sample_ids" / dynamic_view.relative_root / f"{split}.parquet"
        if not source.is_file():
            continue
        frame = _support_frame(paths.shared, dynamic_view, split)
        common = apply_common_requirements(frame, requirements)
        splits[split] = {
            "rows": int(len(common)),
            "source_rows": int(len(frame)),
            "support_hash": support_id_hash(common),
            "support_contract": SUPPORT_CONTRACT,
        }
    result = {
        "status": "PASS",
        "stage": "TIM_E3_COMMON_SUPPORT_METADATA_ONLY_FREEZE",
        "support_contract": SUPPORT_CONTRACT,
        "views": [
            {
                "target_head": dynamic_view.head.head_id,
                "dataset": dynamic_view.head.dataset,
                "information_set": dynamic_view.information_set,
                "availability_scenario": dynamic_view.availability_scenario,
                "proxy_policy": dynamic_view.proxy_policy,
                "requirements": [item.to_json() for item in requirements],
                "splits": splits,
            }
        ],
        "test_y_read": False,
        "test_accessed": False,
    }
    _write_json(paths.leaderboard_support_path, result)
    return result


def _candidate_config_count(result_root: Path, dynamic_view: Any) -> dict[str, int]:
    input_root = (
        result_root
        / "DEVELOPMENT"
        / "K"
        / dynamic_view.head.head_id
        / dynamic_view.proxy_policy
    )
    k_results = [_read(path) for path in input_root.glob("*/RESULT.json")]
    k = sum(
        len(item.get("profile_fold_losses", {}))
        + len(item.get("linear_activation_profile_fold_losses", {}))
        + max(0, len(item.get("structural_fold_losses", {})) - 1)
        + len(item.get("minimal_stabilizing_ridge_audit", []))
        for item in k_results
    )
    values = {"K": k}
    for stage, parts in (
        ("C", (dynamic_view.head.head_id, dynamic_view.proxy_policy)),
        ("W", (dynamic_view.head.head_id, dynamic_view.proxy_policy)),
        (
            "A",
            (
                dynamic_view.head.head_id,
                dynamic_view.availability_scenario,
                dynamic_view.proxy_policy,
            ),
        ),
    ):
        path = result_root / "DEVELOPMENT" / stage
        for part in parts:
            path /= part
        result = _read(path / "RESULT.json")
        values[stage] = len(result.get("candidate_fold_losses", {}))
    values["total"] = sum(values.values())
    return values


def _loss_fit_attempts(losses: Any, *, exclude_neutral: bool = False) -> int:
    if not isinstance(losses, dict):
        return 0
    total = 0
    for key, values in losses.items():
        if exclude_neutral and str(key) in {"EXACT_ZERO", "IDENTITY"}:
            continue
        if isinstance(values, list):
            total += len(values)
    return total


def _candidate_fit_attempts(result_root: Path, dynamic_view: Any) -> dict[str, int]:
    input_root = (
        result_root
        / "DEVELOPMENT"
        / "K"
        / dynamic_view.head.head_id
        / dynamic_view.proxy_policy
    )
    k = 0
    for path in input_root.glob("*/RESULT.json"):
        result = _read(path)
        k += _loss_fit_attempts(result.get("profile_fold_losses"))
        k += _loss_fit_attempts(
            result.get("linear_activation_profile_fold_losses")
        )
        k += _loss_fit_attempts(
            result.get("structural_fold_losses"), exclude_neutral=True
        )
        for audit in result.get("minimal_stabilizing_ridge_audit", []):
            k += 1 + len(audit.get("inner_fold_certificates", []))
    values = {"K": k}
    for stage, parts in (
        ("C", (dynamic_view.head.head_id, dynamic_view.proxy_policy)),
        ("W", (dynamic_view.head.head_id, dynamic_view.proxy_policy)),
        (
            "A",
            (
                dynamic_view.head.head_id,
                dynamic_view.availability_scenario,
                dynamic_view.proxy_policy,
            ),
        ),
    ):
        path = result_root / "DEVELOPMENT" / stage
        for part in parts:
            path /= part
        result = _read(path / "RESULT.json")
        values[stage] = _loss_fit_attempts(result.get("candidate_fold_losses")) + 1
    values["total"] = sum(values.values())
    return values


def _run_uniform_k(
    paths: PublicAllPaths,
    input_view: Any,
    source_results: list[dict[str, Any]],
    selected_history: int,
    workers: int,
) -> list[dict[str, Any]]:
    def run(source: dict[str, Any]) -> dict[str, Any]:
        losses = {
            profile: values
            for profile, values in parse_profile_losses(source).items()
            if int(profile[1]) == selected_history
        }
        return run_k_channel(
            paths.shared,
            paths.project,
            paths.output,
            input_view,
            str(source["channel"]),
            PUBLIC_ALL_PROTOCOL,
            forced_history_steps=selected_history,
            profile_fold_losses_override=losses,
        )

    if workers <= 1:
        return [run(source) for source in source_results]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run, source_results))


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    project = args.project.resolve()
    shared = args.shared.resolve()
    task_root = args.run_root.resolve() / args.task
    if task_root.exists() and any(task_root.iterdir()):
        raise RuntimeError(f"refusing nonempty E3 task root: {task_root}")
    task_root.mkdir(parents=True, exist_ok=True)
    input_view, dynamic_view = _views(shared, args.task)
    source_paths = PublicAllPaths(project, shared, args.scale_aware_run.resolve())
    uniform_paths = PublicAllPaths(project, shared, task_root / "uniform")
    multiscale_paths = PublicAllPaths(project, shared, task_root / "multiscale")

    source_results, source_provenance = _source_k_results(source_paths, input_view)
    v211, v21, _ = load_v211_configs(project, protocol=PUBLIC_ALL_PROTOCOL)
    selection = select_uniform_history(
        source_results,
        maximum_relative_regret=float(
            v211["K"]["profile_selection"]["maximum_relative_regret_vs_best"]
        ),
        minimum_usable_folds=int(v21["selection"]["minimum_usable_folds"]),
    )
    selection["task"] = args.task
    selection["source_k_results"] = source_provenance
    _write_json(task_root / "UNIFORM_HISTORY_SELECTION.json", selection)

    os.environ["PRISM_V211_K_INNER_WORKERS"] = "1"
    k_results = _run_uniform_k(
        uniform_paths,
        input_view,
        source_results,
        int(selection["selected_history"]),
        args.workers,
    )
    c_result = run_c_view(
        shared, project, uniform_paths.output, input_view, PUBLIC_ALL_PROTOCOL
    )
    w_result = run_w_view(
        shared, project, uniform_paths.output, input_view, PUBLIC_ALL_PROTOCOL
    )
    a_result = run_a_view(
        shared, project, uniform_paths.output, dynamic_view, PUBLIC_ALL_PROTOCOL
    )
    statuses = {
        "K": [value.get("status") for value in k_results],
        "C": c_result.get("status"),
        "W": w_result.get("status"),
        "A": a_result.get("status"),
    }
    if not all(value == "PASS" for value in statuses["K"]) or any(
        statuses[stage] != "PASS" for stage in ("C", "W", "A")
    ):
        raise RuntimeError(f"uniform native stage failure: {statuses}")

    _freeze_common_support(uniform_paths, dynamic_view)
    _clone_multiscale_task(source_paths, multiscale_paths, input_view, dynamic_view)
    multiscale_counts = _candidate_config_count(
        multiscale_paths.output, dynamic_view
    )
    uniform_counts = _candidate_config_count(uniform_paths.output, dynamic_view)
    multiscale_attempts = _candidate_fit_attempts(
        multiscale_paths.output, dynamic_view
    )
    uniform_attempts = _candidate_fit_attempts(uniform_paths.output, dynamic_view)
    shared_profile_fits = sum(
        len(value["profile_fold_losses"]) for value in source_results
    )
    uniform_counts["shared_scale_search"] = shared_profile_fits
    multiscale_counts["shared_scale_search"] = shared_profile_fits
    registered_cap = max(uniform_attempts["total"], multiscale_attempts["total"])
    budget = {
        "status": "FROZEN_BEFORE_TEST_ACCESS",
        "budget_definition": "candidate_fit_attempts_including_inner_folds",
        "candidate_fit_budget_scope": "K_C_DeltaW_A_development_selection",
        "fair_budget": registered_cap,
        "B_uniform_max": registered_cap,
        "B_multiscale_max": registered_cap,
        "uniform_candidate_configurations_executed": uniform_counts,
        "multiscale_candidate_configurations_executed": multiscale_counts,
        "uniform_candidate_fit_attempts": uniform_attempts,
        "multiscale_candidate_fit_attempts": multiscale_attempts,
        "shared_profile_screen_reused_without_refit": True,
        "duplicate_candidate_padding": False,
        "test_accessed": False,
    }
    _write_json(task_root / "budget_manifest.json", budget)

    seals = {}
    for arm, paths in (("uniform", uniform_paths), ("multiscale", multiscale_paths)):
        selection_seal = {
            "status": "SELECTION_FROZEN",
            "arm": arm,
            "task": args.task,
            "test_accessed": False,
            "budget_manifest_sha256": _sha256(task_root / "budget_manifest.json"),
        }
        _write_json(task_root / arm / "freeze" / "SELECTION_FREEZE.json", selection_seal)
        checkpoint = fit_prism_checkpoint_for_view(
            paths, dynamic_view, task_root / arm / "checkpoints"
        )
        seal = {
            "status": "CHECKPOINTS_SEALED",
            "arm": arm,
            "checkpoint": checkpoint,
            "test_accessed": False,
        }
        _write_json(task_root / arm / "freeze" / "CHECKPOINTS_SEALED.json", seal)
        seals[arm] = seal

    result = {
        "status": "READY_FOR_ISOLATED_TEST_INFERENCE",
        "task": args.task,
        "dataset": TASKS[args.task]["dataset"],
        "uniform_history": selection["selected_history"],
        "stage_statuses": statuses,
        "seals": seals,
        "wall_seconds": time.time() - started,
        "python": platform.python_version(),
        "test_accessed": False,
    }
    _write_json(task_root / "DEVELOPMENT_COMPLETE.json", result)
    return result


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError(f"{INFERENCE_ONLY_ENV}=1 is required")
    project = args.project.resolve()
    shared = args.shared.resolve()
    task_root = args.run_root.resolve() / args.task
    _, dynamic_view = _views(shared, args.task)
    paths = PublicAllPaths(project, shared, task_root / args.arm)
    checkpoint_root = task_root / args.arm / "checkpoints"
    checkpoint_dir = next((checkpoint_root / "prism").iterdir())
    reload_audit = verify_prism_checkpoint_reload(checkpoint_dir)
    records = predict_prism_checkpoint_for_view(
        paths, dynamic_view, checkpoint_root, split="test"
    )
    result = {
        "status": "PASS",
        "task": args.task,
        "arm": args.arm,
        "reload_audit": reload_audit,
        "records": records,
        "fit_called_in_inference": False,
        "test_accessed": True,
    }
    _write_json(task_root / args.arm / "inference" / "INFERENCE_COMPLETE.json", result)
    return result


def _inference_subprocess(args: argparse.Namespace, arm: str) -> None:
    environment = dict(os.environ)
    environment[INFERENCE_ONLY_ENV] = "1"
    command = [
        sys.executable,
        "-m",
        "experiments.tim_validation.e3_multiscale.runner",
        "infer",
        "--project",
        str(args.project.resolve()),
        "--shared",
        str(args.shared.resolve()),
        "--run-root",
        str(args.run_root.resolve()),
        "--task",
        args.task,
        "--arm",
        arm,
    ]
    subprocess.run(command, cwd=args.project.resolve().parent, env=environment, check=True)


def _full_record(inference: dict[str, Any]) -> dict[str, Any]:
    matches = [
        value
        for value in inference["records"]
        if value.get("model") == FULL_MODEL and value.get("status") == "PASS"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing full PRISM inference record: {len(matches)}")
    return matches[0]


def finalize_task(args: argparse.Namespace) -> dict[str, Any]:
    task_root = args.run_root.resolve() / args.task
    records = {
        arm: _full_record(
            _read(task_root / arm / "inference" / "INFERENCE_COMPLETE.json")
        )
        for arm in ("uniform", "multiscale")
    }
    uniform_rmse = float(records["uniform"]["rmse"])
    multiscale_rmse = float(records["multiscale"]["rmse"])
    result = {
        "status": "COMPLETED",
        "task": args.task,
        "dataset": TASKS[args.task]["dataset"],
        "uniform_history": _read(task_root / "UNIFORM_HISTORY_SELECTION.json")[
            "selected_history"
        ],
        "uniform": records["uniform"],
        "multiscale": records["multiscale"],
        "delta_rmse_uniform_minus_multiscale": uniform_rmse - multiscale_rmse,
        "relative_multiscale_gain": (
            uniform_rmse - multiscale_rmse
        ) / max(abs(uniform_rmse), sys.float_info.epsilon),
        "test_accessed_only_after_both_checkpoint_seals": True,
    }
    _write_json(task_root / "TASK_RESULT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("all", "develop", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--project", type=Path, required=True)
        command.add_argument("--shared", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--scale-aware-run", type=Path)
        command.add_argument("--task", choices=tuple(TASKS), required=True)
        command.add_argument("--workers", type=int, default=4)
    infer = commands.add_parser("infer")
    infer.add_argument("--project", type=Path, required=True)
    infer.add_argument("--shared", type=Path, required=True)
    infer.add_argument("--run-root", type=Path, required=True)
    infer.add_argument("--task", choices=tuple(TASKS), required=True)
    infer.add_argument("--arm", choices=("uniform", "multiscale"), required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "infer":
        run_inference(args)
    elif args.command == "develop":
        if args.scale_aware_run is None:
            raise ValueError("--scale-aware-run is required")
        run_development(args)
    elif args.command == "finalize":
        finalize_task(args)
    else:
        if args.scale_aware_run is None:
            raise ValueError("--scale-aware-run is required")
        run_development(args)
        _inference_subprocess(args, "uniform")
        _inference_subprocess(args, "multiscale")
        finalize_task(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
