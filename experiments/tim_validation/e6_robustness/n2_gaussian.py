"""Gaussian N2 robustness: perturb development, re-identify, then infer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd

from experiments.tim_validation.common.structure_signature import StructureSignature
from prism_benchmark.cpu_data import input_columns
from prism_benchmark.portable_checkpoints import INFERENCE_ONLY_ENV

from .n1_gaussian import (
    DEFAULT_ALPHAS,
    FULL_MODEL,
    _hardlink_copy,
    _remove_case_work,
    _sample_registry_hash,
    _sha256,
    _stable_channel_seed,
    _write_json,
    outer_train_sigma,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _perturb_split(
    source: Path,
    destination: Path,
    *,
    task: str,
    seed: int,
    alpha: float,
    sigma: dict[str, float],
    realization: str,
    target: str,
) -> dict[str, Any]:
    clean = pd.read_parquet(source)
    perturbed = clean.copy()
    for channel, scale in sigma.items():
        values = clean[channel].to_numpy(dtype=np.float64, copy=True)
        finite = np.isfinite(values)
        sequence = _stable_channel_seed(f"{task}:{realization}", seed, channel)
        epsilon = np.random.default_rng(sequence).standard_normal(len(values))
        values[finite] += float(alpha) * float(scale) * epsilon[finite]
        perturbed[channel] = values
    temporary = destination.with_name(f".{destination.name}.e6.tmp")
    perturbed.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, destination)
    if not np.array_equal(clean[target].to_numpy(), perturbed[target].to_numpy(), equal_nan=True):
        raise AssertionError("STOP_E6_N2_CLEAN_TARGET_CHANGED")
    if any(
        not np.array_equal(clean[channel].isna().to_numpy(), perturbed[channel].isna().to_numpy())
        for channel in sigma
    ):
        raise AssertionError("STOP_E6_N2_MISSINGNESS_CHANGED")
    return {
        "split": source.stem,
        "realization": realization,
        "clean_sha256": _sha256(source),
        "perturbed_sha256": _sha256(destination),
        "rows": len(clean),
    }


def _materialize_case(
    clean_shared: Path,
    destination: Path,
    *,
    task: str,
    dataset: str,
    target: str,
    seed: int,
    alpha: float,
    sigma: dict[str, float],
) -> list[dict[str, Any]]:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"REFUSING_EXISTING_E6_N2_CASE_SHARED:{destination}")
    import shutil

    shutil.copytree(clean_shared, destination, copy_function=_hardlink_copy)
    records = []
    for split in ("train", "validation", "test"):
        source = clean_shared / "base_data" / dataset / f"{split}.parquet"
        if not source.is_file():
            continue
        target_path = destination / "base_data" / dataset / f"{split}.parquet"
        records.append(
            _perturb_split(
                source,
                target_path,
                task=task,
                seed=seed,
                alpha=alpha,
                sigma=sigma,
                realization="test_independent" if split == "test" else "development",
                target=target,
            )
        )
    if not any(record["split"] == "train" for record in records):
        raise RuntimeError("STOP_E6_N2_TRAIN_NOT_PERTURBED")
    if not any(record["split"] == "test" for record in records):
        raise RuntimeError("STOP_E6_N2_TEST_NOT_PERTURBED")
    return records


def _run_stage(
    args: argparse.Namespace,
    stage: str,
    shared: Path,
    selection: Path,
    checkpoint: Path,
    prediction: Path,
    log: Path,
) -> None:
    command = [
        sys.executable,
        str(args.project.resolve() / "scripts" / "run_stagewise_ablation_hybrid.py"),
        stage,
        "--project", str(args.project.resolve()),
        "--shared", str(shared),
        "--selection-run-root", str(selection),
        "--head-id", args.head_id,
        "--information-set", args.information_set,
        "--availability-scenario", args.availability_scenario,
        "--proxy-policy", args.proxy_policy,
    ]
    if stage == "development":
        command.extend(["--workers", str(args.workers), "--per-worker-gib", str(args.per_worker_gib)])
    if stage in {"checkpoint", "infer"}:
        command.extend(["--checkpoint-root", str(checkpoint)])
    if stage == "infer":
        command.extend(["--destination-run-root", str(prediction)])
    environment = dict(os.environ)
    if stage == "infer":
        environment[INFERENCE_ONLY_ENV] = "1"
    else:
        environment.pop(INFERENCE_ONLY_ENV, None)
    completed = subprocess.run(
        command,
        cwd=args.project.resolve().parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"STOP_E6_N2_STAGE_FAILED:{stage}:{completed.returncode}:{log}")


def _full_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        record for record in value["records"]
        if record.get("model") == FULL_MODEL and record.get("status") == "PASS"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"STOP_E6_N2_FULL_RECORD_COUNT:{len(matches)}")
    return matches[0]


def _identity_family(value: Any) -> bool:
    text = str(value).upper()
    return "IDENTITY" in text or "EXACT_ZERO" in text


def _structure(
    args: argparse.Namespace,
    shared: Path,
    checkpoint_root: Path,
    record: dict[str, Any],
    seed: int,
    alpha: float,
) -> StructureSignature:
    checkpoint_dir = next((checkpoint_root / "prism").iterdir())
    state = json.loads((checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8"))
    admitted = [str(channel) for channel in state["physical"]["channels"]]
    all_channels = input_columns(shared, state["task"], args.proxy_policy)
    contracts = {
        str(item["channel"]): item for item in state["physical"]["channel_contracts"]
    }
    histories = {
        channel: int(contracts[channel]["profile"][1]) for channel in admitted
    }
    unique = sorted(set(histories.values()))
    classes = {}
    for channel, history in histories.items():
        if len(unique) <= 1:
            classes[channel] = "single_scale"
        elif history == unique[0]:
            classes[channel] = "fast_scale"
        elif history == unique[-1]:
            classes[channel] = "slow_scale"
        else:
            classes[channel] = "intermediate_scale"
    c_family = state["c_contract"].get("family")
    w_family = state["w_contract"].get("family")
    a_family = state["a_contract"].get("family")
    flags = [bool(admitted), not _identity_family(c_family), not _identity_family(w_family), not _identity_family(a_family)]
    return StructureSignature(
        task=args.task,
        head=args.head_id,
        H=args.H,
        W=args.W,
        outer_fold="registered_outer_test",
        run=f"E6_N2_alpha_{alpha}",
        rod=args.rod,
        seed=seed,
        candidate_universe="standard",
        model_variant=FULL_MODEL,
        admitted_channels=admitted,
        rejected_channels=sorted(set(all_channels) - set(admitted)),
        selected_profile_by_channel={channel: contracts[channel]["profile"] for channel in admitted},
        selected_history_by_channel=histories,
        selected_scale_class_by_channel=classes,
        K_admitted=flags[0], C_admitted=flags[1], W_admitted=flags[2], A_admitted=flags[3],
        selected_K_candidate={channel: contracts[channel]["k_contract"].get("family") for channel in admitted},
        selected_C_candidate=c_family,
        selected_W_candidate=w_family,
        selected_A_candidate=a_family,
        parameter_count=record.get("parameter_count"),
        active_channel_count=len(admitted),
        active_stage_count=sum(flags),
        RMSE=float(record["rmse"]), MAE=float(record["mae"]),
        R2=float(record["r2_level_reconstructed"]),
        support_id=str(record["scoring_support_hash"]),
        config_hash=_sha256(args.project.resolve() / "configs" / "stagewise_ablation_hybrid_hw_20260904.json"),
        information_set=args.information_set,
        availability_scenario=args.availability_scenario,
        proxy_policy=args.proxy_policy,
    )


def _jaccard(left: set[str], right: set[str]) -> tuple[float, bool]:
    if not left and not right:
        return 1.0, True
    return len(left & right) / len(left | right), False


def _case_label(seed: int, alpha: float) -> str:
    return f"seed_{seed}_alpha_{str(alpha).replace('.', 'p')}"


def run_n2(args: argparse.Namespace) -> dict[str, Any]:
    clean_shared = args.clean_shared.resolve(strict=True)
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    sigma, train_hash = outer_train_sigma(clean_shared, args.dataset, args.target)
    registry_hash, registry_contract = _sample_registry_hash(clean_shared)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    alphas = sorted(set(args.alpha))
    if 0.0 not in alphas:
        raise RuntimeError("E6 N2 requires alpha=0")
    _write_json(
        run_root / "perturbation_manifest.json",
        {
            "status": "FROZEN_BEFORE_PERTURBED_DEVELOPMENT_OR_TEST_ACCESS",
            "mode": "N2_REIDENTIFICATION", "task": args.task, "rod": args.rod,
            "perturbation": "GAUSSIAN_PROCESS_ONLY", "alpha": alphas, "seeds": seeds,
            "sigma_source": "outer_train_only", "sigma": sigma,
            "outer_train_sha256": train_hash, "clean_registry_sha256": registry_hash,
            "clean_registry_hash_contract": registry_contract,
            "development_and_test_realizations_independent": True,
            "alpha_levels_nested_within_seed_split_channel": True,
            "target_reference": "clean", "algorithmic_training_seed": "FROZEN_PROTOCOL_DEFAULT",
        },
    )
    rows = []
    signatures: dict[tuple[int, float], dict[str, Any]] = {}
    for seed in seeds:
        for alpha in alphas:
            label = _case_label(seed, alpha)
            saved = run_root / "cases" / label
            result_path = saved / "RESULT.json"
            signature_path = saved / "STRUCTURE_SIGNATURE.json"
            if result_path.is_file() and signature_path.is_file():
                rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                signatures[(seed, alpha)] = json.loads(signature_path.read_text(encoding="utf-8"))
                continue
            case = run_root / "_work" / label
            if case.exists():
                _remove_case_work(case, run_root)
            shared = case / "shared"
            started = time.time()
            materialization = _materialize_case(
                clean_shared, shared, task=args.task, dataset=args.dataset,
                target=args.target, seed=seed, alpha=alpha, sigma=sigma,
            )
            selection = case / "selection"
            checkpoint = case / "checkpoint"
            prediction = case / "prediction"
            for stage in ("development", "freeze-bootstrap", "checkpoint", "infer"):
                _run_stage(
                    args, stage, shared, selection, checkpoint, prediction,
                    saved / "logs" / f"{stage}.log",
                )
            record = _full_record(prediction / "STAGEWISE_INFERENCE_COMPLETE.json")
            signature = _structure(args, shared, checkpoint, record, seed, alpha)
            signature.write(signature_path)
            signature_value = signature.to_dict()
            signatures[(seed, alpha)] = signature_value
            row = {
                "status": "COMPLETED", "mode": "N2", "task": args.task,
                "rod": args.rod, "perturbation": "gaussian_process_only",
                "seed": seed, "alpha": alpha, "RMSE": record["rmse"],
                "MAE": record["mae"], "R2": record["r2_level_reconstructed"],
                "delta_RMSE": record["rmse_delta"], "delta_R2": record["r2_delta"],
                "persistence_skill": record["persistence_skill"],
                "support_id": record["scoring_support_hash"],
                "checkpoint_hash": record["checkpoint_hash"],
                "fit_called_in_inference": False,
                "development_test_realizations_independent": True,
                "materialized_splits": materialization,
                "elapsed_seconds": time.time() - started,
            }
            _write_json(result_path, row)
            rows.append(row)
            _remove_case_work(case, run_root)

    clean_rows = {int(row["seed"]): row for row in rows if float(row["alpha"]) == 0.0}
    structural_rows = []
    for row in rows:
        seed, alpha = int(row["seed"]), float(row["alpha"])
        clean = clean_rows[seed]
        row["relative_RMSE_degradation"] = (float(row["RMSE"]) - float(clean["RMSE"])) / float(clean["RMSE"])
        row["R2_change"] = float(row["R2"]) - float(clean["R2"])
        current_signature = signatures[(seed, alpha)]
        clean_signature = signatures[(seed, 0.0)]
        left = set(clean_signature["admitted_channels"])
        right = set(current_signature["admitted_channels"])
        jaccard, empty_empty = _jaccard(left, right)
        common = sorted(left & right)
        scale_matches = [
            clean_signature["selected_scale_class_by_channel"].get(channel)
            == current_signature["selected_scale_class_by_channel"].get(channel)
            for channel in common
        ]
        clean_vector = [bool(clean_signature[name]) for name in ("K_admitted", "C_admitted", "W_admitted", "A_admitted")]
        current_vector = [bool(current_signature[name]) for name in ("K_admitted", "C_admitted", "W_admitted", "A_admitted")]
        structural_rows.append(
            {
                "status": "COMPLETED", "task": args.task, "rod": args.rod,
                "seed": seed, "alpha": alpha, "channel_jaccard": jaccard,
                "empty_empty_agreement": empty_empty,
                "common_admitted_channels": len(common),
                "scale_agreement": None if not common else sum(scale_matches) / len(scale_matches),
                "scale_status": "NOT_RUN" if not common else "COMPLETED",
                "stage_agreement": sum(a == b for a, b in zip(clean_vector, current_vector)) / 4,
                "full_stage_vector_agreement": clean_vector == current_vector,
                "parameter_count_change": (
                    (current_signature.get("parameter_count") or 0)
                    - (clean_signature.get("parameter_count") or 0)
                ),
            }
        )
    _write_csv(run_root / "per_seed.csv", rows)
    _write_csv(run_root / "channel_stability.csv", structural_rows)
    _write_csv(run_root / "scale_stability.csv", structural_rows)
    _write_csv(run_root / "stage_stability.csv", structural_rows)
    aggregate = []
    for alpha in alphas:
        selected = [row for row in rows if float(row["alpha"]) == alpha]
        structures = [row for row in structural_rows if float(row["alpha"]) == alpha]
        for metric, source in (
            ("RMSE", selected), ("R2", selected),
            ("relative_RMSE_degradation", selected), ("R2_change", selected),
            ("channel_jaccard", structures), ("stage_agreement", structures),
        ):
            values = [float(row[metric]) for row in source]
            aggregate.append(
                {
                    "status": "COMPLETED", "task": args.task, "rod": args.rod,
                    "mode": "N2", "alpha": alpha, "metric": metric,
                    "seeds": len(values), "mean": mean(values), "median": median(values),
                    "Q1": float(np.quantile(values, 0.25)), "Q3": float(np.quantile(values, 0.75)),
                    "minimum": min(values), "maximum": max(values),
                }
            )
    _write_csv(run_root / "aggregate.csv", aggregate)
    final = {
        "status": "COMPLETED", "mode": "N2_REIDENTIFICATION",
        "task": args.task, "rod": args.rod, "seeds": len(seeds),
        "alpha": alphas, "cases": len(rows), "structure_evaluated": True,
    }
    _write_json(run_root / "N2_COMPLETE.json", final)
    return final


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--project", type=Path, required=True)
    value.add_argument("--clean-shared", type=Path, required=True)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--task", required=True)
    value.add_argument("--rod")
    value.add_argument("--dataset", required=True)
    value.add_argument("--target", required=True)
    value.add_argument("--head-id", required=True)
    value.add_argument("--H", type=int, required=True)
    value.add_argument("--W", type=int, required=True)
    value.add_argument("--information-set", default="dynamic")
    value.add_argument("--availability-scenario", default="record_time")
    value.add_argument("--proxy-policy", default="primary")
    value.add_argument("--seeds", type=int, default=10)
    value.add_argument("--seed-start", type=int, default=20260910)
    value.add_argument("--alpha", type=float, action="append", default=[])
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--per-worker-gib", type=float, default=4.0)
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.alpha:
        args.alpha = list(DEFAULT_ALPHAS)
    print(json.dumps(run_n2(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
