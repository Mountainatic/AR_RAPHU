"""Gaussian N1 robustness with frozen PRISM checkpoints.

Noise is injected once per raw aligned measurement row and channel before any
history construction.  Recreating a case with the same task/seed uses the same
standard-normal realization for every alpha, making the levels nested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
import zlib
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd

from prism_benchmark.portable_checkpoints import INFERENCE_ONLY_ENV
from prism_benchmark.stagewise_ablation_runner import infer_checkpoints, matching_views
from prism_benchmark.v211_public_all_config import PublicAllPaths


FULL_MODEL = "PRISM_V2_1_1_PHYSICS_FIRST"
IDENTIFIERS = {"entity_id", "row_in_entity"}
DEFAULT_ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_registry_hash(shared: Path) -> tuple[str, str]:
    registry = shared / "SAMPLE_ID_REGISTRY.json"
    if registry.is_file():
        return _sha256(registry), "SAMPLE_ID_REGISTRY.json"
    paths = sorted((shared / "sample_ids").rglob("*.parquet"))
    if not paths:
        raise RuntimeError("STOP_E6_NO_SAMPLE_ID_ARTIFACTS")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(shared).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest(), "COMPOSITE_SAMPLE_PARQUET_PATH_AND_SHA256"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _hardlink_copy(source: str, destination: str) -> str:
    source_path = Path(source)
    if source_path.is_symlink() or not source_path.is_file():
        raise RuntimeError(f"STOP_E6_UNSAFE_SHARED_FILE:{source_path}")
    os.link(source_path.resolve(), destination)
    return destination


def _stable_channel_seed(task: str, seed: int, channel: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [int(seed), zlib.crc32(task.encode()), zlib.crc32(channel.encode())]
    )


def outer_train_sigma(
    clean_shared: Path, dataset: str, target: str
) -> tuple[dict[str, float], str]:
    path = clean_shared / "base_data" / dataset / "train.parquet"
    frame = pd.read_parquet(path)
    channels = [
        column for column in frame.columns
        if column not in IDENTIFIERS and column != target
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    sigma = {
        column: float(np.nanstd(frame[column].to_numpy(dtype=np.float64), ddof=0))
        for column in channels
    }
    invalid = [column for column, value in sigma.items() if not math.isfinite(value) or value <= 0]
    if invalid:
        raise RuntimeError(f"STOP_E6_INVALID_OUTER_TRAIN_SIGMA:{invalid}")
    return sigma, _sha256(path)


def perturb_gaussian_process_only(
    frame: pd.DataFrame,
    *,
    task: str,
    seed: int,
    alpha: float,
    sigma: dict[str, float],
) -> pd.DataFrame:
    result = frame.copy()
    for channel, scale in sigma.items():
        values = frame[channel].to_numpy(dtype=np.float64, copy=True)
        finite = np.isfinite(values)
        generator = np.random.default_rng(_stable_channel_seed(task, seed, channel))
        epsilon = generator.standard_normal(len(values))
        values[finite] += float(alpha) * float(scale) * epsilon[finite]
        result[channel] = values
    return result


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
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"REFUSING_EXISTING_E6_CASE_SHARED:{destination}")
    shutil.copytree(clean_shared, destination, copy_function=_hardlink_copy)
    source = clean_shared / "base_data" / dataset / "test.parquet"
    target_path = destination / "base_data" / dataset / "test.parquet"
    clean = pd.read_parquet(source)
    perturbed = perturb_gaussian_process_only(
        clean, task=task, seed=seed, alpha=alpha, sigma=sigma
    )
    temporary = target_path.with_name(f".{target_path.name}.e6.tmp")
    perturbed.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target_path)
    if not np.array_equal(
        clean[target].to_numpy(), perturbed[target].to_numpy(), equal_nan=True
    ):
        raise AssertionError("STOP_E6_CLEAN_TARGET_CHANGED")
    missing_preserved = all(
        np.array_equal(clean[channel].isna().to_numpy(), perturbed[channel].isna().to_numpy())
        for channel in sigma
    )
    if not missing_preserved:
        raise AssertionError("STOP_E6_MISSINGNESS_CHANGED")
    return {
        "clean_test_sha256": _sha256(source),
        "perturbed_test_sha256": _sha256(target_path),
        "rows": len(clean),
        "target_clean": True,
        "missingness_preserved": True,
        "noise_injection_level": "RAW_ALIGNED_MEASUREMENT_BEFORE_HISTORY_CONSTRUCTION",
    }


def _remove_case_work(path: Path, run_root: Path) -> None:
    resolved = path.resolve()
    work_root = (run_root / "_work").resolve()
    if not resolved.is_relative_to(work_root) or resolved == work_root:
        raise RuntimeError(f"STOP_E6_UNSAFE_WORK_CLEANUP:{resolved}")
    shutil.rmtree(resolved)


def _full_record(result: dict[str, Any]) -> dict[str, Any]:
    records = [
        value for value in result["records"]
        if value.get("model") == FULL_MODEL and value.get("status") == "PASS"
    ]
    if len(records) != 1:
        raise RuntimeError(f"STOP_E6_FULL_MODEL_RECORD_COUNT:{len(records)}")
    return records[0]


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def run_n1(args: argparse.Namespace) -> dict[str, Any]:
    clean_shared = args.clean_shared.resolve(strict=True)
    selection_root = args.selection_root.resolve(strict=True)
    checkpoint_root = args.checkpoint_root.resolve(strict=True)
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    sigma, train_hash = outer_train_sigma(clean_shared, args.dataset, args.target)
    sigma_hash = hashlib.sha256(
        json.dumps(sigma, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    alphas = sorted(set(args.alpha))
    if 0.0 not in alphas:
        raise RuntimeError("E6 N1 requires alpha=0 for paired degradation")
    registry_hash, registry_hash_contract = _sample_registry_hash(clean_shared)
    manifest = {
        "status": "FROZEN_BEFORE_PERTURBED_TEST_ACCESS",
        "mode": "N1_FROZEN_MODEL",
        "task": args.task,
        "rod": args.rod,
        "perturbation": "GAUSSIAN_PROCESS_ONLY",
        "alpha": alphas,
        "seeds": seeds,
        "nested_noise_key": "task/seed/channel plus immutable test row order",
        "sigma_source": "outer_train_only",
        "outer_train_sha256": train_hash,
        "sigma": sigma,
        "sigma_sha256": sigma_hash,
        "target_reference": "clean",
        "structure_evaluated": False,
        "checkpoint_root": str(checkpoint_root),
        "clean_shared": str(clean_shared),
        "clean_registry_sha256": registry_hash,
        "clean_registry_hash_contract": registry_hash_contract,
    }
    _write_json(run_root / "perturbation_manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    os.environ[INFERENCE_ONLY_ENV] = "1"
    for seed in seeds:
        for alpha in alphas:
            label = str(alpha).replace(".", "p")
            case_root = run_root / "_work" / f"seed_{seed}" / f"alpha_{label}"
            case_shared = case_root / "shared"
            prediction_root = case_root / "predictions"
            case_result_path = run_root / "cases" / f"seed_{seed}_alpha_{label}.json"
            if case_result_path.is_file():
                rows.append(_read_case_row(case_result_path))
                continue
            if case_root.exists():
                _remove_case_work(case_root, run_root)
            started = time.time()
            materialization = _materialize_case(
                clean_shared,
                case_shared,
                task=args.task,
                dataset=args.dataset,
                target=args.target,
                seed=seed,
                alpha=alpha,
                sigma=sigma,
            )
            paths = PublicAllPaths(args.project.resolve(), case_shared, selection_root)
            views = matching_views(
                paths,
                args.head_id,
                information_set=args.information_set,
                availability_scenario=args.availability_scenario,
                proxy_policy=args.proxy_policy,
            )
            result = infer_checkpoints(paths, checkpoint_root, prediction_root, views)
            record = _full_record(result)
            row = {
                "status": "COMPLETED", "mode": "N1", "task": args.task,
                "rod": args.rod, "perturbation": "gaussian_process_only",
                "seed": seed, "alpha": alpha, "rows": record["rows"],
                "RMSE": record["rmse"], "MAE": record["mae"],
                "R2": record["r2_level_reconstructed"],
                "delta_RMSE": record["rmse_delta"], "delta_R2": record["r2_delta"],
                "persistence_skill": record["persistence_skill"],
                "support_id": record["scoring_support_hash"],
                "sample_id_order_hash": record["sample_id_order_hash"],
                "checkpoint_hash": record["checkpoint_hash"],
                "fit_called_in_inference": False,
                "structure_evaluated": False,
                "elapsed_seconds": time.time() - started,
                **materialization,
            }
            _write_json(case_result_path, row)
            rows.append(row)
            _remove_case_work(case_root, run_root)

    clean_by_seed = {int(row["seed"]): row for row in rows if float(row["alpha"]) == 0.0}
    for row in rows:
        clean = clean_by_seed[int(row["seed"])]
        row["relative_RMSE_degradation"] = (
            float(row["RMSE"]) - float(clean["RMSE"])
        ) / float(clean["RMSE"])
        row["R2_change"] = float(row["R2"]) - float(clean["R2"])
    _write_csv(run_root / "per_seed.csv", rows)
    aggregate = []
    for alpha in alphas:
        selected = [row for row in rows if float(row["alpha"]) == alpha]
        for metric in ("RMSE", "MAE", "R2", "delta_RMSE", "delta_R2", "persistence_skill", "relative_RMSE_degradation", "R2_change"):
            values = [float(row[metric]) for row in selected]
            aggregate.append(
                {
                    "status": "COMPLETED", "task": args.task, "rod": args.rod,
                    "mode": "N1", "perturbation": "gaussian_process_only",
                    "alpha": alpha, "metric": metric, "seeds": len(values),
                    "mean": mean(values), "median": median(values),
                    "Q1": _quantile(values, 0.25), "Q3": _quantile(values, 0.75),
                    "minimum": min(values), "maximum": max(values),
                }
            )
    _write_csv(run_root / "aggregate.csv", aggregate)
    final = {
        "status": "COMPLETED", "mode": "N1_FROZEN_MODEL",
        "task": args.task, "rod": args.rod, "seeds": len(seeds),
        "alpha": alphas, "cases": len(rows), "structure_evaluated": False,
        "fit_called_in_inference": False,
    }
    _write_json(run_root / "N1_COMPLETE.json", final)
    return final


def _read_case_row(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETED" or value.get("fit_called_in_inference") is not False:
        raise RuntimeError(f"STOP_E6_INVALID_EXISTING_CASE:{path}")
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--project", type=Path, required=True)
    value.add_argument("--clean-shared", type=Path, required=True)
    value.add_argument("--selection-root", type=Path, required=True)
    value.add_argument("--checkpoint-root", type=Path, required=True)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--task", required=True)
    value.add_argument("--rod")
    value.add_argument("--dataset", required=True)
    value.add_argument("--target", required=True)
    value.add_argument("--head-id", required=True)
    value.add_argument("--information-set", default="dynamic")
    value.add_argument("--availability-scenario", default="record_time")
    value.add_argument("--proxy-policy", default="primary")
    value.add_argument("--seeds", type=int, default=30)
    value.add_argument("--seed-start", type=int, default=20260910)
    value.add_argument("--alpha", type=float, action="append", default=[])
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.alpha:
        args.alpha = list(DEFAULT_ALPHAS)
    print(json.dumps(run_n1(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
