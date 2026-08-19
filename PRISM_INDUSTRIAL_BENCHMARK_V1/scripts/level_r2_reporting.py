"""Reporting-only Level-R2 reconstruction for the six-dataset extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import BaseAccessor, load_heads, sha256_file
from prism_benchmark.level_reconstruction import (
    metric_bundle_delta_and_level,
    support_hash,
)
from prism_benchmark.six_dataset_reporting import PredictionSpec, prediction_specs
from prism_benchmark.stage0 import write_json


def _shared_root(
    run_root: Path,
    public_root: Path,
    spec: PredictionSpec,
) -> Path:
    if spec.scope == "public5":
        return public_root / "shared"
    if not spec.direction:
        raise ValueError("CZ prediction is missing direction")
    return run_root / "shared" / spec.direction


def _target_and_window(
    shared: Path,
    spec: PredictionSpec,
) -> tuple[str, int, str]:
    if spec.scope == "public5":
        for head in load_heads(shared, primary_only=False):
            if head.head_id == spec.target_head:
                return head.target, int(head.w0_steps), head.dataset
        raise KeyError(spec.target_head)
    return "crystal_diameter", 12, "cz_czochralski"


def _metadata(shared: Path, spec: PredictionSpec) -> pd.DataFrame:
    path = (
        shared
        / "sample_ids"
        / spec.target_head
        / spec.information_set
        / spec.availability_scenario
        / spec.proxy_policy
        / f"{spec.split}.parquet"
    )
    columns = [
        "base_origin_id",
        "view_sample_id",
        "entity_id",
        "origin",
        "current_start",
        "current_stop_exclusive",
    ]
    return pd.read_parquet(path, columns=columns)


def _prediction(spec: PredictionSpec) -> pd.DataFrame:
    return pd.read_parquet(
        spec.path,
        columns=["sample_id", "base_origin_id", "y_true", "y_pred"],
    )


def _current_levels(
    shared: Path,
    spec: PredictionSpec,
    joined: pd.DataFrame,
    target: str,
    dataset: str,
) -> np.ndarray:
    accessor = BaseAccessor(
        shared,
        dataset,
        spec.split,
        [target],
    )
    start = joined["current_start"].to_numpy(dtype=np.int64)
    stop = joined["current_stop_exclusive"].to_numpy(dtype=np.int64)
    widths = stop - start
    if len(widths) == 0 or not np.all(widths == widths[0]):
        raise ValueError("registered current window has inconsistent widths")
    indices = start[:, None] + np.arange(int(widths[0]), dtype=np.int64)[None, :]
    return accessor.gather(joined, [target], indices).reshape(len(joined), -1).mean(
        axis=1,
        dtype=np.float64,
    )


def collect_level_r2(
    run_root: Path,
    public_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = prediction_specs(run_root, public_root=public_root)
    for spec in specs:
        prediction = _prediction(spec)
        if prediction.empty:
            continue
        shared = _shared_root(run_root, public_root, spec)
        metadata = _metadata(shared, spec)
        joined = prediction.merge(metadata, on="base_origin_id", how="inner")
        if len(joined) != len(prediction):
            raise RuntimeError(f"prediction metadata mismatch: {spec.path}")
        target, _w0, dataset = _target_and_window(shared, spec)
        current = _current_levels(shared, spec, joined, target, dataset)
        bundle = metric_bundle_delta_and_level(
            joined["y_true"].to_numpy(dtype=np.float64),
            joined["y_pred"].to_numpy(dtype=np.float64),
            current,
        )
        sample_ids = joined["sample_id"].astype(str).tolist()
        rows.append(
            {
                "scope": spec.scope,
                "direction": spec.direction or "",
                "dataset": dataset,
                "target_head": spec.target_head,
                "information_set": spec.information_set,
                "availability_scenario": spec.availability_scenario,
                "proxy_policy": spec.proxy_policy,
                "split": spec.split,
                "model": spec.model,
                "rows": int(len(joined)),
                "sample_support_hash": support_hash(sample_ids),
                "prediction_path": str(spec.path),
                "prediction_sha256": sha256_file(spec.path),
                "model_retrained": False,
                "model_reselected": False,
                "sample_support_changed": False,
                "r2_level_reporting": "R2_LEVEL_RECONSTRUCTED",
                "r2_delta": bundle["r2_delta"],
                "r2_level_reconstructed": bundle["r2_level_reconstructed"],
                "mse": bundle["mse"],
                "rmse": bundle["rmse"],
                "mae": bundle["mae"],
                "r2_level_persistence": bundle["r2_level_persistence"],
                "persistence_skill": bundle["persistence_skill"],
                "std_level_target": bundle["std_level_target"],
                "std_delta_target": bundle["std_delta_target"],
                "variance_ratio": bundle["variance_ratio"],
                "mse_identity_max_abs_error": abs(bundle["mse"] - bundle["mse_delta"]),
                "rmse_identity_max_abs_error": abs(bundle["rmse"] - bundle["rmse_delta"]),
                "mae_identity_max_abs_error": abs(bundle["mae"] - bundle["mae_delta"]),
            }
        )
        del prediction, metadata, joined, current
    if not rows:
        raise RuntimeError("no prediction rows available for Level-R2 reporting")
    frame = pd.DataFrame(rows).sort_values(
        ["scope", "direction", "target_head", "information_set", "split", "model"]
    )
    audit = {
        "status": "PASS",
        "reporting_only": True,
        "model_retrained": False,
        "model_reselected": False,
        "test_rerun": False,
        "rows": int(len(frame)),
        "identity_max_mse": float(frame["mse_identity_max_abs_error"].max()),
        "identity_max_rmse": float(frame["rmse_identity_max_abs_error"].max()),
        "identity_max_mae": float(frame["mae_identity_max_abs_error"].max()),
    }
    return frame.reset_index(drop=True), audit


def write_level_r2_outputs(run_root: Path, public_root: Path) -> dict[str, Any]:
    final = run_root / "final"
    final.mkdir(parents=True, exist_ok=True)
    frame, audit = collect_level_r2(run_root, public_root)
    path = final / "SIX_DATASET_LEVEL_R2_METRICS.csv"
    frame.to_csv(path, index=False)
    frame.to_csv(final / "PUBLIC_ALL_LEVEL_R2_METRICS.csv", index=False)
    audit["metrics_path"] = str(path)
    audit["metrics_sha256"] = sha256_file(path)
    write_json(final / "LEVEL_R2_RECONSTRUCTION_AUDIT.json", audit)
    return audit


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_level_r2_outputs(args.run_root, args.public_root)))
