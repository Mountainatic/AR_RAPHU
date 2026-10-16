from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))


@dataclass(frozen=True)
class ViewKey:
    head_id: str
    information_set: str
    availability_scenario: str
    proxy_policy: str

    @property
    def relative_root(self) -> Path:
        return Path(
            self.head_id,
            self.information_set,
            self.availability_scenario,
            self.proxy_policy,
        )


def _primary_heads(shared: Path) -> set[str]:
    registry = json.loads((shared / "TASK_REGISTRY.json").read_text(encoding="utf-8"))
    return {head["head_id"] for head in registry["heads"] if head["primary"]}


def _discover_development_views(shared: Path) -> list[ViewKey]:
    primary = _primary_heads(shared)
    views: list[ViewKey] = []
    for train_path in sorted((shared / "sample_ids").glob("*/*/*/*/train.parquet")):
        relative = train_path.relative_to(shared / "sample_ids")
        head_id, information_set, availability, proxy_policy, _ = relative.parts
        if head_id not in primary:
            continue
        if information_set == "input_only" and availability != "record_time":
            continue
        views.append(ViewKey(head_id, information_set, availability, proxy_policy))
    return views


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mse = float(np.mean(np.square(error), dtype=np.float64))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    centered = y_true - float(np.mean(y_true, dtype=np.float64))
    denominator = float(np.sum(np.square(centered), dtype=np.float64))
    r2 = float("nan") if denominator == 0.0 else 1.0 - float(np.sum(np.square(error))) / denominator
    std = float(np.std(y_true, ddof=0))
    nrmse = float("nan") if std == 0.0 else rmse / std
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse}


def _prediction_frame(samples: pd.DataFrame, model: str, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": samples["view_sample_id"].astype(str),
            "base_origin_id": samples["base_origin_id"].astype(str),
            "dataset": samples["dataset"].astype(str),
            "task": samples["task_id"].astype(str),
            "target_head": samples["target_head"].astype(str),
            "split": samples["split"].astype(str),
            "model": model,
            "y_true": samples["y_true"].astype(np.float64),
            "y_pred": np.full(len(samples), value, dtype=np.float64),
            "information_set": samples["information_set"].astype(str),
            "profile_id": samples["entity_id"].astype(str),
            "availability_scenario": samples["availability_scenario"].astype(str),
            "proxy_policy": samples["proxy_policy"].astype(str),
            "seed": np.full(len(samples), -1, dtype=np.int64),
            "dtype": "float64",
            "parameter_count": 1 if model == "MEAN" else 0,
        }
    )


def run_simple_baselines(shared: Path, output: Path) -> dict[str, Any]:
    """Run C2 analytic development baselines without opening test/OOD lockboxes."""

    validation = json.loads((shared / "C1_VALIDATION.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("failures"):
        raise RuntimeError("C1 independent validation is not PASS")
    lockbox = json.loads((shared / "LOCKBOX.json").read_text(encoding="utf-8"))
    if lockbox.get("protocol_frozen") is not False:
        raise RuntimeError("expected unopened C1 lockbox with protocol_frozen=false")

    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    prediction_root = output / "PREDICTIONS" / "C2_SIMPLE_DEVELOPMENT"
    metric_rows: list[dict[str, Any]] = []
    prediction_files: list[dict[str, Any]] = []

    for view in _discover_development_views(shared):
        source = shared / "sample_ids" / view.relative_root
        train = pd.read_parquet(source / "train.parquet")
        validation_samples = pd.read_parquet(source / "validation.parquet")
        train_mean = float(np.mean(train["y_true"].to_numpy(dtype=np.float64), dtype=np.float64))
        models = [("MEAN", train_mean)]
        if view.information_set == "dynamic":
            models.append(("PERSISTENCE", 0.0))
        for model, prediction in models:
            frame = _prediction_frame(validation_samples, model, prediction)
            relative = view.relative_root / f"{model}.parquet"
            destination = prediction_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(destination, index=False, compression="zstd")
            metric_rows.append(
                {
                    "status": "PASS",
                    "stage": "C2_SIMPLE_DEVELOPMENT",
                    "dataset": str(validation_samples["dataset"].iloc[0]),
                    "task": str(validation_samples["task_id"].iloc[0]),
                    "target_head": view.head_id,
                    "split": "validation",
                    "model": model,
                    "information_set": view.information_set,
                    "availability_scenario": view.availability_scenario,
                    "proxy_policy": view.proxy_policy,
                    "rows": len(frame),
                    "fit_partition": "train_only",
                    "test_accessed": False,
                    "dtype": "float64",
                    "parameter_count": 1 if model == "MEAN" else 0,
                    **_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()),
                }
            )
            prediction_files.append(
                {
                    "path": str(destination.relative_to(output)),
                    "rows": len(frame),
                    "sha256": _sha256(destination),
                }
            )

    for model, reason in (
        ("SEASONAL_PERSISTENCE", "NO_TRAIN_ONLY_REGISTERED_PERIOD"),
        ("LOCAL_LINEAR_TREND", "UNRESOLVED_TREND_WINDOW_GATE"),
    ):
        metric_rows.append(
            {
                "status": "NOT_APPLICABLE" if model.startswith("SEASONAL") else "BLOCKED",
                "stage": "C2_SIMPLE_DEVELOPMENT",
                "model": model,
                "reason": reason,
                "test_accessed": False,
                "dtype": "float64",
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = output / "SIMPLE_BASELINES_DEVELOPMENT.csv"
    metrics.to_csv(metrics_path, index=False)
    manifest = {
        "status": "PASS",
        "scope": "PRIMARY_HEADS_DEVELOPMENT_ONLY",
        "test_accessed": False,
        "c1_registry_sha256": validation["registry_sha256"],
        "metric_rows": len(metrics),
        "prediction_files": prediction_files,
        "metrics_sha256": _sha256(metrics_path),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output / "C2_SIMPLE_MANIFEST.json", manifest)
    return manifest
