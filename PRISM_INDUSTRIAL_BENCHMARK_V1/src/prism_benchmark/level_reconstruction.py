"""Reporting-only reconstruction of registered level targets from deltas.

The benchmark trains and materializes change-target predictions.  This module
keeps the reconstruction and metric identities in one place so a reporting
pass cannot accidentally alter model selection or sample support.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

import numpy as np


R2_LEVEL_RECONSTRUCTED = "R2_LEVEL_RECONSTRUCTED"
R2_DELTA = "R2_DELTA"


def _as_float_array(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _paired_arrays(
    delta_true: Iterable[float],
    delta_pred: Iterable[float],
    current_level: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = _as_float_array(delta_true, "delta_true")
    prediction = _as_float_array(delta_pred, "delta_pred")
    current = _as_float_array(current_level, "current_level")
    if not (len(truth) == len(prediction) == len(current)):
        raise ValueError("delta and current-level arrays must have equal length")
    if not len(truth):
        raise ValueError("metric reconstruction requires at least one row")
    return truth, prediction, current


def reconstruct_registered_level(
    current_level: Iterable[float],
    delta_pred: Iterable[float],
) -> np.ndarray:
    """Add a frozen change prediction to its known current target level."""

    current = _as_float_array(current_level, "current_level")
    prediction = _as_float_array(delta_pred, "delta_pred")
    if len(current) != len(prediction):
        raise ValueError("current_level and delta_pred must have equal length")
    return current + prediction


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mse = float(np.mean(np.square(error), dtype=np.float64))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    denominator = float(
        np.sum(np.square(y_true - np.mean(y_true, dtype=np.float64)), dtype=np.float64)
    )
    r2 = float(1.0 - np.sum(np.square(error), dtype=np.float64) / denominator) if denominator else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def persistence_level_metrics(
    future_level_true: Iterable[float],
    current_level: Iterable[float],
) -> dict[str, float | str]:
    """Evaluate the control-oriented persistence forecast (delta = 0)."""

    truth = _as_float_array(future_level_true, "future_level_true")
    current = _as_float_array(current_level, "current_level")
    if len(truth) != len(current):
        raise ValueError("future_level_true and current_level must have equal length")
    metrics = _regression_metrics(truth, current)
    return {
        "r2_level_persistence": metrics["r2"],
        "mse_persistence": metrics["mse"],
        "rmse_persistence": metrics["rmse"],
        "mae_persistence": metrics["mae"],
    }


def metric_bundle_delta_and_level(
    delta_true: Iterable[float],
    delta_pred: Iterable[float],
    current_level: Iterable[float],
) -> dict[str, float | str]:
    """Return both representations while asserting their shared error identity."""

    truth, prediction, current = _paired_arrays(delta_true, delta_pred, current_level)
    future_truth = current + truth
    future_prediction = reconstruct_registered_level(current, prediction)
    delta_metrics = _regression_metrics(truth, prediction)
    level_metrics = _regression_metrics(future_truth, future_prediction)
    persistence = persistence_level_metrics(future_truth, current)
    residual = truth - prediction
    level_residual = future_truth - future_prediction
    max_abs_error = float(np.max(np.abs(residual - level_residual)))
    tolerance = 1e-10
    if max_abs_error > tolerance:
        raise AssertionError("STOP_LEVEL_RECONSTRUCTION_IDENTITY_FAILED")
    for left, right, name in (
        (delta_metrics["mse"], level_metrics["mse"], "MSE"),
        (delta_metrics["rmse"], level_metrics["rmse"], "RMSE"),
        (delta_metrics["mae"], level_metrics["mae"], "MAE"),
    ):
        if not np.isclose(left, right, rtol=tolerance, atol=tolerance):
            raise AssertionError(f"STOP_LEVEL_RECONSTRUCTION_{name}_IDENTITY_FAILED")
    persistence_mse = float(persistence["mse_persistence"])
    model_mse = float(level_metrics["mse"])
    skill: float | str = (
        "NOT_DEFINED_ZERO_PERSISTENCE_ERROR"
        if persistence_mse == 0.0
        else 1.0 - model_mse / persistence_mse
    )
    variance_delta = float(np.var(truth, dtype=np.float64))
    variance_level = float(np.var(future_truth, dtype=np.float64))
    variance_ratio = (
        float("nan") if variance_delta == 0.0 else variance_level / variance_delta
    )
    return {
        "r2_level_reconstructed": level_metrics["r2"],
        "r2_delta": delta_metrics["r2"],
        "mse": level_metrics["mse"],
        "rmse": level_metrics["rmse"],
        "mae": level_metrics["mae"],
        "mse_delta": delta_metrics["mse"],
        "rmse_delta": delta_metrics["rmse"],
        "mae_delta": delta_metrics["mae"],
        "r2_level_persistence": persistence["r2_level_persistence"],
        "persistence_skill": skill,
        "std_level_target": float(np.std(future_truth, dtype=np.float64)),
        "std_delta_target": float(np.std(truth, dtype=np.float64)),
        "variance_ratio": variance_ratio,
        "future_level_true": future_truth,
        "future_level_pred": future_prediction,
    }


def support_hash(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sample_ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def reconstruction_identity_audit(
    delta_true: Iterable[float],
    delta_pred: Iterable[float],
    current_level: Iterable[float],
) -> dict[str, Any]:
    """Return a serializable audit without changing any prediction values."""

    result = metric_bundle_delta_and_level(delta_true, delta_pred, current_level)
    return {
        "status": "PASS",
        "model_retrained": False,
        "model_reselected": False,
        "hyperparameters_changed": False,
        "sample_support_changed": False,
        "mse_identity_max_abs_error": abs(float(result["mse"] - result["mse_delta"])),
        "rmse_identity_max_abs_error": abs(float(result["rmse"] - result["rmse_delta"])),
        "mae_identity_max_abs_error": abs(float(result["mae"] - result["mae_delta"])),
        "r2_primary_reporting": R2_LEVEL_RECONSTRUCTED,
        "r2_secondary_reporting": R2_DELTA,
    }
