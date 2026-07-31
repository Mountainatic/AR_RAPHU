from __future__ import annotations

from typing import Any

import numpy as np

from .io_data import inner_folds
from .solver import one_se_select, ridge_fit


def residual_design(
    residual: np.ndarray,
    origin_raw_index: np.ndarray,
    future_right_raw_index: np.ndarray,
    *,
    candidate: str,
    cadence_sec: float,
    sample_period_sec: float,
    maximum_history_min: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(residual, dtype=np.float64)
    origins = np.asarray(origin_raw_index, dtype=np.int64)
    future_right = np.asarray(future_right_raw_index, dtype=np.int64)
    cadence_raw = int(round(float(cadence_sec) / float(sample_period_sec)))
    maximum_rows = int(round(maximum_history_min * 60.0 / cadence_sec))
    feature_rows: list[np.ndarray] = []
    eligible: list[int] = []
    maximum_source: list[int] = []
    for current in range(len(residual)):
        if not np.isfinite(residual[current]):
            continue
        mature = np.flatnonzero((future_right <= origins[current]) & np.isfinite(residual))
        if not len(mature):
            continue
        last = int(mature[-1])
        history_index = last - np.arange(maximum_rows, dtype=np.int64)
        if history_index[-1] < 0 or not np.all(np.isfinite(residual[history_index])):
            continue
        if np.any(future_right[history_index] > origins[current]):
            raise RuntimeError("MATURED_RESIDUAL_CAUSALITY_FAILED")
        history = residual[history_index]
        if candidate == "A0":
            feature = np.zeros(0, dtype=np.float64)
        elif candidate.startswith("AR:"):
            minutes = float(candidate.split(":", 1)[1])
            rows = int(round(minutes * 60.0 / cadence_sec))
            feature = history[:rows]
        elif candidate.startswith("STATE:"):
            dimension = int(candidate.split(":", 1)[1])
            time_constants = np.geomspace(2.0, 40.0, dimension)
            lag_minutes = np.arange(maximum_rows, dtype=np.float64) * cadence_sec / 60.0
            weights = np.exp(-lag_minutes[:, None] / time_constants[None, :])
            weights /= np.sum(weights, axis=0, keepdims=True)
            feature = history @ weights
        else:
            raise ValueError(candidate)
        feature_rows.append(feature)
        eligible.append(current)
        maximum_source.append(int(future_right[history_index].max()))
    width = 0 if candidate == "A0" else len(feature_rows[0]) if feature_rows else 0
    matrix = np.vstack(feature_rows) if feature_rows and width else np.zeros((len(feature_rows), width), dtype=np.float64)
    return matrix, np.asarray(eligible, dtype=np.int64), np.asarray(maximum_source, dtype=np.int64)


def select_residual_model(
    oof_residual: np.ndarray,
    origin_raw_index: np.ndarray,
    future_right_raw_index: np.ndarray,
    *,
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    purge_raw = int(round(float(config["purge_min"]) * 60.0 / float(protocol["sample_period_sec"])))
    design_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for candidate in config["residual_candidates"]:
        design, eligible, maximum_source = residual_design(
            oof_residual,
            origin_raw_index,
            future_right_raw_index,
            candidate=candidate,
            cadence_sec=float(config["cadence_sec"]),
            sample_period_sec=float(protocol["sample_period_sec"]),
        )
        design_cache[candidate] = (design, eligible, maximum_source)
        if len(eligible) < 100:
            continue
        folds = inner_folds(origin_raw_index[eligible], protocol["inner_folds"], purge_raw_samples=purge_raw)
        ridge_values = [0.0] if candidate == "A0" else config["residual_ridge_grid"]
        for ridge in ridge_values:
            fold_mse: list[float] = []
            for training, validation in folds:
                if candidate == "A0":
                    prediction = np.zeros(len(validation), dtype=np.float64)
                else:
                    coefficient, intercept = ridge_fit(design[training], oof_residual[eligible[training]], float(ridge))
                    prediction = intercept + design[validation] @ coefficient
                error = oof_residual[eligible[validation]] - prediction
                fold_mse.append(float(np.mean(error * error)))
            rows.append({
                "candidate": candidate,
                "ridge": float(ridge),
                "mean_mse": float(np.mean(fold_mse)),
                "se_mse": float(np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))),
                "fold_mse": fold_mse,
                "eligible_rows": int(len(eligible)),
                "maximum_causality_margin_raw": int(np.min(origin_raw_index[eligible] - maximum_source)),
            })
    if not rows:
        raise RuntimeError("NO_RESIDUAL_CANDIDATE_ELIGIBLE")
    selected = one_se_select(rows)
    candidate = str(selected["candidate"])
    design, eligible, _ = design_cache[candidate]
    if candidate == "A0":
        coefficient = np.zeros(0, dtype=np.float64)
        intercept = 0.0
    else:
        coefficient, intercept = ridge_fit(design, oof_residual[eligible], float(selected["ridge"]))
    return {
        "rows": rows,
        "selected": selected,
        "coefficient": coefficient,
        "intercept": intercept,
    }


def apply_residual_model(
    selected: dict[str, Any],
    residual: np.ndarray,
    origin_raw_index: np.ndarray,
    future_right_raw_index: np.ndarray,
    *,
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = str(selected["selected"]["candidate"])
    correction = np.zeros(len(residual), dtype=np.float64)
    design, eligible, maximum_source = residual_design(
        residual,
        origin_raw_index,
        future_right_raw_index,
        candidate=candidate,
        cadence_sec=float(config["cadence_sec"]),
        sample_period_sec=float(protocol["sample_period_sec"]),
    )
    if candidate != "A0":
        correction[eligible] = float(selected["intercept"]) + design @ np.asarray(selected["coefficient"], dtype=np.float64)
    return correction, eligible, maximum_source
