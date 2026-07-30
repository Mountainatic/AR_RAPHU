"""Physical-window baselines and matched/raw autoregressive features."""

from __future__ import annotations

import numpy as np

from .multiresolution_lags import (
    expand_lag_blocks,
    lag_block_matrix,
)
from .resampling import causal_block_average
from .targets import TargetRows
from .timebase import Timebase


def persistence_prediction(rows: TargetRows) -> np.ndarray:
    return np.zeros_like(rows.target)


def mean_drift_prediction(
    train_target: np.ndarray, samples: int
) -> np.ndarray:
    return np.full(samples, float(np.mean(train_target)), dtype=np.float64)


def local_trend_prediction(
    target: np.ndarray,
    rows: TargetRows,
) -> np.ndarray:
    """Extrapolate the change between adjacent past target windows."""

    origins = rows.origins
    window = rows.window_samples
    previous = causal_block_average(
        target,
        origins,
        start_samples=window,
        stop_samples=2 * window,
    )
    recent_change = rows.current_mean - previous
    center_separation = rows.horizon_samples + rows.window_samples
    return recent_change * (center_separation / rows.window_samples)


def _causal_mean_series(values: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(data)))
    output = np.empty_like(data)
    for index in range(len(data)):
        start = max(0, index - window + 1)
        output[index] = (
            cumulative[index + 1] - cumulative[start]
        ) / (index - start + 1)
    return output


def scale_matched_ar_features(
    target: np.ndarray,
    rows: TargetRows,
    *,
    timebase: Timebase,
    cadence_sec: float,
    history_min: float,
) -> np.ndarray:
    cadence = timebase.cadence_step(cadence_sec)
    count = int(
        np.ceil(
            timebase.samples_for_minutes(history_min) / cadence
        )
    )
    smoothed = _causal_mean_series(target, rows.window_samples)
    offsets = np.arange(count, dtype=np.int64) * cadence
    indices = rows.origins[:, None] - offsets[None, :]
    if int(indices.min()) < 0:
        raise ValueError("SCALE_AR_HISTORY_OUT_OF_RANGE")
    matrix = smoothed[indices]
    return matrix - rows.current_mean[:, None]


def raw_history_ar_features(
    target: np.ndarray,
    rows: TargetRows,
    *,
    timebase: Timebase,
    history_min: float,
    block_specification,
) -> np.ndarray:
    blocks = expand_lag_blocks(
        block_specification,
        history_min=history_min,
    )
    matrix = lag_block_matrix(
        target,
        rows.origins,
        blocks,
        timebase=timebase,
    )
    return matrix - rows.current_mean[:, None]
