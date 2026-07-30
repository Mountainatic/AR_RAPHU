"""Leakage-safe physical-window target and origin construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segmentation import Segment
from .timebase import Timebase


@dataclass(frozen=True, slots=True)
class TargetRows:
    origins: np.ndarray
    target: np.ndarray
    current_mean: np.ndarray
    future_mean: np.ndarray
    horizon_samples: int
    window_samples: int
    history_samples: int


def _window_means(
    values: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.asarray(values, dtype=np.float64)))
    )
    return (cumulative[right] - cumulative[left]) / (right - left)


def build_target_rows(
    target: np.ndarray,
    segment: Segment,
    *,
    timebase: Timebase,
    cadence_sec: float,
    horizon_min: float,
    target_window_min: float,
    history_min: float,
) -> TargetRows:
    horizon = timebase.samples_for_minutes(horizon_min)
    window = timebase.samples_for_minutes(target_window_min)
    history = timebase.samples_for_minutes(history_min)
    cadence = timebase.cadence_step(cadence_sec)
    first = segment.start + max(history, window) - 1
    last_exclusive = segment.stop - horizon - window + 1
    if first >= last_exclusive:
        raise ValueError("PROFILE_HAS_NO_ELIGIBLE_TARGETS")
    first_aligned = first + ((cadence - (first - segment.start) % cadence) % cadence)
    origins = np.arange(first_aligned, last_exclusive, cadence, dtype=np.int64)
    current_left = origins - window + 1
    current_right = origins + 1
    future_left = origins + horizon
    future_right = future_left + window
    if (
        int(current_left.min()) < segment.start
        or np.any(origins >= future_left)
        or int(future_right.max()) > segment.stop
    ):
        raise AssertionError("TARGET_WINDOW_BOUNDARY_OR_FUTURE_LEAKAGE")
    current = _window_means(target, current_left, current_right)
    future = _window_means(target, future_left, future_right)
    return TargetRows(
        origins=origins,
        target=future - current,
        current_mean=current,
        future_mean=future,
        horizon_samples=horizon,
        window_samples=window,
        history_samples=history,
    )
