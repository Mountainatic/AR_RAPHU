from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.neurobem_manifold_switch.metrics import state_errors
from experiments.neurobem_manifold_switch.prism_adapter import FrozenPrismAdapter
from experiments.neurobem_recursive_stability.core import frame_arrays
from prism_benchmark.neurobem_literature import normalize_quaternion

from .resampling import timestamp_seconds


RATES = (100, 200, 400)
ROUTES = ("PF_KCW", "J_KCW")
HORIZONS_MS = (10, 20, 50, 100, 200, 500)
RATE_KEYS = {100: "hz100_h20", 200: "hz200_h40", 400: "hz400_h80"}


@dataclass(frozen=True)
class CompositionEndpoint:
    prediction: np.ndarray
    finite: bool
    steps: int


def state_norms(reference: np.ndarray, value: np.ndarray) -> dict[str, float]:
    if not np.isfinite(value).all():
        return {key: math.inf for key in ("full_state", "velocity", "attitude", "body_rate")}
    errors = state_errors(reference[None], value[None])
    velocity = float(errors["velocity"][0])
    attitude = float(errors["attitude"][0])
    body_rate = float(errors["body_rate"][0])
    return {
        "full_state": float(np.linalg.norm((velocity, attitude, body_rate))),
        "velocity": velocity,
        "attitude": attitude,
        "body_rate": body_rate,
    }


def exact_time_index(frame: pd.DataFrame, time_seconds: float, rate: int) -> int:
    time = timestamp_seconds(frame)
    index = int(np.argmin(np.abs(time - time_seconds)))
    if abs(float(time[index]) - time_seconds) > 0.1 / rate:
        raise RuntimeError("COMPOSITION_TIME_ALIGNMENT_MISMATCH")
    return index


def registered_anchor_times(
    native: pd.DataFrame,
    *, history_ms: int = 200,
    maximum_horizon_ms: int = 500,
    spacing_ms: int = 500,
    maximum_anchors: int = 16,
) -> list[float]:
    end = float(timestamp_seconds(native)[-1])
    start = history_ms / 1000.0
    stop = end - maximum_horizon_ms / 1000.0
    if stop < start:
        return []
    candidates = np.arange(start, stop + 1e-12, spacing_ms / 1000.0)
    if len(candidates) > maximum_anchors:
        positions = np.linspace(0, len(candidates) - 1, maximum_anchors).round().astype(int)
        candidates = candidates[np.unique(positions)]
    return [float(round(value, 10)) for value in candidates]


def composition_endpoint(
    adapter: FrozenPrismAdapter,
    route: str,
    sampled: pd.DataFrame,
    anchor_time_seconds: float,
    horizon_ms: int,
    common_initial_state: np.ndarray,
    rate: int,
) -> CompositionEndpoint:
    state, control = frame_arrays(sampled)
    anchor = exact_time_index(sampled, anchor_time_seconds, rate)
    history = adapter.history
    if anchor - history + 1 < 0:
        raise RuntimeError("COMPOSITION_HISTORY_UNAVAILABLE")
    predicted_history = state[anchor - history + 1:anchor + 1].copy()
    predicted_history[-1] = common_initial_state
    predicted_history[-1, 3:7] = normalize_quaternion(predicted_history[-1:, 3:7])[0]
    steps = int(round(horizon_ms * rate / 1000.0))
    controls = control[anchor - history + 1:]
    prediction = np.full(10, np.nan)
    for step in range(steps):
        try:
            prediction = adapter.next_state(route, predicted_history, controls[step:step + history])
        except Exception:
            prediction = np.full(10, np.nan)
        if not np.isfinite(prediction).all():
            return CompositionEndpoint(prediction, False, steps)
        prediction[3:7] = normalize_quaternion(prediction[None, 3:7])[0]
        predicted_history = np.vstack((predicted_history[1:], prediction))
    return CompositionEndpoint(prediction, True, steps)


def composition_endpoints(
    adapter: FrozenPrismAdapter,
    route: str,
    sampled: pd.DataFrame,
    anchor_time_seconds: float,
    horizons_ms: tuple[int, ...],
    common_initial_state: np.ndarray,
    rate: int,
) -> dict[int, CompositionEndpoint]:
    state, control = frame_arrays(sampled)
    anchor = exact_time_index(sampled, anchor_time_seconds, rate)
    history = adapter.history
    if anchor - history + 1 < 0:
        raise RuntimeError("COMPOSITION_HISTORY_UNAVAILABLE")
    predicted_history = state[anchor - history + 1:anchor + 1].copy()
    predicted_history[-1] = common_initial_state
    predicted_history[-1, 3:7] = normalize_quaternion(predicted_history[-1:, 3:7])[0]
    controls = control[anchor - history + 1:]
    requested = {int(round(value * rate / 1000.0)): value for value in horizons_ms}
    output: dict[int, CompositionEndpoint] = {}
    prediction = np.full(10, np.nan)
    for step in range(1, max(requested) + 1):
        try:
            prediction = adapter.next_state(route, predicted_history, controls[step - 1:step - 1 + history])
        except Exception:
            prediction = np.full(10, np.nan)
        finite = bool(np.isfinite(prediction).all())
        if step in requested:
            output[requested[step]] = CompositionEndpoint(prediction.copy(), finite, step)
        if not finite:
            for later_step, horizon in requested.items():
                if later_step > step:
                    output[horizon] = CompositionEndpoint(np.full(10, np.nan), False, later_step)
            break
        prediction[3:7] = normalize_quaternion(prediction[None, 3:7])[0]
        predicted_history = np.vstack((predicted_history[1:], prediction))
    return output


def one_step_order_supported(one_step_by_rate: dict[int, float]) -> bool:
    return one_step_by_rate[400] < one_step_by_rate[200] < one_step_by_rate[100]


def composition_failure_class(rate_errors: dict[int, float]) -> str:
    if rate_errors[400] > rate_errors[200] > rate_errors[100]:
        return "COMPOSITION_CONSISTENCY_FAILURE_SUPPORTED"
    if rate_errors[400] <= rate_errors[100] and rate_errors[200] <= rate_errors[100]:
        return "COMPOSITION_CONSISTENCY_FAILURE_NOT_SUPPORTED"
    return "COMPOSITION_CONSISTENCY_MIXED"
