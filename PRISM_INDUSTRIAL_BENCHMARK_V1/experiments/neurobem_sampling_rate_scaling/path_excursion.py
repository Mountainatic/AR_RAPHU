from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from experiments.neurobem_manifold_switch.metrics import state_errors
from experiments.neurobem_manifold_switch.prism_adapter import FrozenPrismAdapter
from experiments.neurobem_recursive_stability.core import frame_arrays
from prism_benchmark.neurobem_literature import normalize_quaternion

from .composition_consistency import exact_time_index


HORIZONS_MS = (10, 20, 50, 100, 200, 500)
CHANNELS = ("full_state", "velocity", "attitude", "body_rate")


@dataclass(frozen=True)
class PathReplay:
    errors: dict[str, np.ndarray]
    finite: np.ndarray


def replay_path(
    adapter: FrozenPrismAdapter,
    route: str,
    sampled: pd.DataFrame,
    native: pd.DataFrame,
    anchor_time_seconds: float,
    maximum_horizon_ms: int,
    common_initial_state: np.ndarray,
    rate: int,
) -> PathReplay:
    state, control = frame_arrays(sampled)
    native_state, _ = frame_arrays(native)
    anchor = exact_time_index(sampled, anchor_time_seconds, rate)
    history = adapter.history
    predicted_history = state[anchor - history + 1:anchor + 1].copy()
    predicted_history[-1] = common_initial_state
    predicted_history[-1, 3:7] = normalize_quaternion(predicted_history[-1:, 3:7])[0]
    controls = control[anchor - history + 1:]
    steps = int(round(maximum_horizon_ms * rate / 1000.0))
    predictions = np.full((steps, 10), np.nan)
    targets = np.empty((steps, 10))
    for step in range(1, steps + 1):
        target_time = anchor_time_seconds + step / rate
        targets[step - 1] = native_state[exact_time_index(native, target_time, 400)]
        try:
            prediction = adapter.next_state(route, predicted_history, controls[step - 1:step - 1 + history])
        except Exception:
            prediction = np.full(10, np.nan)
        predictions[step - 1] = prediction
        if not np.isfinite(prediction).all():
            break
        prediction[3:7] = normalize_quaternion(prediction[None, 3:7])[0]
        predicted_history = np.vstack((predicted_history[1:], prediction))
    finite = np.isfinite(predictions).all(axis=1)
    block = state_errors(targets, predictions)
    full = np.sqrt(block["velocity"] ** 2 + block["attitude"] ** 2 + block["body_rate"] ** 2)
    return PathReplay({"full_state": full, **block}, finite)


def inside_envelope(path: PathReplay, bounds: dict[str, float], channel: str) -> np.ndarray:
    if channel == "full_state":
        return path.finite & np.logical_and.reduce([
            path.errors[name] <= float(bounds[name]) for name in ("velocity", "attitude", "body_rate")
        ])
    return path.finite & (path.errors[channel] <= float(bounds[channel]))


def excursion_metrics(path: PathReplay, bounds: dict[str, float], channel: str, steps: int, rate: int) -> dict[str, float | int | None | bool]:
    values = path.errors[channel][:steps]
    inside = inside_envelope(path, bounds, channel)[:steps]
    finite_values = np.where(np.isfinite(values), values, np.inf)
    terminal = float(finite_values[-1])
    maximum = float(np.max(finite_values))
    rms = float(np.sqrt(np.mean(np.square(finite_values))))
    outside = ~inside
    crossing = outside & np.r_[True, inside[:-1]]
    reentry = inside & np.r_[False, outside[:-1]]
    first = np.flatnonzero(outside)
    first_ms = None if len(first) == 0 else float(1000.0 * (first[0] + 1) / rate)
    ratio = math.inf if terminal == 0 and maximum > 0 else (1.0 if terminal == 0 else maximum / terminal)
    return {
        "terminal_error": terminal,
        "max_path_error": maximum,
        "rms_path_error": rms,
        "first_cross_ms": first_ms,
        "envelope_crossing_count": int(crossing.sum()),
        "fraction_inside_envelope": float(inside.mean()),
        "ever_crossing": bool(outside.any()),
        "reentry_count": int(reentry.sum()),
        "max_over_terminal": float(ratio),
    }


def transient_order_class(values: dict[int, float], lower_is_earlier: bool = False) -> str:
    supported = values[400] < values[200] < values[100] if lower_is_earlier else values[400] > values[200] > values[100]
    if supported:
        return "HIGHER_RATE_TRANSIENT_PATH_EXCURSION_SUPPORTED"
    reverse = values[400] >= values[100] and values[200] >= values[100] if not lower_is_earlier else values[400] <= values[100] and values[200] <= values[100]
    return "HIGHER_RATE_TRANSIENT_PATH_EXCURSION_MIXED" if reverse else "HIGHER_RATE_TRANSIENT_PATH_EXCURSION_NOT_SUPPORTED"
