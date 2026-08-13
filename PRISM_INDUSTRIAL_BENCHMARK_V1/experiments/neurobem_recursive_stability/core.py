from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np
import pandas as pd

from prism_benchmark.neurobem_literature import (
    MOTOR_COLUMNS, TRACK_B_STATE_COLUMNS, compose_quaternion_increment,
    normalize_quaternion, quaternion_conjugate, quaternion_multiply,
)
from experiments.neurobem_manifold_switch.metrics import divergence_time, state_errors
from experiments.neurobem_manifold_switch.prism_adapter import FrozenPrismAdapter


BLOCKS = {"velocity": slice(0, 3), "attitude": slice(3, 7), "body_rate": slice(7, 10)}
ABLATION_BLOCKS = {
    "velocity_only": ("velocity",), "attitude_only": ("attitude",),
    "body_rate_only": ("body_rate",), "velocity_attitude": ("velocity", "attitude"),
    "attitude_body_rate": ("attitude", "body_rate"),
    "full_recursive": ("velocity", "attitude", "body_rate"),
}


def frame_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    state = frame.loc[:, TRACK_B_STATE_COLUMNS].to_numpy(dtype=np.float64)
    state[:, 3:7] = normalize_quaternion(state[:, 3:7])
    control = frame.loc[:, MOTOR_COLUMNS].to_numpy(dtype=np.float64)
    return state, control


def effective_delta(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    q = quaternion_multiply(
        quaternion_conjugate(normalize_quaternion(reference[None, 3:7])),
        normalize_quaternion(value[None, 3:7]),
    )[0]
    angle = 2.0 * math.atan2(float(np.linalg.norm(q[1:])), abs(float(q[0])))
    axis = q[1:] / max(float(np.linalg.norm(q[1:])), 1e-15)
    return np.r_[value[:3] - reference[:3], axis * angle, value[7:10] - reference[7:10]]


def perturb_latest(history: np.ndarray, direction: np.ndarray, epsilon: float) -> np.ndarray:
    out = history.copy()
    out[-1, :3] += epsilon * direction[:3]
    dq = compose_quaternion_increment(
        out[-1:, 3:7], (epsilon * direction[3:6])[None],
    )[0]
    out[-1, 3:7] = dq
    out[-1, 7:10] += epsilon * direction[6:9]
    return out


def direction_vector(name: str, seed: int) -> np.ndarray:
    out = np.zeros(9, dtype=np.float64)
    names = ["velocity_x", "velocity_y", "velocity_z", "attitude_x", "attitude_y", "attitude_z", "body_rate_x", "body_rate_y", "body_rate_z"]
    if name in names:
        out[names.index(name)] = 1.0
    elif name.startswith("random_"):
        rng = np.random.default_rng(seed + int(name.rsplit("_", 1)[1]))
        out = rng.normal(size=9); out /= np.linalg.norm(out)
    else:
        raise ValueError(f"UNKNOWN_PERTURBATION_DIRECTION:{name}")
    return out


@dataclass
class RolloutResult:
    predictions: np.ndarray
    errors: dict[str, np.ndarray]
    finite: np.ndarray
    t_diverge: int | None
    elapsed: float


def rollout(
    adapter: FrozenPrismAdapter, route: str, state: np.ndarray, control: np.ndarray,
    steps: int, thresholds: dict[str, float], persistence: int,
    *, resync_interval: int | None = None, recurse_blocks: Iterable[str] = ("velocity", "attitude", "body_rate"),
) -> RolloutResult:
    history = adapter.history
    predicted_history = state[:history].copy()
    predictions = np.full((steps, 10), np.nan)
    blocks = tuple(recurse_blocks)
    start = time.perf_counter()
    for step in range(steps):
        try:
            nxt = adapter.next_state(route, predicted_history, control[step:step + history])
        except Exception:
            nxt = np.full(10, np.nan)
        predictions[step] = nxt
        if not np.isfinite(nxt).all():
            break
        truth = state[history + step]
        feedback = truth.copy()
        for block in blocks:
            feedback[BLOCKS[block]] = nxt[BLOCKS[block]]
        feedback[3:7] = normalize_quaternion(feedback[None, 3:7])[0]
        predicted_history = np.vstack((predicted_history[1:], feedback))
        # After target H+step has been observed, the next prediction may use a
        # history ending at that observation. This is an explicit intervention.
        if resync_interval is not None and (step + 1) % resync_interval == 0:
            predicted_history = state[step + 1:step + history + 1].copy()
    elapsed = time.perf_counter() - start
    target = state[history:history + steps]
    errors = state_errors(target, predictions)
    finite = np.isfinite(predictions).all(axis=1)
    return RolloutResult(predictions, errors, finite, divergence_time(errors, finite, thresholds, persistence), elapsed)


def history_at_anchor(adapter: FrozenPrismAdapter, route: str, state: np.ndarray, control: np.ndarray, anchor: int) -> np.ndarray | None:
    history = state[:adapter.history].copy()
    for step in range(anchor):
        nxt = adapter.next_state(route, history, control[step:step + adapter.history])
        if not np.isfinite(nxt).all(): return None
        history = np.vstack((history[1:], nxt))
    return history


def paired_growth(
    adapter: FrozenPrismAdapter, route: str, base_history: np.ndarray,
    controls: np.ndarray, direction: np.ndarray, epsilon: float, horizon: int,
) -> np.ndarray:
    base = base_history.copy(); perturbed = perturb_latest(base_history, direction, epsilon)
    growth = np.full(horizon, np.nan)
    for step in range(horizon):
        a = adapter.next_state(route, base, controls[step:step + adapter.history])
        b = adapter.next_state(route, perturbed, controls[step:step + adapter.history])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            growth[step:] = np.inf; break
        growth[step] = np.linalg.norm(effective_delta(a, b)) / epsilon
        base = np.vstack((base[1:], a)); perturbed = np.vstack((perturbed[1:], b))
    return growth


def paired_growth_with_channels(
    adapter: FrozenPrismAdapter, route: str, base_history: np.ndarray,
    controls: np.ndarray, direction: np.ndarray, epsilon: float, horizon: int,
) -> dict[str, np.ndarray]:
    base = base_history.copy(); perturbed = perturb_latest(base_history, direction, epsilon)
    values = {key: np.full(horizon, np.nan) for key in ("total", "velocity", "attitude", "body_rate")}
    for step in range(horizon):
        a = adapter.next_state(route, base, controls[step:step + adapter.history])
        b = adapter.next_state(route, perturbed, controls[step:step + adapter.history])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            for value in values.values(): value[step:] = np.inf
            break
        delta = effective_delta(a, b) / epsilon
        values["total"][step] = np.linalg.norm(delta)
        values["velocity"][step] = np.linalg.norm(delta[:3])
        values["attitude"][step] = np.linalg.norm(delta[3:6])
        values["body_rate"][step] = np.linalg.norm(delta[6:9])
        base = np.vstack((base[1:], a)); perturbed = np.vstack((perturbed[1:], b))
    return values


def newest_state_jacobian(
    adapter: FrozenPrismAdapter, route: str, history: np.ndarray,
    controls: np.ndarray, epsilon: float,
) -> np.ndarray:
    matrix = np.empty((9, 9), dtype=np.float64)
    for column in range(9):
        direction = np.eye(9)[column]
        plus = adapter.next_state(route, perturb_latest(history, direction, epsilon), controls[:adapter.history])
        minus = adapter.next_state(route, perturb_latest(history, direction, -epsilon), controls[:adapter.history])
        matrix[:, column] = effective_delta(minus, plus) / (2.0 * epsilon)
    return matrix


def newest_state_block_product(
    adapter: FrozenPrismAdapter, route: str, history: np.ndarray,
    controls: np.ndarray, epsilon: float, horizon: int,
) -> tuple[float, float]:
    """Finite product of newest-state blocks, not the full augmented Jacobian."""
    current = history.copy(); product = np.eye(9, dtype=np.float64); used = 0
    for step in range(horizon):
        matrix = newest_state_jacobian(adapter, route, current, controls[step:], epsilon)
        product = matrix @ product; used += 1
        nxt = adapter.next_state(route, current, controls[step:step + adapter.history])
        if not np.isfinite(nxt).all(): return float("inf"), float("inf")
        current = np.vstack((current[1:], nxt))
    amplification = float(np.linalg.svd(product, compute_uv=False)[0])
    return amplification, float(np.log(max(amplification, 1e-300)) / used)


def block_growth(growth_delta: np.ndarray, epsilon: float) -> dict[str, float]:
    return {
        "velocity": float(np.linalg.norm(growth_delta[:3]) / epsilon),
        "attitude": float(np.linalg.norm(growth_delta[3:6]) / epsilon),
        "body_rate": float(np.linalg.norm(growth_delta[6:9]) / epsilon),
    }
