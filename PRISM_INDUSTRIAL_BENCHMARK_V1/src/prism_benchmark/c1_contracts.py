from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def round_half_up_steps(seconds: float, cadence_seconds: float, *, minimum: int) -> tuple[int, float]:
    if seconds < 0 or cadence_seconds <= 0:
        raise ValueError("seconds must be nonnegative and cadence must be positive")
    if seconds == 0:
        if minimum > 0:
            raise ValueError("a zero duration cannot satisfy a positive minimum")
        return 0, 0.0
    steps = max(minimum, math.floor(seconds / cadence_seconds + 0.5))
    error = abs(steps * cadence_seconds - seconds) / seconds
    return steps, error


def ceil_steps(seconds: float, cadence_seconds: float) -> int:
    if seconds < 0 or cadence_seconds <= 0:
        raise ValueError("seconds must be nonnegative and cadence must be positive")
    return math.ceil(seconds / cadence_seconds)


def target_change(y: np.ndarray, origins: np.ndarray, h: int, w: int, w0: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    if y.ndim != 1 or origins.ndim != 1 or h < 0 or w < 1 or w0 < 1:
        raise ValueError("invalid target arguments")
    if len(origins) == 0:
        return np.empty(0, dtype=np.float64)
    if origins.min() - w0 < 0 or origins.max() + h + w > len(y):
        raise IndexError("target support leaves the entity")
    prefix = np.concatenate(([0.0], np.cumsum(y, dtype=np.float64)))
    current = (prefix[origins] - prefix[origins - w0]) / w0
    future = (prefix[origins + h + w] - prefix[origins + h]) / w
    return future - current


def dependency_bounds(origins: np.ndarray, lmax: int, h: int, w: int, delay: int) -> tuple[np.ndarray, np.ndarray]:
    origins = np.asarray(origins, dtype=np.int64)
    if min(lmax, h, delay) < 0 or w < 1:
        raise ValueError("invalid dependency arguments")
    return origins - lmax, origins + h + w + delay


def latest_available_target_index(origin: int, delay: int) -> int:
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    return origin - 1 - delay


def valid_origins_for_interval(
    start: int,
    stop: int,
    *,
    lmax: int,
    h: int,
    w: int,
    delay: int,
    left_buffer: int = 0,
) -> np.ndarray:
    first = start + left_buffer + lmax
    last = stop - h - w - delay
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


def stable_identifier(*parts: object) -> str:
    canonical = "|".join(str(part) for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def maximum_registered_history(h_steps: int, w0_steps: int, train_entity_lengths: Iterable[int]) -> int:
    if h_steps > 0:
        return max(w0_steps, 8 * h_steps)
    lengths = [int(length) for length in train_entity_lengths]
    if not lengths or min(lengths) <= 0:
        raise ValueError("positive training entity lengths are required")
    candidates = sorted(
        {multiplier * cadence_multiple for multiplier in (4, 16, 64) for cadence_multiple in (1, 2, 4, 8, 16, 32, 64)}
    )
    feasible = [candidate for candidate in candidates if candidate < min(lengths)]
    if not feasible:
        return w0_steps
    return max(w0_steps, feasible[-1])


@dataclass(frozen=True)
class RealizedHead:
    task_id: str
    dataset: str
    target: str
    cadence_seconds: float
    requested_horizon_seconds: float
    requested_window_seconds: float
    h_steps: int
    w_steps: int
    w0_steps: int
    horizon_error: float
    window_error: float
    status: str
    primary: bool

    @property
    def head_id(self) -> str:
        return f"{self.task_id}__H{self.h_steps}__W{self.w_steps}"


def realize_heads(config_path: Path) -> tuple[dict[str, Any], list[RealizedHead]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    heads: list[RealizedHead] = []
    for task in config["tasks"]:
        cadence = float(task["cadence_seconds"])
        w_steps, w_error = round_half_up_steps(float(task["window_seconds"]), cadence, minimum=1)
        for horizon in task["horizons_seconds"]:
            h_steps, h_error = round_half_up_steps(float(horizon), cadence, minimum=0)
            status = "PASS" if max(h_error, w_error) <= 0.10 else "UNSUPPORTED_BY_CADENCE"
            heads.append(
                RealizedHead(
                    task_id=task["task_id"],
                    dataset=task["dataset"],
                    target=task["target"],
                    cadence_seconds=cadence,
                    requested_horizon_seconds=float(horizon),
                    requested_window_seconds=float(task["window_seconds"]),
                    h_steps=h_steps,
                    w_steps=w_steps,
                    w0_steps=w_steps,
                    horizon_error=h_error,
                    window_error=w_error,
                    status=status,
                    primary=float(horizon) == float(task["primary_horizon_seconds"]),
                )
            )
    return config, heads
