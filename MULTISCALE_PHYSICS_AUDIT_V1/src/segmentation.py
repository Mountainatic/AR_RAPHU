"""Hard breakpoint detection and leakage-safe segment boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Segment:
    start: int
    stop: int
    role: str

    def contains_window(self, start: int, stop: int) -> bool:
        return self.start <= start and stop <= self.stop


def detect_breakpoints(target: np.ndarray, threshold: float) -> list[int]:
    values = np.asarray(target, dtype=np.float64)
    return (np.flatnonzero(np.abs(np.diff(values)) > float(threshold)) + 1).tolist()


def build_segments(samples: int, breakpoints: Iterable[int]) -> list[Segment]:
    points = sorted(set(int(value) for value in breakpoints))
    if any(point <= 0 or point >= samples for point in points):
        raise ValueError("BREAKPOINT_OUT_OF_RANGE")
    boundaries = [0, *points, samples]
    return [
        Segment(
            start=boundaries[index],
            stop=boundaries[index + 1],
            role="main" if index == len(boundaries) - 2 else "sensitivity",
        )
        for index in range(len(boundaries) - 1)
    ]


def verify_frozen_breakpoints(
    detected: Iterable[int], frozen: Iterable[int], *, sheet: str
) -> None:
    detected_values = [int(value) for value in detected]
    frozen_values = [int(value) for value in frozen]
    if detected_values != frozen_values:
        raise ValueError(
            f"BREAKPOINT_MISMATCH:{sheet}:"
            f"detected={detected_values}:frozen={frozen_values}"
        )
