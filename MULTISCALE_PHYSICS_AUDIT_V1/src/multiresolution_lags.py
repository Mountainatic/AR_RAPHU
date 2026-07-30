"""Physical lag-block specifications and cached feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .resampling import causal_block_average, causal_ema
from .timebase import Timebase


@dataclass(frozen=True, slots=True)
class LagBlock:
    start_min: float
    stop_min: float
    midpoint_min: float


def expand_lag_blocks(
    specification: Iterable[Iterable[float]],
    *,
    history_min: float,
    resolution_multiplier: float = 1.0,
) -> list[LagBlock]:
    blocks: list[LagBlock] = []
    for raw_start, raw_stop, raw_width in specification:
        start = float(raw_start)
        stop = min(float(raw_stop), float(history_min))
        width = float(raw_width) * float(resolution_multiplier)
        if stop <= start:
            continue
        cursor = start
        while cursor < stop - 1.0e-10:
            end = min(cursor + width, stop)
            blocks.append(LagBlock(cursor, end, 0.5 * (cursor + end)))
            cursor = end
    if not blocks:
        raise ValueError("NO_LAG_BLOCKS_FOR_PROFILE")
    return blocks


def lag_block_matrix(
    values: np.ndarray,
    origins: np.ndarray,
    blocks: Iterable[LagBlock],
    *,
    timebase: Timebase,
) -> np.ndarray:
    columns = []
    for block in blocks:
        start = timebase.samples_for_minutes(block.start_min, minimum=0)
        stop = timebase.samples_for_minutes(block.stop_min)
        columns.append(
            causal_block_average(
                values,
                origins,
                start_samples=start,
                stop_samples=stop,
            )
        )
    return np.column_stack(columns)


def thermal_state_bank(
    power: np.ndarray,
    tau_minutes: Iterable[float],
    *,
    timebase: Timebase,
) -> np.ndarray:
    return np.column_stack(
        [
            causal_ema(
                power,
                tau_samples=float(tau) * 60.0 / timebase.sample_period_sec,
            )
            for tau in tau_minutes
        ]
    )


def thermal_state_features(
    bank: np.ndarray, origins: np.ndarray
) -> np.ndarray:
    return np.asarray(bank, dtype=np.float64)[
        np.asarray(origins, dtype=np.int64)
    ]
