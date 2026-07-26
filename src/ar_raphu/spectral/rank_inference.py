"""Pre-registered moving-block bootstrap helpers for rank inference."""

from __future__ import annotations

import numpy as np


def circular_moving_block_indices(
    length: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if length <= 0 or not 1 <= block_length <= length:
        raise ValueError("Invalid sequence or block length.")
    blocks = int(np.ceil(length / block_length))
    starts = rng.integers(0, length, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % length).reshape(-1)[:length]


def bootstrap_upper_tail_pvalue(
    observed: float, bootstrap_statistics: np.ndarray
) -> float:
    values = np.asarray(bootstrap_statistics, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Bootstrap statistics must be finite and one-dimensional.")
    return float((1 + np.count_nonzero(values >= observed)) / (len(values) + 1))
