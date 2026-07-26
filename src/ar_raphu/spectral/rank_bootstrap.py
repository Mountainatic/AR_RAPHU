"""Circular moving-block bootstrap helpers for frozen-smoothing rank profiles."""

from __future__ import annotations

import numpy as np


def circular_block_indices(
    length: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if length <= 0 or block_length <= 0:
        raise ValueError("length and block_length must be positive.")
    blocks = int(np.ceil(length / block_length))
    starts = rng.integers(0, length, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % length).reshape(-1)[:length]


def rank_interval(
    ranks: np.ndarray,
    *,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> tuple[int, int, int]:
    values = np.asarray(ranks, dtype=np.float64)
    low, high = np.quantile(values, quantiles, method="nearest")
    median = np.quantile(values, 0.5, method="nearest")
    return int(low), int(median), int(high)
