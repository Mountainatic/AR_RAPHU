"""Normalized tensor smoothness and Gram-ridge penalties."""

from __future__ import annotations

import numpy as np


def normalized_second_difference(size: int) -> np.ndarray:
    if size < 3:
        return np.zeros((size, size), dtype=np.float64)
    difference = np.diff(np.eye(size, dtype=np.float64), n=2, axis=0)
    penalty = difference.T @ difference
    trace = float(np.trace(penalty))
    return penalty * (size / trace)


def tensor_penalty(
    lag_gram: np.ndarray,
    amplitude_grams: list[np.ndarray],
    *,
    lag_smoothness: float,
    amplitude_smoothness: float,
    ridge_weight: float,
) -> np.ndarray:
    m_lag = lag_gram.shape[0]
    m_amp = amplitude_grams[0].shape[0]
    r_lag = normalized_second_difference(m_lag)
    r_amp = normalized_second_difference(m_amp)
    blocks = []
    for amplitude_gram in amplitude_grams:
        blocks.append(
            lag_smoothness * np.kron(r_lag, np.eye(m_amp))
            + amplitude_smoothness * np.kron(np.eye(m_lag), r_amp)
            + ridge_weight * np.kron(lag_gram, amplitude_gram)
        )
    total = sum(block.shape[0] for block in blocks)
    result = np.zeros((total, total), dtype=np.float64)
    offset = 0
    for block in blocks:
        width = block.shape[0]
        result[offset : offset + width, offset : offset + width] = block
        offset += width
    return result
