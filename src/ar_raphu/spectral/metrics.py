"""Small deterministic metrics used by spectral experiments."""

from __future__ import annotations

import numpy as np


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    return float(np.mean(error**2))


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mse(target, prediction)))


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(target, prediction) else float("nan")
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator)


def normalized_root_mean_square_error(
    truth: np.ndarray, estimate: np.ndarray
) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    denominator = float(np.sqrt(np.mean(truth**2)))
    return float(
        np.sqrt(np.mean((truth - estimate) ** 2))
        / max(denominator, np.finfo(np.float64).eps)
    )
