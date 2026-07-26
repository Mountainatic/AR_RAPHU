"""FP64 strongly convex full-kernel solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True, slots=True)
class SpectralFit:
    coefficients: np.ndarray
    predictions: np.ndarray
    relative_kkt_residual: float
    condition_number: float


def solve_full_kernel(
    matrix: np.ndarray,
    target: np.ndarray,
    penalty: np.ndarray,
    *,
    numerical_jitter_relative: float = 1.0e-10,
) -> SpectralFit:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(penalty, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or p.shape != (x.shape[1],) * 2:
        raise ValueError("Incompatible design, target, or penalty shape.")
    system = x.T @ x / len(x) + p
    rhs = x.T @ y / len(x)
    scale = max(float(np.trace(system) / len(system)), 1.0)
    jitter = numerical_jitter_relative * scale
    solved_system = system + jitter * np.eye(len(system))
    factor = scipy.linalg.cho_factor(solved_system, lower=True, check_finite=True)
    coefficients = scipy.linalg.cho_solve(factor, rhs, check_finite=True)
    residual = solved_system @ coefficients - rhs
    denominator = max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
    return SpectralFit(
        coefficients=coefficients,
        predictions=x @ coefficients,
        relative_kkt_residual=float(np.linalg.norm(residual) / denominator),
        condition_number=float(np.linalg.cond(solved_system)),
    )
