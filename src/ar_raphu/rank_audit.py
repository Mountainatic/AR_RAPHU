"""Orthogonal surface residuals and Gram-whitened rank diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sym_psd_sqrt(matrix: np.ndarray, *, tolerance: float = 1.0e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square Gram matrix.")
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    cutoff = tolerance * max(1.0, float(np.max(np.abs(values))))
    if np.min(values) < -cutoff:
        raise ValueError("Gram matrix is not positive semidefinite.")
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def orthogonalize_lag_basis(
    lag_basis: np.ndarray,
    anchor_kernel: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Project every lag-basis column off the anchored A-kernel direction."""

    basis = np.asarray(lag_basis, dtype=np.float64)
    q = np.asarray(anchor_kernel, dtype=np.float64).reshape(-1)
    if basis.ndim != 2 or basis.shape[0] != len(q):
        raise ValueError("lag_basis must be [lags, basis] and match q.")
    w = (
        np.ones_like(q)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if len(w) != len(q) or np.any(w <= 0):
        raise ValueError("weights must be positive and match q.")
    denominator = float(q @ (w * q))
    if denominator <= 0:
        raise ValueError("anchor kernel has zero weighted norm.")
    coefficients = (q * w) @ basis / denominator
    projected = basis - q[:, None] * coefficients[None, :]
    return projected


def empirical_amplitude_gram(
    amplitude_basis: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    basis = np.asarray(amplitude_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("amplitude_basis must be [observations, basis].")
    if weights is None:
        return basis.T @ basis / len(basis)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) != len(basis) or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("Invalid empirical weights.")
    normalized = w / w.sum()
    return basis.T @ (basis * normalized[:, None])


def discrete_lag_gram(
    lag_basis: np.ndarray, *, weights: np.ndarray | None = None
) -> np.ndarray:
    basis = np.asarray(lag_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("lag_basis must be [lags, basis].")
    w = (
        np.ones(len(basis), dtype=np.float64) / len(basis)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if len(w) != len(basis) or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("Invalid lag weights.")
    w = w / w.sum()
    return basis.T @ (basis * w[:, None])


@dataclass(frozen=True, slots=True)
class WhitenedRankAudit:
    singular_values: np.ndarray
    first_singular_energy: float
    nonseparability: float
    whitened_coefficients: np.ndarray


def gram_whitened_rank_audit(
    coefficients: np.ndarray,
    lag_gram: np.ndarray,
    amplitude_gram: np.ndarray,
) -> WhitenedRankAudit:
    theta = np.asarray(coefficients, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError("coefficients must be a matrix.")
    left = sym_psd_sqrt(lag_gram)
    right = sym_psd_sqrt(amplitude_gram)
    if left.shape[0] != theta.shape[0] or right.shape[0] != theta.shape[1]:
        raise ValueError("Coefficient and Gram dimensions do not match.")
    whitened = left @ theta @ right
    singular_values = np.linalg.svd(whitened, compute_uv=False)
    energy = np.square(singular_values)
    total = float(energy.sum())
    first = float(energy[0] / total) if total > 0 else 1.0
    return WhitenedRankAudit(
        singular_values=singular_values,
        first_singular_energy=first,
        nonseparability=1.0 - first,
        whitened_coefficients=whitened,
    )
