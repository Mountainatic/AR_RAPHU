"""Efficient two-sided least-squares projection of lag/amplitude surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import normalized_root_mean_square_error


@dataclass(frozen=True, slots=True)
class SurfaceProjectionResult:
    coefficients: np.ndarray
    estimate: np.ndarray
    nrmse: float


def identity_lag_basis(length: int) -> np.ndarray:
    """Return the discrete mother-space lag basis."""

    if length <= 0:
        raise ValueError("length must be positive.")
    return np.eye(length, dtype=np.float64)


def project_tensor_surface(
    truth_surface: np.ndarray,
    lag_basis: np.ndarray,
    amplitude_basis: np.ndarray,
) -> SurfaceProjectionResult:
    """Project without materializing the equivalent Kronecker design."""

    truth = np.asarray(truth_surface, dtype=np.float64)
    lag = np.asarray(lag_basis, dtype=np.float64)
    amplitude = np.asarray(amplitude_basis, dtype=np.float64)
    if truth.ndim != 2 or lag.ndim != 2 or amplitude.ndim != 2:
        raise ValueError("truth_surface and both bases must be matrices.")
    if truth.shape != (lag.shape[0], amplitude.shape[0]):
        raise ValueError("Basis evaluation dimensions do not match the surface.")
    intermediate = np.linalg.lstsq(lag, truth, rcond=None)[0]
    theta_t = np.linalg.lstsq(amplitude, intermediate.T, rcond=None)[0]
    theta = theta_t.T
    estimate = lag @ theta @ amplitude.T
    return SurfaceProjectionResult(
        coefficients=theta,
        estimate=estimate,
        nrmse=normalized_root_mean_square_error(truth, estimate),
    )
