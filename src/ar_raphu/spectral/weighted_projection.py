"""Weighted orthogonal tensor-surface projection for Spectral v0.3.3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


def normalized_trapezoidal_weights(grid: np.ndarray) -> np.ndarray:
    """Return positive trapezoidal quadrature weights summing to one."""

    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("grid must be a one-dimensional array with at least two points.")
    differences = np.diff(values)
    if not np.all(differences > 0):
        raise ValueError("grid must be strictly increasing.")
    weights = np.empty_like(values)
    weights[0] = differences[0] / 2.0
    weights[-1] = differences[-1] / 2.0
    if len(values) > 2:
        weights[1:-1] = (differences[:-1] + differences[1:]) / 2.0
    weights /= weights.sum()
    return weights


def orthogonal_projector(
    basis: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Project onto a basis after whitening by diagonal quadrature weights."""

    matrix = np.asarray(basis, dtype=np.float64)
    diagonal = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or diagonal.shape != (matrix.shape[0],):
        raise ValueError("basis and weights have incompatible shapes.")
    if not np.all(diagonal > 0) or not np.isclose(diagonal.sum(), 1.0):
        raise ValueError("weights must be positive and normalized.")
    whitened = np.sqrt(diagonal)[:, None] * matrix
    return whitened @ np.linalg.pinv(whitened, rcond=1.0e-12)


@dataclass(frozen=True, slots=True)
class WeightedProjectionResult:
    coefficients: np.ndarray
    estimate: np.ndarray
    whitened_truth: np.ndarray
    whitened_estimate: np.ndarray
    epsilon_lag: float
    epsilon_amplitude: float
    epsilon_amplitude_given_lag: float
    epsilon_lag_given_amplitude: float
    epsilon_joint: float
    operator_error: float
    lag_decomposition_error: float
    amplitude_decomposition_error: float


def weighted_tensor_projection(
    truth_surface: np.ndarray,
    lag_basis: np.ndarray,
    amplitude_basis: np.ndarray,
    *,
    lag_weights: np.ndarray,
    amplitude_weights: np.ndarray,
) -> WeightedProjectionResult:
    """Compute the v0.3.3 weighted projection and both orthogonal decompositions."""

    truth = np.asarray(truth_surface, dtype=np.float64)
    lag = np.asarray(lag_basis, dtype=np.float64)
    amplitude = np.asarray(amplitude_basis, dtype=np.float64)
    w_lag = np.asarray(lag_weights, dtype=np.float64)
    w_amplitude = np.asarray(amplitude_weights, dtype=np.float64)
    if truth.shape != (lag.shape[0], amplitude.shape[0]):
        raise ValueError("truth and basis evaluation dimensions do not agree.")

    p_lag = orthogonal_projector(lag, w_lag)
    p_amplitude = orthogonal_projector(amplitude, w_amplitude)
    lag_root = np.sqrt(w_lag)
    amplitude_root = np.sqrt(w_amplitude)
    whitened_truth = lag_root[:, None] * truth * amplitude_root[None, :]
    whitened_estimate = p_lag @ whitened_truth @ p_amplitude

    denominator = max(
        float(np.linalg.norm(whitened_truth, ord="fro")),
        np.finfo(np.float64).eps,
    )
    identity_lag = np.eye(len(w_lag), dtype=np.float64)
    identity_amplitude = np.eye(len(w_amplitude), dtype=np.float64)
    lag_residual = (identity_lag - p_lag) @ whitened_truth
    amplitude_residual = whitened_truth @ (identity_amplitude - p_amplitude)
    amplitude_given_lag = p_lag @ whitened_truth @ (
        identity_amplitude - p_amplitude
    )
    lag_given_amplitude = (
        identity_lag - p_lag
    ) @ whitened_truth @ p_amplitude
    joint_residual = whitened_truth - whitened_estimate

    lag_gram = lag.T @ (w_lag[:, None] * lag)
    amplitude_gram = amplitude.T @ (w_amplitude[:, None] * amplitude)
    cross = lag.T @ (w_lag[:, None] * truth * w_amplitude[None, :]) @ amplitude
    coefficients = np.linalg.pinv(lag_gram, rcond=1.0e-12) @ cross @ np.linalg.pinv(
        amplitude_gram, rcond=1.0e-12
    )
    estimate = lag @ coefficients @ amplitude.T

    epsilon_lag = float(np.linalg.norm(lag_residual, ord="fro") / denominator)
    epsilon_amplitude = float(
        np.linalg.norm(amplitude_residual, ord="fro") / denominator
    )
    epsilon_amplitude_given_lag = float(
        np.linalg.norm(amplitude_given_lag, ord="fro") / denominator
    )
    epsilon_lag_given_amplitude = float(
        np.linalg.norm(lag_given_amplitude, ord="fro") / denominator
    )
    epsilon_joint = float(np.linalg.norm(joint_residual, ord="fro") / denominator)
    return WeightedProjectionResult(
        coefficients=coefficients,
        estimate=estimate,
        whitened_truth=whitened_truth,
        whitened_estimate=whitened_estimate,
        epsilon_lag=epsilon_lag,
        epsilon_amplitude=epsilon_amplitude,
        epsilon_amplitude_given_lag=epsilon_amplitude_given_lag,
        epsilon_lag_given_amplitude=epsilon_lag_given_amplitude,
        epsilon_joint=epsilon_joint,
        operator_error=float(scipy.linalg.svdvals(joint_residual)[0]),
        lag_decomposition_error=abs(
            epsilon_joint**2
            - epsilon_lag**2
            - epsilon_amplitude_given_lag**2
        ),
        amplitude_decomposition_error=abs(
            epsilon_joint**2
            - epsilon_amplitude**2
            - epsilon_lag_given_amplitude**2
        ),
    )
