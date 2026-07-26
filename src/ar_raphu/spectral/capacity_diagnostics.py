"""Truth/operator closure helpers for v0.3.2 capacity diagnostics."""

from __future__ import annotations

import numpy as np

from ar_raphu.synthetic import (
    SyntheticSequence,
    _normalized_gaussian,
    second_truth_response,
    truth_response,
)

from .spline_basis import CenteredSplineBasis


class ModelClassMismatchError(ValueError):
    """Raised when a conditional kernel is requested as a 2D surface."""


def direct_apply_truth_kernel(
    sequence: SyntheticSequence,
    variable: int,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Apply the declared 2D truth kernel with target-minus-one semantics."""

    if sequence.scenario == "AR-S4":
        raise ModelClassMismatchError("AR-S4C is not a 2D additive Urysohn kernel.")
    targets = np.asarray(target_indices, dtype=np.int64)
    truth = sequence.truth
    L_x = int(truth["L_x"])
    offsets = np.arange(L_x)
    windows = sequence.x[targets[:, None] - 1 - offsets[None, :], variable]
    q_primary = np.asarray(truth["q_primary"], dtype=np.float64)[variable]
    if sequence.scenario == "AR-S3":
        q_secondary = np.asarray(truth["q_secondary"], dtype=np.float64)[variable]
        return (
            0.6
            * np.sum(
                q_primary[None, :] * truth_response(variable, windows), axis=1
            )
            + 0.4
            * np.sum(
                q_secondary[None, :]
                * second_truth_response(variable, windows),
                axis=1,
            )
        )
    if sequence.scenario == "AR-S4U":
        result = np.zeros(len(targets), dtype=np.float64)
        response = truth_response(variable, windows)
        for lag in range(L_x):
            amplitudes = windows[:, lag]
            centers = 8.0 + 12.0 / (1.0 + np.exp(-2.0 * amplitudes))
            weights = np.empty(len(amplitudes), dtype=np.float64)
            for index, center in enumerate(centers):
                weights[index] = _normalized_gaussian(L_x, center, 2.0)[lag]
            result += weights * response[:, lag]
        return result
    return np.sum(
        q_primary[None, :] * truth_response(variable, windows), axis=1
    )


def direct_apply_projected_kernel(
    x: np.ndarray,
    *,
    variable: int,
    target_indices: np.ndarray,
    horizon: int,
    lag_basis: np.ndarray,
    amplitude_basis: CenteredSplineBasis,
    coefficients: np.ndarray,
) -> np.ndarray:
    targets = np.asarray(target_indices, dtype=np.int64)
    origins = targets - int(horizon)
    offsets = np.arange(lag_basis.shape[0], dtype=np.int64)
    windows = np.asarray(x, dtype=np.float64)[
        origins[:, None] - offsets[None, :], variable
    ]
    amplitude = amplitude_basis.transform(windows.reshape(-1)).reshape(
        len(targets), len(offsets), -1
    )
    return np.einsum(
        "la,ab,nlb->n",
        lag_basis,
        np.asarray(coefficients, dtype=np.float64),
        amplitude,
        optimize=True,
    )
