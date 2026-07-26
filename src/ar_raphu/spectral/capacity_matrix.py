"""Capacity-grid helpers shared by v0.3.3 experiments and tests."""

from __future__ import annotations

import itertools

import numpy as np


def smoothing_pairs(config: dict[str, object]) -> list[tuple[float, float]]:
    capacity = config["capacity"]
    return [
        (float(lag), float(amplitude))
        for lag, amplitude in itertools.product(
            capacity["smoothing_lag_candidates"],
            capacity["smoothing_amplitude_candidates"],
        )
    ]


def apply_tensor_coefficients(
    matrix: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    design = np.asarray(matrix, dtype=np.float64)
    theta = np.asarray(coefficients, dtype=np.float64)
    return design @ theta.reshape(-1)


def build_single_variable_matrix(
    values: np.ndarray,
    *,
    origin_indices: np.ndarray,
    lag_basis: np.ndarray,
    amplitude_basis,
) -> np.ndarray:
    """Build a one-variable lag-amplitude tensor design for arbitrary lag bases."""

    sequence = np.asarray(values, dtype=np.float64)
    origins = np.asarray(origin_indices, dtype=np.int64)
    lag = np.asarray(lag_basis, dtype=np.float64)
    if sequence.ndim != 1 or origins.ndim != 1 or lag.ndim != 2:
        raise ValueError("Expected a scalar sequence, origins, and lag basis.")
    offsets = np.arange(lag.shape[0], dtype=np.int64)
    if origins.min() - offsets[-1] < 0 or origins.max() >= len(sequence):
        raise ValueError("Origin histories are outside the available sequence.")
    windows = sequence[origins[:, None] - offsets[None, :]]
    amplitude = amplitude_basis.transform(windows.reshape(-1)).reshape(
        len(origins), lag.shape[0], -1
    )
    tensor = np.einsum("la,nlb->nab", lag, amplitude, optimize=True)
    return tensor.reshape(len(origins), -1)


def select_minimum_validation_mse(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("No smoothing candidates were evaluated.")
    return min(
        rows,
        key=lambda row: (
            float(row["validation_contribution_mse"]),
            int(row.get("configuration_order", 0)),
        ),
    )
