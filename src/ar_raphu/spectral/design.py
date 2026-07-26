"""Tensor-product lag/amplitude design with causal origin-time indexing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .amplitude_domain import AmplitudeDomain
from .spline_basis import CenteredSplineBasis, clamped_knots, evaluate_basis


@dataclass(slots=True)
class SpectralDesign:
    matrix: np.ndarray
    variable_slices: dict[int, slice]
    lag_basis: np.ndarray
    amplitude_bases: list[CenteredSplineBasis]
    lag_gram: np.ndarray
    amplitude_grams: list[np.ndarray]
    target_indices: np.ndarray
    origin_indices: np.ndarray


def build_ar_nuisance_design(
    y: np.ndarray,
    *,
    target_indices: np.ndarray,
    train_target_stop: int,
    horizon: int,
    L_y: int,
    lag_basis_count: int,
    amplitude_basis_count: int,
    degree: int = 3,
) -> np.ndarray:
    """Build the AR-history nuisance design using only data through each origin."""

    y = np.asarray(y, dtype=np.float64)
    targets = np.asarray(target_indices, dtype=np.int64)
    origins = targets - int(horizon)
    if origins.min() - L_y + 1 < 0:
        raise ValueError("AR history precedes sequence start.")
    lag_knots = clamped_knots(0.0, float(L_y - 1), lag_basis_count, degree)
    lag_basis = evaluate_basis(np.arange(L_y), lag_knots, degree)
    amplitude_basis = CenteredSplineBasis.fit(
        y[:train_target_stop],
        n_basis=amplitude_basis_count,
        degree=degree,
    )
    offsets = np.arange(L_y, dtype=np.int64)
    windows = y[origins[:, None] - offsets[None, :]]
    amplitude = amplitude_basis.legacy_transform_for_audit(
        windows.reshape(-1)
    ).reshape(
        len(targets), L_y, amplitude_basis_count
    )
    tensor = np.einsum("la,nlb->nab", lag_basis, amplitude, optimize=True)
    return tensor.reshape(len(targets), lag_basis_count * amplitude_basis_count)


def build_spectral_design(
    x: np.ndarray,
    *,
    target_indices: np.ndarray,
    train_target_stop: int,
    horizon: int,
    L_x: int,
    lag_basis_count: int,
    amplitude_basis_count: int,
    degree: int = 3,
    amplitude_quantiles: tuple[float, float] = (0.01, 0.99),
    amplitude_domains: list[AmplitudeDomain] | None = None,
) -> SpectralDesign:
    x = np.asarray(x, dtype=np.float64)
    targets = np.asarray(target_indices, dtype=np.int64)
    if x.ndim != 2 or targets.ndim != 1 or not len(targets):
        raise ValueError("Expected x [time, variable] and non-empty target indices.")
    origins = targets - int(horizon)
    if origins.min() - L_x + 1 < 0:
        raise ValueError("External history precedes sequence start.")
    if targets.max() >= len(x):
        raise ValueError("Target index exceeds sequence.")
    lag_knots = clamped_knots(0.0, float(L_x - 1), lag_basis_count, degree)
    lag_basis = evaluate_basis(np.arange(L_x), lag_knots, degree)
    lag_gram = lag_basis.T @ lag_basis / L_x
    train_x = x[:train_target_stop]
    domains = amplitude_domains or [
        AmplitudeDomain.fit(
            train_x[:, variable],
            padding_fraction=0.10,
            core_quantiles=amplitude_quantiles,
        )
        for variable in range(x.shape[1])
    ]
    if len(domains) != x.shape[1]:
        raise ValueError("One amplitude domain is required per variable.")
    bases = [
        CenteredSplineBasis.fit(
            train_x[:, variable],
            n_basis=amplitude_basis_count,
            degree=degree,
            domain=domains[variable],
            quantiles=amplitude_quantiles,
        )
        for variable in range(x.shape[1])
    ]
    block_width = lag_basis_count * amplitude_basis_count
    matrix = np.empty((len(targets), x.shape[1] * block_width), dtype=np.float64)
    slices: dict[int, slice] = {}
    amplitude_grams: list[np.ndarray] = []
    lag_offsets = np.arange(L_x, dtype=np.int64)
    for variable, basis in enumerate(bases):
        train_eval = basis.transform(train_x[:, variable])
        amplitude_grams.append(train_eval.T @ train_eval / len(train_eval))
        window_values = x[origins[:, None] - lag_offsets[None, :], variable]
        amplitude_eval = basis.transform(window_values.reshape(-1)).reshape(
            len(targets), L_x, amplitude_basis_count
        )
        tensor = np.einsum("la,nlb->nab", lag_basis, amplitude_eval, optimize=True)
        block = slice(variable * block_width, (variable + 1) * block_width)
        matrix[:, block] = tensor.reshape(len(targets), block_width)
        slices[variable] = block
    return SpectralDesign(
        matrix=matrix,
        variable_slices=slices,
        lag_basis=lag_basis,
        amplitude_bases=bases,
        lag_gram=lag_gram,
        amplitude_grams=amplitude_grams,
        target_indices=targets,
        origin_indices=origins,
    )
