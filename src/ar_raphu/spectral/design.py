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
    continuation_diagnostics: list[dict[str, float | int | str]]


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
    continuation_scale_factor: float | None = None,
) -> np.ndarray:
    """Build the AR-history nuisance design using only data through each origin."""

    y = np.asarray(y, dtype=np.float64)
    targets = np.asarray(target_indices, dtype=np.int64)
    origins = targets - int(horizon)
    if origins.min() - L_y + 1 < 0:
        raise ValueError("AR history precedes sequence start.")
    lag_knots = clamped_knots(0.0, float(L_y - 1), lag_basis_count, degree)
    lag_basis = evaluate_basis(np.arange(L_y), lag_knots, degree)
    ar_domain = (
        AmplitudeDomain.fit(y[:train_target_stop], padding_fraction=0.10)
        if continuation_scale_factor is not None
        else None
    )
    amplitude_basis = CenteredSplineBasis.fit(
        y[:train_target_stop],
        n_basis=amplitude_basis_count,
        degree=degree,
        domain=ar_domain,
    )
    offsets = np.arange(L_y, dtype=np.int64)
    windows = y[origins[:, None] - offsets[None, :]]
    if continuation_scale_factor is None:
        amplitude = amplitude_basis.legacy_transform_for_audit(
            windows.reshape(-1)
        )
    else:
        amplitude, _ = amplitude_basis.bounded_c1_transform(
            windows.reshape(-1),
            scale_factor=continuation_scale_factor,
        )
    amplitude = amplitude.reshape(len(targets), L_y, amplitude_basis_count)
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
    continuation_scale_factor: float | None = None,
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
    continuation_diagnostics: list[dict[str, float | int | str]] = []
    lag_offsets = np.arange(L_x, dtype=np.int64)
    for variable, basis in enumerate(bases):
        train_eval = basis.transform(train_x[:, variable])
        amplitude_grams.append(train_eval.T @ train_eval / len(train_eval))
        window_values = x[origins[:, None] - lag_offsets[None, :], variable]
        if continuation_scale_factor is None:
            amplitude_eval = basis.transform(window_values.reshape(-1))
            in_domain = np.ones(window_values.size, dtype=bool)
        else:
            amplitude_eval, in_domain = basis.bounded_c1_transform(
                window_values.reshape(-1),
                scale_factor=continuation_scale_factor,
            )
        flattened = window_values.reshape(-1)
        train_range = max(
            float(np.ptp(train_x[:, variable])),
            np.finfo(np.float64).eps,
        )
        below = np.maximum(basis.lower - flattened, 0.0)
        above = np.maximum(flattened - basis.upper, 0.0)
        continuation_diagnostics.append(
            {
                "variable_index": variable,
                "source": "RECORDED_EXTERNAL_INPUT",
                "total_calls": int(flattened.size),
                "out_of_domain_calls": int(np.count_nonzero(~in_domain)),
                "out_of_domain_fraction": float(np.mean(~in_domain)),
                "maximum_normalized_distance": float(
                    np.max(np.maximum(below, above)) / train_range
                ),
            }
        )
        amplitude_eval = amplitude_eval.reshape(
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
        continuation_diagnostics=continuation_diagnostics,
    )
