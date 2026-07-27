"""Sequence-safe adapter from public WindowedTask data to spectral design."""

from __future__ import annotations

import numpy as np

from ar_raphu.datasets.windowing import WindowedTask

from .amplitude_domain import AmplitudeDomain
from .design import SpectralDesign
from .spline_basis import CenteredSplineBasis, clamped_knots, evaluate_basis


def build_spectral_design_from_windows(
    task: WindowedTask,
    *,
    basis_fit_x: np.ndarray,
    lag_basis_count: int,
    amplitude_basis_count: int,
    degree: int = 3,
    amplitude_quantiles: tuple[float, float] = (0.01, 0.99),
    amplitude_domains: list[AmplitudeDomain] | None = None,
) -> SpectralDesign:
    """Build a design without reconstructing or joining source sequences.

    ``basis_fit_x`` must contain only quality-valid training rows, each raw row
    once. It is explicit so callers cannot silently fit spline support or Gram
    matrices on validation/test windows.
    """

    histories = np.asarray(task.x_history, dtype=np.float64)
    fit_x = np.asarray(basis_fit_x, dtype=np.float64)
    if histories.ndim != 3:
        raise ValueError("x_history must have shape [sample, variable, lag].")
    n_sample, n_variable, history = histories.shape
    if history != task.L_x:
        raise ValueError("Window history width differs from task.L_x.")
    if fit_x.ndim != 2 or fit_x.shape[1] != n_variable or not len(fit_x):
        raise ValueError("basis_fit_x must be non-empty [train_time, variable].")
    if not np.isfinite(fit_x).all():
        raise ValueError("basis_fit_x contains a non-finite training value.")
    lag_knots = clamped_knots(
        0.0, float(history - 1), lag_basis_count, degree
    )
    lag_basis = evaluate_basis(np.arange(history), lag_knots, degree)
    lag_gram = lag_basis.T @ lag_basis / history
    domains = amplitude_domains or [
        AmplitudeDomain.fit(
            fit_x[:, variable],
            padding_fraction=0.10,
            core_quantiles=amplitude_quantiles,
        )
        for variable in range(n_variable)
    ]
    if len(domains) != n_variable:
        raise ValueError("One amplitude domain is required per variable.")
    bases = [
        CenteredSplineBasis.fit(
            fit_x[:, variable],
            n_basis=amplitude_basis_count,
            degree=degree,
            domain=domains[variable],
            quantiles=amplitude_quantiles,
        )
        for variable in range(n_variable)
    ]
    block_width = lag_basis_count * amplitude_basis_count
    matrix = np.empty((n_sample, n_variable * block_width), dtype=np.float64)
    slices: dict[int, slice] = {}
    amplitude_grams: list[np.ndarray] = []
    for variable, basis in enumerate(bases):
        fit_eval = basis.transform(fit_x[:, variable])
        amplitude_grams.append(fit_eval.T @ fit_eval / len(fit_eval))
        amplitude = basis.transform(histories[:, variable, :].reshape(-1))
        amplitude = amplitude.reshape(n_sample, history, amplitude_basis_count)
        tensor = np.einsum(
            "la,nlb->nab", lag_basis, amplitude, optimize=True
        )
        block = slice(variable * block_width, (variable + 1) * block_width)
        matrix[:, block] = tensor.reshape(n_sample, block_width)
        slices[variable] = block
    return SpectralDesign(
        matrix=matrix,
        variable_slices=slices,
        lag_basis=lag_basis,
        amplitude_bases=bases,
        lag_gram=lag_gram,
        amplitude_grams=amplitude_grams,
        target_indices=np.asarray(task.target_index, dtype=np.int64),
        origin_indices=np.asarray(task.origin_index, dtype=np.int64),
    )
