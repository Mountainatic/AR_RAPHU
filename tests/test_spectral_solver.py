import numpy as np

from ar_raphu.spectral.penalties import tensor_penalty
from ar_raphu.spectral.solver import solve_full_kernel, solve_full_kernel_pcg


def test_fp64_cholesky_solver_has_small_kkt_residual():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(500, 24))
    truth = rng.normal(size=24)
    y = x @ truth + 0.01 * rng.normal(size=500)
    lag_gram = np.eye(3)
    amplitude_grams = [np.eye(4), np.eye(4)]
    penalty = tensor_penalty(
        lag_gram,
        amplitude_grams,
        lag_smoothness=1e-4,
        amplitude_smoothness=1e-4,
        ridge_weight=1e-6,
    )
    fit = solve_full_kernel(x, y, penalty)
    assert fit.relative_kkt_residual <= 1e-8
    assert np.mean((fit.predictions - y) ** 2) < 0.01


def test_fp64_cholesky_solver_fits_unpenalized_intercept():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(400, 12))
    truth = rng.normal(size=12)
    y = 3.25 + x @ truth
    fit = solve_full_kernel(
        x,
        y,
        np.eye(12) * 1e-10,
        fit_intercept=True,
    )
    assert abs(fit.intercept - 3.25) <= 1e-8
    assert np.max(np.abs(fit.predictions - y)) <= 1e-7


def test_pcg_matches_direct_on_random_200_dimensional_problem():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(700, 200))
    target = rng.normal(size=700)
    raw = rng.normal(size=(200, 200))
    penalty = raw.T @ raw / 200 + 0.1 * np.eye(200)
    direct = solve_full_kernel(
        x,
        target,
        penalty,
        compute_condition_number=False,
    )
    pcg = solve_full_kernel_pcg(
        x,
        target,
        penalty,
        relative_tolerance=1e-10,
        max_iterations=2000,
        block_slices=tuple(slice(start, start + 50) for start in range(0, 200, 50)),
    )
    relative_difference = np.linalg.norm(
        pcg.coefficients - direct.coefficients
    ) / np.linalg.norm(direct.coefficients)
    assert pcg.converged
    assert relative_difference <= 1e-7
