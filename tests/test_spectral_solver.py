import numpy as np

from ar_raphu.spectral.penalties import tensor_penalty
from ar_raphu.spectral.solver import solve_full_kernel


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
