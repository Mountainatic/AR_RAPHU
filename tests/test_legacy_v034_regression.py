import numpy as np

from ar_raphu.spectral.gram_svd import gram_whitened_svd
from ar_raphu.spectral.rank_profile import build_rank_profile
from ar_raphu.spectral.solver import solve_full_kernel


def test_v034_solver_spectrum_and_rank_snapshot():
    design = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    target = 2.0 + 3.0 * design[:, 0] - 4.0 * design[:, 1]
    fit = solve_full_kernel(
        design,
        target,
        np.eye(2) * 1.0e-12,
        fit_intercept=True,
    )
    np.testing.assert_allclose(fit.coefficients, [3.0, -4.0], atol=1.0e-8)
    np.testing.assert_allclose(fit.predictions, target, atol=1.0e-8)
    assert fit.relative_kkt_residual <= 1.0e-12

    spectrum = gram_whitened_svd(
        np.diag([3.0, 1.0]),
        np.eye(2),
        np.eye(2),
    )
    np.testing.assert_allclose(spectrum.singular_values, [3.0, 1.0])
    profile = build_rank_profile(
        spectrum.singular_values,
        rank_max=2,
        budgets=(0.10, 0.05, 0.02),
    )
    assert profile.effective_ranks == {0.10: 2, 0.05: 2, 0.02: 2}
    np.testing.assert_allclose(profile.tail_curve, [np.sqrt(0.1), 0.0])
