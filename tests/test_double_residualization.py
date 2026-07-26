import numpy as np

from ar_raphu.spectral.crossfit import forward_crossfit


def test_forward_crossfit_removes_nuisance_predictable_feature_component():
    rng = np.random.default_rng(12)
    n = 1600
    z = np.zeros(n)
    for index in range(1, n):
        z[index] = 0.9 * z[index - 1] + rng.normal()
    psi = np.column_stack([np.ones(n), z, z**2])
    phi = np.column_stack(
        [0.8 * z + rng.normal(scale=0.5, size=n), rng.normal(size=n)]
    )
    y = 1.5 * z + phi @ np.array([0.7, -0.2]) + rng.normal(scale=0.2, size=n)
    result = forward_crossfit(
        y,
        phi,
        psi,
        folds=4,
        initial_nuisance_prefix_targets=500,
        purge_gap=20,
        ridge_candidates=(1e-4, 1e-3, 1e-2, 1e-1),
        nuisance_selection_tail_fraction=0.2,
    )
    before = abs(np.corrcoef(phi[500:, 0], z[500:])[0, 1])
    after = abs(
        np.corrcoef(result.phi_residual[:, 0], z[result.evaluation_indices])[0, 1]
    )
    assert after < before * 0.25
    assert len(result.fold_summaries) == 4
