import numpy as np

from ar_raphu.spectral.design import (
    build_ar_nuisance_design,
    build_spectral_design,
)


def test_spectral_design_is_causal_and_train_centered():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(180, 3))
    targets = np.arange(80, 130)
    design = build_spectral_design(
        x,
        target_indices=targets,
        train_target_stop=100,
        horizon=5,
        L_x=16,
        lag_basis_count=5,
        amplitude_basis_count=8,
    )
    mutated = x.copy()
    mutated[126:] += 1e6
    repeated = build_spectral_design(
        mutated,
        target_indices=targets[:1],
        train_target_stop=100,
        horizon=5,
        L_x=16,
        lag_basis_count=5,
        amplitude_basis_count=8,
    )
    assert design.matrix.shape == (50, 3 * 5 * 8)
    assert np.allclose(design.matrix[0], repeated.matrix[0])
    for basis in design.amplitude_bases:
        assert np.allclose(basis.transform(x[:100, 0]).mean(axis=0), 0.0, atol=0.2)


def test_ar_nuisance_design_uses_origin_history():
    y = np.linspace(-2.0, 2.0, 200)
    targets = np.arange(100, 120)
    design = build_ar_nuisance_design(
        y,
        target_indices=targets,
        train_target_stop=100,
        horizon=5,
        L_y=32,
        lag_basis_count=6,
        amplitude_basis_count=8,
    )
    assert design.shape == (20, 48)
