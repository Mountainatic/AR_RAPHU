import numpy as np

from ar_raphu.spectral.amplitude_domain import AmplitudeDomain
from ar_raphu.spectral.capacity_diagnostics import (
    direct_apply_projected_kernel,
)
from ar_raphu.spectral.design import build_spectral_design
from ar_raphu.synthetic import generate_synthetic_sequence


def test_design_matrix_matches_direct_projected_kernel_application():
    sequence = generate_synthetic_sequence("AR-S1", seed=4, n_samples=300)
    start, stop = sequence.split_target_intervals["validation"]
    targets = np.arange(start, stop)
    train_stop = sequence.split_target_intervals["train"][1]
    train_values = sequence.x[:train_stop, 0]
    domain = AmplitudeDomain.fit(train_values)
    design = build_spectral_design(
        sequence.x[:, [0]],
        target_indices=targets,
        train_target_stop=train_stop,
        horizon=1,
        L_x=64,
        lag_basis_count=32,
        amplitude_basis_count=16,
        amplitude_domains=[domain],
    )
    rng = np.random.default_rng(44)
    coefficients = rng.normal(size=(32, 16))
    via_matrix = design.matrix @ coefficients.reshape(-1)
    via_direct = direct_apply_projected_kernel(
        sequence.x,
        variable=0,
        target_indices=targets,
        horizon=1,
        lag_basis=design.lag_basis,
        amplitude_basis=design.amplitude_bases[0],
        coefficients=coefficients,
    )
    assert np.max(np.abs(via_matrix - via_direct)) <= 1e-10
