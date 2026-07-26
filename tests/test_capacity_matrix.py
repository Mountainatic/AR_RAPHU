from pathlib import Path

import numpy as np
import json

from ar_raphu.spectral.capacity_matrix import (
    apply_tensor_coefficients,
    build_single_variable_matrix,
    select_minimum_validation_mse,
    smoothing_pairs,
)
from ar_raphu.spectral.amplitude_domain import AmplitudeDomain
from ar_raphu.spectral.spline_basis import CenteredSplineBasis


ROOT = Path(__file__).resolve().parents[1]


def test_capacity_smoothing_matrix_and_projected_target_closure():
    config = json.loads(
        (ROOT / "configs" / "spectral_v033.yaml").read_text(encoding="utf-8")
    )
    assert len(smoothing_pairs(config)) == 25
    rng = np.random.default_rng(333)
    matrix = rng.normal(size=(50, 24))
    coefficients = rng.normal(size=(6, 4))
    np.testing.assert_allclose(
        apply_tensor_coefficients(matrix, coefficients),
        matrix @ coefficients.reshape(-1),
    )


def test_capacity_selection_uses_validation_contribution_mse_only():
    selected = select_minimum_validation_mse(
        [
            {"validation_contribution_mse": 0.2, "configuration_order": 0},
            {"validation_contribution_mse": 0.1, "configuration_order": 1},
        ]
    )
    assert selected["configuration_order"] == 1


def test_arbitrary_lag_matrix_matches_explicit_tensor_application():
    values = np.linspace(-1.0, 1.0, 80)
    domain = AmplitudeDomain.fit(values, padding_fraction=0.10)
    basis = CenteredSplineBasis.fit(
        values, n_basis=6, degree=3, domain=domain
    )
    lag = np.eye(8)
    origins = np.arange(20, 70)
    matrix = build_single_variable_matrix(
        values,
        origin_indices=origins,
        lag_basis=lag,
        amplitude_basis=basis,
    )
    theta = np.arange(48, dtype=np.float64).reshape(8, 6) / 48
    windows = values[origins[:, None] - np.arange(8)[None, :]]
    amplitude = basis.transform(windows.reshape(-1)).reshape(50, 8, 6)
    expected = np.einsum("la,nlb,ab->n", lag, amplitude, theta)
    np.testing.assert_allclose(
        matrix @ theta.reshape(-1), expected, atol=1.0e-14
    )
