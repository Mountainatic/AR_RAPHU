import numpy as np

from ar_raphu.spectral.weighted_projection import (
    normalized_trapezoidal_weights,
    orthogonal_projector,
    weighted_tensor_projection,
)


def test_weighted_projector_is_symmetric_and_idempotent():
    grid = np.linspace(-2.0, 3.0, 41)
    basis = np.column_stack([np.ones_like(grid), grid, grid**2])
    projector = orthogonal_projector(
        basis, normalized_trapezoidal_weights(grid)
    )
    np.testing.assert_allclose(projector, projector.T, atol=1e-12)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)


def test_identity_lag_has_zero_marginal_error():
    rng = np.random.default_rng(33)
    truth = rng.normal(size=(8, 51))
    amplitudes = np.linspace(-1.0, 1.0, 51)
    amplitude_basis = np.column_stack(
        [np.ones_like(amplitudes), amplitudes, amplitudes**2]
    )
    result = weighted_tensor_projection(
        truth,
        np.eye(8),
        amplitude_basis,
        lag_weights=np.full(8, 1 / 8),
        amplitude_weights=normalized_trapezoidal_weights(amplitudes),
    )
    assert result.epsilon_lag < 1e-12
    assert result.epsilon_lag_given_amplitude < 1e-12
