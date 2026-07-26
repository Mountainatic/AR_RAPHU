import numpy as np

from ar_raphu.spectral.weighted_projection import (
    normalized_trapezoidal_weights,
    weighted_tensor_projection,
)


def test_both_weighted_error_decompositions_close():
    lag = np.arange(12, dtype=np.float64)
    amplitude = np.linspace(-2.0, 2.0, 101)
    truth = np.sin(lag[:, None] / 3.0 + amplitude[None, :]) + 0.1 * (
        lag[:, None] * amplitude[None, :]
    )
    lag_basis = np.column_stack([np.ones_like(lag), lag, lag**2])
    amplitude_basis = np.column_stack(
        [np.ones_like(amplitude), amplitude, amplitude**2, amplitude**3]
    )
    result = weighted_tensor_projection(
        truth,
        lag_basis,
        amplitude_basis,
        lag_weights=np.full(len(lag), 1 / len(lag)),
        amplitude_weights=normalized_trapezoidal_weights(amplitude),
    )
    assert result.lag_decomposition_error <= 1e-10
    assert result.amplitude_decomposition_error <= 1e-10
