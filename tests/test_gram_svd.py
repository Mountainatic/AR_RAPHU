import numpy as np

from ar_raphu.spectral.gram_svd import gram_whitened_svd


def test_gram_svd_recovers_rank_and_tail_energy():
    rng = np.random.default_rng(8)
    left = rng.normal(size=(5, 2))
    right = rng.normal(size=(2, 8))
    theta = left @ right
    lag_gram = np.diag(np.linspace(0.5, 1.5, 5))
    amplitude_gram = np.diag(np.linspace(0.8, 1.2, 8))
    spectrum = gram_whitened_svd(theta, lag_gram, amplitude_gram)
    assert spectrum.singular_values[2] < 1e-10
    assert spectrum.tail_energy_ratio(2) < 1e-20
    reconstructed = spectrum.truncate(2)
    assert np.allclose(reconstructed, theta, atol=1e-9)
