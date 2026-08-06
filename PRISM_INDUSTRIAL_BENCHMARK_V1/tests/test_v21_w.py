import numpy as np

from prism_benchmark.v21_w import (
    IDENTITY,
    NATURAL_CUBIC,
    fit_w_correction,
    predict_w_correction,
)


def test_identity_is_exact_zero_correction():
    values = np.linspace(-1.0, 1.0, 20)
    correction, contract = fit_w_correction(values, values, values, family=IDENTITY)
    np.testing.assert_array_equal(correction, np.zeros_like(values))
    assert contract["hard_feature_residualization"] is False


def test_mu_zero_matches_unpenalized_overlap_path_and_round_trips():
    latent = np.linspace(-2.0, 2.0, 80)
    target = 0.2 * latent**2
    correction, contract = fit_w_correction(
        latent, target, latent, family=NATURAL_CUBIC, knot_count=4,
        smoothness=1e-3, mu=0.0, upstream_predictions=latent,
    )
    np.testing.assert_allclose(predict_w_correction(latent, contract), correction, rtol=1e-11, atol=1e-11)
    assert contract["basis"]["hard_feature_residualization"] is False
    assert "projection" not in contract["basis"]


def test_larger_soft_overlap_does_not_increase_training_overlap():
    latent = np.linspace(-2.0, 2.0, 100)
    target = latent + 0.1 * latent**2
    _, free = fit_w_correction(
        latent, target, latent, family=NATURAL_CUBIC, knot_count=4,
        smoothness=1e-4, mu=0.0, upstream_predictions=latent,
    )
    _, guarded = fit_w_correction(
        latent, target, latent, family=NATURAL_CUBIC, knot_count=4,
        smoothness=1e-4, mu=30.0, upstream_predictions=latent,
    )
    assert guarded["soft_overlap_norm"] <= free["soft_overlap_norm"] + 1e-10
