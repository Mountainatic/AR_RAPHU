import numpy as np

from prism_benchmark.v21_a import MATURE_RESIDUAL_AR, fit_mature_residual_ar


def test_a_mu_zero_is_exactly_unpenalized_and_has_no_hard_projection():
    x = np.arange(24, dtype=np.float64).reshape(8, 3)
    y = np.linspace(-1.0, 1.0, 8)
    evaluation = x[:3] + 0.25
    plain, plain_contract = fit_mature_residual_ar(x, y, evaluation, alpha=0.1, mu=0.0)
    routed, routed_contract = fit_mature_residual_ar(x, y, evaluation, alpha=0.1, mu=0.0, upstream_predictions=np.column_stack([y, y * y]))
    np.testing.assert_array_equal(plain, routed)
    assert routed_contract["family"] == MATURE_RESIDUAL_AR
    assert routed_contract["hard_feature_residualization"] is False
    assert routed_contract["soft_overlap_mu"] == 0.0
    assert plain_contract["coefficient"] == routed_contract["coefficient"]
