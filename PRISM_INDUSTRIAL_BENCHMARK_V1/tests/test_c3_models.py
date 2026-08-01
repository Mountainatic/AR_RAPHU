from __future__ import annotations

import numpy as np

from prism_benchmark.c3_models import _narx_expand, _nonlinear_features, _ridge_block_predict


def test_block_ridge_fp64_certificate_and_prediction() -> None:
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = 2.0 * x[:, 0] - x[:, 1] + 3.0
    prediction, certificate = _ridge_block_predict(x, y, x, 1e-8)
    np.testing.assert_allclose(prediction, y, atol=1e-7)
    assert certificate["relative_kkt"] <= 1e-8
    assert np.isfinite(certificate["condition_number"])


def test_hammerstein_feature_families_are_exactly_nested() -> None:
    train = np.linspace(-2.0, 2.0, 20, dtype=np.float64).reshape(10, 2)
    linear_train, linear_eval = _nonlinear_features(train, train, "linear")
    quadratic_train, quadratic_eval = _nonlinear_features(train, train, "quadratic_centered")
    cubic_train, cubic_eval = _nonlinear_features(train, train, "cubic_centered")
    np.testing.assert_allclose(quadratic_train[:, :2], linear_train)
    np.testing.assert_allclose(cubic_train[:, : quadratic_train.shape[1]], quadratic_train)
    np.testing.assert_allclose(linear_eval, linear_train)
    np.testing.assert_allclose(quadratic_eval, quadratic_train)
    np.testing.assert_allclose(cubic_eval, cubic_train)


def test_linear_narx_expansion_has_linear_and_square_terms_only() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 30))
    y = x[:, 7] + 0.1 * rng.normal(size=100)
    train, evaluation, indices = _narx_expand(x, y, x[:5], maximum=20)
    assert train.shape == (100, 40)
    assert evaluation.shape == (5, 40)
    assert len(indices) == 20
    np.testing.assert_allclose(train[:, 20:], np.square(train[:, :20]))
