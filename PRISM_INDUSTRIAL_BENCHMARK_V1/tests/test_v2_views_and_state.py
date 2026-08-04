from __future__ import annotations

import numpy as np

from prism_benchmark.v2_state import _quadratic, _standardized_fit


def test_state_quadratic_has_no_pairwise_interactions() -> None:
    x = np.arange(30, dtype=np.float64).reshape(10, 3)
    result = _quadratic(x, 2)
    assert result.shape == (10, 5)
    np.testing.assert_array_equal(result[:, 3:], x[:, :2] ** 2)


def test_state_fit_is_fp64_and_finite() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 5))
    y = x[:, 0] - 2 * x[:, 2]
    prediction, contract = _standardized_fit(x, y, x, 1e-8)
    assert prediction.dtype == np.float64
    assert np.mean((prediction - y) ** 2) < 1e-8
    assert contract["numerical_certificate"]["status"] in {"PASS", "PASS_WITH_WARNING"}
