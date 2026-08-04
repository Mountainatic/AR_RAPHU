from __future__ import annotations

import numpy as np

from prism_benchmark.v2_w import _ispline_raw, fit_w_candidate


def test_ispline_columns_are_monotone_and_w_identity_is_exact() -> None:
    x = np.linspace(-2, 2, 400)
    columns = _ispline_raw(x, np.array([-1.0, 0.0, 1.0]))
    assert columns.shape[1] > 0
    assert np.all(np.diff(columns, axis=0) >= -1e-12)
    prediction, contract = fit_w_candidate(x, np.sin(x), x, "IDENTITY", 4, 0.0)
    np.testing.assert_array_equal(prediction, x)
    assert contract["family"] == "IDENTITY"
