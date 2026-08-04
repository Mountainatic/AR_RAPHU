from __future__ import annotations

import numpy as np

from prism_benchmark.v2_w import _ispline_raw, fit_w_candidate, predict_w_contract


def test_ispline_columns_are_monotone_and_w_identity_is_exact() -> None:
    x = np.linspace(-2, 2, 400)
    columns = _ispline_raw(x, np.array([-1.0, 0.0, 1.0]))
    assert columns.shape[1] > 0
    assert np.all(np.diff(columns, axis=0) >= -1e-12)
    prediction, contract = fit_w_candidate(x, np.sin(x), x, "IDENTITY", 4, 0.0)
    np.testing.assert_array_equal(prediction, x)
    assert contract["family"] == "IDENTITY"


def test_w_contract_round_trip() -> None:
    x=np.linspace(-2,2,400);target=x+0.1*x*x
    for family in ("MONOTONE","NATURAL_CUBIC"):
        prediction,contract=fit_w_candidate(x,target,x,family,4,1e-3,1)
        np.testing.assert_allclose(predict_w_contract(x,contract),prediction,atol=1e-10)
