import numpy as np
import pandas as pd

from prism_benchmark.v21_a import mature_residual_features


def _samples(latest):
    return pd.DataFrame({
        "entity_id": ["run-a"],
        "origin": [20],
        "latest_available_target_index": [latest],
    })


def _residuals():
    return pd.DataFrame({
        "entity_id": ["run-a"] * 20 + ["run-b"],
        "origin": list(range(20)) + [14],
        "residual": np.arange(21, dtype=np.float64),
    })


def test_maturity_uses_latest_available_target_index_including_delay():
    immediate, _, audit = mature_residual_features(
        _samples(19), _residuals(), h_steps=2, w_steps=3,
        delta=1, history=1, maximum_lags=1, residual_mean=0.0,
    )
    delayed, _, _ = mature_residual_features(
        _samples(17), _residuals(), h_steps=2, w_steps=3,
        delta=1, history=1, maximum_lags=1, residual_mean=0.0,
    )
    assert immediate[0, 0] == 15.0
    assert delayed[0, 0] == 13.0
    assert audit["uses_latest_available_target_index"] is True


def test_maturity_never_crosses_entity_boundary():
    features, coverage, _ = mature_residual_features(
        pd.DataFrame({"entity_id": ["missing"], "origin": [20], "latest_available_target_index": [19]}),
        _residuals(), h_steps=2, w_steps=3, delta=1, history=1,
        maximum_lags=1, residual_mean=0.0,
    )
    assert features[0, 0] == 0.0
    assert coverage == 0.0
