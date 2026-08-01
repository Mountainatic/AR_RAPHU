from __future__ import annotations

import numpy as np
import pandas as pd

from prism_benchmark.c5_models import mature_residual_features, rolling_oof_folds


def _samples() -> pd.DataFrame:
    rows = []
    for entity in ("a", "b"):
        for origin in range(20, 120):
            rows.append({"entity_id": entity, "origin": origin, "dependency_start": origin - 10, "dependency_stop_exclusive": origin + 3})
    return pd.DataFrame(rows)


def test_rolling_oof_is_expanding_and_disjoint() -> None:
    samples = _samples()
    for train, validation in rolling_oof_folds(samples, 4):
        assert not set(train).intersection(validation)
        for entity in ("a", "b"):
            train_rows = samples.iloc[train].query("entity_id == @entity")
            validation_rows = samples.iloc[validation].query("entity_id == @entity")
            assert train_rows["dependency_stop_exclusive"].max() <= validation_rows["dependency_start"].min()


def test_mature_residual_features_never_use_unmatured_rows() -> None:
    residuals = pd.DataFrame({"entity_id": ["a"] * 10, "origin": np.arange(10), "residual": np.arange(10, dtype=float)})
    samples = pd.DataFrame({"entity_id": ["a"], "origin": [10]})
    features = mature_residual_features(samples, residuals, 2, 2, 1, 3, 64, 0.0)
    np.testing.assert_array_equal(features, [[6.0, 5.0, 4.0, 3.0]])


def test_missing_residual_history_is_centered_zero() -> None:
    residuals = pd.DataFrame({"entity_id": ["a"], "origin": [5], "residual": [7.0]})
    samples = pd.DataFrame({"entity_id": ["b"], "origin": [10]})
    features = mature_residual_features(samples, residuals, 1, 1, 1, 4, 64, 7.0)
    assert np.all(features == 0.0)
