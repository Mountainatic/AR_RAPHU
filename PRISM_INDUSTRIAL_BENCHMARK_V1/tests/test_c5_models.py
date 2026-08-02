from __future__ import annotations

import numpy as np
import pandas as pd

from prism_benchmark.c3_models import _ridge_block_predict
from prism_benchmark.c5_models import _ridge_path_losses, mature_residual_features, rolling_oof_folds
from prism_benchmark.cpu_selection import mse


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


def test_vectorized_residual_lookup_preserves_last_duplicate_value() -> None:
    residuals = pd.DataFrame(
        {"entity_id": ["a", "a", "a"], "origin": [5, 5, 6], "residual": [1.0, 9.0, 4.0]}
    )
    samples = pd.DataFrame({"entity_id": ["a"], "origin": [8]})
    features = mature_residual_features(samples, residuals, 1, 1, 1, 1, 64, 0.0)
    np.testing.assert_array_equal(features, [[4.0, 9.0]])


def test_residual_ridge_path_reuses_statistics_without_changing_losses() -> None:
    rng = np.random.default_rng(20260802)
    x_train = rng.normal(size=(128, 7))
    y_train = rng.normal(size=128)
    x_evaluation = rng.normal(size=(41, 7))
    y_evaluation = rng.normal(size=41)
    alphas = [1e-8, 1e-4, 1.0, 100.0]
    expected = [
        mse(y_evaluation, _ridge_block_predict(x_train, y_train, x_evaluation, alpha)[0])
        for alpha in alphas
    ]
    actual = _ridge_path_losses(x_train, y_train, x_evaluation, y_evaluation, alphas)
    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-15)
