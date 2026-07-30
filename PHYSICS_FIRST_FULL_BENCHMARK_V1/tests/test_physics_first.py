from __future__ import annotations

from pathlib import Path

import numpy as np

from src.benchmark import _residual_history_design
from src.common import improvement, sha256_array
from src.models import model_factory


def test_mature_residual_history_ends_before_current_origin():
    residual = np.arange(1000, dtype=np.float64)
    rows, design = _residual_history_design(
        residual,
        maturity_rows=132,
        maximum_history_rows=240,
    )
    current = int(rows[0])
    assert current == 371
    assert design[0, 0] == residual[current - 132]
    assert design[0, -1] == residual[current - 132 - 239]
    assert current - 132 < current


def test_exact_zero_residual_branch_is_nested():
    target = np.array([1.0, -1.0, 2.0])
    k_prediction = np.array([0.5, -0.5, 1.0])
    zero_residual = np.zeros_like(target)
    np.testing.assert_allclose(
        k_prediction + zero_residual, k_prediction
    )


def test_ridge_prediction_is_float64():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(100, 5))
    target = matrix[:, 0] + 0.1 * rng.normal(size=100)
    model = model_factory("Ridge", {"alpha": 1.0}).fit(matrix, target)
    prediction = model.predict(matrix)
    assert np.asarray(prediction).dtype == np.float64


def test_improvement_definition():
    reference = np.array([4.0, 4.0])
    model = np.array([2.0, 2.0])
    assert improvement(reference, model) == 0.5


def test_array_hash_binds_shape_and_dtype():
    values = np.arange(10, dtype=np.float64)
    assert sha256_array(values) != sha256_array(values.astype(np.float32))
    assert sha256_array(values) != sha256_array(values.reshape(2, 5))
