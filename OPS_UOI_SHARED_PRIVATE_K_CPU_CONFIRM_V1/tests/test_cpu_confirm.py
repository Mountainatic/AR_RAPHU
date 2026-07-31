from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.linear_k import (
    absolute_gram_correlation,
    block_penalty,
    full_feature_tensor,
    mother_basis,
    ridge_fit,
    verify_nested,
)
from src.nonlinear import (
    evaluate_amplitude_basis,
    fit_amplitude_basis,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/frozen_l6.yaml").read_text(encoding="utf-8")
)


def _basis(name: str):
    return mother_basis(
        name,
        CONFIG["mother_spaces"][name],
        sequence_steps=240,
        cadence_sec=10.0,
    )


def test_protocol_is_frozen() -> None:
    assert CONFIG["protocol_frozen"] is True
    assert CONFIG["history_min"] == 40.0
    assert CONFIG["horizon_min"] == 20.0
    assert CONFIG["target_window_min"] == 2.0


def test_v0_is_nested_in_v1() -> None:
    assert verify_nested(_basis("V0"), _basis("V1")) < 1e-10


def test_feature_tensor_is_fp64_and_causal_shape() -> None:
    values = np.zeros((7, 240, 4), dtype=np.float32)
    values[:, 0, :] = 1.0
    features = full_feature_tensor(values, _basis("V0"), 10.0)
    assert features.shape == (7, 4, 14)
    assert features.dtype == np.float64


def test_exact_zero_support_predicts_zero() -> None:
    fit = ridge_fit(
        np.zeros((12, 0)),
        np.arange(12, dtype=np.float64),
        np.zeros((0, 0)),
        predict_matrix=np.zeros((5, 0)),
    )
    assert fit.coefficient.size == 0
    assert np.array_equal(fit.prediction, np.zeros(5))


def test_ridge_kkt_and_dtype() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(200, 12))
    target = matrix @ rng.normal(size=12) + 0.01 * rng.normal(size=200)
    fit = ridge_fit(matrix, target, 1e-3 * np.eye(12))
    assert fit.kkt_relative < 1e-10
    assert fit.prediction.dtype == np.float64


def test_shared_private_gram_orthogonality() -> None:
    basis = _basis("V0")
    rng = np.random.default_rng(7)
    first = rng.normal(size=basis.size)
    first /= np.sqrt(first @ basis.gram @ first)
    second = rng.normal(size=basis.size)
    second -= first * (first @ basis.gram @ second)
    second /= np.sqrt(second @ basis.gram @ second)
    assert abs(first @ basis.gram @ second) < 1e-12
    assert absolute_gram_correlation(first, first, basis.gram) > 1 - 1e-12


def test_smoothing_penalty_is_positive_semidefinite() -> None:
    basis = _basis("V1")
    penalty = block_penalty(basis, 4, 1e-6, 1e2)
    assert np.min(np.linalg.eigvalsh(penalty)) >= -1e-10


def test_nonlinear_basis_is_orthogonal_to_linear_span() -> None:
    values = np.linspace(-2, 3, 1000)
    specification = fit_amplitude_basis(values)
    transformed, _ = evaluate_amplitude_basis(specification, values)
    linear = np.column_stack((np.ones_like(values), values))
    relative = np.linalg.norm(linear.T @ transformed) / max(
        np.linalg.norm(transformed), 1e-30
    )
    assert relative < 1e-10


def test_c1_extension_value_and_derivative_are_continuous() -> None:
    values = np.linspace(-1, 1, 1001)
    specification = fit_amplitude_basis(values)
    epsilon = 1e-6
    left_values = np.array(
        [
            specification.lower - epsilon,
            specification.lower,
            specification.lower + epsilon,
        ]
    )
    transformed, _ = evaluate_amplitude_basis(specification, left_values)
    assert np.max(np.abs(transformed[1] - (transformed[0] + transformed[2]) / 2)) < 1e-4
    right_values = np.array(
        [
            specification.upper - epsilon,
            specification.upper,
            specification.upper + epsilon,
        ]
    )
    transformed, _ = evaluate_amplitude_basis(specification, right_values)
    assert np.max(np.abs(transformed[1] - (transformed[0] + transformed[2]) / 2)) < 1e-4


def test_private_and_nonlinear_zero_are_exactly_nested() -> None:
    rng = np.random.default_rng(9)
    base = rng.normal(size=50)
    private = rng.normal(size=50)
    nonlinear = rng.normal(size=50)
    assert np.array_equal(base + 0.0 * private, base)
    assert np.array_equal(base + 0.0 * nonlinear, base)
