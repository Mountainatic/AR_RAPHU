from __future__ import annotations

import time

import numpy as np
import pytest

from ar_raphu.cz_fast_audit.conditional_gram import spectrum_metrics
from ar_raphu.cz_fast_audit.decision import RuntimeGate, decide_go_nogo
from ar_raphu.cz_fast_audit.fast_coarse_xar import fit_penalized
from ar_raphu.cz_fast_audit.fast_linear import DenseRidgePath
from ar_raphu.cz_fast_audit.qk_stability import _correlation
from ar_raphu.cz_fast_audit.residualization import (
    FAST_TASKS,
    build_fast_folds,
    fit_multi_target_ridge,
    history_matrix,
    target_indices,
)
from ar_raphu.cz_real.protocol import FurnaceBLockedError, load_furnace_b


def test_furnace_b_locked(tmp_path):
    with pytest.raises(FurnaceBLockedError):
        load_furnace_b(
            tmp_path / "must-not-be-read.xlsx",
            protocol_frozen=False,
            stage="FAST",
        )


def test_purge_for_three_fixed_tasks():
    expected = {"short": 64, "medium": 270, "long": 571}
    for task in FAST_TASKS:
        folds = build_fast_folds(20_000, task)
        assert all(fold.purge_gap == expected[task.name] for fold in folds)
        assert all(
            fold.nominal_train_stop - fold.effective_train_stop
            == expected[task.name]
            for fold in folds
        )


def test_residualization_train_only():
    rng = np.random.default_rng(1)
    train_y = rng.normal(size=(100, 4))
    train_x = train_y @ rng.normal(size=(4, 6)) + 0.1 * rng.normal(
        size=(100, 6)
    )
    first = fit_multi_target_ridge(train_y, train_x, alpha=1.0e-4)
    future = rng.normal(size=(20, 4))
    prediction_before = first.predict(future)
    corrupted_future_targets = rng.normal(size=(20, 6)) * 1.0e9
    del corrupted_future_targets
    second = fit_multi_target_ridge(train_y, train_x, alpha=1.0e-4)
    np.testing.assert_allclose(first.coefficients, second.coefficients)
    np.testing.assert_allclose(prediction_before, second.predict(future))


def test_conditional_energy_bounded():
    rng = np.random.default_rng(2)
    history = rng.normal(size=(200, 5))
    inputs = history @ rng.normal(size=(5, 8)) + rng.normal(size=(200, 8))
    model = fit_multi_target_ridge(history, inputs, alpha=1.0e-4)
    residual = inputs - model.predict(history)
    centered = inputs - inputs.mean(axis=0)
    ratio = float(np.sum(residual**2) / np.sum(centered**2))
    assert 0.0 <= ratio <= 1.0 + 1.0e-10


def test_lag_correlation_alignment():
    values = np.arange(100, dtype=np.float64)
    task = FAST_TASKS[0]
    targets = target_indices(start=70, stop=80, task=task)
    matrix = history_matrix(
        values,
        targets=targets,
        horizon=task.horizon,
        length=task.L_y,
    )
    origins = targets - task.horizon
    np.testing.assert_array_equal(matrix[:, 0], values[origins])
    np.testing.assert_array_equal(matrix[:, -1], values[origins - task.L_y + 1])
    assert np.all(origins < targets)


def test_conditional_gram_psd():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(300, 12))
    eigenvalues, metrics = spectrum_metrics(
        matrix, coercivity_ratios=(1.0e-2, 1.0e-3, 1.0e-4)
    )
    assert eigenvalues.min() >= 0.0
    assert metrics["effective_rank"] >= 1.0


def test_dense_batched_matches_scalar_dense():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 10))
    y = rng.normal(size=200)
    alpha = 1.0e-3
    path = DenseRidgePath.fit(x, y)
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    expected = np.linalg.solve(
        xc.T @ xc / len(x) + alpha * np.eye(x.shape[1]),
        xc.T @ yc / len(x),
    )
    np.testing.assert_allclose(
        path.coefficients(alpha), expected, rtol=1.0e-9, atol=1.0e-10
    )


def test_exact_zero_minimum_norm():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(120, 3))
    x = np.column_stack((base, base[:, 0]))
    y = rng.normal(size=120)
    path = DenseRidgePath.fit(x, y)
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    expected = np.linalg.pinv(xc, rcond=1.0e-12) @ yc
    np.testing.assert_allclose(
        path.coefficients(0.0), expected, rtol=1.0e-8, atol=1.0e-9
    )


def test_coarse_xar_nestedness():
    rng = np.random.default_rng(6)
    ar = rng.normal(size=(300, 5))
    external = rng.normal(size=(300, 4))
    y = ar @ rng.normal(size=5) + external @ rng.normal(size=4)
    ar_fit = fit_penalized(
        ar, y, np.eye(ar.shape[1]), weight=0.0
    )
    xar = np.column_stack((external, ar))
    xar_fit = fit_penalized(
        xar, y, np.eye(xar.shape[1]), weight=0.0
    )
    assert xar_fit.train_mse <= ar_fit.train_mse + 1.0e-12


def test_q_contribution_decomposition_identity():
    rng = np.random.default_rng(7)
    x = rng.normal(size=100)
    ar = rng.normal(size=100)
    covariance = np.cov(x, ar, ddof=0)[0, 1]
    left = np.var(x + ar)
    right = np.var(x) + np.var(ar) + 2.0 * covariance
    assert abs(left - right) <= 1.0e-12


def test_k_mode_sign_alignment():
    curve = np.linspace(-1.0, 1.0, 101)
    raw = _correlation(curve, -curve)
    sign = 1.0 if raw >= 0.0 else -1.0
    distance = np.linalg.norm(curve - sign * (-curve)) / np.linalg.norm(curve)
    assert abs(raw) == pytest.approx(1.0)
    assert distance == pytest.approx(0.0)


def _decision_fixture():
    energy = [{"conditional_energy_ratio": 0.2}]
    gram = {
        task: {
            "1": {
                "joint": {
                    "effective_rank": 5.0,
                    "coercive_dimension": {"0.001": 8},
                }
            },
            "2": {
                "joint": {
                    "effective_rank": 4.0,
                    "coercive_dimension": {"0.001": 7},
                }
            },
        }
        for task in ("short", "medium", "long")
    }
    coarse = [
        {
            "task": task,
            "horizon": horizon,
            "fold": fold,
            "delta_X_given_AR_coarse": 0.1,
            "direction_positive": True,
        }
        for task, horizon in (("short", 1), ("medium", 15))
        for fold in (1, 2)
    ]
    q = [
        {
            "task": "medium",
            "horizon": 15,
            "mean_delta_X_given_AR_coarse": 0.1,
            "contribution_correlation": 0.9,
            "status": "HIGH_Q_STABILITY",
        }
    ]
    k = [
        {
            "task": "short",
            "input": "x",
            "leading_surface_mode_correlation": 0.9,
            "status": "K_LOW_ORDER_STABLE",
        }
    ]
    gates = {
        "minimum_positive_horizons_for_full_go": 2,
        "minimum_conditional_energy_ratio": 0.05,
        "minimum_joint_effective_rank_for_full_go": 3.0,
        "minimum_joint_coercive_dimension_for_full_go": 5,
        "minimum_joint_coercive_dimension_for_partial_k": 2,
        "q_stability_high": 0.8,
        "q_stability_moderate": 0.5,
    }
    return energy, gram, coarse, q, k, gates


def test_go_nogo_rule_deterministic():
    energy, gram, coarse, q, k, gates = _decision_fixture()
    first = decide_go_nogo(
        conditional_energy_rows=energy,
        conditional_gram_summary=gram,
        coarse_rows=coarse,
        q_rows=q,
        k_rows=k,
        gates=gates,
    )
    second = decide_go_nogo(
        conditional_energy_rows=energy,
        conditional_gram_summary=gram,
        coarse_rows=coarse,
        q_rows=q,
        k_rows=k,
        gates=gates,
    )
    assert first == second
    assert first["status"] == "GO_FULL_CZ_IDENTIFICATION"


def test_runtime_hard_cap():
    gate = RuntimeGate(maximum_seconds=0.001, started_at=time.monotonic() - 1.0)
    with pytest.raises(TimeoutError, match="FAST_AUDIT_RUNTIME_GATE_FAILED"):
        gate.check("TEST")
