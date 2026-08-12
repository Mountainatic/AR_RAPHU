from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prism_benchmark.neurobem_exact_training as exact
from prism_benchmark.neurobem_literature import (
    MOTOR_COLUMNS,
    TRACK_B_STATE_COLUMNS,
    LiteratureTrajectory,
    normalize_quaternion,
)


def _trajectory(rows: int = 36) -> LiteratureTrajectory:
    t = np.arange(rows, dtype=np.float64)
    q = normalize_quaternion(np.column_stack((np.ones(rows), 0.002 * t, 0.001 * t, -0.001 * t)))
    state = np.column_stack((
        10.0 + t, 20.0 + t, 30.0 + t,
        q,
        40.0 + t, 50.0 + t, 60.0 + t,
    ))
    frame = pd.DataFrame(state, columns=TRACK_B_STATE_COLUMNS)
    for index, column in enumerate(MOTOR_COLUMNS, 1):
        frame[column] = 1000.0 * index + t
    return LiteratureTrajectory("fixture.csv", "train", frame)


def _collect(monkeypatch: pytest.MonkeyPatch, branch: str, prediction: np.ndarray):
    def fake(_contracts, _route, selected_branch, xk, _state):
        assert selected_branch == branch
        return np.broadcast_to(prediction, (len(xk), len(prediction))).copy()
    monkeypatch.setattr(exact, "_decoupled_branch_prediction", fake)
    return list(exact.iter_published_training_rollin(
        {}, "PF_KC", branch, _trajectory(), history=20, unroll=10,
    ))


def test_training_velocity_unroll_uses_predicted_velocity(monkeypatch):
    rows = _collect(monkeypatch, "velocity", np.array([1., 2., 3., 4., 5., 6.]))
    first_state = rows[0][1].reshape(-1, 20, 10)
    second_state = rows[1][1].reshape(-1, 20, 10)
    np.testing.assert_allclose(second_state[:, -1, :3], first_state[:, -1, :3] + [1., 2., 3.])


def test_training_velocity_unroll_uses_predicted_angular_rate(monkeypatch):
    rows = _collect(monkeypatch, "velocity", np.array([1., 2., 3., 4., 5., 6.]))
    first_state = rows[0][1].reshape(-1, 20, 10)
    second_state = rows[1][1].reshape(-1, 20, 10)
    np.testing.assert_allclose(second_state[:, -1, 7:10], first_state[:, -1, 7:10] + [4., 5., 6.])


def test_training_velocity_unroll_injects_gt_attitude(monkeypatch):
    rows = _collect(monkeypatch, "velocity", np.ones(6))
    second = rows[1][1].reshape(-1, 20, 10)
    measured = _trajectory().frame.loc[:, TRACK_B_STATE_COLUMNS].to_numpy(dtype=np.float64)
    measured[:, 3:7] = normalize_quaternion(measured[:, 3:7])
    np.testing.assert_allclose(second[:, -1, 3:7], measured[20:26, 3:7])


def test_training_attitude_unroll_uses_predicted_attitude(monkeypatch):
    rows = _collect(monkeypatch, "attitude", np.array([0.01, -0.02, 0.03]))
    first = rows[0][1].reshape(-1, 20, 10)
    second = rows[1][1].reshape(-1, 20, 10)
    assert not np.allclose(second[:, -1, 3:7], first[:, -1, 3:7])
    np.testing.assert_allclose(np.linalg.norm(second[:, -1, 3:7], axis=1), 1.0, atol=1e-14)


def test_training_attitude_unroll_injects_gt_velocity(monkeypatch):
    rows = _collect(monkeypatch, "attitude", np.zeros(3))
    second = rows[1][1].reshape(-1, 20, 10)
    measured = _trajectory().frame.loc[:, TRACK_B_STATE_COLUMNS].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(second[:, -1, :3], measured[20:26, :3])


def test_training_attitude_unroll_injects_gt_angular_rate(monkeypatch):
    rows = _collect(monkeypatch, "attitude", np.zeros(3))
    second = rows[1][1].reshape(-1, 20, 10)
    measured = _trajectory().frame.loc[:, TRACK_B_STATE_COLUMNS].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(second[:, -1, 7:10], measured[20:26, 7:10])


def test_training_uses_future_control_only(monkeypatch):
    rows = _collect(monkeypatch, "velocity", np.zeros(6))
    xk0, xk1 = rows[0][0], rows[1][0]
    # The first 80 K columns are the squared 20x4 motor history.  Step one
    # shifts by one published future command and does not expose another field.
    np.testing.assert_allclose(xk1[:, :76], xk0[:, 4:80])


def test_training_never_reads_future_target_residual():
    audit = exact.audit_published_training_contract()
    assert audit["future_target_residual"] is False
    assert audit["velocity"]["future_owned_ground_truth"] is False
    assert audit["attitude"]["future_owned_ground_truth"] is False


def test_training_contract_matches_published_evaluator_contract():
    assert exact.published_training_state_updates.__doc__
    assert exact.audit_published_training_contract()["state_update_implementation_shared_with_evaluator"]


def test_H_20():
    assert exact.PUBLISHED_HISTORY == 20


def test_U_10():
    assert exact.PUBLISHED_TRAINING_UNROLL == 10


def test_T_60():
    assert exact.PUBLISHED_TEST_ROLLOUT == 60


def test_A_disabled():
    assert exact.audit_published_training_contract()["a_enabled"] is False


def test_quaternion_norm_preserved(monkeypatch):
    rows = _collect(monkeypatch, "attitude", np.array([0.2, -0.1, 0.3]))
    for _, state, _ in rows[1:]:
        q = state.reshape(-1, 20, 10)[:, -1, 3:7]
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-13)


def test_streaming_ridge_matches_materialized_fit():
    from prism_benchmark.neurobem_linear import fit_numerical_ridge
    rng = np.random.default_rng(9)
    x = rng.normal(size=(101, 7))
    y = rng.normal(size=(101, 3))
    moments = exact.StreamingRidgeMoments(7, 3)
    moments.update(x[:37], y[:37])
    moments.update(x[37:], y[37:])
    streamed = moments.solve([1e-4], 1e12, 1e-8)
    direct = fit_numerical_ridge(x, y, [1e-4], 1e12, 1e-8)
    np.testing.assert_allclose(streamed.feature_mean, direct.feature_mean, atol=1e-14)
    np.testing.assert_allclose(streamed.feature_scale, direct.feature_scale, atol=1e-14)
    np.testing.assert_allclose(streamed.coefficient_standardized, direct.coefficient_standardized, atol=2e-13)


def test_nonfinite_training_design_is_not_clipped_or_hidden():
    moments = exact.StreamingRidgeMoments(2, 1)
    with pytest.raises(exact.NonfinitePublishedTrainingError, match="NONFINITE_TRAINING_DESIGN"):
        moments.update(np.array([[np.inf, 0.0]]), np.zeros((1, 1)))


def test_zero_ridge_is_exact_neutral_boundary():
    from prism_benchmark.neurobem_linear import RidgeContract, predict_ridge
    template = RidgeContract(
        1e-4, np.array([2.0, 3.0]), np.array([4.0, 5.0]), np.array([7.0]),
        np.ones((2, 1)), 2.0, 1e-12, 10,
    )
    neutral = exact._zero_ridge(template)
    np.testing.assert_array_equal(predict_ridge(neutral, np.array([[99.0, -8.0]])), np.zeros((1, 1)))
