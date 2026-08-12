from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.neurobem_literature import (
    CANONICAL_W_CANDIDATES,
    FORMAL_ROUTE_IDS,
    K_CHANNEL_REGISTRY,
    TRACK_A_COLUMNS,
    LatentWContract,
    assert_no_future_state_access,
    candidate_binding_audit,
    compose_quaternion_increment,
    delta_q_metric,
    delta_z_metric,
    fit_route_contracts,
    fit_track_b_route_contracts,
    force_torque_metrics,
    latent_w_basis,
    legacy_aero_w_classification,
    normalize_quaternion,
    official_prediction_force_torque,
    official_prediction_ground_truth_force_torque,
    resample_track_b_100hz,
    route_contract_from_json,
    route_contract_to_json,
    select_w_family,
    select_track_b_w_family_rollout,
    track_a_design,
    track_a_force_torque_target,
    track_b_design,
    track_b_rollout,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "PRISM_V2_1_1_NEUROBEM_LITERATURE_ALIGNED_DUAL_BENCHMARK_PACKAGE"


def frame(rows: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, len(TRACK_A_COLUMNS)))
    result = pd.DataFrame(values, columns=TRACK_A_COLUMNS)
    result["t"] = np.arange(rows) * 0.0025
    q = normalize_quaternion(rng.normal(size=(rows, 4)))
    result.loc[:, ["quat w", "quat x", "quat y", "quat z"]] = q
    result.loc[:, ["mot 1", "mot 2", "mot 3", "mot 4"]] = 1000.0 + rng.normal(0, 30, size=(rows, 4))
    return result


def small_contracts():
    current = frame(170)
    xk, state, y, _ = track_b_design(resample_track_b_100hz(current), history=5)
    return fit_track_b_route_contracts(
        xk,
        state,
        y,
        "SIGNED_QUADRATIC_LATENT",
        [1e-8, 1e-4, 1.0],
        1e16,
        1e-7,
        history=5,
    )


def test_theory_compliance_w_latent_only():
    signature = inspect.signature(latent_w_basis)
    assert tuple(signature.parameters) == ("family", "latent", "knots")
    source = inspect.getsource(latent_w_basis)
    for forbidden in ("velocity", "body_rate", "speed", "context"):
        assert forbidden not in source
    latent = np.array([[-2.0, 3.0]])
    np.testing.assert_allclose(latent_w_basis("SIGNED_QUADRATIC_LATENT", latent), [[-4.0, 9.0]])


def test_registered_context_k_and_interpretation_labels():
    assert K_CHANNEL_REGISTRY["motor_actuator"]["interpretation"] == "ACTUATOR_PHYSICS_CONSISTENCY"
    for channel in ("linear_velocity_context", "attitude_context", "body_rate_context"):
        assert K_CHANNEL_REGISTRY[channel]["interpretation"] == "PREDICTIVE_MOTION_CONTEXT"
        assert "CAUSAL" not in K_CHANNEL_REGISTRY[channel]["interpretation"]
    assert K_CHANNEL_REGISTRY["attitude_context"]["tracks"] == ["B"]


def test_track_a_published_information_set_excludes_attitude():
    original = frame()
    changed = original.copy()
    changed.loc[:, ["quat w", "quat x", "quat y", "quat z"]] = normalize_quaternion(
        np.random.default_rng(77).normal(size=(len(changed), 4))
    )
    xk_original, state_original, y_original, origin_original = track_a_design(original)
    xk_changed, state_changed, y_changed, origin_changed = track_a_design(changed)
    np.testing.assert_array_equal(origin_original, origin_changed)
    np.testing.assert_allclose(xk_original, xk_changed)
    np.testing.assert_allclose(state_original, state_changed)
    np.testing.assert_allclose(y_original, y_changed)
    np.testing.assert_array_equal(state_original.reshape(len(state_original), -1, 10)[:, :, 3:7], 0.0)


def test_cross_context_terms_belong_to_c_and_w_is_latent_curvature():
    source = inspect.getsource(__import__("prism_benchmark.neurobem_literature", fromlist=["_context_design"])._context_design)
    assert "pairwise" in source and "interactions" in source
    assert CANONICAL_W_CANDIDATES == (
        "IDENTITY_CORRECTION",
        "NATURAL_CUBIC_LATENT",
        "SIGNED_QUADRATIC_LATENT",
    )


def test_old_aero_w_preserved_as_historical_extension():
    assert legacy_aero_w_classification("SIGNED_QUADRATIC_AERO_CONTEXT") == "AERODYNAMIC_CONTEXT_W_EXTENSION_DIAGNOSTIC"
    assert legacy_aero_w_classification("NATURAL_CUBIC_SPEED_CONTEXT") == "AERODYNAMIC_CONTEXT_W_EXTENSION_DIAGNOSTIC"


def test_track_a_6d_target_units_frames_and_metrics():
    current = frame()
    target = track_a_force_torque_target(current)
    assert target.shape == (len(current), 6)
    assert np.isfinite(target).all()
    assert force_torque_metrics(target, target) == {key: 0.0 for key in ("Fxy", "Fz", "Mxy", "Mz", "F", "M")}


def test_track_a_official_prediction_column_semantics():
    current = frame(30)
    for index, name in enumerate((
        "predicted_fx", "predicted_fy", "predicted_fz", "predicted_tx", "predicted_ty", "predicted_tz",
        "error_residual_fx", "error_residual_fy", "error_residual_fz",
        "error_residual_tx", "error_residual_ty", "error_residual_tz",
    )):
        current[name] = float(index + 1)
    prediction = official_prediction_force_torque(current)
    target = official_prediction_ground_truth_force_torque(current)
    np.testing.assert_allclose(prediction[0], np.arange(1.0, 7.0))
    np.testing.assert_allclose(target[0], np.arange(1.0, 7.0) + np.arange(7.0, 13.0))


def test_track_a_prism_information_and_no_a():
    xk, state, y, origins = track_a_design(frame(), history=20)
    assert xk.shape[0] == state.shape[0] == y.shape[0] == len(origins)
    assert state.shape[1] == 200
    assert xk.shape[1] == 80 + 3 + 3 + 3 + 4 + 3 + 3
    contracts = fit_route_contracts(xk, state, y, "IDENTITY_CORRECTION", [1e-6, 1.0], 1e18, 1e-6, target_kind="TEST", history=20)
    assert candidate_binding_audit(contracts)["a_enabled"] is False
    assert tuple(contracts) == FORMAL_ROUTE_IDS


def test_track_b_resample_history_unroll_and_control_scaling():
    raw = frame(240)
    sampled = resample_track_b_100hz(raw)
    assert len(sampled) == 60
    xk, state, y, origins = track_b_design(sampled, history=20)
    assert origins[0] == 19 and origins[-1] == len(sampled) - 2
    assert state.shape[1] == 200 and y.shape[1] == 9
    assert xk.shape[1] == 80 + 19
    expected = np.square(sampled.loc[:19, ["mot 1", "mot 2", "mot 3", "mot 4"]].to_numpy() * 0.001).reshape(-1)
    np.testing.assert_allclose(xk[0, :80], expected)


def test_track_b_quaternion_unit_norm_and_known_error():
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])
    rotated = compose_quaternion_increment(identity, np.array([[0.0, 0.0, np.pi / 2]]))
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), 1.0, atol=1e-14)
    assert delta_q_metric(rotated, identity) == pytest.approx(np.pi / 2)
    assert delta_q_metric(-rotated, rotated) == pytest.approx(0.0, abs=1e-14)


def test_track_b_delta_z_metric_matches_reference():
    target = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    prediction = target + 1.0
    assert delta_z_metric(target, prediction) == pytest.approx(6.0)


def test_track_b_rollout_access_guards_and_no_a():
    with pytest.raises(RuntimeError, match="FUTURE_MEASURED_STATE_ACCESS"):
        assert_no_future_state_access(19, range(21), range(80), 60)
    assert_no_future_state_access(19, range(20), range(80), 60)
    contracts = small_contracts()
    result = track_b_rollout(contracts, "PF_KC", resample_track_b_100hz(frame(500)), history=5, rollout=6)
    assert result["future_controls_used"] is True
    assert result["future_measured_states_used"] is False
    assert result["future_target_residual_used"] is False
    assert result["maximum_quaternion_norm_error"] < 1e-12


def test_candidate_loss_prediction_materialization_ids_match_roundtrip():
    contracts = small_contracts()
    assert candidate_binding_audit(contracts)["passed"] is True
    restored = {key: route_contract_from_json(route_contract_to_json(value)) for key, value in contracts.items()}
    assert tuple(restored) == FORMAL_ROUTE_IDS
    assert all(restored[key].route_id == key for key in FORMAL_ROUTE_IDS)


def test_track_b_velocity_and_attitude_are_decoupled_contracts():
    contracts = small_contracts()
    for contract in contracts.values():
        assert contract.velocity_contract.target_kind == "DELTA_Z_6D"
        assert contract.attitude_contract.target_kind == "ROTATION_VECTOR_3D"
        assert contract.velocity_contract is not contract.attitude_contract


def test_no_published_score_used_for_model_selection():
    signature = inspect.signature(select_w_family)
    assert "published" not in " ".join(signature.parameters)


def test_frozen_protocol_values_and_access_order():
    config = json.loads((PACKAGE / "DUAL_BENCHMARK_CONFIG_FROZEN.json").read_text(encoding="utf-8"))
    assert config["track_b"]["history_samples"] == 20
    assert config["track_b"]["training_unroll_samples"] == 10
    assert config["track_b"]["evaluation_rollout_samples"] == 60
    assert config["track_b"]["future_controls_allowed"] is True
    assert config["track_b"]["future_measured_states_allowed"] is False
    assert config["track_a"]["a_enabled"] is False and config["track_b"]["a_enabled"] is False
    assert config["global_dual_freeze_required_before_test"] is True
    assert config["published_scores_used_for_selection"] is False


def test_track_b_u10_development_selection_uses_recursive_rollout():
    source = inspect.getsource(select_track_b_w_family_rollout)
    assert "track_b_rollout" in source
    assert "rollout=rollout" in source
    assert "published" not in " ".join(inspect.signature(select_track_b_w_family_rollout).parameters)
