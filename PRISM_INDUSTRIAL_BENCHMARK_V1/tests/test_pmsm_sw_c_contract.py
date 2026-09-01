from __future__ import annotations

import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark import pmsm_sw_c_contract as contract  # noqa: E402


def _d_result(kind: str = "linear") -> dict:
    return {
        "status": "PASS",
        "channel": "u_q",
        "validation_used_for_selection": False,
        "test_accessed": False,
        "active": False,  # deliberately contradictory; must be ignored by C
        "mse": 123.0,
        "r2": -999.0,
        "selection": {
            "selection_partition": "train_only",
            "validation_used_for_selection": False,
            "test_accessed": False,
            "selected_profile": [15, 2400],
            "selected_kind": kind,
            "selected_m_tau": 8,
            "selected_m_x": 4,
            "selected_lambdas": [1e-4, 1e-3, 1e-3],
        },
    }


def test_corrected_c_freeze_is_exactly_inherited() -> None:
    freeze = contract.assert_c_freeze_consistency(PROJECT_ROOT)
    c = freeze["C_contract"]
    assert c["candidate_scope"] == contract.EXPECTED_CANDIDATES
    assert [float(value) for value in c["ridge_alpha_grid"]] == contract.EXPECTED_ALPHA_GRID
    assert c["input_path_gate"]["split"] == "train_inner_oof_only"
    assert c["fallback_on_gate_failure"] == "BEST_ACTIVE_K_CHANNEL"


def test_c_activity_is_derived_only_from_train_frozen_kind() -> None:
    descriptor = contract.d_selection_descriptor(_d_result("linear"))
    assert descriptor["active"] is True
    descriptor = contract.d_selection_descriptor(_d_result("exact_zero"))
    assert descriptor["active"] is False


def test_d_validation_metrics_and_stored_active_flag_are_not_consumed() -> None:
    source = inspect.getsource(contract.d_selection_descriptor)
    for forbidden in ("mse", "rmse", "mae", "r2"):
        assert f'result["{forbidden}"]' not in source
        assert f"result.get(\"{forbidden}\")" not in source
    assert 'result["active"]' not in source
    assert 'result.get("active")' not in source


def test_c_loader_rejects_any_reported_selection_leakage() -> None:
    value = _d_result()
    value["validation_used_for_selection"] = True
    try:
        contract.d_selection_descriptor(value)
    except RuntimeError as error:
        assert "validation" in str(error)
    else:
        raise AssertionError("validation leakage was accepted")

    value = _d_result()
    value["selection"]["test_accessed"] = True
    try:
        contract.d_selection_descriptor(value)
    except RuntimeError as error:
        assert "test" in str(error)
    else:
        raise AssertionError("test leakage was accepted")
