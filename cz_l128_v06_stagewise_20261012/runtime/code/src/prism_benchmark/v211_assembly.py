from __future__ import annotations

from typing import Any, Mapping

from .v21_a import EXACT_ZERO
from .v21_selection import assert_final_prediction_contract
from .v211_w import IDENTITY


JOINT_CANDIDATES = ("J_K", "J_KW", "J_KA", "J_KWA")
PF_ASSEMBLIES = (
    "PRISM_V2_1_1_PF_K",
    "PRISM_V2_1_1_PF_K_W",
    "PRISM_V2_1_1_PF_K_A",
    "PRISM_V2_1_1_PF_K_W_A",
)


def _selected(result: Mapping[str, Any]) -> str:
    value = result.get("final_selected_candidate")
    if value is None:
        raise RuntimeError("module result lacks final_selected_candidate")
    return str(value)


def input_path_passed(result: Mapping[str, Any]) -> bool:
    return bool(result.get("input_path_preservation", {}).get("pass", False))


_GATE_IDENTITY_FIELDS = (
    "gate_version",
    "gate_parameters_sha256",
    "input_prediction_sha256",
    "target_sha256",
    "best_k_comparator_sha256",
)


def pf_joint_input_gate_inconsistent(
    physics_first: Mapping[str, Any],
    joint: Mapping[str, Any],
) -> bool:
    """Detect contradictory outcomes only for the exact same gate evaluation."""
    if input_path_passed(physics_first) == input_path_passed(joint):
        return False
    pf_gate = physics_first.get("input_path_preservation", {})
    joint_gate = joint.get("input_path_preservation", {})
    pf_identity = pf_gate.get("gate_evaluation_identity", pf_gate)
    joint_identity = joint_gate.get("gate_evaluation_identity", joint_gate)
    pf_values = tuple(pf_identity.get(name) for name in _GATE_IDENTITY_FIELDS)
    joint_values = tuple(joint_identity.get(name) for name in _GATE_IDENTITY_FIELDS)
    return all(value is not None for value in pf_values) and pf_values == joint_values


def pf_and_joint_input_status_match(
    physics_first: Mapping[str, Any],
    joint: Mapping[str, Any],
) -> bool:
    """Compatibility wrapper: different predictions may have different outcomes."""
    return not pf_joint_input_gate_inconsistent(physics_first, joint)


def build_physics_first_card(
    k_result: Mapping[str, Any],
    c_result: Mapping[str, Any],
    w_result: Mapping[str, Any],
    a_result: Mapping[str, Any],
) -> dict[str, Any]:
    for result in (k_result, c_result, w_result, a_result):
        if result.get("status") not in {"PASS", "K_EXACT_ZERO"}:
            raise RuntimeError("v2.1.1 PF prerequisite is not complete")
    assert_final_prediction_contract(w_result)
    assert_final_prediction_contract(a_result)
    gate = c_result.get("input_path_preservation", {})
    if not bool(gate.get("pass", False)):
        return {
            "status": "PHYSICS_ROUTE_NOT_SUPPORTED",
            "assembly": None,
            "input_path_preservation": gate,
            "input_path_nonzero": False,
            "a_only_fallback_allowed": False,
            "test_accessed": False,
        }
    w_selected = _selected(w_result)
    a_selected = _selected(a_result)
    w_active = w_selected != IDENTITY
    a_active = a_selected != EXACT_ZERO
    if w_active and a_active:
        assembly = "PRISM_V2_1_1_PF_K_W_A"
    elif w_active:
        assembly = "PRISM_V2_1_1_PF_K_W"
    elif a_active:
        assembly = "PRISM_V2_1_1_PF_K_A"
    else:
        assembly = "PRISM_V2_1_1_PF_K"
    return {
        "status": "PHYSICS_FIRST_STAGEWISE",
        "assembly": assembly,
        "input_path_preservation": gate,
        "input_path_nonzero": True,
        "a_only_fallback_allowed": False,
        "global_assembly_one_se": False,
        "selection_scope": ["K_LOCAL", "C_GIVEN_K", "W_GIVEN_KC", "A_GIVEN_KCW"],
        "selected_K": k_result.get("final_selected_candidate"),
        "selected_C": c_result.get("final_selected_candidate"),
        "selected_W": w_selected,
        "selected_A": a_selected,
        "module_status": {
            "W": "W_RESIDUAL_VALIDATED" if w_active else "WIENER_IDENTITY",
            "A": "A_RESIDUAL_VALIDATED" if a_active else "STATE_EXACT_ZERO",
        },
        "test_accessed": False,
    }


def build_joint_card(joint_result: Mapping[str, Any]) -> dict[str, Any]:
    gate = joint_result.get("input_path_preservation", {})
    selected = joint_result.get("final_selected_candidate")
    if selected is not None and selected not in JOINT_CANDIDATES:
        raise RuntimeError("Joint result contains an unregistered or AR-only candidate")
    if not bool(gate.get("pass", False)):
        return {
            "status": "JOINT_NOT_SUPPORTED_ON_DEVELOPMENT",
            "assembly": None,
            "selected_candidate": selected,
            "input_path_preservation": gate,
            "input_path_failure_class": gate.get(
                "input_path_failure_class",
                joint_result.get("input_path_failure_class"),
            ),
            "formal_test_eligible": False,
            "selection_eligible_for_test": False,
            "evidence_role": "DEVELOPMENT_DIAGNOSTIC_ONLY",
            "ar_only_fallback_allowed": False,
            "test_accessed": False,
        }
    selected = _selected(joint_result)
    assert_final_prediction_contract(joint_result)
    return {
        "status": "PREDICTIVE_JOINT_KWA",
        "assembly": f"PRISM_V2_1_1_{selected}",
        "selected_candidate": selected,
        "input_path_preservation": gate,
        "input_path_status": "INPUT_PATH_PRESERVED",
        "physical_interpretation": "TOTAL_PREDICTION_ONLY",
        "formal_test_eligible": True,
        "selection_eligible_for_test": True,
        "evidence_role": "FORMAL_PREDICTIVE_CANDIDATE",
        "ar_only_fallback_allowed": False,
        "test_accessed": False,
    }


def assert_no_state_only_assembly(card: Mapping[str, Any]) -> None:
    assembly = str(card.get("assembly") or "")
    forbidden = ("A_ONLY", "AR_ONLY", "EXACT_K_ZERO", "EXACT_BOTH_ZERO")
    if any(token in assembly for token in forbidden):
        raise RuntimeError("state-only route entered a v2.1.1 PRISM assembly")
