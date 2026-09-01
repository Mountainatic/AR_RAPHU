from __future__ import annotations

"""Contract-only helpers for the PMSM_SW corrected-C adapter.

This module intentionally contains no validation-driven model selection.  It
freezes the inherited v2.1.1/v2.2 corrected-C semantics and extracts only the
pre-validation D selection payload needed to replay already-selected K models.
"""

import json
from pathlib import Path
from typing import Any

from .v211_support import SUPPORT_CONTRACT


IMPLEMENTATION_FREEZE = "configs/prism_v22_pmsm_sw_implementation_semantics_freeze_20260901.json"
EXPECTED_CANDIDATES = ["ADDITIVE_COMPRESSED", "BEST_ACTIVE_K_CHANNEL_FALLBACK"]
EXPECTED_ALPHA_GRID = [
    1e-8,
    2.848035868435799e-7,
    8.111308307896873e-6,
    0.0002310129700083158,
    0.006579332246575682,
    0.18738174228603832,
    5.336699231206313,
    151.99110829529332,
    4328.7612810830615,
    123284.67394420659,
    3511191.7342151273,
    100000000.0,
]
EXPECTED_GATE = {
    "minimum_variance_ratio_to_target": 1e-8,
    "minimum_fraction_of_best_active_k_variance_ratio": 0.10,
    "maximum_mse_ratio_vs_best_active_k": 1.02,
    "minimum_nonintercept_coefficient_abs": 1e-10,
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_c_freeze_consistency(project: Path) -> dict[str, Any]:
    freeze = _json(project / IMPLEMENTATION_FREEZE)
    if freeze["status"] != "PRE_MODEL_FIT_IMPLEMENTATION_SEMANTICS_FROZEN":
        raise RuntimeError("PMSM SW implementation freeze status drift")
    if freeze["model_target_results_seen_before_this_freeze"] is not False:
        raise RuntimeError("implementation freeze is not pre-result")
    if freeze["test_target_access_before_this_freeze"] is not False:
        raise RuntimeError("test target was accessed before implementation freeze")
    if freeze["sample_support"]["contract"] != SUPPORT_CONTRACT:
        raise RuntimeError("assembly support contract drift")

    c = freeze["C_contract"]
    if c["inherit_source"] != "PRISM_V2_2_SRU_C_CONTRACT_CORRECTION_20260831_V1":
        raise RuntimeError("corrected-C inheritance source drift")
    if c["candidate_scope"] != EXPECTED_CANDIDATES:
        raise RuntimeError("corrected-C candidate scope drift")
    if c["ridge_semantics"] != "NUMERICAL_STABILIZATION_ONLY":
        raise RuntimeError("C ridge semantics drift")
    if c["ridge_selection"] != "SMALLEST_STABLE_RIDGE":
        raise RuntimeError("C ridge selection drift")
    if [float(value) for value in c["ridge_alpha_grid"]] != EXPECTED_ALPHA_GRID:
        raise RuntimeError("C alpha grid drift")
    if c["input_path_gate"]["split"] != "train_inner_oof_only":
        raise RuntimeError("C gate split drift")
    for key, expected in EXPECTED_GATE.items():
        if float(c["input_path_gate"][key]) != float(expected):
            raise RuntimeError(f"C gate threshold drift: {key}")
    if c["fallback_on_gate_failure"] != "BEST_ACTIVE_K_CHANNEL":
        raise RuntimeError("C fallback drift")
    if c["silent_active_k_erasure_forbidden"] is not True:
        raise RuntimeError("silent K erasure was re-enabled")

    selection = freeze["selection_partition"]
    if int(selection["inner_fold_count"]) != 4:
        raise RuntimeError("C fold-count drift")
    if int(selection["minimum_usable_folds"]) != 3:
        raise RuntimeError("C minimum-fold drift")
    caps = selection["row_caps"]
    if int(caps["D_C4_fit_row_cap"]) != 100000:
        raise RuntimeError("C/D fit row-cap drift")
    if int(caps["D_C4_selection_validation_row_cap"]) != 30000:
        raise RuntimeError("C/D fold-evaluation row-cap drift")
    if caps["subsample_rule"] != "smallest_sha256_of_base_origin_id_within_training_fold":
        raise RuntimeError("C support-before-cap subsample rule drift")
    return freeze


def d_selection_descriptor(result: dict[str, Any]) -> dict[str, Any]:
    """Whitelist only the train-frozen D selection fields consumed by C.

    In particular, validation MSE/RMSE/MAE/R2 and the stored ``active`` flag are
    deliberately ignored.  C derives activity from the pre-validation selected
    K family, preventing D holdout metrics from becoming a C selection path.
    """
    if result.get("status") != "PASS":
        raise RuntimeError("C may consume only successful D audit records")
    if result.get("validation_used_for_selection") is not False:
        raise RuntimeError("D record reports validation selection leakage")
    if result.get("test_accessed") is not False:
        raise RuntimeError("D record reports test access")
    selection = result.get("selection", {})
    if selection.get("selection_partition") != "train_only":
        raise RuntimeError("D selection was not frozen on train only")
    if selection.get("validation_used_for_selection") is not False:
        raise RuntimeError("D selection payload reports validation leakage")
    if selection.get("test_accessed") is not False:
        raise RuntimeError("D selection payload reports test access")

    profile = selection.get("selected_profile")
    if not isinstance(profile, list) or len(profile) != 2:
        raise RuntimeError("D selected_profile missing or malformed")
    kind = str(selection.get("selected_kind"))
    descriptor = {
        "channel": str(result["channel"]),
        "support_contract": SUPPORT_CONTRACT,
        "selected_profile": [int(profile[0]), int(profile[1])],
        "selected_profile_history_steps": int(profile[1]),
        "selected_kind": kind,
        "selected_m_tau": int(selection["selected_m_tau"]),
        "selected_m_x": int(selection["selected_m_x"]),
        "selected_lambdas": [float(value) for value in selection["selected_lambdas"]],
        "active": kind != "exact_zero",
        "selection_source": "D_SELECTION_FROZEN_BEFORE_VALIDATION_MATERIALIZATION",
    }
    return descriptor


def load_d_selection_descriptors(d_output: Path) -> list[dict[str, Any]]:
    root = d_output / "D_ONLY" / "PMSM_SW__H600__W60" / "proxy_excluded"
    paths = sorted(root.glob("*/RESULT.json"))
    if len(paths) != 8:
        raise RuntimeError(f"expected eight D channel results, found {len(paths)}")
    descriptors = [d_selection_descriptor(_json(path)) for path in paths]
    channels = [item["channel"] for item in descriptors]
    expected = ["ambient", "coolant", "i_d", "i_q", "motor_speed", "torque", "u_d", "u_q"]
    if channels != expected:
        raise RuntimeError(f"unexpected D channel set/order: {channels}")
    return descriptors


def active_d_descriptors(d_output: Path) -> list[dict[str, Any]]:
    return [item for item in load_d_selection_descriptors(d_output) if item["active"]]
