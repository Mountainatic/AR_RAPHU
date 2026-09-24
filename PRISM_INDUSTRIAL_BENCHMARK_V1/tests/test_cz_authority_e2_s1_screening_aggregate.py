from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "aggregate_cz_authority_e2_s1_screening.py"
)
SPEC = importlib.util.spec_from_file_location("cz_e2_s1_screening", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_s1_screening_has_no_phase_a_gate_authority() -> None:
    rows = []
    for direction, recovered in (("Rod_1_to_Rod_2", 5), ("Rod_2_to_Rod_1", 6)):
        for seed in range(10):
            active = seed < recovered
            rows.append(
                {
                    "direction": direction,
                    "seed": seed,
                    "unit_status": (
                        "PASS" if active else "COMPLETED_WITH_RETAINED_FAILURES"
                    ),
                    "true_k_recovered": active,
                    "rmse": 0.004 if active else None,
                    "r2": 0.9 if active else None,
                }
            )
    summary = MODULE.summarize(rows)
    assert summary["pooled"]["recovery_rate"] == 0.55
    assert summary["pooled"]["phase_a_gate_authority"] is False
    assert all(
        item["phase_a_gate_authority"] is False
        for item in summary["directions"]
    )
    assert summary["phase_a_gate_applicability"] == "NONE_S1_K_SCREENING_ONLY"
    assert summary["authority_phase_a_recovery_scope"] == [
        "S2:C",
        "S3:W",
        "S4:A",
    ]
    assert summary["authority_phase_a_evidence"] == "formal_30seed_only"


def test_cz_plan_matches_authority_phase_a_scope() -> None:
    project = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (
            project
            / "configs"
            / "cz_raw2s_h4_authority_e1_e6_rerun_plan_20260922.json"
        ).read_text(encoding="utf-8")
    )
    gate = plan["experiments"]["E2_SEMISYNTHETIC_RECOVERY"]["go_gate"]
    assert gate["true_stage_recovery_scope"] == ["S2:C", "S3:W", "S4:A"]
    assert gate["true_stage_recovery_evidence"] == "formal_30seed_only"
    assert gate["s1_k_screening_gate_authority"] is False
    correction = json.loads(
        (
            project
            / "configs"
            / "cz_raw2s_h4_phase_a_gate_scope_correction_20260924.json"
        ).read_text(encoding="utf-8")
    )
    assert correction["status"] == "ACTIVE"
    assert correction["authority_commit"].startswith("2ee6273b")
    assert correction["authority_behavior"]["recovery_gate_scope"] == [
        "S2:C",
        "S3:W",
        "S4:A",
    ]
    assert correction["affected_evidence"]["prior_s1_k_go_stop_interpretation"] == (
        "INVALID_DO_NOT_CITE"
    )
