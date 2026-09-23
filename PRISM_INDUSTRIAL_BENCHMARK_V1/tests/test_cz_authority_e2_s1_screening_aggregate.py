from __future__ import annotations

import importlib.util
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


def test_gate_interpretation_preserves_directional_disagreement() -> None:
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
    assert summary["pooled"]["exclusive_gt_0_5_gate"] is True
    assert summary["directions"][0]["exclusive_gt_0_5_gate"] is False
    assert summary["directions"][1]["exclusive_gt_0_5_gate"] is True
    assert summary["gate_interpretation"] == "REQUIRES_DIRECTIONAL_SCOPE_DECISION"
