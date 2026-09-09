from __future__ import annotations

import json

from experiments.tim_validation.e4_candidate.runner import (
    _minimum_fold_support_block,
    _valid_k_result,
    build_report,
)


def test_minimum_fold_support_block_requires_every_k_channel() -> None:
    blocked = {
        "status": "SOLVER_FAILED_RETAINED",
        "error": "no K profile has the minimum usable folds",
    }
    assert _minimum_fold_support_block([blocked, blocked])
    assert not _minimum_fold_support_block([blocked, {"status": "PASS"}])


def test_resume_accepts_a_tagged_minimum_fold_support_failure(tmp_path) -> None:
    path = tmp_path / "RESULT.json"
    path.write_text(
        json.dumps(
            {
                "status": "SOLVER_FAILED_RETAINED",
                "error": "no K profile has the minimum usable folds",
                "test_accessed": False,
                "tim_e4_candidate_universe": "expanded",
            }
        ),
        encoding="utf-8",
    )

    assert _valid_k_result(path, "expanded", [64, 128, 256, 512]) is not None


def test_report_surfaces_protocol_block_without_claiming_completion(tmp_path) -> None:
    marker = tmp_path / "TEP_G12" / "expanded" / "DEVELOPMENT_PROTOCOL_BLOCKED.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "status": "PROTOCOL_BLOCKED",
                "stage": "K",
                "reason": "INSUFFICIENT_INNER_FOLD_SUPPORT_FOR_EXPANDED_HISTORY_GRID",
            }
        ),
        encoding="utf-8",
    )

    result = build_report(tmp_path)

    assert result["status"] == "PARTIAL"
    assert result["protocol_blocked"] == [
        {
            "task": "TEP_G12",
            "universe": "expanded",
            "stage": "K",
            "reason": "INSUFFICIENT_INNER_FOLD_SUPPORT_FOR_EXPANDED_HISTORY_GRID",
        }
    ]
