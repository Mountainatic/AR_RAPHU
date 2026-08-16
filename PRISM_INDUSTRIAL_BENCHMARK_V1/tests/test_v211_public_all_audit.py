from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism_benchmark.v211_public_all_audit import audit_k_result
from prism_benchmark.v211_support import SUPPORT_CONTRACT


def _result() -> dict[str, object]:
    return {
        "status": "PASS",
        "support_contract": SUPPORT_CONTRACT,
        "test_accessed": False,
        "ood_accessed": False,
        "selected_profile_history_steps": 20,
        "selected_native_train_rows": 100,
        "selected_native_validation_rows": 20,
        "local_scoring_rows_by_fold": [10, 10],
        "local_scoring_support_hash_by_fold": ["a", "b"],
        "native_fit_rows_by_fold": [20, 30],
        "native_fit_support_hash_by_fold": ["c", "d"],
        "row_cap_applied_after_native_mask": True,
        "cross_channel_loss_comparable": False,
        "historical_global_lmax_used": False,
        "exact_zero_scoring_support_hash": ["a", "b"],
        "nonzero_scoring_support_hash": ["a", "b"],
        "selected_native_support_audit": {
            "train": {"support_contract": SUPPORT_CONTRACT},
            "validation": {"support_contract": SUPPORT_CONTRACT},
        },
    }


def test_k_result_audit_accepts_native_common_support_contract() -> None:
    assert audit_k_result(_result()) == []


def test_k_result_audit_rejects_candidate_scoring_hash_drift() -> None:
    result = _result()
    result["nonzero_scoring_support_hash"] = ["different", "b"]
    assert any("support hashes disagree" in error for error in audit_k_result(result))


def test_k_result_audit_allows_selected_oof_support_to_be_less_restrictive() -> None:
    result = _result()
    result["local_scoring_support_hash_by_fold"] = ["selected-a", "selected-b"]
    assert audit_k_result(result) == []
