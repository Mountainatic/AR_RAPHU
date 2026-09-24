from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_cz_e2_authority_branch_logic import (
    AUTHORITY_COMMIT,
    E2_MODULE_SHA256,
    SELECTOR_SHA256,
)


PROJECT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_authority_e2_blobs_are_pinned() -> None:
    assert AUTHORITY_COMMIT == "2ee6273b8f915cbcdff2f46d56bc80047ddae4a7"
    assert _sha256(PROJECT / "src/prism_benchmark/e1e6_revalidation.py") == E2_MODULE_SHA256
    assert _sha256(PROJECT / "src/prism_benchmark/strict_oof_selection.py") == SELECTOR_SHA256


def test_raw_cz_e2_adapter_is_invalidated_and_scope_is_explicit() -> None:
    correction = json.loads(
        (PROJECT / "configs/cz_authority_branch_logic_correction_20260924.json").read_text(
            encoding="utf-8"
        )
    )
    registry = json.loads(
        (PROJECT / "configs/cz_raw2s_h4_e2_truth_operator_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert correction["status"] == "ACTIVE_SUPERSEDES_RAW_CZ_E2_ADAPTER"
    assert correction["E2"]["generator"] == "IDENTIFIABLE_E2_V2"
    assert correction["E2"]["private_cz_data_accessed"] is False
    assert correction["CZ_E3_E6"]["status"] == "PROTOCOL_BLOCKED_UNDER_AUTHORITY_BRANCH"
    assert registry["status"] == "INVALIDATED_DO_NOT_RUN_OR_CITE"
