from __future__ import annotations

import json
from pathlib import Path


FREEZE_PATH = Path("configs/public_benchmarks/PB1_PROTOCOL_FREEZE.json")


def test_partial_freeze_never_authorizes_test_selection() -> None:
    if not FREEZE_PATH.is_file():
        return
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["selection_uses_official_test"] is False
    assert freeze["datasets"]["pwh"]["status"] == "FROZEN"
    assert freeze["datasets"]["whpn"]["status"] == "FROZEN"
    assert freeze["datasets"]["cascaded_tanks"]["status"].startswith("PENDING")
    assert freeze["datasets"]["silverbox"]["formal_experiment"] == "BLOCKED"


def test_pwh_validation_is_phase_atomic_and_amplitude_complete() -> None:
    if not FREEZE_PATH.is_file():
        return
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    pwh = freeze["datasets"]["pwh"]
    assert pwh["periods_are_atomic"] is True
    assert pwh["validation_record_count"] == 20
    ids = pwh["validation_record_ids"]
    for phase in range(16, 20):
        assert sum(f"phase_zero_based={phase}," in value for value in ids) == 5


def test_whpn_validation_ids_and_hashes_are_unambiguous() -> None:
    if not FREEZE_PATH.is_file():
        return
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    whpn = freeze["datasets"]["whpn"]
    assert whpn["validation_record_ids"] == [
        "WH_EstimationExample.mat::dataMeas[realization_zero_based=8]",
        "WH_EstimationExample.mat::dataMeas[realization_zero_based=9]",
    ]
    assert len(whpn["validation_record_sha256"]) == 2
    assert all(len(value) == 64 for value in whpn["validation_record_sha256"])
    assert whpn["alignment"]["primary"] == "RAW_UNSHIFTED"
    assert whpn["alignment"]["may_change_primary"] is False
