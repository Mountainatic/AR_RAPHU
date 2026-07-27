from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/public_benchmarks/PB1_PROTOCOL_FREEZE.json"


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


def test_frozen_dataset_configs_match_protocol_freeze() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    pwh = json.loads(
        (ROOT / "configs/public_benchmarks/pb1_pwh.yaml").read_text(
            encoding="utf-8"
        )
    )
    whpn = json.loads(
        (ROOT / "configs/public_benchmarks/pb1_whpn.yaml").read_text(
            encoding="utf-8"
        )
    )
    pwh_split = pwh["dataset"]["development_split"]
    assert pwh_split["status"] == "FROZEN"
    assert pwh_split["train_phase_zero_based"] == [0, 15]
    assert pwh_split["validation_phase_zero_based"] == [16, 19]
    assert pwh_split["periods_are_atomic"] is True
    whpn_split = whpn["dataset"]["development_split"]
    assert whpn_split["status"] == "FROZEN"
    assert (
        whpn_split["validation_record_ids"]
        == freeze["datasets"]["whpn"]["validation_record_ids"]
    )
    assert (
        whpn_split["validation_record_sha256"]
        == freeze["datasets"]["whpn"]["validation_record_sha256"]
    )
    alignment = whpn["dataset"]["channel_alignment"]
    assert alignment["input_shift_samples"] == 0
    assert alignment["may_change_primary"] is False
