from __future__ import annotations

import csv
import json

from experiments.tim_validation.e3_multiscale.report import (
    _canonical_task,
    _register_cz_single_scale_block,
)


def test_legacy_tep_run_key_is_reported_as_frozen_h0_task() -> None:
    assert _canonical_task("TEP_G12") == "TEP_H0"
    assert _canonical_task("DEB_C4") == "DEB_C4"


def test_cz_single_frozen_history_is_protocol_blocked(tmp_path) -> None:
    registry = tmp_path / "registry.csv"
    with registry.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["task", "head", "H_steps", "W_steps", "history_steps"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": "CZ_H4",
                "head": "CZ_DIAM_RAW2S_CURRENT_L256_H4",
                "H_steps": 4,
                "W_steps": 1,
                "history_steps": "[256]",
            }
        )
    result = _register_cz_single_scale_block(tmp_path / "runs", registry)
    assert result is not None
    assert result["status"] == "PROTOCOL_BLOCKED"
    assert result["test_accessed"] is False
    assert json.loads(
        (tmp_path / "runs" / "CZ_H4" / "PROTOCOL_STATUS.json").read_text()
    )["scale_arms_identical_by_construction"] is True
