from __future__ import annotations

import json

from experiments.tim_validation.e5_structural.runner import (
    _entropy,
    _jaccard,
    build_report,
)


def _signature(run: str, channels: list[str], rmse: float) -> dict:
    return {
        "task": "T", "head": "T__H1__W1", "H": 1, "W": 1,
        "outer_fold": "registered_outer_test", "run": run, "rod": None,
        "seed": 0, "candidate_universe": "standard",
        "model_variant": "PRISM", "information_set": "dynamic",
        "availability_scenario": "record_time", "proxy_policy": "primary",
        "admitted_channels": channels, "rejected_channels": [],
        "selected_scale_class_by_channel": {channel: "fast" for channel in channels},
        "K_admitted": bool(channels), "C_admitted": False,
        "W_admitted": False, "A_admitted": False,
        "RMSE": rmse, "support_id": "same", "active_channel_count": len(channels),
        "active_stage_count": int(bool(channels)), "parameter_count": len(channels),
    }


def test_empty_empty_is_flagged_separately() -> None:
    assert _jaccard(set(), set()) == (1.0, True)


def test_entropy_uses_observed_category_count() -> None:
    entropy, normalized, k = _entropy(["a", "b", "a", "b"])
    assert k == 2
    assert entropy > 0
    assert normalized == 1.0


def test_report_keeps_rashomon_and_empty_empty_semantics(tmp_path) -> None:
    source = tmp_path / "source"
    for run, channels, rmse in (("a", [], 1.0), ("b", [], 1.005), ("c", ["x"], 1.009)):
        path = source / run / "STRUCTURE_SIGNATURE.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_signature(run, channels, rmse)), encoding="utf-8")
    output = tmp_path / "output"
    result = build_report([source], output)
    assert result["status"] == "COMPLETED"
    assert result["empty_empty_pair_count"] == 1
    assert result["rashomon_structure_group_count"] == 1
    assert "EMPTY_EMPTY_AGREEMENT" in (output / "pairwise_jaccard.csv").read_text()
