from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prism_benchmark.strict_oof_selection import (
    ACTIVE,
    ZERO_IDENTITY,
    apply_stage_increment,
    strict_nested_oof_select,
)


@pytest.mark.parametrize("gain", [0.02, 0.005, 0.001])
def test_every_positive_nested_oof_gain_is_active(gain: float) -> None:
    parent = [1.0, 1.0, 1.0, 1.0]
    # Deliberately inconsistent fold signs: fold-count stability has no authority.
    child = [1.02, 1.02, 1.02, 0.94 - 4.0 * gain]
    selection = strict_nested_oof_select(
        {"ACTIVE_MODEL": child}, parent, identity="ZERO"
    )
    assert selection.incremental_gain == pytest.approx(gain)
    assert selection.routing_status == ACTIVE
    assert selection.final_selected_candidate == "ACTIVE_MODEL"


def test_identity_is_external_to_nonzero_tuning() -> None:
    with pytest.raises(ValueError, match="must not be a hyperparameter candidate"):
        strict_nested_oof_select(
            {"ZERO": [1.0] * 4, "ACTIVE": [0.9] * 4},
            [1.0] * 4,
            identity="ZERO",
        )


def test_outer_fold_is_not_used_to_choose_its_nonzero_model() -> None:
    selection = strict_nested_oof_select(
        {
            "A": [0.0, 10.0, 10.0, 10.0],
            "B": [9.0, 9.0, 9.0, 9.0],
        },
        [20.0] * 4,
        identity="ZERO",
    )
    # On outer fold 0, A's tempting held-out value is invisible; B wins on folds 1-3.
    assert selection.outer_selected_nonzero_candidates[0] == "B"


def test_complexity_and_reporting_evidence_cannot_change_route() -> None:
    selection = strict_nested_oof_select(
        {"VERY_COMPLEX": [0.999] * 4, "SIMPLE": [1.0] * 4},
        [1.0] * 4,
        identity="ZERO",
    )
    assert selection.final_selected_candidate == "VERY_COMPLEX"
    reported = selection.with_reporting_evidence(
        {
            "bootstrap_positive_probability": 0.01,
            "ci95": [-1.0, 1.0],
            "positive_fold_fraction": 0.0,
        }
    )
    assert reported.routing_status == selection.routing_status == ACTIVE
    assert reported.final_selected_candidate == selection.final_selected_candidate
    with pytest.raises(ValueError, match="cannot contain routing fields"):
        selection.with_reporting_evidence({"routing_status": ZERO_IDENTITY})


def test_only_machine_precision_tie_returns_identity() -> None:
    selection = strict_nested_oof_select(
        {"ACTIVE": [1.0 - 1e-15] * 4},
        [1.0] * 4,
        identity="ZERO",
    )
    assert selection.routing_status == ZERO_IDENTITY
    assert selection.final_selected_candidate == "ZERO"


def test_zero_increment_does_not_block_a_later_stage() -> None:
    parent = np.array([1.0, 2.0])
    after_c = apply_stage_increment(parent, [10.0, 10.0], active=False)
    after_w = apply_stage_increment(after_c, [0.5, -0.5], active=True)
    np.testing.assert_array_equal(after_c, parent)
    np.testing.assert_array_equal(after_w, np.array([1.5, 1.5]))


def test_frozen_protocol_forbids_statistical_routing_authority() -> None:
    project = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project / "configs" / "strict_nested_oof_selection_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["nonzero_family"]["zero_in_candidates"] is False
    assert config["nonzero_family"]["one_se"] is False
    assert config["routing"]["zero_blocks_downstream"] is False
    assert config["evidence"] == {
        "bootstrap": "REPORTING_ONLY",
        "confidence_interval": "REPORTING_ONLY",
        "fold_stability": "REPORTING_ONLY",
    }


def test_active_v211_stages_do_not_import_one_se_selectors() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "prism_benchmark"
    for name in ("v211_k.py", "v211_c.py", "v211_w.py", "v211_a.py", "v211_joint.py"):
        text = (source / name).read_text(encoding="utf-8")
        assert "one_se_select" not in text
        assert "guarded_local_one_se_select" not in text
