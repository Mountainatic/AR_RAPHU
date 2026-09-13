from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.e1e6_revalidation import (
    _aggregate_metric_sufficient_statistics,
    _markdown_table,
    _metric_sufficient_statistics,
    _selection_report,
    run_identifiable_seed,
)
from prism_benchmark.level_reconstruction import metric_bundle_delta_and_level
from prism_benchmark.v211_joint_stability import _revalidation_counterfactual_specs


def test_identifiable_generator_is_deterministic_and_separates_truth() -> None:
    left = run_identifiable_seed(7, 512, "S4")
    right = run_identifiable_seed(7, 512, "S4")
    assert left["prediction_hash"] == right["prediction_hash"]
    assert left["support_hash"] == right["support_hash"]
    assert left["C_truth"] and left["W_truth"] and left["A_truth"]
    assert not run_identifiable_seed(7, 512, "S1")["C_truth"]


def test_selection_reporting_margin_has_no_routing_authority() -> None:
    report = _selection_report(
        {
            "parent_oof_risk": 10.0,
            "child_oof_risk": 9.0,
            "incremental_gain": 1.0,
            "parent_outer_fold_losses": [10.0, 10.0, 10.0, 10.0],
            "child_outer_fold_losses": [8.0, 9.0, 10.0, 11.0],
            "routing_status": "ACTIVE",
            "final_selected_candidate": "candidate",
            "identity": "zero",
            "epsilon_num": np.finfo(np.float64).eps,
        }
    )
    assert report["relative_admission_margin"] == 0.1
    assert report["margin_signs"] == [1, 1, 0, -1]
    assert report["reporting_only"] is True


def test_joint_counterfactual_parser_rejects_unregistered_eta(monkeypatch) -> None:
    monkeypatch.setenv(
        "PRISM_REVALIDATION_JOINT_COUNTERFACTUALS",
        "J_KA|CHANNEL_COMPRESSED|0.01,J_KA|CHANNEL_COMPRESSED|1.0",
    )
    assert _revalidation_counterfactual_specs() == [
        ("J_KA", "CHANNEL_COMPRESSED", 0.01),
        ("J_KA", "CHANNEL_COMPRESSED", 1.0),
    ]
    monkeypatch.setenv(
        "PRISM_REVALIDATION_JOINT_COUNTERFACTUALS",
        "J_KA|CHANNEL_COMPRESSED|2.0",
    )
    try:
        _revalidation_counterfactual_specs()
    except ValueError as error:
        assert "outside the frozen grid" in str(error)
    else:
        raise AssertionError("an unregistered eta must not be materialized")


def test_markdown_table_has_no_optional_dependency() -> None:
    rendered = _markdown_table(pd.DataFrame([{"name": "a|b", "value": 1.25}]))
    assert "| name | value |" in rendered
    assert "a\\|b" in rendered
    assert "1.25" in rendered


def test_metric_sufficient_statistics_reconstruct_fold_subset_exactly() -> None:
    target = np.asarray([0.2, -0.1, 0.4, -0.3])
    prediction = np.asarray([0.1, -0.2, 0.35, -0.1])
    current = np.asarray([10.0, 10.2, 9.8, 10.1])
    pieces = [
        _metric_sufficient_statistics(target[:2], prediction[:2], current[:2]),
        _metric_sufficient_statistics(target[2:], prediction[2:], current[2:]),
    ]
    observed = _aggregate_metric_sufficient_statistics(pieces)
    expected = metric_bundle_delta_and_level(target, prediction, current)
    assert observed["rmse_delta"] == pytest.approx(expected["rmse_delta"])
    assert observed["mae_delta"] == pytest.approx(expected["mae_delta"])
    assert observed["r2_delta"] == pytest.approx(expected["r2_delta"])
    assert observed["r2_level_reconstructed"] == pytest.approx(
        expected["r2_level_reconstructed"]
    )
    assert observed["persistence_skill"] == pytest.approx(
        expected["persistence_skill"]
    )
