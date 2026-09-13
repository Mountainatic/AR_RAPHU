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
from prism_benchmark.e1e6_phase_b import (
    E3_RIDGES,
    E4_UNIVERSES,
    _e3_candidates,
    run_synthetic_variant,
)
from prism_benchmark.e1e6_robustness import _keyed_normal, perturbation_conditions
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
    assert report["stage_decision_recomputed"] is True
    assert report["reporting_only"] is True


def test_cached_stage_route_must_replay_from_oof_evidence() -> None:
    with pytest.raises(RuntimeError, match="does not replay"):
        _selection_report(
            {
                "parent_oof_risk": 1.0,
                "child_oof_risk": 0.9,
                "incremental_gain": 0.1,
                "parent_outer_fold_losses": [1.0, 1.0, 1.0],
                "child_outer_fold_losses": [0.9, 0.9, 0.9],
                "routing_status": "ZERO_IDENTITY",
                "final_selected_candidate": "zero",
                "tuned_nonzero_candidate": "candidate",
                "identity": "zero",
                "epsilon_num": np.finfo(np.float64).eps,
            }
        )


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


def test_e3_equal_budget_candidate_registration_is_exact() -> None:
    channels = ("a", "b", "c")
    histories = (8, 16, 32)
    uniform = _e3_candidates(channels, histories, 3, "UNIFORM_SCALE")
    multiscale = _e3_candidates(
        channels, histories, 3, "CHANNEL_SPECIFIC_MULTISCALE"
    )
    expected = len(histories) * len(E3_RIDGES)
    assert len(uniform) == expected
    assert len(multiscale) == expected
    assert len(set(uniform)) == expected
    assert len(set(multiscale)) == expected
    assert all(len(set(assignment)) == 1 for assignment, _ in uniform)
    assert any(len(set(assignment)) > 1 for assignment, _ in multiscale)


def test_e4_grids_are_nested_and_family_removal_is_external_zero() -> None:
    histories = [set(E4_UNIVERSES[name]["histories"]) for name in ("COARSE", "STANDARD", "EXPANDED")]
    ridges = [set(E4_UNIVERSES[name]["ridges"]) for name in ("COARSE", "STANDARD", "EXPANDED")]
    assert histories[0] < histories[1] < histories[2]
    assert ridges[0] < ridges[1] < ridges[2]
    left = run_synthetic_variant(
        2,
        "S4",
        histories=(1, 4),
        ridges=(1e-3, 1.0),
        include_w=False,
        n=512,
    )
    right = run_synthetic_variant(
        2,
        "S4",
        histories=(1, 4),
        ridges=(1e-3, 1.0),
        include_w=False,
        n=512,
    )
    assert left["W_active"] is False
    assert left["W_margin"] == 0.0
    assert left["prediction_hash"] == right["prediction_hash"]


def test_e6_noise_is_timestamp_keyed_and_condition_grid_is_frozen() -> None:
    indices = np.asarray([10, 11, 10, 15], dtype=np.int64)
    left = _keyed_normal(indices, 3, 7)
    right = _keyed_normal(indices, 3, 7)
    assert np.array_equal(left, right)
    assert left[0] == left[2]
    assert not np.array_equal(left, _keyed_normal(indices, 4, 7))
    conditions = perturbation_conditions()
    assert len(conditions) == 19
    assert ("GAUSSIAN", 0.0) in conditions
    assert ("RANDOM_WALK_DRIFT", 0.05) in conditions
