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
    _group_lag_block,
    _history_supported,
    run_synthetic_variant,
)
from prism_benchmark.e1e6_robustness import (
    _causal_group_lag_block,
    _keyed_normal,
    _outer_train_target_sigma,
    _perturb_raw_process_block,
    perturbation_conditions,
)
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


def test_e3_history_support_is_entity_local() -> None:
    class Accessor:
        entities = {
            "a": (np.asarray([0, 1, 2, 3, 4]), {}),
            "b": (np.asarray([10, 11, 12, 13, 14]), {}),
        }

    samples = pd.DataFrame(
        {
            "entity_id": ["a", "a", "b", "b"],
            "origin": [2, 4, 12, 14],
        }
    )
    observed = _history_supported(samples, Accessor(), 3)
    assert observed[["entity_id", "origin"]].to_records(index=False).tolist() == [
        ("a", 4),
        ("b", 14),
    ]


def test_e6_group_lags_and_outer_folds_do_not_cross_entities() -> None:
    values = np.asarray([1.0, 2.0, 100.0, 200.0])
    groups = np.asarray([0, 0, 1, 1])
    observed = _group_lag_block(values, (1,), groups)
    assert observed[:, 0].tolist() == [1.0, 1.0, 100.0, 100.0]
    rng = np.random.default_rng(4)
    x = rng.normal(size=(128, 4))
    y = rng.normal(size=128)
    labels = np.repeat(np.arange(4), 32)
    result = run_synthetic_variant(
        1,
        "REAL_GROUPED",
        histories=(1,),
        ridges=(1e-3,),
        x_override=x,
        y_override=y,
        precomputed_k=True,
        groups_override=labels,
    )
    assert len(result["stage_vector"]) == 3


def test_e6_raw_measurement_noise_precedes_history_fusion() -> None:
    clean = np.zeros((2, 3), dtype=np.float64)
    indices = np.asarray([[5, 6, 5], [5, 7, 8]], dtype=np.int64)
    groups = np.asarray([0, 1], dtype=np.int64)
    low = _perturb_raw_process_block(
        clean,
        indices,
        groups,
        sigma=2.0,
        channel_key=4,
        condition="GAUSSIAN",
        magnitude=0.05,
        seed=3,
    )
    high = _perturb_raw_process_block(
        clean,
        indices,
        groups,
        sigma=2.0,
        channel_key=4,
        condition="GAUSSIAN",
        magnitude=0.10,
        seed=3,
    )
    assert np.allclose(high, 2.0 * low)
    assert low[0, 0] == low[0, 2]
    assert low[0, 0] != low[1, 0]


def test_e6_residual_history_is_strict_past_at_entity_boundaries() -> None:
    values = np.asarray([11.0, 12.0, 100.0, 200.0])
    groups = np.asarray([0, 0, 1, 1])
    observed = _causal_group_lag_block(values, (1, 2), groups)
    assert observed.tolist() == [
        [0.0, 0.0],
        [11.0, 0.0],
        [0.0, 0.0],
        [100.0, 0.0],
    ]


def test_e6_target_scale_excludes_outer_evaluation_entity() -> None:
    data = {
        "groups": np.asarray([0, 0, 1, 1]),
        "origins": np.asarray([10, 11, 10, 11]),
        "anchor": np.asarray([1.0, 3.0, 1000.0, 2000.0]),
        "latest_target": np.asarray([6, 7, 6, 7]),
        "x": np.asarray([[9.0], [11.0], [900.0], [1900.0]]),
        "process_columns": 0,
    }
    fit_mask = np.asarray([True, True, False, False])
    observed = _outer_train_target_sigma(data, fit_mask)
    assert observed == pytest.approx(np.std([1.0, 3.0, 9.0, 11.0]))
