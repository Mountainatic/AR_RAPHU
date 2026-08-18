from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.cpu_data import HeadSpec, ViewSpec
from prism_benchmark.v211_public_all_baselines import (
    DYNAMIC_MODELS,
    SIMPLE_MODELS,
    SupportRequirement,
    _development_summary_status,
    _has_contiguous_entity_sequence,
    apply_common_requirements,
    candidate_fold_supports,
    requirement_mask,
)
from prism_benchmark.v211_support import SUPPORT_CONTRACT, support_id_hash


def _view() -> ViewSpec:
    return ViewSpec(
        head=HeadSpec(
            head_id="TEST_HEAD__H1__W1",
            task_id="TEST_HEAD",
            dataset="synthetic",
            target="y",
            cadence_seconds=600.0,
            h_steps=1,
            w_steps=1,
            w0_steps=1,
            primary=True,
        ),
        information_set="dynamic",
        availability_scenario="record_time",
        proxy_policy="primary",
    )


def _samples() -> pd.DataFrame:
    origins = np.arange(1, 121, dtype=np.int64)
    return pd.DataFrame(
        {
            "base_origin_id": [f"b{value}" for value in origins],
            "view_sample_id": [f"v{value}" for value in origins],
            "dataset": "synthetic",
            "entity_id": "entity=1",
            "task_id": "TEST_HEAD",
            "target_head": "TEST_HEAD__H1__W1",
            "split": "train",
            "origin": origins,
            "dependency_start": origins,
            "dependency_stop_exclusive": origins + 1,
            "latest_available_target_index": origins - 1,
            "y_true": origins.astype(np.float64),
            "causal_history_floor": np.zeros(len(origins), dtype=np.int64),
            "anchor_history_steps": np.ones(len(origins), dtype=np.int64),
            "sample_support_contract": SUPPORT_CONTRACT,
        }
    )


def test_candidate_native_fit_and_common_scoring_are_both_enforced() -> None:
    short = SupportRequirement(input_history_steps=2)
    long = SupportRequirement(input_history_steps=10)
    folds, audit = candidate_fold_supports(
        _samples(),
        _view(),
        [short, long],
        fit_cap=10_000,
        evaluation_cap=10_000,
    )
    assert all(
        short_fold.fit_native_rows >= long_fold.fit_native_rows
        for short_fold, long_fold in zip(folds[short], folds[long], strict=True)
    )
    assert [fold.evaluation_support_hash for fold in folds[short]] == [
        fold.evaluation_support_hash for fold in folds[long]
    ]
    assert audit["candidate_native_fitting"] is True
    assert audit["common_validation_scoring"] is True
    assert audit["row_cap_applied_after_native_mask"] is True


def test_target_support_uses_registered_label_availability() -> None:
    samples = _samples().iloc[[20]].copy()
    samples["origin"] = 10
    samples["latest_available_target_index"] = 4
    supported = SupportRequirement(
        target_delta_steps=2,
        target_history_steps=6,
    )
    unsupported = SupportRequirement(
        target_delta_steps=2,
        target_history_steps=8,
    )
    assert requirement_mask(samples, supported).tolist() == [True]
    assert requirement_mask(samples, unsupported).tolist() == [False]


def test_common_support_is_the_candidate_intersection() -> None:
    samples = _samples()
    short = SupportRequirement(input_history_steps=2)
    long = SupportRequirement(input_history_steps=10)
    common = apply_common_requirements(samples, [short, long])
    expected = samples.loc[requirement_mask(samples, long)]
    assert support_id_hash(common) == support_id_hash(expected)


def test_frozen_inventory_keeps_seasonal_status_and_excludes_unregistered_trend() -> None:
    assert "SEASONAL_PERSISTENCE" in SIMPLE_MODELS
    assert "LOCAL_LINEAR_TREND" not in DYNAMIC_MODELS


def test_development_summary_preserves_retained_failure_signal() -> None:
    assert _development_summary_status([{"status": "PASS"}]) == "PASS"
    assert (
        _development_summary_status(
            [
                {"status": "PASS"},
                {"status": "NOT_RUN_PROTOCOL_INCOMPATIBLE"},
            ]
        )
        == "PASS"
    )
    assert (
        _development_summary_status(
            [{"status": "PASS"}, {"status": "FAILED_RETAINED"}]
        )
        == "PASS_WITH_RETAINED_FAILURES"
    )
    assert _development_summary_status([{"status": "UNKNOWN"}]) == "FAILED"


def test_n4sid_sequence_check_rejects_grouped_or_gapped_training_rows() -> None:
    contiguous = pd.DataFrame(
        {"entity_id": ["a", "a", "a"], "row_in_entity": [4, 5, 6]}
    )
    gapped = pd.DataFrame(
        {"entity_id": ["a", "a", "a"], "row_in_entity": [4, 5, 7]}
    )
    grouped = pd.DataFrame(
        {"entity_id": ["a", "b", "b"], "row_in_entity": [4, 5, 6]}
    )
    assert _has_contiguous_entity_sequence(contiguous) is True
    assert _has_contiguous_entity_sequence(gapped) is False
    assert _has_contiguous_entity_sequence(grouped) is False


def test_unavailable_long_baseline_history_is_filtered_before_scoring() -> None:
    short = SupportRequirement(input_history_steps=2)
    long = SupportRequirement(input_history_steps=1_000)
    folds, audit = candidate_fold_supports(
        _samples(),
        _view(),
        [short, long],
        fit_cap=10_000,
        evaluation_cap=10_000,
    )

    assert short in folds
    assert long not in folds
    assert audit["unavailable_requirements"] == [long.to_json()]
