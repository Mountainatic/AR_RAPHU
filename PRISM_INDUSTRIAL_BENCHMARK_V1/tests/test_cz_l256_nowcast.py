from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.cz_l256_nowcast import (
    BREAKPOINTS,
    HISTORY_STEPS,
    INPUT_COLUMNS,
    TASK_ID,
    RawSegment,
    _raw_segments,
    _sample_rows,
    _split_source_origins,
    materialize_target_direction,
    view,
)
from prism_benchmark.level_reconstruction import metric_bundle_delta_and_level
from prism_benchmark.representative_formal import assert_scope_request


def _canonical_frame(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float64)
    return pd.DataFrame(
        {
            "main_heater_power": index + 1.0,
            "crystal_lift": index + 2.0,
            "crucible_lift": index + 3.0,
            "joint_lift": index + 4.0,
            "crystal_rotation": index + 5.0,
            "crucible_rotation": index + 6.0,
            "crystal_diameter": 100.0 + index * 0.25,
            "row_in_entity": np.arange(rows, dtype=np.int64),
            "entity_id": "Rod_1_segment_0",
        }
    )


def _raw_chinese_frame(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float64)
    return pd.DataFrame(
        {
            "主加热功率": index + 1.0,
            "晶升速度": index + 2.0,
            "埚升速度": index + 3.0,
            "晶转速度": index + 4.0,
            "埚转速度": index + 5.0,
            "晶体直径": 100.0 + index * 0.25,
        }
    )


def test_l256_delta_nowcast_has_exact_past_only_boundaries() -> None:
    segment = RawSegment("Rod_1", "Rod_1_segment_0", 0, 600, _canonical_frame(600))
    samples = _sample_rows(
        [segment],
        direction="Rod_1_to_Rod_2",
        split="train",
        origin_filter={segment.segment_id: {HISTORY_STEPS}},
    )
    assert len(samples) == 2
    for row in samples.itertuples():
        assert row.history_start == 0
        assert row.history_stop_exclusive == HISTORY_STEPS
        assert row.latest_available_target_index == HISTORY_STEPS - 1
        assert row.target_start == HISTORY_STEPS
        assert row.target_stop_exclusive == HISTORY_STEPS + 1
        assert row.dependency_start == 0
        assert row.dependency_stop_exclusive == HISTORY_STEPS + 1
        assert row.current_level == pytest.approx(100.0 + 255 * 0.25)
        assert row.future_level_true == pytest.approx(100.0 + 256 * 0.25)
        assert row.y_true == pytest.approx(0.25)
    assert set(samples["information_set"]) == {"input_only", "dynamic"}


def test_raw_segments_preserve_breakpoints_without_aggregation() -> None:
    segments = _raw_segments("Rod_1", _raw_chinese_frame(1000))
    assert [(segment.raw_start, segment.raw_stop) for segment in segments] == [
        (0, 334),
        (334, 688),
        (688, 734),
        (734, 1000),
    ]
    assert tuple(BREAKPOINTS["Rod_1"]) == (334, 688, 734)
    assert sum(len(segment.frame) for segment in segments) == 1000


def test_source_split_never_overlaps_dependency_intervals_across_breakpoints() -> None:
    segments = [
        RawSegment("Rod_1", "Rod_1_segment_0", 0, 334, _canonical_frame(334)),
        RawSegment("Rod_1", "Rod_1_segment_1", 334, 688, _canonical_frame(354)),
        RawSegment("Rod_1", "Rod_1_segment_2", 688, 734, _canonical_frame(46)),
        RawSegment("Rod_1", "Rod_1_segment_3", 734, 1734, _canonical_frame(1000)),
    ]
    train, validation = _split_source_origins(segments)
    assert any(train.values())
    assert any(validation.values())
    for entity in train:
        if train[entity] and validation[entity]:
            assert min(validation[entity]) - max(train[entity]) >= HISTORY_STEPS + 1


def test_cz_view_contract_is_independent_of_legacy_cz_d20() -> None:
    input_view = view("input_only")
    dynamic_view = view("dynamic")
    assert input_view.head.task_id == TASK_ID
    assert input_view.head.cadence_seconds == 2.0
    assert input_view.head.h_steps == 0
    assert dynamic_view.information_set == "dynamic"
    assert tuple(INPUT_COLUMNS) == (
        "main_heater_power",
        "joint_lift",
        "crystal_rotation",
        "crucible_rotation",
    )


def test_level_reporting_reconstructs_cz_delta_target() -> None:
    result = metric_bundle_delta_and_level(
        delta_true=[0.25, -0.5, 1.0],
        delta_pred=[0.2, -0.25, 0.75],
        current_level=[100.0, 101.0, 102.0],
    )
    np.testing.assert_allclose(result["future_level_pred"], [100.2, 100.75, 102.75])
    assert abs(float(result["mse"]) - float(result["mse_delta"])) <= 1e-10


def test_target_materialization_requires_both_sealed_artifacts(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    checkpoint = tmp_path / "checkpoint.json"
    freeze.write_text(json.dumps({"status": "GLOBAL_SELECTION_FROZEN", "sealed": False}))
    checkpoint.write_text(json.dumps({"status": "CHECKPOINTS_SEALED", "sealed": True}))
    with pytest.raises(RuntimeError, match="sealed artifact"):
        materialize_target_direction(
            tmp_path / "raw.xlsx",
            tmp_path / "shared",
            "Rod_1_to_Rod_2",
            global_freeze_path=freeze,
            checkpoint_manifest_path=checkpoint,
        )


def test_scope_guards_neural_stage2_and_reserved_datasets() -> None:
    assert_scope_request(datasets=["tep", "sru", "cz_czochralski"])
    with pytest.raises(RuntimeError, match="NEURAL3"):
        assert_scope_request(datasets=["tep"], neural3=True)
    with pytest.raises(RuntimeError, match="STAGE2"):
        assert_scope_request(datasets=["tep"], stage2=True)
    with pytest.raises(RuntimeError, match="RESERVED_DATASET"):
        assert_scope_request(datasets=["debutanizer"])
