from pathlib import Path

import numpy as np
import pandas as pd

from scripts import level_r2_reporting
from prism_benchmark.six_dataset_reporting import PredictionSpec


def _spec(path: Path, *, model: str = "LSTM") -> PredictionSpec:
    return PredictionSpec(
        path=path,
        scope="public5",
        direction=None,
        split="test",
        model=model,
        target_head="DEB_C4__H5__W1",
        information_set="dynamic",
        availability_scenario="record_time",
        proxy_policy="primary",
    )


def test_collect_level_r2_filters_to_frozen_common_support(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prediction_path = tmp_path / "test.parquet"
    prediction_path.write_bytes(b"prediction-placeholder")
    selected = _spec(prediction_path)
    ignored = _spec(tmp_path / "ignored.parquet", model="PRISM")
    prediction = pd.DataFrame(
        {
            "sample_id": ["keep-1", "keep-2", "keep-3", "drop"],
            "base_origin_id": ["base-1", "base-2", "base-3", "base-4"],
            "y_true": [1.0, -1.0, 0.5, 99.0],
            "y_pred": [0.5, -0.5, 0.5, -99.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "base_origin_id": ["base-1", "base-2", "base-3", "base-4"],
            "view_sample_id": ["keep-1", "keep-2", "keep-3", "drop"],
            "entity_id": ["entity"] * 4,
            "origin": [10, 11, 12, 13],
            "current_start": [9, 10, 11, 12],
            "current_stop_exclusive": [10, 11, 12, 13],
        }
    )

    monkeypatch.setattr(
        level_r2_reporting,
        "prediction_specs",
        lambda *_args, **_kwargs: [selected, ignored],
    )
    monkeypatch.setattr(
        level_r2_reporting,
        "frozen_support_records",
        lambda _run_root: {"frozen": True},
    )
    monkeypatch.setattr(
        level_r2_reporting,
        "allowed_support_for_prediction",
        lambda *_args, **_kwargs: (
            {"keep-1", "keep-2", "keep-3"},
            "frozen-support-hash",
            3,
        ),
    )
    monkeypatch.setattr(
        level_r2_reporting,
        "_prediction",
        lambda _spec: prediction.copy(),
    )
    monkeypatch.setattr(
        level_r2_reporting,
        "_shared_root",
        lambda *_args: tmp_path,
    )
    monkeypatch.setattr(
        level_r2_reporting,
        "_metadata",
        lambda *_args: metadata.copy(),
    )
    monkeypatch.setattr(
        level_r2_reporting,
        "_target_and_window",
        lambda *_args: ("y", 1, "debutanizer"),
    )
    monkeypatch.setattr(
        level_r2_reporting,
        "_current_levels",
        lambda *_args: np.array([10.0, 11.0, 12.0]),
    )

    frame, audit = level_r2_reporting.collect_level_r2(
        tmp_path,
        tmp_path,
        common_support_only=True,
        scopes={"public5"},
        splits={"test"},
        models={"LSTM"},
        target_heads={"DEB_C4__H5__W1"},
    )

    assert len(frame) == 1
    assert frame.iloc[0]["rows"] == 3
    assert frame.iloc[0]["sample_support_hash"] == "frozen-support-hash"
    assert bool(frame.iloc[0]["frozen_common_support"]) is True
    assert frame.iloc[0]["frozen_common_support_rows"] == 3
    assert audit["status"] == "PASS"
    assert audit["frozen_common_support"] is True
    assert audit["model_retrained"] is False
    assert audit["test_rerun"] is False
