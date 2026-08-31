from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.neural3 import native_support
from prism_benchmark.tep_nowcast_c1 import (
    COMMON_SUPPORT_HISTORY_STEPS,
    HEAD,
    TASK,
    build_development,
    build_test,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source(root: Path) -> Path:
    source = root / "source"
    _write_json(
        source / "PROTOCOL.json",
        {
            "tasks": [
                {
                    "task_id": "TEP_G_REP_H1",
                    "proxy_policies": ["proxy_excluded"],
                }
            ]
        },
    )
    _write_json(
        source / "dataset_views" / "VIEW_REGISTRY.json",
        [
            {
                "task_id": "TEP_G_REP_H1",
                "proxy_policy": "proxy_excluded",
                "input_columns": ["xmv_1"],
            }
        ],
    )
    for name in (
        "DATASET_HASHES.json",
        "SPLIT_REGISTRY.json",
        "SAMPLE_ID_REGISTRY.json",
        "LOCKBOX.json",
    ):
        _write_json(source / name, {"name": name})
    for split in ("train", "validation", "test"):
        rows = 600
        frame = pd.DataFrame(
            {
                "entity_id": [f"{split}_entity"] * rows,
                "row_in_entity": np.arange(rows, dtype=np.int64),
                "xmeas_40": np.linspace(50.0, 55.0, rows),
                "xmv_1": np.linspace(0.0, 1.0, rows),
            }
        )
        path = source / "base_data" / "tep" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return source


def _samples(destination: Path, information: str, availability: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(
        destination
        / "sample_ids"
        / HEAD
        / information
        / availability
        / "proxy_excluded"
        / f"{split}.parquet"
    )


def test_development_is_strict_past_and_uses_l256_common_support(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "nowcast"
    audit = build_development(source, destination)
    assert audit["test_accessed"] is False
    assert audit["common_support_history_steps"] == COMMON_SUPPORT_HISTORY_STEPS
    assert not (destination / "base_data" / "tep" / "test.parquet").exists()

    for information, availability, delay in (
        ("input_only", "record_time", 0),
        ("dynamic", "record_time", 0),
        ("dynamic", "analyzer_maturity_5_steps", 5),
    ):
        samples = _samples(destination, information, availability, "validation")
        assert samples["origin"].min() == COMMON_SUPPORT_HISTORY_STEPS + delay
        assert (samples["target_start"] == samples["origin"]).all()
        assert (samples["current_stop_exclusive"] == samples["origin"]).all()
        assert (samples["latest_available_target_index"] < samples["origin"]).all()
        dynamic = information == "dynamic"
        support128 = native_support(samples, 128, dynamic=dynamic)
        support256 = native_support(samples, 256, dynamic=dynamic)
        assert support128["view_sample_id"].tolist() == support256["view_sample_id"].tolist()

    source_train = source / "base_data" / "tep" / "train.parquet"
    linked_train = destination / "base_data" / "tep" / "train.parquet"
    assert not linked_train.is_symlink()
    assert source_train.stat().st_ino == linked_train.stat().st_ino


def test_test_materialization_requires_both_freezes(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "nowcast"
    build_development(source, destination)
    global_freeze = tmp_path / "GLOBAL_SELECTION_FREEZE.json"
    selected = tmp_path / "SELECTED_CHECKPOINT_MANIFEST.json"
    _write_json(global_freeze, {"status": "GLOBAL_SELECTION_FROZEN", "sealed": True})
    _write_json(selected, {"status": "NOT_SEALED", "sealed": False})
    with pytest.raises(RuntimeError, match="SELECTED_CHECKPOINT"):
        build_test(source, destination, global_freeze, selected)

    _write_json(selected, {"status": "SELECTED_CHECKPOINTS_SEALED", "sealed": True})
    audit = build_test(source, destination, global_freeze, selected)
    assert audit["test_accessed"] is True
    test = _samples(destination, "dynamic", "record_time", "test")
    assert test["origin"].min() == COMMON_SUPPORT_HISTORY_STEPS
    assert (test["latest_available_target_index"] == test["origin"] - 1).all()


def test_registered_head_is_h0_w1(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "nowcast"
    build_development(source, destination)
    registry = json.loads((destination / "TASK_REGISTRY.json").read_text())
    head = registry["heads"][0]
    assert head["task_id"] == TASK
    assert head["head_id"] == HEAD
    assert (head["h_steps"], head["w_steps"], head["w0_steps"]) == (0, 1, 1)

