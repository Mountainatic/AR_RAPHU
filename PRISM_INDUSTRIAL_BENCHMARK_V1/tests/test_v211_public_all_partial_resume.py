from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark import v211_public_all_baseline_materialization as baseline
from prism_benchmark.cpu_data import HeadSpec, ViewSpec
from prism_benchmark.v211_public_all_partial_resume import (
    _hardlink,
    _link_regular_files,
    _logical_support_hash,
    _physical_support_fingerprint,
    _validate_prediction,
)
from prism_benchmark.v211_support import support_id_hash


def _view(information_set: str = "input_only") -> ViewSpec:
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
        information_set=information_set,
        availability_scenario="record_time",
        proxy_policy="primary",
    )


def _prediction_frame(
    view: ViewSpec,
    model: str,
    *,
    rows: int = 7,
    split: str = "test",
) -> pd.DataFrame:
    origin = np.arange(20, 20 + rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "sample_id": [f"view-{value}" for value in origin],
            "base_origin_id": [f"base-{value}" for value in origin],
            "dataset": view.head.dataset,
            "entity_id": "entity=1",
            "task_id": view.head.task_id,
            "target_head": view.head.head_id,
            "split": split,
            "origin": origin,
            "y_true": origin.astype(np.float64),
            "y_pred": origin.astype(np.float64) + 0.25,
            "model": model,
            "information_set": view.information_set,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "parameter_count": 3,
            "dtype": "float64",
        }
    )


def _support_hash(frame: pd.DataFrame) -> str:
    samples = frame[["base_origin_id", "sample_id"]].rename(
        columns={"sample_id": "view_sample_id"}
    )
    return support_id_hash(samples)


def test_baseline_materialization_can_target_only_pending_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        baseline,
        "_result",
        lambda *args: {
            "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE",
            "reason": "TEST",
        },
    )
    audits = baseline.materialize_baseline_view(
        SimpleNamespace(),
        _view(),
        models={"RIDGE", "PLS"},
    )
    assert [audit["model"] for audit in audits] == ["RIDGE", "PLS"]
    assert all(
        audit["status"] == "NOT_RUN_PROTOCOL_INCOMPATIBLE"
        for audit in audits
    )

    with pytest.raises(ValueError, match="unregistered"):
        baseline.materialize_baseline_view(
            SimpleNamespace(),
            _view(),
            models={"NOT_REGISTERED"},
        )


def test_prediction_validation_checks_support_metrics_and_contract(
    tmp_path: Path,
) -> None:
    view = _view()
    model = "RIDGE"
    frame = _prediction_frame(view, model)
    path = tmp_path / "prediction.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    expected_support = _support_hash(frame)

    observed = _validate_prediction(
        path,
        view,
        model,
        "test",
        len(frame),
        expected_support,
        compute_logical_support=True,
    )

    assert observed["logical_support_hash"] == expected_support
    assert observed["rows"] == len(frame)
    assert observed["parameter_count"] == 3
    assert observed["metrics"]["mse"] == pytest.approx(0.0625)
    assert _logical_support_hash(path) == expected_support


def test_prediction_validation_rejects_identity_drift(tmp_path: Path) -> None:
    view = _view()
    frame = _prediction_frame(view, "RIDGE")
    frame["model"] = "PLS"
    path = tmp_path / "prediction.parquet"
    frame.to_parquet(path, index=False, compression="zstd")

    with pytest.raises(RuntimeError, match="model mismatch"):
        _validate_prediction(
            path,
            view,
            "RIDGE",
            "test",
            len(frame),
            _support_hash(frame),
            compute_logical_support=True,
        )


def test_physical_support_fingerprint_ignores_prediction_values(
    tmp_path: Path,
) -> None:
    view = _view()
    first = _prediction_frame(view, "RIDGE")
    second = _prediction_frame(view, "PLS")
    second["y_pred"] += 100.0
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    first.to_parquet(first_path, index=False, compression="zstd")
    second.to_parquet(second_path, index=False, compression="zstd")

    assert _physical_support_fingerprint(
        first_path
    ) == _physical_support_fingerprint(second_path)


def test_hardlink_reuse_preserves_inode_and_sha(tmp_path: Path) -> None:
    source = tmp_path / "source" / "artifact.json"
    source.parent.mkdir()
    source.write_text('{"status":"PASS"}\n', encoding="utf-8")
    destination = tmp_path / "destination" / "artifact.json"

    record = _hardlink(source, destination)

    assert source.stat().st_ino == destination.stat().st_ino
    assert record["sha256"]
    assert record["destination_link_count"] == 2


def test_nonrecursive_metadata_link_does_not_copy_prediction_tree(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "audit.json").write_text("{}\n", encoding="utf-8")
    nested = source / "final" / "test_predictions"
    nested.mkdir(parents=True)
    (nested / "large.parquet").write_bytes(b"large-placeholder")

    records = _link_regular_files(
        source,
        destination,
        recursive=False,
    )

    assert [Path(record["destination"]).name for record in records] == [
        "audit.json"
    ]
    assert not (destination / "final").exists()
