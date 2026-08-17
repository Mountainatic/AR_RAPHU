from __future__ import annotations

import json
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
    _canonical_baseline_audits,
    _partial_resume_prefix,
    _resume_identity,
    _hardlink,
    _link_regular_files,
    _reuse_baseline_test,
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


def test_pending_baseline_manifest_uses_json_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prism_benchmark import v211_public_all_partial_resume as module

    view = _view()
    monkeypatch.setattr(
        module,
        "baseline_candidates",
        lambda selected_view: (
            ("C2", "RIDGE", "RIDGE"),
            ("C2", "PLS", "PLS"),
        ),
    )
    monkeypatch.setattr(
        module,
        "_result",
        lambda _paths, _family, model, _view: (
            None
            if model == "PLS"
            else {"status": "PASS", "selection": {}}
        ),
    )
    monkeypatch.setattr(
        module,
        "_expected_support",
        lambda *args: (7, "a" * 64),
    )
    paths = SimpleNamespace(
        project=tmp_path,
        shared=tmp_path / "shared",
        output=tmp_path / "results",
        final=tmp_path / "destination",
    )
    parent = SimpleNamespace(final=tmp_path / "parent")

    slots, pending, records = _reuse_baseline_test(
        paths,
        parent,
        [view],
        {str(view.relative_root): "b" * 64},
    )

    assert records == []
    assert pending == {str(view.relative_root): {"RIDGE"}}
    json.dumps(
        {relative_root: sorted(models) for relative_root, models in pending.items()}
    )
    root = str(view.relative_root)
    assert (root, "PLS") in slots
    slots[(root, "RIDGE")] = {"model": "RIDGE"}
    audits = _canonical_baseline_audits([view], slots)
    assert [audit["model"] for audit in audits] == ["RIDGE", "PLS"]


def test_resume_identity_uses_manifest_generation(tmp_path: Path) -> None:
    manifest = {
        "repair_generation": 5,
        "lockbox_access_attempts": 6,
        "lockbox_failure_history": [
            {"attempt": attempt}
            for attempt in range(1, 6)
        ],
    }
    path = tmp_path / "POST_FREEZE_MATERIALIZATION_REPAIR.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    paths = SimpleNamespace(freeze=tmp_path)

    assert _partial_resume_prefix(5) == "R5_PARTIAL_RESUME"
    assert _resume_identity(paths) == (5, 6)

    manifest["lockbox_failure_history"].pop()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="failure history is incomplete"):
        _resume_identity(paths)
