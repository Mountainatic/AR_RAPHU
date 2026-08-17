from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from prism_benchmark.v211_public_all_reporting import (
    MAX_SMALL_PACKAGE_ARTIFACT_BYTES,
    REPAIR_MANIFEST_NAME,
    REUSED_ARTIFACT_MANIFEST_NAME,
    _add_ranks,
    _full_repro_manifest,
    _holm,
    _is_small_package_artifact,
    _raw_audit_summary,
    _repair_manifest,
)


def test_reporting_ranks_only_pass_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "task_id": "TASK",
                "target_head": "HEAD",
                "information_set": "dynamic",
                "availability_scenario": "record_time",
                "proxy_policy": "primary",
                "split": "test",
                "status": "PASS",
                "model": "A",
                "mse": 1.0,
            },
            {
                "task_id": "TASK",
                "target_head": "HEAD",
                "information_set": "dynamic",
                "availability_scenario": "record_time",
                "proxy_policy": "primary",
                "split": "test",
                "status": "PASS",
                "model": "B",
                "mse": 2.0,
            },
            {
                "task_id": "TASK",
                "target_head": "HEAD",
                "information_set": "dynamic",
                "availability_scenario": "record_time",
                "proxy_policy": "primary",
                "split": "test",
                "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE",
                "model": "C",
                "mse": None,
            },
        ]
    )
    ranked = _add_ranks(frame)
    assert ranked.loc[ranked["model"] == "A", "rank"].item() == 1
    assert ranked.loc[ranked["model"] == "B", "rank"].item() == 2
    assert pd.isna(ranked.loc[ranked["model"] == "C", "rank"].item())


def test_reporting_holm_uses_finite_sample_p_values() -> None:
    rows = [
        {
            "task_id": "TASK",
            "information_set": "dynamic",
            "comparison_family": "PAIR",
            "raw_p_value": 2.0 / 501.0,
            "holm_p_value": None,
        },
        {
            "task_id": "TASK",
            "information_set": "dynamic",
            "comparison_family": "PAIR",
            "raw_p_value": 4.0 / 501.0,
            "holm_p_value": None,
        },
    ]
    _holm(rows)
    assert all(row["holm_p_value"] is not None for row in rows)
    assert all(row["holm_p_value"] > 0.0 for row in rows)


def test_reporting_excludes_large_files_from_small_package(tmp_path: Path) -> None:
    small = tmp_path / "small.csv"
    small.write_text("value\n1\n", encoding="utf-8")
    large = tmp_path / "large.csv"
    with large.open("wb") as stream:
        stream.seek(MAX_SMALL_PACKAGE_ARTIFACT_BYTES)
        stream.write(b"x")

    assert _is_small_package_artifact(small)
    assert not _is_small_package_artifact(large)


def test_reporting_requires_an_accepted_repair_without_reselection(
    tmp_path: Path,
) -> None:
    paths = SimpleNamespace(freeze=tmp_path)
    assert _repair_manifest(paths) == {}
    manifest = tmp_path / REPAIR_MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "status": "ACCEPTED_AUDITED_REUSE",
                "post_test_reselection": False,
                "lockbox_access_attempts": 2,
            }
        ),
        encoding="utf-8",
    )
    assert _repair_manifest(paths)["lockbox_access_attempts"] == 2
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "COMPLETED_AUDITED_PARTIAL_RESUME"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert _repair_manifest(paths)["lockbox_access_attempts"] == 2
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["post_test_reselection"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot include reselection"):
        _repair_manifest(paths)


def test_reporting_preserves_multiple_lockbox_failures(tmp_path: Path) -> None:
    paths = SimpleNamespace(freeze=tmp_path)
    history = [
        {"attempt": 1, "failure_sha256": "a" * 64},
        {"attempt": 2, "failure_sha256": "b" * 64},
    ]
    (tmp_path / REPAIR_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "status": "ACCEPTED_AUDITED_REUSE",
                "post_test_reselection": False,
                "lockbox_access_attempts": 3,
                "lockbox_failure_history": history,
            }
        ),
        encoding="utf-8",
    )
    observed = _repair_manifest(paths)
    assert observed["lockbox_access_attempts"] == 3
    assert observed["lockbox_failure_history"] == history


def test_full_repro_manifest_includes_reused_development_files(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    freeze = run_root / "freeze"
    freeze.mkdir(parents=True)
    reused = {
        "path": "/source/results/DEVELOPMENT/K/RESULT.json",
        "bytes": 12,
        "sha256": "a" * 64,
        "role": "reused_development_artifact",
        "generated_by_stage": "D1_K",
    }
    (freeze / REUSED_ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({"status": "PASS", "files": [reused]}), encoding="utf-8"
    )
    paths = SimpleNamespace(
        run_root=run_root,
        return_root=run_root / "return",
        final=run_root / "final",
        freeze=freeze,
    )
    observed = _full_repro_manifest(paths)
    assert reused in observed["files"]


def test_full_repro_manifest_follows_routed_prediction_directory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    final = run_root / "final"
    freeze = run_root / "freeze"
    return_root = run_root / "return"
    external = tmp_path / "external_predictions"
    final.mkdir(parents=True)
    freeze.mkdir()
    external.mkdir()
    prediction = external / "PREDICTION.parquet"
    prediction.write_bytes(b"prediction")
    routed = final / "baseline_test_predictions"
    routed.symlink_to(external, target_is_directory=True)
    paths = SimpleNamespace(
        run_root=run_root,
        return_root=return_root,
        final=final,
        freeze=freeze,
    )

    observed = _full_repro_manifest(paths)

    record = next(
        item
        for item in observed["files"]
        if item["path"]
        == "final/baseline_test_predictions/PREDICTION.parquet"
    )
    assert record["bytes"] == len(b"prediction")
    assert record["role"] == "prediction"


def test_raw_audit_summary_uses_registry_audit_schema() -> None:
    observed = _raw_audit_summary(
        {
            "status": "PASS",
            "summary": {
                "datasets_total": 5,
                "datasets_pass": 5,
                "files_total": 8,
                "files_pass": 8,
            },
            "dataset_status": {"dataset": "PASS"},
            "files": [{"match": True}],
        }
    )
    assert observed == {
        "datasets": 5,
        "datasets_pass": 5,
        "files": 8,
        "files_pass": 8,
        "pass": True,
    }
