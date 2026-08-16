from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from prism_benchmark.v211_public_all_reporting import (
    REPAIR_MANIFEST_NAME,
    REUSED_ARTIFACT_MANIFEST_NAME,
    _add_ranks,
    _full_repro_manifest,
    _holm,
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
    payload["post_test_reselection"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot include reselection"):
        _repair_manifest(paths)


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
