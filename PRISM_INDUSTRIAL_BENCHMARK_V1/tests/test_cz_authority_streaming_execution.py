from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_cz_raw2s_authority_e1_e6_r2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_cz_raw2s_authority_e1_e6_r2_streaming_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_streaming_amendment_does_not_change_statistical_protocol() -> None:
    project = Path(__file__).resolve().parents[1]
    amendment = json.loads(
        (
            project
            / "configs"
            / "cz_raw2s_h4_authority_e2_e6_streaming_amendment_20260923.json"
        ).read_text(encoding="utf-8")
    )
    assert amendment["status"] == "EXECUTION_AMENDMENT_ACTIVE"
    assert amendment["statistical_protocol_changed"] is False
    assert amendment["model_or_selector_changed"] is False
    assert amendment["h_w_protocol_changed"] is False
    assert amendment["evidence_boundary_changed"] is False
    assert amendment["streaming_execution"]["maximum_concurrent_units"] == 2


def test_streaming_gate_reserves_low_watermarks(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    project = Path(__file__).resolve().parents[1]
    gib = 1024**3
    values = iter(
        [
            shutil._ntuple_diskusage(50 * gib, 34 * gib, 16 * gib),
            shutil._ntuple_diskusage(30 * gib, 0, 30 * gib),
        ]
    )
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: next(values))
    result = module.streaming_storage_gate(project, tmp_path / "run", tmp_path)
    assert result["status"] == "PASS"
    assert result["authorized_outer_units"] == 1
    assert result["two_way_parallelism"] == "BLOCKED_PENDING_WORKER_EQUIVALENCE"
    assert result["private_required_before_unit_gib"] == 12
    assert result["scratch_required_before_unit_gib"] == 20


def test_streaming_gate_fails_closed_below_reserved_capacity(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    project = Path(__file__).resolve().parents[1]
    gib = 1024**3
    values = iter(
        [
            shutil._ntuple_diskusage(50 * gib, 40 * gib, 10 * gib),
            shutil._ntuple_diskusage(30 * gib, 0, 30 * gib),
        ]
    )
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: next(values))
    with pytest.raises(RuntimeError, match="STOP_INSUFFICIENT_STREAMING_STORAGE"):
        module.streaming_storage_gate(project, tmp_path / "run", tmp_path)
    blocked = json.loads(
        (tmp_path / "run" / "STREAMING_STORAGE_GATE.json").read_text(
            encoding="utf-8"
        )
    )
    assert blocked["status"] == "BLOCKED"
