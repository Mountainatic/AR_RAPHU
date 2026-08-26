from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_benchmark.neural_resource_guard import (
    GIB,
    ResourceSnapshot,
    ResourceThresholds,
    atomic_write_json,
    build_workload_matrix,
    collect_resource_snapshot,
    controlled_thread_environment,
    decide_resource_action,
    load_workload_config,
    query_gpu_memory,
    read_effective_memory,
)
import prism_benchmark.neural_resource_guard as resource_guard


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "active3_neural3_parallel_cache_20260826.json"
)


def _snapshot(
    *,
    ram_gib: float = 1.0,
    vram_gib: float = 1.0,
    disk_gib: float = 10.0,
    errors: tuple[str, ...] = (),
) -> ResourceSnapshot:
    return ResourceSnapshot(
        observed_at_utc="2026-08-26T00:00:00+00:00",
        memory_current_bytes=int(ram_gib * GIB),
        memory_limit_bytes=90 * GIB,
        memory_source="cgroup_v2",
        gpu_index=0,
        gpu_used_bytes=int(vram_gib * GIB),
        gpu_total_bytes=32 * GIB,
        data_disk_path="/root/autodl-tmp",
        data_disk_free_bytes=int(disk_gib * GIB),
        data_disk_total_bytes=100 * GIB,
        telemetry_errors=errors,
    )


def test_cgroup_v2_memory_is_preferred_over_host_meminfo(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text(str(11 * GIB), encoding="utf-8")
    (cgroup / "memory.max").write_text(str(90 * GIB), encoding="utf-8")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       800000000 kB\nMemAvailable:   700000000 kB\n",
        encoding="utf-8",
    )
    current, limit, source = read_effective_memory(
        cgroup_root=cgroup, meminfo_path=meminfo
    )
    assert source == "cgroup_v2"
    assert current == 11 * GIB
    assert limit == 90 * GIB


def test_required_cgroup_does_not_fallback_to_host_meminfo(tmp_path: Path) -> None:
    cgroup = tmp_path / "missing-cgroup"
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       800000000 kB\nMemAvailable:   700000000 kB\n",
        encoding="utf-8",
    )
    current, limit, source = read_effective_memory(
        cgroup_root=cgroup, meminfo_path=meminfo, require_cgroup=True
    )
    assert (current, limit, source) == (None, None, "cgroup_unavailable")


def test_required_cgroup_rejects_unbounded_memory_limit(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text(str(11 * GIB), encoding="utf-8")
    (cgroup / "memory.max").write_text("max", encoding="utf-8")

    current, limit, source = read_effective_memory(
        cgroup_root=cgroup,
        meminfo_path=tmp_path / "missing-meminfo",
        require_cgroup=True,
    )
    assert (current, limit, source) == (None, None, "cgroup_unavailable")


def test_snapshot_parses_nvidia_smi_and_disk(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text(str(2 * GIB), encoding="utf-8")
    (cgroup / "memory.max").write_text("max", encoding="utf-8")

    snapshot = collect_resource_snapshot(
        tmp_path,
        cgroup_root=cgroup,
        command_runner=lambda _command: "0, 2048, 32607\n",
    )
    assert snapshot.memory_source == "cgroup_v2"
    assert snapshot.memory_current_bytes == 2 * GIB
    assert snapshot.memory_limit_bytes is None
    assert snapshot.gpu_used_bytes == 2048 * 1024**2
    assert snapshot.gpu_total_bytes == 32607 * 1024**2
    assert snapshot.data_disk_free_bytes > 0
    assert snapshot.telemetry_errors == ()


def test_disk_telemetry_failure_is_recorded_and_stops_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text(str(2 * GIB), encoding="utf-8")
    (cgroup / "memory.max").write_text(str(90 * GIB), encoding="utf-8")

    def fail_disk_usage(_path: Path):
        raise OSError("disk probe failed")

    monkeypatch.setattr(resource_guard.shutil, "disk_usage", fail_disk_usage)
    snapshot = collect_resource_snapshot(
        tmp_path,
        cgroup_root=cgroup,
        command_runner=lambda _command: "0, 2048, 32607\n",
    )
    assert snapshot.data_disk_free_bytes is None
    assert any(
        error.startswith("DISK_TELEMETRY_UNAVAILABLE")
        for error in snapshot.telemetry_errors
    )
    decision = decide_resource_action(snapshot)
    assert not decision.dispatch_allowed
    assert any(
        reason.startswith("DISK_TELEMETRY_UNAVAILABLE") for reason in decision.reasons
    )


def test_dispatch_budget_and_runtime_stoplines() -> None:
    thresholds = ResourceThresholds()
    allowed = decide_resource_action(
        _snapshot(ram_gib=70, vram_gib=10),
        expected_job_rss_bytes=4 * GIB,
        expected_job_vram_bytes=8 * GIB,
        thresholds=thresholds,
    )
    assert allowed.action == "ALLOW_DISPATCH"
    assert allowed.dispatch_allowed

    over_budget = decide_resource_action(
        _snapshot(ram_gib=74, vram_gib=10),
        expected_job_rss_bytes=2 * GIB,
        thresholds=thresholds,
    )
    assert over_budget.action == "STOP_DISPATCH"
    assert "PROJECTED_RAM_EXCEEDS_DISPATCH_BUDGET" in over_budget.reasons

    paused = decide_resource_action(_snapshot(ram_gib=82), thresholds=thresholds)
    assert paused.action == "STOP_DISPATCH"
    assert not paused.request_checkpoint
    assert "RAM_PAUSE_DISPATCH_REACHED" in paused.reasons

    stopped = decide_resource_action(_snapshot(ram_gib=85), thresholds=thresholds)
    assert stopped.action == "CHECKPOINT_AND_SAFE_STOP"
    assert stopped.request_checkpoint
    assert stopped.safe_stop


def test_vram_disk_and_missing_telemetry_fail_closed() -> None:
    vram = decide_resource_action(
        _snapshot(vram_gib=20), expected_job_vram_bytes=8 * GIB
    )
    assert not vram.dispatch_allowed
    assert "PROJECTED_VRAM_EXCEEDS_DISPATCH_BUDGET" in vram.reasons

    disk = decide_resource_action(_snapshot(disk_gib=4.99))
    assert not disk.dispatch_allowed
    assert "DATA_DISK_BELOW_STOPLINE" in disk.reasons

    missing = _snapshot(errors=("NVIDIA_SMI_FAILED:TEST",))
    missing = ResourceSnapshot(
        **{
            **missing.to_json(),
            "gpu_used_bytes": None,
            "gpu_total_bytes": None,
        }
    )
    decision = decide_resource_action(missing)
    assert not decision.dispatch_allowed
    assert "NVIDIA_SMI_FAILED:TEST" in decision.reasons


def test_non_authoritative_memory_and_invalid_gpu_values_stop_dispatch() -> None:
    snapshot = ResourceSnapshot(
        observed_at_utc="2026-08-26T00:00:00+00:00",
        memory_current_bytes=1 * GIB,
        memory_limit_bytes=90 * GIB,
        memory_source="proc_meminfo",
        gpu_index=0,
        gpu_used_bytes=1 * GIB,
        gpu_total_bytes=32 * GIB,
        data_disk_path="/root/autodl-tmp",
        data_disk_free_bytes=10 * GIB,
        data_disk_total_bytes=100 * GIB,
    )
    decision = decide_resource_action(snapshot)
    assert not decision.dispatch_allowed
    assert "MEMORY_CGROUP_NOT_AUTHORITATIVE" in decision.reasons

    invalid_gpu = query_gpu_memory(
        command_runner=lambda _command: "0, -1, 32607\n",
    )
    assert invalid_gpu[0:2] == (None, None)
    assert invalid_gpu[2] == "GPU_MEMORY_VALUES_INVALID:0"


def test_invalid_disk_telemetry_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text(str(2 * GIB), encoding="utf-8")
    (cgroup / "memory.max").write_text(str(90 * GIB), encoding="utf-8")

    monkeypatch.setattr(
        resource_guard.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 11 * GIB, "total": -1})(),
    )
    snapshot = collect_resource_snapshot(
        tmp_path,
        cgroup_root=cgroup,
        command_runner=lambda _command: "0, 2048, 32607\n",
    )
    assert snapshot.data_disk_free_bytes is None
    decision = decide_resource_action(snapshot)
    assert not decision.dispatch_allowed
    assert any(
        reason.startswith("DISK_TELEMETRY_UNAVAILABLE")
        for reason in decision.reasons
    )


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("memory_current_bytes", float("nan"), "MEMORY_CURRENT_NON_FINITE"),
        ("memory_limit_bytes", float("inf"), "MEMORY_LIMIT_NON_FINITE"),
        ("gpu_used_bytes", 1.5, "GPU_USED_NON_INTEGER"),
        ("gpu_total_bytes", float("-inf"), "GPU_TOTAL_NON_FINITE"),
        ("data_disk_free_bytes", 2.25, "DISK_FREE_NON_INTEGER"),
        ("data_disk_total_bytes", float("nan"), "DISK_TOTAL_NON_FINITE"),
    ],
)
def test_non_finite_or_non_integer_snapshot_values_fail_closed(
    field: str, value: object, detail: str
) -> None:
    payload = _snapshot().to_json()
    payload[field] = value
    snapshot = ResourceSnapshot(**payload)
    decision = decide_resource_action(snapshot)
    assert not decision.dispatch_allowed
    assert detail in decision.reasons


def test_gpu_total_is_required_even_when_used_memory_is_present() -> None:
    payload = _snapshot().to_json()
    payload["gpu_total_bytes"] = None
    decision = decide_resource_action(ResourceSnapshot(**payload))
    assert not decision.dispatch_allowed
    assert "GPU_TELEMETRY_UNAVAILABLE" in decision.reasons


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.5, True])
def test_invalid_expected_increment_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="finite non-negative integer"):
        decide_resource_action(_snapshot(), expected_job_rss_bytes=value)


def test_active3_workload_matrix_is_exact_and_unique() -> None:
    config, matrix = load_workload_config(CONFIG_PATH)
    assert matrix.counts() == {
        "profiles": 81,
        "selection_candidates": 456,
        "selected_checkpoint_references": 81,
        "total_training_units": 456,
    }
    scope_profiles = {
        scope: [profile for profile in matrix.profiles if profile["scope"] == scope]
        for scope in ("tep", "sru", "cz")
    }
    assert {scope: len(items) for scope, items in scope_profiles.items()} == {
        "tep": 9,
        "sru": 12,
        "cz": 60,
    }
    assert config["declared_view_counts"] == {
        "tep": 3,
        "sru": 4,
        "cz": 20,
        "total": 27,
    }
    assert all(
        item["requires_global_selection_freeze"]
        and item["fit_called"] is False
        and item["refit_called"] is False
        for item in matrix.selected_checkpoint_references
    )
    assert {
        item["task_id"] for item in matrix.profiles if item["scope"] == "cz"
    } == {
        f"CZ_DIAM_RAW2S_CURRENT_L256_H{h}" for h in (1, 2, 4, 8, 16)
    }
    assert all(
        item["availability_scenario"] == "record_time"
        and item["proxy_policy"] == "primary"
        for item in matrix.profiles
        if item["scope"] in {"sru", "cz"}
    )
    assert len({item["candidate_id"] for item in matrix.selection_candidates}) == 456
    assert {
        item["candidate_label"]
        for item in matrix.selection_candidates
        if item["scope"] == "tep" and item["model"] == "LSTM"
    } >= {
        "LSTM__L128pts__SMALL__lr0.0003",
        "LSTM__L256pts__MEDIUM__lr0.001",
    }
    assert any(
        item["candidate_label"] == "iTransformer__L2h__SMALL__lr0.0003"
        for item in matrix.selection_candidates
        if item["scope"] == "sru"
    )
    assert (
        len(
            {
                item["checkpoint_reference_id"]
                for item in matrix.selected_checkpoint_references
            }
        )
        == 81
    )


def test_bad_declared_workload_count_is_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["declared_workload_counts"]["selection_candidates"] = 455
    try:
        build_workload_matrix(config)
    except ValueError as error:
        assert "workload count mismatch" in str(error)
    else:
        raise AssertionError("invalid declared workload count was accepted")


def test_atomic_json_and_controlled_thread_environment(tmp_path: Path) -> None:
    target = tmp_path / "status" / "RESOURCE_STATUS.json"
    atomic_write_json(target, {"generation": 1})
    atomic_write_json(target, {"generation": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 2}
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))
    assert set(controlled_thread_environment(1).values()) == {"1"}
