from __future__ import annotations

import hashlib
import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_runner() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_active3_neural3_parallel_cache.py"
    spec = importlib.util.spec_from_file_location("active3_runner_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _prediction_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_result(profile: dict[str, Any], prediction_path: Path) -> dict[str, Any]:
    metric_names = {
        "r2_level_reconstructed",
        "r2_delta",
        "r2_level_persistence",
        "mse",
        "rmse",
        "mae",
        "mse_delta",
        "rmse_delta",
        "mae_delta",
        "std_level_target",
        "std_level_prediction",
        "std_delta_target",
        "std_delta_prediction",
        "variance_ratio_level_prediction_to_target",
        "variance_ratio_delta_prediction_to_target",
        "residual_identity_max_abs_error",
        "persistence_skill",
    }
    metrics = {name: 0.1 for name in metric_names}
    metrics["residual_identity_max_abs_error"] = 0.0
    return {
        "status": "PASS",
        "dataset": profile["dataset"],
        "task_id": profile["task_id"],
        "target_head": profile["head_id"],
        "information_set": profile["information_set"],
        "availability_scenario": profile["availability_scenario"],
        "proxy_policy": profile["proxy_policy"],
        "direction": profile.get("direction"),
        "horizon_steps": int(profile["horizon_steps"]),
        "model": profile["model"],
        "metrics": metrics,
        "evaluation_rows": 1,
        "checkpoint_contract": "CACHED_DEVELOPMENT_BEST_WEIGHT_NO_RETRAIN",
        "prediction_path": str(prediction_path),
        "prediction_file_sha256": _prediction_sha(prediction_path),
        "fit_called": False,
        "refit_called": False,
        "select_called": False,
        "test_accessed": True,
    }


@pytest.fixture()
def complete_report_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config, matrix = RUNNER.load_scope()
    monkeypatch.setattr(RUNNER, "load_scope", lambda: (config, matrix))
    monkeypatch.setattr(RUNNER, "update_status", lambda *args, **kwargs: {})
    # The strict provenance helper is exercised by runner integration tests;
    # this fixture isolates report aggregation and supplies synthetic outputs.
    monkeypatch.setattr(RUNNER, "_validate_test_result", lambda *args, **kwargs: None)
    run_root = tmp_path / "run"
    for index, profile in enumerate(matrix.profiles):
        result_dir = run_root / "test_results" / str(index)
        result_dir.mkdir(parents=True, exist_ok=True)
        prediction = result_dir / "test.parquet"
        prediction.write_bytes(f"prediction-{index}".encode("ascii"))
        result = _make_result(dict(profile), prediction)
        (result_dir / "TEST_RESULT.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
    return run_root, matrix


def test_build_report_accepts_complete_profile_bound_results(complete_report_fixture) -> None:
    run_root, matrix = complete_report_fixture
    report = RUNNER.build_report(run_root)
    assert report["status"] == "PASS"
    assert report["result_count"] == len(matrix.profiles) == 81
    assert report["leaderboard_count"] > 0


def test_build_report_rejects_duplicate_profile_even_when_count_is_81(
    complete_report_fixture,
) -> None:
    run_root, matrix = complete_report_fixture
    first = run_root / "test_results" / "0" / "TEST_RESULT.json"
    value = json.loads(first.read_text(encoding="utf-8"))
    # Profiles 0 and 1 share the view but use different model identities.
    value["model"] = matrix.profiles[1]["model"]
    first.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="STOP_REPORT_RESULT_PROFILE_DUPLICATE"):
        RUNNER.build_report(run_root)


def test_build_report_rejects_prediction_hash_drift(complete_report_fixture) -> None:
    run_root, _ = complete_report_fixture
    prediction = run_root / "test_results" / "0" / "test.parquet"
    prediction.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="STOP_REPORT_RESULT_PREDICTION_SHA_MISMATCH"):
        RUNNER.build_report(run_root)


def _minimal_freeze() -> dict[str, Any]:
    selections = [{"profile_id": str(index), "logical_candidate_id": str(index)} for index in range(81)]
    return {
        "status": "GLOBAL_SELECTION_FROZEN",
        "sealed": True,
        "scope": "ACTIVE3_NEURAL3_456_CANDIDATES_81_PROFILES",
        "selection_candidate_count": 456,
        "selected_checkpoint_reference_count": 81,
        "test_accessed": False,
        "deletion_forbidden": True,
        "selections": selections,
        "selections_sha256": RUNNER.stable_hash(selections),
        "code": {"branch": RUNNER.EXPECTED_BRANCH, "commit": "old", "dirty": False},
    }


def test_existing_freeze_rejects_selection_hash_drift(tmp_path: Path) -> None:
    value = _minimal_freeze()
    value["selections_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="STOP_EXISTING_GLOBAL_SELECTION_FREEZE_HASH_MISMATCH"):
        RUNNER._validate_existing_global_freeze(tmp_path, value)


def test_existing_freeze_rejects_code_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _minimal_freeze()
    monkeypatch.setattr(
        RUNNER,
        "code_identity",
        lambda *, require_clean: {"branch": RUNNER.EXPECTED_BRANCH, "commit": "current", "dirty": False},
    )
    with pytest.raises(RuntimeError, match="STOP_EXISTING_GLOBAL_SELECTION_FREEZE_CODE_BINDING_MISMATCH"):
        RUNNER._validate_existing_global_freeze(tmp_path, value)


def test_safe_drain_signals_and_reaps_worker_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 1234

        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float = 0.0) -> int:
            if self.returncode is None:
                raise TimeoutError
            return self.returncode

    process = FakeProcess()
    signals: list[tuple[int, int]] = []

    monkeypatch.setattr(RUNNER.os, "name", "posix")
    monkeypatch.setattr(RUNNER.os, "getpgid", lambda _pid: 4321, raising=False)
    sigkill = getattr(signal, "SIGKILL", 9)
    monkeypatch.setattr(RUNNER.signal, "SIGKILL", sigkill, raising=False)

    def fake_killpg(group: int, signum: int) -> None:
        signals.append((group, signum))
        if signum == sigkill:
            process.returncode = -signum

    monkeypatch.setattr(RUNNER.os, "killpg", fake_killpg, raising=False)
    running = {process: {"candidate_id": "candidate"}}
    forced = RUNNER._drain_running_workers(running, timeout_seconds=0.0)

    assert running == {}
    assert signals == [(4321, signal.SIGTERM), (4321, sigkill)]
    assert forced[-1]["reason"] == "SAFE_DRAIN_TIMEOUT_KILL_SENT"


def test_pilot_worker_wrapper_uses_isolated_process_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        pid = 5678

        def __init__(self) -> None:
            self.wait_calls: list[float | None] = []

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls.append(timeout)
            return 0

        def poll(self) -> int:
            return 0

    process = FakeProcess()
    commands: list[list[str]] = []

    def fake_spawn(command):
        commands.append(list(command))
        return process

    monkeypatch.setattr(RUNNER, "_spawn_worker_subprocess", fake_spawn)
    returncode = RUNNER._run_worker_subprocess(
        tmp_path,
        {"candidate_id": "candidate"},
    )

    assert returncode == 0
    assert commands and commands[0][-2:] == ["--candidate-id", "candidate"]
    assert process.wait_calls[0] is None
    assert 0.0 in process.wait_calls[1:]


def test_safe_drain_window_is_bounded_to_thirty_seconds() -> None:
    assert RUNNER.SAFE_DRAIN_TIMEOUT_SECONDS == 30.0


def test_test_worker_wrapper_uses_isolated_process_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        pid = 6789

        def __init__(self) -> None:
            self.wait_calls: list[float | None] = []

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls.append(timeout)
            return 0

        def poll(self) -> int:
            return 0

    process = FakeProcess()
    commands: list[list[str]] = []
    environments: list[dict[str, str] | None] = []

    def fake_spawn(command, *, environment=None):
        commands.append(list(command))
        environments.append(None if environment is None else dict(environment))
        return process

    monkeypatch.setattr(RUNNER, "_spawn_worker_subprocess", fake_spawn)
    returncode = RUNNER._run_test_worker_subprocess(
        tmp_path,
        "profile-id",
        environment={"PRISM_NEURAL3_INFERENCE_ONLY": "1"},
    )

    assert returncode == 0
    assert commands and commands[0][-2:] == ["--profile-id", "profile-id"]
    assert commands[0][2] == "test-worker"
    assert environments == [{"PRISM_NEURAL3_INFERENCE_ONLY": "1"}]
    assert process.wait_calls[0] is None
    assert 0.0 in process.wait_calls[1:]


def test_spawn_worker_uses_new_session_and_captures_stable_posix_pgid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        pid = 4321

    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_popen(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        return FakeProcess()

    monkeypatch.setattr(RUNNER.os, "name", "posix")
    monkeypatch.setattr(RUNNER.subprocess, "Popen", fake_popen)
    process = RUNNER._spawn_worker_subprocess(
        ["python", "worker"], environment={"TEST": "1"}
    )

    assert process._active3_process_group == 4321
    assert calls == [
        (
            ["python", "worker"],
            {
                "cwd": RUNNER._REPOSITORY,
                "env": {"TEST": "1"},
                "start_new_session": True,
            },
        )
    ]


def test_aborted_worker_wait_drains_and_confirms_empty_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del tmp_path

    class FakeProcess:
        pid = 9876

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            raise RuntimeError("simulated interrupt while loading data")

        def poll(self) -> None:
            return None

    process = FakeProcess()
    drained: list[dict[Any, Any]] = []

    def fake_drain(running, **kwargs):
        del kwargs
        drained.append(running)
        running.clear()
        return []

    monkeypatch.setattr(RUNNER, "_drain_running_workers", fake_drain)
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        RUNNER._wait_isolated_subprocess(process, {"candidate_id": "loading"})
    assert len(drained) == 1
    assert drained[0] == {}


def test_nonzero_worker_exit_kills_descendant_group_before_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedProcess:
        pid = 2468

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 17

        def poll(self) -> int:
            return 17

    signals: list[dict[str, Any]] = []
    monkeypatch.setattr(
        RUNNER,
        "_signal_worker_group",
        lambda process, **kwargs: signals.append(
            {"pid": process.pid, **kwargs}
        ),
    )
    assert RUNNER._wait_isolated_subprocess(
        FailedProcess(), {"candidate_id": "failed"}
    ) == 17
    assert signals == [{"pid": 2468, "force": True, "allow_exited": True}]


def test_safe_drain_keeps_unreaped_worker_for_explicit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StuckProcess:
        pid = 9753

        def poll(self) -> None:
            return None

    process = StuckProcess()
    monkeypatch.setattr(RUNNER, "_signal_worker_group", lambda *a, **k: None)
    monkeypatch.setattr(RUNNER, "_reap_worker", lambda *a, **k: False)
    running = {process: {"candidate_id": "stuck-loader"}}
    forced = RUNNER._drain_running_workers(running, timeout_seconds=0.0)
    assert process in running
    assert forced[-1]["reason"] == "SAFE_DRAIN_TIMEOUT_KILL_SENT_UNREAPED"


def test_seal_revalidates_existing_selected_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected_path = tmp_path / "freeze" / "SELECTED_CHECKPOINT_MANIFEST.json"
    selected_path.parent.mkdir(parents=True)
    selected_path.write_text("{}", encoding="utf-8")
    expected = {"status": "SELECTED_CHECKPOINTS_SEALED", "sealed": True}
    calls: list[tuple[Path, dict[str, Any]]] = []

    def validate(root: Path, value: dict[str, Any]) -> dict[str, Any]:
        calls.append((root, value))
        return expected

    monkeypatch.setattr(RUNNER, "_validate_existing_selected_manifest", validate)
    assert RUNNER.seal_checkpoints(tmp_path) == expected
    assert calls == [(tmp_path, {})]


def test_pilot_resource_gate_runs_before_each_worker_and_persists_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, matrix = RUNNER.load_scope()
    monkeypatch.setattr(RUNNER, "load_scope", lambda: (config, matrix))
    monkeypatch.setattr(RUNNER, "prepare_data_bindings", lambda *args, **kwargs: {})
    monkeypatch.setattr(RUNNER, "require_gpu", lambda: object())
    monkeypatch.setattr(RUNNER, "update_status", lambda *args, **kwargs: {})

    class FakeSnapshot:
        def to_json(self) -> dict[str, Any]:
            return {"memory_current_bytes": 1, "gpu_used_bytes": 2}

    class FakeDecision:
        dispatch_allowed = True
        reasons: tuple[str, ...] = ()

        def to_json(self) -> dict[str, Any]:
            return {"action": "ALLOW_DISPATCH", "dispatch_allowed": True}

    sequence: list[tuple[str, str]] = []

    def collect(*args, **kwargs):
        del args, kwargs
        sequence.append(("snapshot", ""))
        return FakeSnapshot()

    def decide(snapshot, *, thresholds):
        del snapshot, thresholds
        sequence.append(("decision", ""))
        return FakeDecision()

    monkeypatch.setattr(RUNNER, "collect_resource_snapshot", collect)
    monkeypatch.setattr(RUNNER, "decide_resource_action", decide)

    class FakeCache:
        def candidate_dir(self, candidate_id: str) -> Path:
            return tmp_path / "candidates" / candidate_id

    monkeypatch.setattr(RUNNER, "cache_for", lambda root: FakeCache())

    def candidate_record(root: Path, task: dict[str, Any]) -> dict[str, Any]:
        del root
        return {
            "candidate_id": f"logical-{task['model']}",
            "files": [],
            "validation_metrics": {
                "peak_vram_bytes": 2,
                "peak_process_rss_bytes": 3,
                "best_epoch": 1,
                "validation_mse": 0.1,
                "parameter_count": 1,
            },
        }

    monkeypatch.setattr(RUNNER, "_candidate_record", candidate_record)

    def worker(root: Path, task: dict[str, Any]) -> int:
        del root
        sequence.append(("worker", str(task["model"])))
        return 0

    monkeypatch.setattr(RUNNER, "_run_worker_subprocess", worker)
    result = RUNNER.run_pilots(tmp_path)

    assert result["status"] == "PASS"
    assert [kind for kind, _ in sequence] == [
        "snapshot",
        "decision",
        "worker",
        "snapshot",
        "decision",
        "worker",
        "snapshot",
        "decision",
        "worker",
    ]
    dispatch_log = json.loads(
        (tmp_path / "logs" / "GPU_RESOURCE_PILOT_DISPATCHES.json").read_text(
            encoding="utf-8"
        )
    )
    assert dispatch_log["status"] == "PASS"
    assert len(dispatch_log["records"]) == 3
    assert all(item["status"] == "DISPATCH_ALLOWED" for item in dispatch_log["records"])


def test_pilot_resource_gate_stops_before_worker_when_snapshot_denies_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, matrix = RUNNER.load_scope()
    monkeypatch.setattr(RUNNER, "load_scope", lambda: (config, matrix))
    monkeypatch.setattr(RUNNER, "prepare_data_bindings", lambda *args, **kwargs: {})
    monkeypatch.setattr(RUNNER, "require_gpu", lambda: object())
    monkeypatch.setattr(RUNNER, "update_status", lambda *args, **kwargs: {})

    snapshot = SimpleNamespace(to_json=lambda: {"memory_current_bytes": 99})
    decision = SimpleNamespace(
        dispatch_allowed=False,
        reasons=("TEST_STOPLINE",),
        to_json=lambda: {"action": "STOP_DISPATCH", "dispatch_allowed": False},
    )
    monkeypatch.setattr(RUNNER, "collect_resource_snapshot", lambda *a, **k: snapshot)
    monkeypatch.setattr(RUNNER, "decide_resource_action", lambda *a, **k: decision)
    monkeypatch.setattr(
        RUNNER,
        "_run_worker_subprocess",
        lambda *a, **k: pytest.fail("worker must not run after a denied pilot gate"),
    )

    with pytest.raises(RuntimeError, match="STOP_PILOT_RESOURCE_STOPLINE:TEST_STOPLINE"):
        RUNNER.run_pilots(tmp_path)
    dispatch_log = json.loads(
        (tmp_path / "logs" / "GPU_RESOURCE_PILOT_DISPATCHES.json").read_text(
            encoding="utf-8"
        )
    )
    assert dispatch_log["status"] == "STOPPED"
    assert dispatch_log["records"][0]["status"] == "DISPATCH_STOPPED"
    assert (tmp_path / "STOP_REQUESTED").is_file()


@pytest.mark.parametrize("scope", ["tep", "cz"])
def test_run_tests_materializes_cz_only_for_cz_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    config, full_matrix = RUNNER.load_scope()
    profile = next(item for item in full_matrix.profiles if item["scope"] == scope)
    matrix = SimpleNamespace(profiles=(dict(profile),))
    monkeypatch.setattr(RUNNER, "load_scope", lambda: (config, matrix))
    monkeypatch.setattr(RUNNER, "update_status", lambda *args, **kwargs: {})
    monkeypatch.setattr(RUNNER, "_validate_test_result", lambda *args, **kwargs: None)
    materialized: list[Path] = []
    monkeypatch.setattr(
        RUNNER, "materialize_cz_targets", lambda root: materialized.append(root)
    )
    result_path = RUNNER.test_output(tmp_path, profile) / "TEST_RESULT.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("{}", encoding="utf-8")

    result = RUNNER.run_tests(tmp_path, dataset=scope, model=None)
    assert result["status"] == "PASS"
    assert bool(materialized) is (scope == "cz")


def test_verify_immutability_rejects_tampered_pretest_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = {
        "status": "PASS",
        "files": [],
        "file_count": 0,
        "inventory_sha256": "0" * 64,
        "deletion_forbidden": True,
    }
    path = tmp_path / "freeze" / "PRE_TEST_IMMUTABILITY_INVENTORY.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(RuntimeError, match="STOP_PRE_TEST_IMMUTABILITY_INVENTORY_INVALID"):
        RUNNER.verify_immutability(tmp_path)


def test_privacy_audit_scans_untracked_paths_and_redacts_content_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "safe.py").write_text("print('ok')\n", encoding="utf-8")
    (repository / "raw_cz.xlsx").write_bytes(b"private workbook")
    (repository / "id_rsa").write_text("private key material", encoding="utf-8")
    secret_label = "to" + "ken"
    secret_value = "abc" + "defghijk" + "lmn"
    (repository / "settings.txt").write_text(
        f"{secret_label}: {secret_value}\n", encoding="utf-8"
    )
    monkeypatch.setattr(RUNNER, "_REPOSITORY", repository)
    monkeypatch.setattr(RUNNER, "RAW_CZ_ROOT", tmp_path / "raw")

    def git_paths(*args: str) -> str:
        if args == ("ls-files",):
            return "safe.py\n"
        if args == ("ls-files", "--others", "--exclude-standard"):
            return "raw_cz.xlsx\nid_rsa\nsettings.txt\n"
        raise AssertionError(args)

    monkeypatch.setattr(RUNNER, "git_output", git_paths)
    run_root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="STOP_PRIVACY_AUDIT_FAILED"):
        RUNNER.privacy_audit(run_root)
    audit_path = run_root / "final" / "REPORT_PRIVACY_AUDIT.json"
    audit_text = audit_path.read_text(encoding="utf-8")
    audit = json.loads(audit_text)
    assert "raw_cz.xlsx" in audit["forbidden_cz_data_files"]
    assert "id_rsa" in audit["credential_path_hits"]
    assert {item["pattern"] for item in audit["high_risk_content_hits"]} == {
        "api_or_secret_assignment"
    }
    assert "abcdefghijklmn" not in audit_text
    assert RUNNER._privacy_path_flags("test_results/cz/test.parquet", repository=False)[0] is False
    assert RUNNER._privacy_path_flags("raw/source_cz.parquet", repository=False)[0] is True


def test_privacy_audit_rejects_run_root_inside_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(RUNNER, "_REPOSITORY", repository)
    monkeypatch.setattr(RUNNER, "RAW_CZ_ROOT", tmp_path / "raw")
    monkeypatch.setattr(
        RUNNER,
        "git_output",
        lambda *args: "safe.py\n" if args == ("ls-files",) else "",
    )
    run_root = repository / "run"
    with pytest.raises(RuntimeError, match="STOP_PRIVACY_AUDIT_FAILED"):
        RUNNER.privacy_audit(run_root)
    audit = json.loads(
        (run_root / "final" / "REPORT_PRIVACY_AUDIT.json").read_text(encoding="utf-8")
    )
    assert audit["run_root_is_private_and_outside_git"] is False


def _inventory_payload(files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "files": files,
        "file_count": len(files),
        "inventory_sha256": RUNNER.stable_hash(files),
        "test_results_included": True,
        "new_test_results_allowed_after_freeze": True,
        "deletion_forbidden": True,
    }


def test_verify_immutability_allows_new_test_result_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = {"name": "test_results/existing/TEST_RESULT.json", "bytes": 1, "sha256": "a", "mtime_ns": 1}
    frozen = {"name": "checkpoint_cache/CANDIDATE_CHECKPOINT_MANIFEST.json", "bytes": 1, "sha256": "b", "mtime_ns": 2}
    before = _inventory_payload([existing, frozen])
    before_path = tmp_path / "freeze" / "PRE_TEST_IMMUTABILITY_INVENTORY.json"
    before_path.parent.mkdir(parents=True, exist_ok=True)
    before_path.write_text(json.dumps(before), encoding="utf-8")
    added = {"name": "test_results/new/TEST_RESULT.json", "bytes": 3, "sha256": "c", "mtime_ns": 3}
    monkeypatch.setattr(
        RUNNER,
        "immutable_inventory",
        lambda root: _inventory_payload([existing, frozen, added]),
    )
    monkeypatch.setattr(RUNNER, "update_status", lambda *args, **kwargs: {})

    result = RUNNER.verify_immutability(tmp_path)
    assert result["status"] == "PASS"
    assert result["new_test_result_files"] == ["test_results/new/TEST_RESULT.json"]


@pytest.mark.parametrize("mutation", ["modified", "deleted"])
def test_verify_immutability_rejects_existing_test_result_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    existing = {"name": "test_results/existing/TEST_RESULT.json", "bytes": 1, "sha256": "a", "mtime_ns": 1}
    before = _inventory_payload([existing])
    before_path = tmp_path / "freeze" / "PRE_TEST_IMMUTABILITY_INVENTORY.json"
    before_path.parent.mkdir(parents=True, exist_ok=True)
    before_path.write_text(json.dumps(before), encoding="utf-8")
    if mutation == "modified":
        after_files = [{"name": existing["name"], "bytes": 2, "sha256": "z", "mtime_ns": 1}]
    else:
        after_files = []
    monkeypatch.setattr(
        RUNNER, "immutable_inventory", lambda root: _inventory_payload(after_files)
    )
    with pytest.raises(RuntimeError, match="STOP_POST_TEST_IMMUTABILITY_CHANGED"):
        RUNNER.verify_immutability(tmp_path)
