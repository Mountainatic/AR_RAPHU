from __future__ import annotations

import os

# These limits must be in the environment before NumPy/PyTorch are imported.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_PROJECT = Path(__file__).resolve().parents[1]
_REPOSITORY = _PROJECT.parent
for _path in (_PROJECT, _PROJECT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch

# Keep the per-worker CPU footprint bounded in addition to BLAS/OpenMP env vars.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from prism_benchmark.cpu_data import HeadSpec, ViewSpec, load_heads
from prism_benchmark.cz_l256_nowcast import (
    build_development_direction,
    materialize_target_direction,
    view as cz_view,
)
from prism_benchmark.neural3_cached import (
    SELECTION_PHASE,
    build_data_binding,
    logical_candidate_id,
    prepare_candidate,
    train_selection_candidate,
    validate_data_binding,
)
from prism_benchmark.neural3_inference import (
    INFERENCE_ENVIRONMENT,
    predict_cached_ensemble,
)
from prism_benchmark.neural_candidate_cache import (
    NeuralCandidateCache,
    atomic_write_json,
    file_record,
    sha256_file,
    stable_hash,
)
from prism_benchmark.neural_resource_guard import (
    GIB,
    ResourceThresholds,
    build_workload_matrix,
    collect_resource_snapshot,
    decide_resource_action,
    load_workload_config,
)
from prism_benchmark.tep_nowcast_c1 import (
    build_development as build_tep_nowcast_development_c1,
    build_test as build_tep_nowcast_test_c1,
    protocol_summary as tep_nowcast_protocol_summary,
)
from prism_benchmark.v211_support import SUPPORT_CONTRACT


DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/PRISM_V211_ACTIVE3_NEURAL3_TEP_NOWCAST_H0_20260831_R1"
)
PUBLIC_FORECAST_SHARED = Path(
    "/root/autodl-tmp/PRISM_V211_REPRESENTATIVE_TEP_SRU_CZ_L256_FORMAL_20260825_R3/public/shared"
)
RAW_CZ_ROOT = Path(
    "/root/autodl-tmp/PRISM_DATASETS_V1/raw_sources/cz_czochralski"
)
CONFIG_PATH = _PROJECT / "configs" / "active3_neural3_parallel_cache_20260826.json"
EXPECTED_BRANCH = "prism-v2-1-1-active3-neural3-tep-nowcast-h0-20260831"
SCREENING_SEED = 20260817
# A resource stopline gives workers a bounded checkpoint window before the
# supervisor terminates their whole process groups.  Keep this short enough
# that a stuck data read cannot hold the run open indefinitely.
SAFE_DRAIN_TIMEOUT_SECONDS = 30.0


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def event(name: str, **payload: Any) -> None:
    print(json.dumps({"event": name, "utc": utc(), **payload}, sort_keys=True), flush=True)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_REPOSITORY, check=True, capture_output=True, text=True
    ).stdout.strip()


def code_identity(*, require_clean: bool) -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    commit = git_output("rev-parse", "HEAD")
    dirty = git_output("status", "--porcelain=v1", "--untracked-files=all")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"STOP_EXECUTION_BRANCH_MISMATCH:{branch}")
    if require_clean and dirty:
        raise RuntimeError("STOP_EXECUTION_WORKTREE_DIRTY")
    return {"branch": branch, "commit": commit, "dirty": bool(dirty)}


def require_gpu() -> torch.device:
    """Fail closed: this namespace is a GPU experiment, never a CPU fallback."""

    if not torch.cuda.is_available():
        raise RuntimeError("STOP_ACTIVE3_NEURAL3_REQUIRES_CUDA")
    return torch.device("cuda")


def cache_for(run_root: Path) -> NeuralCandidateCache:
    return NeuralCandidateCache(run_root / "checkpoint_cache").initialize()


def load_scope() -> tuple[dict[str, Any], Any]:
    return load_workload_config(CONFIG_PATH)


def configured_resource_thresholds(config: Mapping[str, Any] | None = None) -> ResourceThresholds:
    """Build every runtime threshold from the frozen config in decimal GB."""

    if config is None:
        config, _ = load_scope()
    resources = config.get("resources")
    if not isinstance(resources, Mapping):
        raise RuntimeError("STOP_ACTIVE3_RESOURCE_CONFIG_MISSING")
    if resources.get("resource_unit") != "decimal_GB":
        raise RuntimeError("STOP_ACTIVE3_RESOURCE_UNIT_MISMATCH")
    try:
        thresholds = ResourceThresholds.from_gib(resources)
        hard_limit_bytes = int(float(resources["container_memory_hard_limit_gib"]) * GIB)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("STOP_ACTIVE3_RESOURCE_THRESHOLDS_INVALID") from error
    if thresholds.ram_safe_stop_bytes >= hard_limit_bytes:
        raise RuntimeError("STOP_ACTIVE3_SAFE_STOP_MUST_PRECEDE_CGROUP_LIMIT")
    return thresholds


def _pilot_protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen protocol fields that govern a resource pilot."""

    # Keep the protocol binding explicit and independent from JSON formatting
    # so a semantically changed protocol cannot reuse an old pilot.  The raw
    # config file hash is recorded separately below for byte-for-byte binding.
    fields = (
        "protocol_id",
        "status",
        "active_scope",
        "neural_protocol",
        "declared_view_counts",
        "declared_workload_counts",
        "resources",
        "cache_and_freeze",
    )
    return {field: config[field] for field in fields if field in config}


def _pilot_hash_aggregate(
    records: Sequence[Mapping[str, Any]], field: str, *, allow_missing: bool = False
) -> str:
    """Hash the per-model provenance values in deterministic model order."""

    values = []
    for record in sorted(records, key=lambda item: str(item.get("model", ""))):
        model = str(record.get("model", ""))
        value = record.get(field)
        if not isinstance(value, str) or not value:
            if not allow_missing:
                raise RuntimeError(f"STOP_RESOURCE_PILOT_{field.upper()}_MISSING:{model}")
            value = "UNAVAILABLE"
        values.append({"model": model, field: value})
    return stable_hash(values)


def _pilot_provenance(
    run_root: Path,
    config: Mapping[str, Any],
    matrix: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    data_bindings: Mapping[str, Any] | None = None,
    require_clean_code: bool = False,
    allow_missing_record_hashes: bool = False,
) -> dict[str, Any]:
    """Build the exact code/config/protocol/data/support pilot binding."""

    identity = code_identity(require_clean=require_clean_code)
    binding_payload: Any = []
    if isinstance(data_bindings, Mapping):
        binding_payload = data_bindings.get("bindings", [])
        if not isinstance(binding_payload, list):
            raise RuntimeError("STOP_RESOURCE_PILOT_DATA_BINDING_SCHEMA_INVALID")
    # ``matrix`` is intentionally part of the protocol digest: a changed
    # workload matrix must never inherit resource estimates from a prior run.
    protocol_payload = {
        **_pilot_protocol_payload(config),
        "workload_matrix": {
            "profiles": [dict(item) for item in matrix.profiles],
            "selection_candidates": [
                dict(item) for item in matrix.selection_candidates
            ],
            "selected_checkpoint_references": [
                dict(item) for item in matrix.selected_checkpoint_references
            ],
        },
    }
    return {
        "code": identity,
        "code_hash": stable_hash(identity),
        "code_commit": identity["commit"],
        "config_hash": sha256_file(CONFIG_PATH),
        "protocol_hash": stable_hash(protocol_payload),
        "data_binding_hash": stable_hash(binding_payload),
        "data_hash": _pilot_hash_aggregate(
            records, "data_hash", allow_missing=allow_missing_record_hashes
        ),
        "support_hash": _pilot_hash_aggregate(
            records, "support_hash", allow_missing=allow_missing_record_hashes
        ),
        "sample_order_hash": _pilot_hash_aggregate(
            records, "sample_order_hash", allow_missing=allow_missing_record_hashes
        ),
    }


def _history_labels(task: Mapping[str, Any]) -> dict[int, str]:
    return {int(key): str(value) for key, value in task["history_labels"].items()}


def _data_binding_path(run_root: Path, shared: Path, view: ViewSpec) -> Path:
    key = stable_hash(
        {"shared_root": str(Path(shared).resolve()), "view": view.relative_root.as_posix()}
    )[:32]
    return run_root / "data_bindings" / f"{key}.json"


def data_binding_for(run_root: Path, shared: Path, view: ViewSpec) -> dict[str, Any]:
    """Load a prehashed view binding, creating it atomically on first use."""

    path = _data_binding_path(run_root, shared, view)
    if path.is_file():
        binding = read_json(path)
        validate_data_binding(shared, view, binding, verify_sha256=False)
        return binding
    binding = build_data_binding(shared, view)
    atomic_write_json(path, binding)
    return binding


def prepare_data_bindings(
    run_root: Path, tasks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Hash each train/validation view once and fail on any later drift."""

    records: dict[str, dict[str, Any]] = {}
    for task in tasks:
        shared, view = task_view(run_root, task)
        key = _data_binding_path(run_root, shared, view).stem
        if key in records:
            continue
        path = _data_binding_path(run_root, shared, view)
        if path.is_file():
            binding = read_json(path)
            validate_data_binding(shared, view, binding, verify_sha256=True)
        else:
            binding = build_data_binding(shared, view)
            atomic_write_json(path, binding)
        records[key] = binding
    result = {
        "status": "PASS",
        "view_count": len(records),
        "bindings": [records[key] for key in sorted(records)],
        "train_validation_only": True,
        "test_accessed": False,
    }
    atomic_write_json(run_root / "logs" / "DATA_BINDINGS.json", result)
    return result


def task_view(run_root: Path, task: Mapping[str, Any]) -> tuple[Path, ViewSpec]:
    if task["scope"] in {"tep", "sru"}:
        shared = (
            run_root / "tep_nowcast" / "shared"
            if task["scope"] == "tep"
            else PUBLIC_FORECAST_SHARED
        )
        heads = [head for head in load_heads(shared) if head.task_id == task["task_id"]]
        if len(heads) != 1:
            raise RuntimeError(f"STOP_PUBLIC_HEAD_NOT_UNIQUE:{task['task_id']}")
        config, _ = load_scope()
        scope = config["active_scope"][str(task["scope"])]
        configured = [
            item for item in scope["tasks"] if item["task_id"] == task["task_id"]
        ]
        if len(configured) != 1:
            raise RuntimeError(f"STOP_CONFIG_HEAD_NOT_UNIQUE:{task['task_id']}")
        expected = configured[0]
        head = heads[0]
        checks = {
            "head_id": (head.head_id, expected["head_id"]),
            "dataset": (head.dataset, task["dataset"]),
            "task_id": (head.task_id, task["task_id"]),
            "h_steps": (head.h_steps, expected["h_steps"]),
            "w_steps": (head.w_steps, expected["w_steps"]),
            "w0_steps": (head.w0_steps, expected["w0_steps"]),
            "cadence_seconds": (float(head.cadence_seconds), float(scope["cadence_seconds"])),
        }
        for field, (observed, configured_value) in checks.items():
            if observed != configured_value:
                raise RuntimeError(
                    f"STOP_PUBLIC_HEAD_PROTOCOL_MISMATCH:{task['task_id']}:{field}:"
                    f"{observed}!={configured_value}"
                )
        return shared, ViewSpec(
            head,
            str(task["information_set"]),
            str(task["availability_scenario"]),
            str(task["proxy_policy"]),
        )
    horizon = int(task["horizon_steps"])
    direction = str(task["direction"])
    shared = run_root / "cz" / f"h{horizon}" / "shared" / direction
    return shared, cz_view(
        str(task["information_set"]),
        h_steps=horizon,
        task_id=str(task["task_id"]),
    )


def candidate_logical_id(run_root: Path, task: Mapping[str, Any]) -> str:
    shared, view = task_view(run_root, task)
    del shared
    # Candidate labels in the matrix are generated by the same frozen label
    # contract used by prepare_candidate.
    from prism_benchmark.neural3 import Candidate

    candidate = Candidate(
        str(task["model"]),
        0,
        int(task["history_steps"]),
        str(task["capacity"]),
        float(task["learning_rate"]),
        str(task["candidate_label"]).split("__")[1][1:],
    )
    # The preceding reconstruction is asserted instead of trusted.
    if candidate.candidate_id != task["candidate_label"]:
        raise RuntimeError("STOP_MATRIX_CANDIDATE_LABEL_RECONSTRUCTION_MISMATCH")
    placeholder = type("PreparedKey", (), {"view": view, "candidate": candidate})()
    return logical_candidate_id(
        placeholder,  # type: ignore[arg-type]
        phase=SELECTION_PHASE,
        seed=int(task["seed"]),
        direction=task.get("direction"),
        horizon_steps=int(task["horizon_steps"]),
    )


def select_tasks(
    matrix: Any,
    *,
    dataset: str | None,
    model: str | None,
) -> list[dict[str, Any]]:
    result = []
    for raw in matrix.selection_candidates:
        task = dict(raw)
        if dataset and task["scope"] != dataset and task["dataset"] != dataset:
            continue
        if model and task["model"] != model:
            continue
        result.append(task)
    return result


def update_status(run_root: Path, stage: str, **extra: Any) -> dict[str, Any]:
    config, matrix = load_scope()
    cache = cache_for(run_root)
    manifest = cache.validate_manifest(validate_candidates=False)
    complete = {
        candidate_id
        for candidate_id, value in manifest["candidates"].items()
        if value.get("status") == "COMPLETE"
    }
    expected = {
        candidate_logical_id(run_root, task) for task in matrix.selection_candidates
    }
    value = {
        "status": "RUNNING" if len(complete & expected) < len(expected) else "DEVELOPMENT_COMPLETE",
        "stage": stage,
        "updated_utc": utc(),
        "pid": os.getpid(),
        "run_root": str(run_root),
        "declared_candidate_count": len(expected),
        "completed_candidate_count": len(complete & expected),
        "remaining_candidate_count": len(expected - complete),
        "unexpected_complete_candidate_count": len(complete - expected),
        "workload_counts": matrix.counts(),
        "out_of_scope": config["out_of_scope"],
        **extra,
    }
    atomic_write_json(run_root / "STATUS.json", value)
    return value


def initialize_run(run_root: Path, *, resume: bool) -> dict[str, Any]:
    identity = code_identity(require_clean=True)
    require_gpu()
    if run_root.exists() and any(run_root.iterdir()) and not resume:
        raise RuntimeError(f"STOP_REFUSING_NONEMPTY_RUN_ROOT_WITHOUT_RESUME:{run_root}")
    for name in (
        "logs",
        "freeze",
        "final",
        "reports",
        "cz",
        "test_results",
    ):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    cache_for(run_root)
    config, matrix = load_scope()
    tep_nowcast_audit = build_tep_nowcast_development_c1(
        PUBLIC_FORECAST_SHARED,
        run_root / "tep_nowcast" / "shared",
    )
    thresholds = configured_resource_thresholds(config)
    dependency_files = [
        file_record(_REPOSITORY / "pyproject.toml", relative_to=_REPOSITORY),
        file_record(_REPOSITORY / "uv.lock", relative_to=_REPOSITORY),
    ]
    environment = {
        "status": "PASS",
        "created_utc": utc(),
        "code": identity,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "resource_unit": "decimal_GB",
        "resource_thresholds_bytes": {
            "ram_dispatch_budget": thresholds.ram_dispatch_budget_bytes,
            "ram_pause_dispatch": thresholds.ram_pause_dispatch_bytes,
            "ram_safe_stop": thresholds.ram_safe_stop_bytes,
            "disk_stopline": thresholds.disk_stopline_bytes,
            "vram_dispatch_budget": thresholds.vram_dispatch_budget_bytes,
        },
        "dependency_files": dependency_files,
        "config": file_record(CONFIG_PATH, relative_to=_PROJECT),
        "protocol_hash": stable_hash(config),
        "workload_counts": matrix.counts(),
        "public_forecast_shared_readonly": str(PUBLIC_FORECAST_SHARED),
        "tep_nowcast_shared": str(run_root / "tep_nowcast" / "shared"),
        "tep_nowcast_development_audit": tep_nowcast_audit,
        "tep_nowcast_protocol": dict(tep_nowcast_protocol_summary()),
        "raw_cz_root_private": str(RAW_CZ_ROOT),
        "test_accessed": False,
        "password_persisted": False,
    }
    atomic_write_json(run_root / "logs" / "ENVIRONMENT_AND_BINDINGS.json", environment)
    atomic_write_json(run_root / "logs" / "RUN_CONFIG.json", config)
    update_status(run_root, "INITIALIZED", environment_path="logs/ENVIRONMENT_AND_BINDINGS.json")
    return environment


def build_cz_development(run_root: Path) -> dict[str, Any]:
    raw_files = sorted(RAW_CZ_ROOT.glob("*.xlsx"))
    if len(raw_files) != 1:
        raise RuntimeError(f"STOP_CZ_RAW_FILE_COUNT:{len(raw_files)}")
    raw = raw_files[0]
    config, _ = load_scope()
    records = []
    for horizon in config["active_scope"]["cz"]["horizons_steps"]:
        task_id = config["active_scope"]["cz"]["task_id_template"].format(h=horizon)
        output_root = run_root / "cz" / f"h{int(horizon)}" / "shared"
        for direction in config["active_scope"]["cz"]["directions"]:
            audit_path = output_root / direction / "C1_NATIVE_SUPPORT_AUDIT.json"
            if audit_path.is_file():
                audit = read_json(audit_path)
                if audit.get("status") != "PASS" or audit.get("test_accessed") is not False:
                    raise RuntimeError(f"STOP_INVALID_EXISTING_CZ_DEVELOPMENT:{audit_path}")
            else:
                audit = build_development_direction(
                    raw,
                    output_root,
                    direction,
                    h_steps=int(horizon),
                    task_id=task_id,
                )
            records.append(audit)
    result = {
        "status": "PASS",
        "stage": "CZ_SOURCE_ROD_DEVELOPMENT_ONLY",
        "records": records,
        "target_rod_accessed_for_direction_selection": False,
        "test_accessed": False,
    }
    atomic_write_json(run_root / "logs" / "CZ_DEVELOPMENT_BUILD.json", result)
    update_status(run_root, "CZ_DEVELOPMENT_BUILT")
    return result


class WorkerStopline:
    def __init__(
        self,
        run_root: Path,
        *,
        thresholds: ResourceThresholds,
        gpu_index: int = 0,
    ) -> None:
        self.run_root = run_root
        self.thresholds = thresholds
        self.gpu_index = int(gpu_index)
        self.next_check = 0.0
        self.reason: str | None = None

    def __call__(self) -> str | None:
        if self.reason:
            return self.reason
        now = time.monotonic()
        if now < self.next_check:
            return None
        self.next_check = now + 2.0
        if (self.run_root / "STOP_REQUESTED").exists():
            self.reason = "SUPERVISOR_STOP_REQUESTED"
            return self.reason
        try:
            snapshot = collect_resource_snapshot(
                run_root_data_disk(self.run_root),
                gpu_index=self.gpu_index,
                require_cgroup=True,
            )
            decision = decide_resource_action(
                snapshot,
                thresholds=self.thresholds,
            )
        except BaseException as error:
            # A worker cannot safely continue when resource telemetry itself is
            # unavailable (for example, a broken cgroup mount or disk probe).
            self.reason = f"RESOURCE_TELEMETRY_UNAVAILABLE:{type(error).__name__}"
            return self.reason
        if decision.safe_stop:
            self.reason = "RESOURCE_SAFE_STOP:" + "|".join(decision.reasons)
        elif not decision.dispatch_allowed:
            self.reason = "RESOURCE_STOPLINE:" + "|".join(decision.reasons)
        return self.reason


def run_root_data_disk(run_root: Path) -> Path:
    return run_root if run_root.exists() else run_root.parent


def candidate_worker(run_root: Path, matrix_candidate_id: str) -> dict[str, Any]:
    identity = code_identity(require_clean=True)
    device = require_gpu()
    config, matrix = load_scope()
    thresholds = configured_resource_thresholds(config)
    matches = [
        dict(item)
        for item in matrix.selection_candidates
        if item["candidate_id"] == matrix_candidate_id
    ]
    if len(matches) != 1:
        raise KeyError(f"candidate task not unique: {matrix_candidate_id}")
    task = matches[0]
    shared, view = task_view(run_root, task)
    binding = data_binding_for(run_root, shared, view)
    prepared = prepare_candidate(
        shared=shared,
        view=view,
        model_name=str(task["model"]),
        histories=sorted(_history_labels(task)),
        history_labels=_history_labels(task),
        candidate_id=str(task["candidate_label"]),
        data_binding=binding,
    )
    result = train_selection_candidate(
        cache=cache_for(run_root),
        prepared=prepared,
        code_commit=identity["commit"],
        device=device,
        direction=task.get("direction"),
        horizon_steps=int(task["horizon_steps"]),
        seed=int(task["seed"]),
        stop_requested=WorkerStopline(
            run_root,
            thresholds=thresholds,
            gpu_index=int(config["resources"].get("gpu_index", 0)),
        ),
    )
    event("CANDIDATE_WORKER_DONE", matrix_candidate_id=matrix_candidate_id, status=result["status"])
    return result


def _wait_isolated_subprocess(
    process: subprocess.Popen[Any], task: Mapping[str, Any]
) -> int:
    """Wait for one isolated worker and drain its group if waiting is aborted."""

    try:
        returncode = process.wait()
    except BaseException:
        running: dict[subprocess.Popen[Any], Mapping[str, Any]] = {
            process: task
        }
        _drain_running_workers(running)
        # ``_drain_running_workers`` removes only the handles it attempted to
        # stop.  A remaining handle means the OS did not confirm termination;
        # surface that fact instead of allowing a caller to claim a clean stop.
        if running:
            raise RuntimeError("STOP_WORKER_PROCESS_GROUP_NOT_REAPED")
        raise
    else:
        if returncode != 0:
            # The group leader may have exited while a data-loading descendant
            # is still alive.  Use the group id captured at spawn time before
            # reaping the leader, otherwise that descendant would be orphaned.
            _signal_worker_group(process, force=True, allow_exited=True)
    finally:
        if process.poll() is not None:
            _reap_worker(process)
    return int(returncode)


def _run_worker_subprocess(
    run_root: Path,
    task: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--run-root",
        str(run_root),
        "--candidate-id",
        str(task["candidate_id"]),
    ]
    process = (
        _spawn_worker_subprocess(command)
        if environment is None
        else _spawn_worker_subprocess(command, environment=environment)
    )
    return _wait_isolated_subprocess(process, task)


def _run_test_worker_subprocess(
    run_root: Path,
    profile_id: str,
    *,
    environment: Mapping[str, str],
) -> int:
    """Run inference/data loading in an isolated process group."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "test-worker",
        "--run-root",
        str(run_root),
        "--profile-id",
        str(profile_id),
    ]
    process = _spawn_worker_subprocess(command, environment=environment)
    return _wait_isolated_subprocess(process, {"candidate_id": str(profile_id)})


def _spawn_worker_subprocess(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> subprocess.Popen[Any]:
    """Start a worker in its own process group for bounded safe draining."""

    options: dict[str, Any] = {"cwd": _REPOSITORY}
    if environment is not None:
        options["env"] = dict(environment)
    if os.name == "posix":
        options["start_new_session"] = True
    elif os.name == "nt":
        options["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    process = subprocess.Popen(list(command), **options)
    if os.name == "posix":
        # ``start_new_session=True`` makes the child its own process-group
        # leader, so its PGID is its PID even if it exits before ``getpgid``
        # could be queried.  Capturing this deterministically closes the
        # leader-exit/descendant-read race during forced cleanup.
        setattr(process, "_active3_process_group", int(process.pid))
    return process


def _signal_worker_group(
    process: subprocess.Popen[Any], *, force: bool, allow_exited: bool = False
) -> None:
    """Terminate one worker group, falling back to the process handle."""

    if process.poll() is not None and not allow_exited:
        return
    try:
        if os.name == "posix":
            process_group = getattr(process, "_active3_process_group", None)
            if process_group is None:
                process_group = os.getpgid(process.pid)
            os.killpg(process_group, signal.SIGKILL if force else signal.SIGTERM)
        elif not force and hasattr(signal, "CTRL_BREAK_EVENT"):
            process.send_signal(signal.CTRL_BREAK_EVENT)
        elif force:
            # ``Popen.kill`` only stops the group leader on Windows.  taskkill
            # with ``/T`` reaches descendants that may be blocked in parquet
            # or other data-loading code before falling back to the handle.
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                process.kill()
        else:
            process.terminate()
    except (OSError, ProcessLookupError, AttributeError, ValueError):
        # The worker may have exited between poll() and signalling.  A direct
        # handle fallback still covers platforms without process-group APIs.
        try:
            process.kill() if force else process.terminate()
        except (OSError, ProcessLookupError, AttributeError, ValueError):
            pass


def _reap_worker(process: subprocess.Popen[Any], *, timeout_seconds: float = 0.0) -> bool:
    """Reap a finished worker without allowing shutdown to block indefinitely."""

    if process.poll() is not None:
        try:
            process.wait(timeout=0.0)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired, AttributeError):
            pass
        return True
    try:
        process.wait(timeout=max(0.0, float(timeout_seconds)))
    except subprocess.TimeoutExpired:
        return False
    except (OSError, ProcessLookupError, AttributeError):
        return process.poll() is not None
    return True


def _drain_running_workers(
    running: dict[subprocess.Popen[Any], Mapping[str, Any]],
    *,
    timeout_seconds: float = SAFE_DRAIN_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Wait for workers to checkpoint, then force-stop only after a deadline."""

    if not running:
        return []
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    forced: list[dict[str, Any]] = []
    while running and time.monotonic() < deadline:
        for process, task in list(running.items()):
            returncode = process.poll()
            if returncode is not None:
                if returncode != 0:
                    _signal_worker_group(process, force=True, allow_exited=True)
                _reap_worker(process)
                del running[process]
        if running:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    if not running:
        return forced
    for process, task in list(running.items()):
        _signal_worker_group(process, force=False)
        forced.append(
            {
                "pid": int(getattr(process, "pid", -1)),
                "candidate_id": str(task.get("candidate_id", "")),
                "reason": "SAFE_DRAIN_TIMEOUT_TERM_SENT",
            }
        )
    grace_deadline = time.monotonic() + min(5.0, max(0.0, timeout_seconds))
    while running and time.monotonic() < grace_deadline:
        for process in list(running):
            if _reap_worker(process, timeout_seconds=0.0):
                if process.poll() not in (None, 0):
                    _signal_worker_group(process, force=True, allow_exited=True)
                del running[process]
        if running:
            time.sleep(0.1)
    for process, task in list(running.items()):
        _signal_worker_group(process, force=True)
        reaped = _reap_worker(process, timeout_seconds=1.0)
        forced.append(
            {
                "pid": int(getattr(process, "pid", -1)),
                "candidate_id": str(task.get("candidate_id", "")),
                "reason": (
                    "SAFE_DRAIN_TIMEOUT_KILL_SENT"
                    if reaped
                    else "SAFE_DRAIN_TIMEOUT_KILL_SENT_UNREAPED"
                ),
            }
        )
        if reaped:
            del running[process]
    return forced


def _write_stop_request(run_root: Path, reasons: Sequence[str]) -> None:
    atomic_write_json(
        run_root / "STOP_REQUESTED",
        {
            "status": "CHECKPOINT_AND_SAFE_STOP_REQUESTED",
            "reasons": list(dict.fromkeys(str(reason) for reason in reasons)),
            "requested_utc": utc(),
        },
    )


def _pilot_tasks(matrix: Any) -> list[dict[str, Any]]:
    result = []
    for model in ("LSTM", "iTransformer", "TimeMixer"):
        matches = [
            dict(task)
            for task in matrix.selection_candidates
            if task["scope"] == "tep"
            and task["task_id"] == "TEP_G_NOWCAST_H0"
            and task["information_set"] == "input_only"
            and task["availability_scenario"] == "record_time"
            and task["model"] == model
            and task["history_steps"] == 128
            and task["capacity"] == "SMALL"
            and abs(float(task["learning_rate"]) - 3e-4) < 1e-15
        ]
        if len(matches) != 1:
            raise RuntimeError(f"STOP_PILOT_TASK_NOT_UNIQUE:{model}")
        result.append(matches[0])
    return result


def _candidate_record(run_root: Path, task: Mapping[str, Any]) -> dict[str, Any]:
    return cache_for(run_root).validate_candidate(candidate_logical_id(run_root, task))


def _pilot_resource_gate(
    run_root: Path,
    task: Mapping[str, Any],
    *,
    thresholds: ResourceThresholds,
    gpu_index: int,
) -> ResourceDecision:
    """Persist one resource decision before starting a pilot worker."""

    dispatches_path = run_root / "logs" / "GPU_RESOURCE_PILOT_DISPATCHES.json"
    if dispatches_path.is_file():
        prior = read_json(dispatches_path)
        prior_records = prior.get("records", [])
        if not isinstance(prior_records, list):
            raise RuntimeError("STOP_PILOT_RESOURCE_DISPATCH_LOG_INVALID")
    else:
        prior_records = []

    try:
        snapshot = collect_resource_snapshot(
            run_root_data_disk(run_root),
            gpu_index=gpu_index,
            require_cgroup=True,
        )
        decision = decide_resource_action(snapshot, thresholds=thresholds)
        record = {
            "status": "DISPATCH_ALLOWED" if decision.dispatch_allowed else "DISPATCH_STOPPED",
            "stage": "PILOT_PRE_DISPATCH",
            "created_utc": utc(),
            "model": str(task["model"]),
            "matrix_candidate_id": str(task["candidate_id"]),
            "snapshot": snapshot.to_json(),
            "decision": decision.to_json(),
            "test_accessed": False,
        }
    except BaseException as error:
        record = {
            "status": "RESOURCE_GATE_FAILED",
            "stage": "PILOT_PRE_DISPATCH",
            "created_utc": utc(),
            "model": str(task["model"]),
            "matrix_candidate_id": str(task["candidate_id"]),
            "error": f"{type(error).__name__}: {error}",
            "test_accessed": False,
        }
        prior_records.append(record)
        atomic_write_json(
            dispatches_path,
            {
                "status": "FAILED",
                "stage": "PILOT_RESOURCE_GATES",
                "records": prior_records,
                "test_accessed": False,
            },
        )
        raise

    prior_records.append(record)
    atomic_write_json(
        dispatches_path,
        {
            "status": "PASS" if decision.dispatch_allowed else "STOPPED",
            "stage": "PILOT_RESOURCE_GATES",
            "records": prior_records,
            "test_accessed": False,
        },
    )
    if not decision.dispatch_allowed:
        reasons = list(decision.reasons) or ["RESOURCE_DISPATCH_NOT_ALLOWED"]
        _write_stop_request(run_root, reasons)
        raise RuntimeError("STOP_PILOT_RESOURCE_STOPLINE:" + "|".join(reasons))
    return decision


def run_pilots(run_root: Path) -> dict[str, Any]:
    config, matrix = load_scope()
    thresholds = configured_resource_thresholds(config)
    data_bindings = prepare_data_bindings(run_root, matrix.selection_candidates)
    require_gpu()
    gpu_index = int(config["resources"].get("gpu_index", 0))
    records = []
    for task in _pilot_tasks(matrix):
        _pilot_resource_gate(
            run_root,
            task,
            thresholds=thresholds,
            gpu_index=gpu_index,
        )
        event("PILOT_START", model=task["model"], candidate=task["candidate_id"])
        returncode = _run_worker_subprocess(run_root, task)
        if returncode != 0:
            raise RuntimeError(f"STOP_PILOT_WORKER_FAILED:{task['model']}:{returncode}")
        try:
            record = _candidate_record(run_root, task)
        except Exception as error:
            atomic_write_json(
                run_root / "logs" / "GPU_RESOURCE_PILOTS_FAILED.json",
                {
                    "status": "SAFE_PAUSED_OR_FAILED",
                    "model": task["model"],
                    "matrix_candidate_id": task["candidate_id"],
                    "error": f"{type(error).__name__}: {error}",
                    "test_accessed": False,
                },
            )
            raise
        metrics = record["validation_metrics"]
        hashes = record.get("hashes", {})
        records.append(
            {
                "model": task["model"],
                "matrix_candidate_id": task["candidate_id"],
                "logical_candidate_id": record["candidate_id"],
                "candidate_dir": str(cache_for(run_root).candidate_dir(record["candidate_id"])),
                "peak_vram_bytes": int(metrics["peak_vram_bytes"]),
                "peak_process_rss_bytes": int(metrics["peak_process_rss_bytes"]),
                "cache_bytes": int(sum(item["bytes"] for item in record["files"])),
                "best_epoch": int(metrics["best_epoch"]),
                "formal_max_epochs": 100,
                "formal_patience": 12,
                "reusable_formal_candidate": True,
                # Persist the candidate-level provenance alongside the
                # resource estimate so a later development resume can verify
                # the exact data/support inputs used by this pilot.
                "candidate_record_hash": record.get("record_hash"),
                "config_hash": hashes.get("config_hash"),
                "data_hash": hashes.get("data_hash"),
                "support_hash": hashes.get("support_hash"),
                "sample_order_hash": hashes.get("sample_order_hash"),
            }
        )
        event("PILOT_DONE", **records[-1])
    worst_vram = max(item["peak_vram_bytes"] for item in records)
    worst_rss = max(item["peak_process_rss_bytes"] for item in records)
    vram_capacity = thresholds.vram_dispatch_budget_bytes // max(1, worst_vram)
    ram_capacity = thresholds.ram_dispatch_budget_bytes // max(1, worst_rss)
    adopted = max(1, min(3, int(vram_capacity), int(ram_capacity)))
    identity = code_identity(require_clean=False)
    provenance = _pilot_provenance(
        run_root,
        config,
        matrix,
        records,
        data_bindings=data_bindings,
        require_clean_code=False,
        allow_missing_record_hashes=not all(
            isinstance(item.get("data_hash"), str)
            and isinstance(item.get("support_hash"), str)
            and isinstance(item.get("sample_order_hash"), str)
            for item in records
        ),
    )
    result = {
        "status": "PASS",
        "stage": "FORMAL_CANDIDATE_RESOURCE_PILOT",
        "code": identity,
        "code_hash": provenance["code_hash"],
        "code_commit": provenance["code_commit"],
        "config_hash": provenance["config_hash"],
        "protocol_hash": provenance["protocol_hash"],
        "data_binding_hash": provenance["data_binding_hash"],
        "data_hash": provenance["data_hash"],
        "support_hash": provenance["support_hash"],
        "sample_order_hash": provenance["sample_order_hash"],
        "provenance": provenance,
        "records": records,
        "worst_peak_vram_bytes": worst_vram,
        "worst_peak_process_rss_bytes": worst_rss,
        "adopted_max_concurrency": adopted,
        "vram_budget_bytes": thresholds.vram_dispatch_budget_bytes,
        "ram_budget_bytes": thresholds.ram_dispatch_budget_bytes,
        "pilot_candidates_are_part_of_formal_456": True,
        "test_accessed": False,
    }
    atomic_write_json(run_root / "logs" / "GPU_RESOURCE_PILOTS.json", result)
    update_status(run_root, "PILOTS_COMPLETE", adopted_max_concurrency=adopted)
    return result


def _pilot_estimates(run_root: Path) -> tuple[dict[str, tuple[int, int]], int]:
    pilot = read_json(run_root / "logs" / "GPU_RESOURCE_PILOTS.json")
    if (
        pilot.get("status") != "PASS"
        or pilot.get("stage") != "FORMAL_CANDIDATE_RESOURCE_PILOT"
        or pilot.get("test_accessed") is not False
        or pilot.get("pilot_candidates_are_part_of_formal_456") is not True
    ):
        raise RuntimeError("STOP_RESOURCE_PILOTS_NOT_PASS")
    records = pilot.get("records")
    expected_models = {"LSTM", "iTransformer", "TimeMixer"}
    if not isinstance(records, list) or len(records) != len(expected_models):
        raise RuntimeError("STOP_RESOURCE_PILOT_RECORD_COUNT_INVALID")
    estimates: dict[str, tuple[int, int]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            raise RuntimeError("STOP_RESOURCE_PILOT_RECORD_INVALID")
        model = str(item.get("model", ""))
        if model in estimates or model not in expected_models:
            raise RuntimeError("STOP_RESOURCE_PILOT_MODEL_SET_INVALID")
        try:
            rss = int(item["peak_process_rss_bytes"])
            vram = int(item["peak_vram_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("STOP_RESOURCE_PILOT_ESTIMATE_INVALID") from error
        if rss < 0 or vram < 0:
            raise RuntimeError("STOP_RESOURCE_PILOT_ESTIMATE_NEGATIVE")
        estimates[model] = (rss, vram)
    if set(estimates) != expected_models:
        raise RuntimeError("STOP_RESOURCE_PILOT_MODEL_SET_INVALID")
    config, _ = load_scope()
    thresholds = configured_resource_thresholds(config)
    if (
        pilot.get("ram_budget_bytes") != thresholds.ram_dispatch_budget_bytes
        or pilot.get("vram_budget_bytes") != thresholds.vram_dispatch_budget_bytes
    ):
        raise RuntimeError("STOP_RESOURCE_PILOT_BUDGET_BINDING_INVALID")
    try:
        adopted = int(pilot["adopted_max_concurrency"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("STOP_RESOURCE_PILOT_CONCURRENCY_INVALID") from error
    if adopted < 1 or adopted > 3:
        raise RuntimeError("STOP_RESOURCE_PILOT_CONCURRENCY_INVALID")

    # A resource estimate is reusable only when every input that can affect
    # candidate memory/support is byte-for-byte bound to the current run.
    config, matrix = load_scope()
    pilot_tasks = _pilot_tasks(matrix)
    data_bindings = prepare_data_bindings(run_root, matrix.selection_candidates)
    current_records: list[dict[str, Any]] = []
    for task in pilot_tasks:
        current = _candidate_record(run_root, task)
        hashes = current.get("hashes")
        if not isinstance(hashes, Mapping):
            raise RuntimeError("STOP_RESOURCE_PILOT_CANDIDATE_HASHES_MISSING")
        current_records.append(
            {
                "model": str(task["model"]),
                "candidate_record_hash": current.get("record_hash"),
                "config_hash": hashes.get("config_hash"),
                "data_hash": hashes.get("data_hash"),
                "support_hash": hashes.get("support_hash"),
                "sample_order_hash": hashes.get("sample_order_hash"),
            }
        )
    expected_provenance = _pilot_provenance(
        run_root,
        config,
        matrix,
        current_records,
        data_bindings=data_bindings,
        require_clean_code=True,
    )
    observed_provenance = pilot.get("provenance")
    if observed_provenance != expected_provenance:
        raise RuntimeError("STOP_RESOURCE_PILOT_PROVENANCE_HASH_MISMATCH")
    for field in (
        "code",
        "code_hash",
        "code_commit",
        "config_hash",
        "protocol_hash",
        "data_binding_hash",
        "data_hash",
        "support_hash",
        "sample_order_hash",
    ):
        if pilot.get(field) != expected_provenance.get(field):
            raise RuntimeError(f"STOP_RESOURCE_PILOT_{field.upper()}_BINDING_INVALID")
    pilot_records_by_model = {
        str(item.get("model")): item
        for item in records
        if isinstance(item, Mapping)
    }
    for current in current_records:
        model = str(current["model"])
        observed = pilot_records_by_model.get(model)
        if observed is None:
            raise RuntimeError("STOP_RESOURCE_PILOT_RECORD_PROVENANCE_MISSING")
        for field in (
            "candidate_record_hash",
            "config_hash",
            "data_hash",
            "support_hash",
            "sample_order_hash",
        ):
            if observed.get(field) != current.get(field):
                raise RuntimeError(
                    f"STOP_RESOURCE_PILOT_RECORD_{field.upper()}_MISMATCH:{model}"
                )
    return estimates, adopted


def _completed_logical_ids(run_root: Path) -> set[str]:
    manifest = cache_for(run_root).validate_manifest(validate_candidates=False)
    return {
        candidate_id
        for candidate_id, item in manifest["candidates"].items()
        if item.get("status") == "COMPLETE"
    }


def _resume_abandoned_locks(
    run_root: Path, tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    cache = cache_for(run_root)
    evidence = []
    for task in tasks:
        candidate_id = candidate_logical_id(run_root, task)
        try:
            result = cache.reclaim_abandoned_lock(
                candidate_id, requested_by="active3 runner --resume"
            )
        except Exception as error:
            if "MAY_STILL_BE_ACTIVE" in str(error):
                continue
            raise
        if result.get("status") != "NO_LOCK":
            evidence.append(result)
    if evidence:
        atomic_write_json(
            run_root / "logs" / f"ABANDONED_LOCK_RECOVERY_{int(time.time())}.json",
            {"status": "PASS", "records": evidence},
        )
    return evidence


def run_development(
    run_root: Path,
    *,
    dataset: str | None,
    model: str | None,
    max_concurrency: int | None,
    resume: bool,
) -> dict[str, Any]:
    config, matrix = load_scope()
    tasks = select_tasks(matrix, dataset=dataset, model=model)
    prepare_data_bindings(run_root, tasks)
    if resume:
        _resume_abandoned_locks(run_root, tasks)
    # Every task is passed through the worker on resume.  The worker computes
    # the current data/support/config hashes before calling reuse_status, so a
    # stale COMPLETE summary can never be silently accepted as reusable.
    # Exact matches return REUSED without retraining.
    pending = list(tasks)
    estimates, adopted = _pilot_estimates(run_root)
    concurrency = adopted if max_concurrency is None else min(adopted, max(1, max_concurrency))
    thresholds = configured_resource_thresholds(config)
    gpu_index = int(config["resources"].get("gpu_index", 0))
    running: dict[subprocess.Popen[Any], dict[str, Any]] = {}
    stopped_reason: list[str] = []
    worker_failure: RuntimeError | None = None
    drain_deadline: float | None = None
    started = time.time()
    stop_flag = run_root / "STOP_REQUESTED"
    if stop_flag.exists():
        stop_flag.unlink()
    try:
        while pending or running:
            finished = []
            for process, task in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                if returncode != 0:
                    # Preserve the worker's process-group id long enough to
                    # terminate any data-loading descendant before reaping
                    # the failed group leader.
                    _signal_worker_group(process, force=True, allow_exited=True)
                _reap_worker(process)
                finished.append(process)
                if returncode != 0:
                    if worker_failure is None:
                        worker_failure = RuntimeError(
                            f"STOP_CANDIDATE_WORKER_FAILED:{task['candidate_id']}:{returncode}"
                        )
                    pending.append(task)
                    stopped_reason.append(str(worker_failure))
                    if drain_deadline is None:
                        drain_deadline = time.monotonic() + SAFE_DRAIN_TIMEOUT_SECONDS
                    _write_stop_request(run_root, stopped_reason)
                    continue
                # A worker can exit successfully after checkpointing a resource
                # stopline pause. Keep that candidate queued for --resume rather
                # than reporting a false development pass.
                logical_id = candidate_logical_id(run_root, task)
                complete = logical_id in _completed_logical_ids(run_root)
                if not complete:
                    pending.append(task)
                    stopped_reason.append(
                        f"WORKER_SAFE_PAUSED_OR_INCOMPLETE:{task['candidate_id']}"
                    )
                    if drain_deadline is None:
                        drain_deadline = time.monotonic() + SAFE_DRAIN_TIMEOUT_SECONDS
                    _write_stop_request(run_root, stopped_reason)
                    event(
                        "DEVELOPMENT_CANDIDATE_REQUEUED",
                        candidate=task["candidate_id"],
                        reason="worker exited without a complete cached candidate",
                    )
                event(
                    "DEVELOPMENT_CANDIDATE_DONE",
                    candidate=task["candidate_id"],
                    complete=complete,
                )
            for process in finished:
                del running[process]
            while pending and len(running) < concurrency and not stopped_reason:
                task = pending[0]
                expected_rss, expected_vram = estimates[str(task["model"])]
                snapshot = collect_resource_snapshot(
                    run_root_data_disk(run_root),
                    gpu_index=gpu_index,
                    require_cgroup=True,
                )
                decision = decide_resource_action(
                    snapshot,
                    expected_job_rss_bytes=expected_rss,
                    expected_job_vram_bytes=expected_vram,
                    thresholds=thresholds,
                )
                atomic_write_json(
                    run_root / "logs" / "RESOURCE_STATUS.json",
                    {"snapshot": snapshot.to_json(), "decision": decision.to_json()},
                )
                if not decision.dispatch_allowed:
                    stopped_reason.extend(decision.reasons)
                    if drain_deadline is None:
                        drain_deadline = time.monotonic() + SAFE_DRAIN_TIMEOUT_SECONDS
                    _write_stop_request(run_root, stopped_reason)
                    break
                pending.pop(0)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--run-root",
                    str(run_root),
                    "--candidate-id",
                    str(task["candidate_id"]),
                ]
                try:
                    process = _spawn_worker_subprocess(command)
                except BaseException as error:
                    if worker_failure is None:
                        worker_failure = RuntimeError(
                            f"STOP_CANDIDATE_WORKER_SPAWN_FAILED:{task['candidate_id']}"
                        )
                        worker_failure.__cause__ = error
                    pending.insert(0, task)
                    stopped_reason.append(str(worker_failure))
                    if drain_deadline is None:
                        drain_deadline = time.monotonic() + SAFE_DRAIN_TIMEOUT_SECONDS
                    _write_stop_request(run_root, stopped_reason)
                    break
                running[process] = task
                event(
                    "DEVELOPMENT_CANDIDATE_DISPATCH",
                    pid=process.pid,
                    candidate=task["candidate_id"],
                    active=len(running),
                    concurrency=concurrency,
                )
            update_status(
                run_root,
                "DEVELOPMENT_RUNNING" if not stopped_reason else "RESOURCE_STOPLINE",
                running_pids=[process.pid for process in running],
                scoped_pending=len(pending),
                adopted_max_concurrency=concurrency,
                stop_reasons=stopped_reason,
            )
            if stopped_reason and running:
                if drain_deadline is None:
                    drain_deadline = time.monotonic() + SAFE_DRAIN_TIMEOUT_SECONDS
                if time.monotonic() >= drain_deadline:
                    forced = _drain_running_workers(
                        running, timeout_seconds=0.0
                    )
                    stopped_reason.extend(
                        f"{item['candidate_id']}:{item['reason']}" for item in forced
                    )
                    if running:
                        stopped_reason.append("WORKER_PROCESS_GROUP_NOT_REAPED")
                        raise RuntimeError("STOP_WORKER_PROCESS_GROUP_NOT_REAPED")
            if stopped_reason and not running:
                break
            time.sleep(2.0)
    except BaseException as error:
        stopped_reason.append(f"SUPERVISOR_EXCEPTION:{type(error).__name__}")
        try:
            _write_stop_request(run_root, stopped_reason)
        finally:
            try:
                forced = _drain_running_workers(running)
                if running:
                    raise RuntimeError("STOP_WORKER_PROCESS_GROUP_NOT_REAPED")
            except RuntimeError as drain_error:
                stopped_reason.append(str(drain_error))
                raise
            else:
                stopped_reason.extend(
                    f"{item['candidate_id']}:{item['reason']}" for item in forced
                )
        raise
    status = "SAFE_PAUSED_RESOURCE_STOPLINE" if stopped_reason else "PASS"
    result = {
        "status": "FAILED_AFTER_SAFE_DRAIN" if worker_failure else status,
        "stage": "ACTIVE3_NEURAL3_DEVELOPMENT",
        "scope_dataset": dataset,
        "scope_model": model,
        "max_concurrency": concurrency,
        "elapsed_seconds": time.time() - started,
        "stop_reasons": stopped_reason,
        "test_accessed": False,
    }
    atomic_write_json(run_root / "logs" / "DEVELOPMENT_RUN.json", result)
    update_status(
        run_root,
        result["status"],
        adopted_max_concurrency=concurrency,
    )
    if worker_failure:
        raise worker_failure
    return result


def _profile_tasks(matrix: Any, profile_id: str) -> list[dict[str, Any]]:
    return [
        dict(task)
        for task in matrix.selection_candidates
        if task["profile_id"] == profile_id
    ]


def _validate_existing_global_freeze(
    run_root: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate a persisted selection freeze before allowing resume/use."""

    if value.get("status") != "GLOBAL_SELECTION_FROZEN" or value.get("sealed") is not True:
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_INVALID")
    if value.get("scope") != "ACTIVE3_NEURAL3_456_CANDIDATES_81_PROFILES":
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_SCOPE_INVALID")
    if value.get("selection_candidate_count") != 456 or value.get("selected_checkpoint_reference_count") != 81:
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_COUNT_INVALID")
    if value.get("test_accessed") is not False or value.get("deletion_forbidden") is not True:
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_ACCESS_CONTRACT_INVALID")
    selections = value.get("selections")
    if not isinstance(selections, list) or len(selections) != 81:
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_SELECTION_COUNT_INVALID")
    if value.get("selections_sha256") != stable_hash(selections):
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_HASH_MISMATCH")

    identity = code_identity(require_clean=True)
    if value.get("code") != identity:
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_CODE_BINDING_MISMATCH")
    config, matrix = load_scope()
    expected_profiles = {str(profile["profile_id"]) for profile in matrix.profiles}
    observed_profiles = {str(selection.get("profile_id")) for selection in selections}
    if observed_profiles != expected_profiles:
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_PROFILE_SET_MISMATCH")
    task_by_selection = {
        (str(task["profile_id"]), str(task["candidate_id"])): task
        for task in matrix.selection_candidates
    }
    logical_ids = [str(selection.get("logical_candidate_id")) for selection in selections]
    if len(set(logical_ids)) != len(logical_ids):
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_LOGICAL_ID_DUPLICATE")
    expected_logical_ids = set()
    for selection in selections:
        key = (str(selection.get("profile_id")), str(selection.get("selected_matrix_candidate_id")))
        task = task_by_selection.get(key)
        if task is None or selection.get("candidate_id") != task.get("candidate_label"):
            raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_SELECTED_TASK_MISMATCH")
        expected_logical = candidate_logical_id(run_root, task)
        expected_logical_ids.add(expected_logical)
        if selection.get("logical_candidate_id") != expected_logical:
            raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_LOGICAL_ID_MISMATCH")
    if set(logical_ids) != expected_logical_ids:
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_LOGICAL_ID_UNEXPECTED")
    completed = _completed_logical_ids(run_root)
    if set(logical_ids) - completed:
        raise RuntimeError("STOP_EXISTING_GLOBAL_SELECTION_FREEZE_SELECTED_CANDIDATE_INCOMPLETE")
    return dict(value)


def freeze_selection(run_root: Path) -> dict[str, Any]:
    path = run_root / "freeze" / "GLOBAL_SELECTION_FREEZE.json"
    if path.is_file():
        return _validate_existing_global_freeze(run_root, read_json(path))
    _, matrix = load_scope()
    # Re-hash all train/validation inputs immediately before selection freeze;
    # a drift is a hard stop rather than a silent stale-cache reuse.
    prepare_data_bindings(run_root, matrix.selection_candidates)
    complete = _completed_logical_ids(run_root)
    expected = {
        candidate_logical_id(run_root, task) for task in matrix.selection_candidates
    }
    missing = sorted(expected - complete)
    if missing:
        raise RuntimeError(f"STOP_SELECTION_FREEZE_MISSING_CANDIDATES:{len(missing)}")
    cache = cache_for(run_root)
    selections = []
    for profile in matrix.profiles:
        tasks = _profile_tasks(matrix, profile["profile_id"])
        candidates = []
        for task in tasks:
            logical_id = candidate_logical_id(run_root, task)
            record = cache.validate_candidate(logical_id)
            metrics = record["validation_metrics"]
            candidates.append(
                {
                    "task": task,
                    "logical_candidate_id": logical_id,
                    "record": record,
                    "validation_mse": float(metrics["validation_mse"]),
                    "parameter_count": int(metrics["parameter_count"]),
                    "best_epoch": int(metrics["best_epoch"]),
                }
            )
        candidates.sort(
            key=lambda item: (
                item["validation_mse"],
                item["parameter_count"],
                int(item["task"]["history_steps"]),
                0 if item["task"]["capacity"] == "SMALL" else 1,
                float(item["task"]["learning_rate"]),
            )
        )
        best = candidates[0]["validation_mse"]
        eligible = [item for item in candidates if item["validation_mse"] <= best * 1.01]
        selected = min(
            eligible,
            key=lambda item: (
                item["parameter_count"],
                int(item["task"]["history_steps"]),
                0 if item["task"]["capacity"] == "SMALL" else 1,
                float(item["task"]["learning_rate"]),
            ),
        )
        task = selected["task"]
        record = selected["record"]
        selections.append(
            {
                **dict(profile),
                "selected_matrix_candidate_id": task["candidate_id"],
                "logical_candidate_id": selected["logical_candidate_id"],
                "candidate_id": task["candidate_label"],
                "history_steps": int(task["history_steps"]),
                "capacity": task["capacity"],
                "learning_rate": float(task["learning_rate"]),
                "seed": int(task["seed"]),
                "best_epoch": selected["best_epoch"],
                "validation_mse": selected["validation_mse"],
                "candidate_record_hash": record["record_hash"],
                "config_hash": record["hashes"]["config_hash"],
                "data_hash": record["hashes"]["data_hash"],
                "support_hash": record["hashes"]["support_hash"],
                "sample_order_hash": record["hashes"]["sample_order_hash"],
                "validation_prediction_hash": record["validation_prediction_hash"],
                "candidate_count": len(candidates),
                "selection_rule": "MIN_VALIDATION_MSE_THEN_1PCT_SIMPLICITY_TIEBREAK",
            }
        )
    if len(selections) != 81:
        raise RuntimeError(f"STOP_SELECTION_PROFILE_COUNT:{len(selections)}")
    value = {
        "status": "GLOBAL_SELECTION_FROZEN",
        "sealed": True,
        "sealed_utc": utc(),
        "scope": "ACTIVE3_NEURAL3_456_CANDIDATES_81_PROFILES",
        "code": code_identity(require_clean=True),
        "selection_candidate_count": 456,
        "selected_checkpoint_reference_count": 81,
        "final_fit_status": "NOT_RUN_RETRAIN_FORBIDDEN_BY_CURRENT_USER_RULE",
        "selections": selections,
        "selections_sha256": stable_hash(selections),
        "test_accessed": False,
        "target_rod_accessed_for_direction_selection": False,
        "deletion_forbidden": True,
    }
    atomic_write_json(path, value)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    update_status(run_root, "GLOBAL_SELECTION_FROZEN")
    return value


def _candidate_manifest_sha(cache: NeuralCandidateCache, candidate_id: str) -> str:
    return sha256_file(cache.candidate_dir(candidate_id) / "CANDIDATE_MANIFEST.json")


def _selected_profile_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        value.get("dataset"),
        value.get("task_id"),
        value.get("head_id", value.get("target_head")),
        value.get("information_set"),
        value.get("availability_scenario"),
        value.get("proxy_policy"),
        value.get("direction"),
        int(value["horizon_steps"]),
        value.get("model"),
    )


def _validate_existing_selected_manifest(
    run_root: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate an already sealed checkpoint manifest before reuse."""

    if value.get("status") != "SELECTED_CHECKPOINTS_SEALED" or value.get("sealed") is not True:
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_INVALID")
    if value.get("selected_checkpoint_count") != 81:
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_COUNT_INVALID")
    if value.get("checkpoint_contract") != "REFERENCE_CACHED_DEVELOPMENT_BEST_WEIGHT_NO_RETRAIN":
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_CHECKPOINT_CONTRACT_INVALID")
    if value.get("deletion_forbidden") is not True or value.get("test_accessed") is not False:
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_ACCESS_CONTRACT_INVALID")
    selected = value.get("selected_checkpoints")
    if not isinstance(selected, list) or len(selected) != 81:
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_SELECTION_COUNT_INVALID")
    if value.get("selected_checkpoints_sha256") != stable_hash(selected):
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_HASH_MISMATCH")

    freeze_path = run_root / "freeze" / "GLOBAL_SELECTION_FREEZE.json"
    freeze = _validate_existing_global_freeze(run_root, read_json(freeze_path))
    cache = cache_for(run_root)
    candidate_manifest = cache.validate_manifest(validate_candidates=True)
    if (
        candidate_manifest.get("status") != "CANDIDATE_CHECKPOINTS_SEALED"
        or candidate_manifest.get("sealed") is not True
        or candidate_manifest.get("deletion_forbidden") is not True
    ):
        raise RuntimeError("STOP_EXISTING_CANDIDATE_MANIFEST_NOT_SEALED")
    if value.get("global_selection_freeze_sha256") != sha256_file(freeze_path):
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_GLOBAL_FREEZE_SHA_MISMATCH")
    if value.get("candidate_checkpoint_manifest_sha256") != sha256_file(cache.manifest_path):
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_CANDIDATE_MANIFEST_SHA_MISMATCH")

    config, matrix = load_scope()
    expected_profiles = {_selected_profile_key(profile) for profile in matrix.profiles}
    if len(expected_profiles) != len(matrix.profiles):
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_EXPECTED_PROFILE_KEYS_INVALID")
    observed_profiles = set()
    freeze_by_profile = {
        _selected_profile_key(selection): selection
        for selection in freeze["selections"]
    }
    if set(freeze_by_profile) != expected_profiles:
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_FREEZE_PROFILE_SET_MISMATCH")
    expected_seed = int(config["neural_protocol"]["active_checkpoint_seed"])
    cache_root = cache.candidates_dir.resolve()
    for entry in selected:
        if not isinstance(entry, Mapping):
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_ENTRY_INVALID")
        try:
            key = _selected_profile_key(entry)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_PROFILE_FIELDS_INVALID") from error
        if key not in expected_profiles or key in observed_profiles:
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_PROFILE_SET_INVALID")
        observed_profiles.add(key)
        freeze_entry = freeze_by_profile[key]
        logical_id = entry.get("logical_candidate_id")
        if not isinstance(logical_id, str) or logical_id != freeze_entry.get("logical_candidate_id"):
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_LOGICAL_ID_MISMATCH")
        if entry.get("candidate_id") != freeze_entry.get("candidate_id"):
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_CANDIDATE_ID_MISMATCH")
        try:
            entry_seed = int(entry.get("seed", -1))
        except (TypeError, ValueError) as error:
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_SEED_INVALID") from error
        if entry_seed != expected_seed:
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_SEED_MISMATCH")
        if entry.get("sealed") is not True or entry.get("deletion_forbidden") is not True:
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_ENTRY_NOT_SEALED")
        record = cache.validate_candidate(logical_id)
        if record.get("sealed") is not True or record.get("deletion_forbidden") is not True:
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_CANDIDATE_NOT_SEALED")
        candidate_dir = cache.candidate_dir(logical_id).resolve()
        try:
            candidate_dir.relative_to(cache_root)
        except ValueError as error:
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_CANDIDATE_OUTSIDE_CACHE") from error
        if entry.get("candidate_dir") != str(candidate_dir):
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_CANDIDATE_DIR_MISMATCH")
        if entry.get("candidate_manifest_sha256") != _candidate_manifest_sha(cache, logical_id):
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_CANDIDATE_MANIFEST_SHA_MISMATCH")
        if entry.get("candidate_record_hash") != record.get("record_hash"):
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_CANDIDATE_RECORD_HASH_MISMATCH")
        if entry.get("config_hash") != record.get("hashes", {}).get("config_hash"):
            raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_CANDIDATE_CONFIG_HASH_MISMATCH")
    if observed_profiles != expected_profiles:
        raise RuntimeError("STOP_EXISTING_SELECTED_MANIFEST_PROFILE_SET_MISMATCH")
    return dict(value)


def seal_checkpoints(run_root: Path) -> dict[str, Any]:
    selected_path = run_root / "freeze" / "SELECTED_CHECKPOINT_MANIFEST.json"
    if selected_path.is_file():
        return _validate_existing_selected_manifest(run_root, read_json(selected_path))
    freeze_path = run_root / "freeze" / "GLOBAL_SELECTION_FREEZE.json"
    freeze = _validate_existing_global_freeze(run_root, read_json(freeze_path))
    cache = cache_for(run_root)
    _, matrix = load_scope()
    for index, task in enumerate(matrix.selection_candidates, start=1):
        logical_id = candidate_logical_id(run_root, task)
        record = cache.validate_candidate(logical_id)
        if not record.get("sealed"):
            with cache.lock(logical_id, {"operation": "post-selection seal"}):
                cache.seal_candidate(logical_id)
        if index % 25 == 0:
            event("CANDIDATE_SEAL_PROGRESS", completed=index, total=456)
    candidate_manifest = cache.seal_manifest()
    if candidate_manifest.get("status") != "CANDIDATE_CHECKPOINTS_SEALED":
        raise RuntimeError("STOP_CANDIDATE_GLOBAL_MANIFEST_NOT_SEALED")
    selected = []
    for selection in freeze["selections"]:
        logical_id = selection["logical_candidate_id"]
        record = cache.validate_candidate(logical_id)
        candidate_dir = cache.candidate_dir(logical_id)
        selected.append(
            {
                "logical_candidate_id": logical_id,
                "candidate_dir": str(candidate_dir.resolve()),
                "candidate_manifest_sha256": _candidate_manifest_sha(cache, logical_id),
                "candidate_record_hash": record["record_hash"],
                "config_hash": record["hashes"]["config_hash"],
                "dataset": selection["dataset"],
                "task_id": selection["task_id"],
                "head_id": selection["head_id"],
                "information_set": selection["information_set"],
                "availability_scenario": selection["availability_scenario"],
                "proxy_policy": selection["proxy_policy"],
                "direction": selection.get("direction"),
                "horizon_steps": int(selection["horizon_steps"]),
                "model": selection["model"],
                "candidate_id": selection["candidate_id"],
                "seed": int(selection["seed"]),
                "deletion_forbidden": True,
                "sealed": True,
            }
        )
    value = {
        "status": "SELECTED_CHECKPOINTS_SEALED",
        "sealed": True,
        "sealed_utc": utc(),
        "global_selection_freeze_sha256": sha256_file(freeze_path),
        "candidate_checkpoint_manifest_sha256": sha256_file(cache.manifest_path),
        "selected_checkpoint_count": len(selected),
        "selected_checkpoints": selected,
        "selected_checkpoints_sha256": stable_hash(selected),
        "checkpoint_contract": "REFERENCE_CACHED_DEVELOPMENT_BEST_WEIGHT_NO_RETRAIN",
        "deletion_forbidden": True,
        "test_accessed": False,
    }
    if len(selected) != 81:
        raise RuntimeError("STOP_SELECTED_CHECKPOINT_COUNT_NOT_81")
    atomic_write_json(selected_path, value)
    selected_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    bridge = {
        "status": "CHECKPOINTS_SEALED",
        "sealed": True,
        "sealed_utc": utc(),
        "selected_checkpoint_manifest_sha256": sha256_file(selected_path),
        "global_selection_freeze_sha256": sha256_file(freeze_path),
        "deletion_forbidden": True,
    }
    bridge_path = run_root / "freeze" / "CZ_CHECKPOINT_UNLOCK.json"
    atomic_write_json(bridge_path, bridge)
    bridge_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    inventory = immutable_inventory(run_root)
    atomic_write_json(run_root / "freeze" / "PRE_TEST_IMMUTABILITY_INVENTORY.json", inventory)
    update_status(run_root, "CHECKPOINTS_SEALED")
    return value


def immutable_inventory(run_root: Path) -> dict[str, Any]:
    cache = cache_for(run_root)
    files = []

    def collect_tree(root: Path) -> list[dict[str, Any]]:
        if root.is_symlink():
            raise RuntimeError(f"STOP_IMMUTABILITY_INVENTORY_SYMLINK_ROOT:{root}")
        if not root.is_dir():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"STOP_IMMUTABILITY_INVENTORY_SYMLINK:{path}")
            if path.is_file():
                records.append(file_record(path, relative_to=run_root))
        return records

    files.extend(collect_tree(cache.candidates_dir))
    # Test results are created only after the pre-test inventory is sealed.  An
    # empty pre-test tree is therefore intentional; post-test verification
    # permits new files under this namespace while protecting every prior file.
    files.extend(collect_tree(run_root / "test_results"))
    for path in (
        cache.manifest_path,
        run_root / "freeze" / "GLOBAL_SELECTION_FREEZE.json",
        run_root / "freeze" / "SELECTED_CHECKPOINT_MANIFEST.json",
        run_root / "freeze" / "CZ_CHECKPOINT_UNLOCK.json",
    ):
        if path.is_symlink():
            raise RuntimeError(f"STOP_IMMUTABILITY_INVENTORY_SYMLINK:{path}")
        if path.is_file():
            files.append(file_record(path, relative_to=run_root))
    return {
        "status": "PASS",
        "created_utc": utc(),
        "files": files,
        "file_count": len(files),
        "inventory_sha256": stable_hash(files),
        "test_results_included": True,
        "new_test_results_allowed_after_freeze": True,
        "deletion_forbidden": True,
    }


def materialize_cz_targets(run_root: Path) -> dict[str, Any]:
    raw_files = sorted(RAW_CZ_ROOT.glob("*.xlsx"))
    if len(raw_files) != 1:
        raise RuntimeError("STOP_CZ_RAW_FILE_COUNT_BEFORE_TARGET_UNLOCK")
    raw = raw_files[0]
    config, _ = load_scope()
    records = []
    for horizon in config["active_scope"]["cz"]["horizons_steps"]:
        task_id = config["active_scope"]["cz"]["task_id_template"].format(h=horizon)
        output_root = run_root / "cz" / f"h{int(horizon)}" / "shared"
        for direction in config["active_scope"]["cz"]["directions"]:
            audit_path = output_root / direction / "CZ_TARGET_TEST_ACCESS_AUDIT.json"
            if audit_path.is_file():
                audit = read_json(audit_path)
                if audit.get("target_rod_first_access_after_freeze") is not True:
                    raise RuntimeError("STOP_EXISTING_CZ_TARGET_AUDIT_INVALID")
            else:
                audit = materialize_target_direction(
                    raw,
                    output_root,
                    direction,
                    global_freeze_path=run_root / "freeze" / "GLOBAL_SELECTION_FREEZE.json",
                    checkpoint_manifest_path=run_root / "freeze" / "CZ_CHECKPOINT_UNLOCK.json",
                    h_steps=int(horizon),
                    task_id=task_id,
                )
            records.append(audit)
    result = {
        "status": "PASS",
        "stage": "CZ_TARGET_ROD_UNLOCK_AFTER_FREEZE",
        "records": records,
        "all_target_access_after_freeze": all(
            item.get("target_rod_first_access_after_freeze") is True for item in records
        ),
        "test_accessed": True,
    }
    atomic_write_json(run_root / "logs" / "CZ_TARGET_UNLOCK.json", result)
    return result


def materialize_tep_nowcast_test(run_root: Path) -> dict[str, Any]:
    """Open only the H0/W1 TEP test partition after both GPU freezes."""

    result = build_tep_nowcast_test_c1(
        PUBLIC_FORECAST_SHARED,
        run_root / "tep_nowcast" / "shared",
        run_root / "freeze" / "GLOBAL_SELECTION_FREEZE.json",
        run_root / "freeze" / "SELECTED_CHECKPOINT_MANIFEST.json",
    )
    atomic_write_json(run_root / "logs" / "TEP_NOWCAST_TEST_UNLOCK.json", result)
    return result


def _safe_component(value: Any) -> str:
    return "".join(character if str(character).isalnum() or character in "-_." else "_" for character in str(value))


def test_output(run_root: Path, profile: Mapping[str, Any]) -> Path:
    parts = [
        profile["scope"],
        profile.get("direction") or "NO_DIRECTION",
        f"H{profile['horizon_steps']}",
        profile["head_id"],
        profile["information_set"],
        profile["availability_scenario"],
        profile["proxy_policy"],
        profile["model"],
    ]
    path = run_root / "test_results"
    for part in parts:
        path /= _safe_component(part)
    return path


def test_worker(run_root: Path, profile_id: str) -> dict[str, Any]:
    identity = code_identity(require_clean=True)
    device = require_gpu()
    _, matrix = load_scope()
    profiles = [dict(item) for item in matrix.profiles if item["profile_id"] == profile_id]
    if len(profiles) != 1:
        raise KeyError(f"test profile not unique: {profile_id}")
    profile = profiles[0]
    selected_manifest = read_json(
        run_root / "freeze" / "SELECTED_CHECKPOINT_MANIFEST.json"
    )
    entries = [
        item
        for item in selected_manifest["selected_checkpoints"]
        if item["dataset"] == profile["dataset"]
        and item["task_id"] == profile["task_id"]
        and item["information_set"] == profile["information_set"]
        and item["availability_scenario"] == profile["availability_scenario"]
        and item["proxy_policy"] == profile["proxy_policy"]
        and item.get("direction") == profile.get("direction")
        and int(item["horizon_steps"]) == int(profile["horizon_steps"])
        and item["model"] == profile["model"]
    ]
    if len(entries) != 1:
        raise RuntimeError("STOP_SELECTED_TEST_PROFILE_NOT_UNIQUE")
    shared, view = task_view(run_root, profile)
    os.environ[INFERENCE_ENVIRONMENT] = "1"
    return predict_cached_ensemble(
        shared=shared,
        view=view,
        checkpoint_dirs=[Path(entries[0]["candidate_dir"])],
        split="test",
        output=test_output(run_root, profile),
        device=device,
        global_selection_freeze_path=run_root / "freeze" / "GLOBAL_SELECTION_FREEZE.json",
        candidate_manifest_path=run_root / "checkpoint_cache" / "CANDIDATE_CHECKPOINT_MANIFEST.json",
        selected_checkpoint_manifest_path=run_root / "freeze" / "SELECTED_CHECKPOINT_MANIFEST.json",
        code_commit=identity["commit"],
        direction=profile.get("direction"),
        horizon_steps=int(profile["horizon_steps"]),
    )


_TEST_RESULT_STAGE = "NEURAL3_SEALED_CHECKPOINT_TEST_INFERENCE_ONLY"
_TEST_RESULT_CHECKPOINT_CONTRACT = "CACHED_DEVELOPMENT_BEST_WEIGHT_NO_RETRAIN"
_TEST_RESULT_METRICS = frozenset(
    {
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
)


def _test_result_failure(result_path: Path, reason: str) -> RuntimeError:
    return RuntimeError(f"STOP_TEST_RESULT_INVALID:{result_path}:{reason}")


def _require_sha256(value: Any, *, field: str, result_path: Path) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise _test_result_failure(result_path, f"{field}_NOT_SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise _test_result_failure(result_path, f"{field}_NOT_LOWERCASE_HEX")
    return value


def _resolve_run_path(run_root: Path, value: Any, *, field: str, result_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise _test_result_failure(result_path, f"{field}_MISSING")
    path = Path(value)
    if not path.is_absolute():
        path = run_root / path
    return path.resolve()


def _require_sealed_result_artifact(
    path: Path, *, expected_status: str, result_path: Path
) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise _test_result_failure(
            result_path, f"SEALED_ARTIFACT_UNREADABLE:{path}"
        ) from error
    if (
        not isinstance(value, Mapping)
        or value.get("status") != expected_status
        or value.get("sealed") is not True
    ):
        raise _test_result_failure(result_path, f"SEALED_ARTIFACT_INVALID:{path}")
    return dict(value)


def _validate_test_result(
    run_root: Path,
    profile: Mapping[str, Any],
    value: Mapping[str, Any],
    result_path: Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Validate a persisted test result before treating it as reusable evidence.

    A result is reusable only when its identity, inference-only contract,
    prediction bytes, and all three sealed manifests still bind to the current
    run.  This is intentionally independent of the worker that originally
    produced the result, so a resume cannot silently trust a stale PASS file.
    """

    result_path = Path(result_path).resolve()
    run_root = Path(run_root).resolve()
    if not isinstance(value, Mapping):
        raise _test_result_failure(result_path, "RESULT_NOT_OBJECT")
    expected_output = test_output(run_root, profile).resolve()
    if result_path.parent != expected_output:
        raise _test_result_failure(result_path, "RESULT_PATH_NOT_PROFILE_OUTPUT")
    if value.get("status") != "PASS":
        raise _test_result_failure(result_path, "STATUS_NOT_PASS")
    if value.get("stage") != _TEST_RESULT_STAGE:
        raise _test_result_failure(result_path, "STAGE_INVALID")
    expected_identity = {
        "dataset": profile["dataset"],
        "task_id": profile["task_id"],
        "target_head": profile["head_id"],
        "information_set": profile["information_set"],
        "availability_scenario": profile["availability_scenario"],
        "proxy_policy": profile["proxy_policy"],
        "direction": profile.get("direction"),
        "model": profile["model"],
    }
    for field, expected in expected_identity.items():
        if value.get(field) != expected:
            raise _test_result_failure(result_path, f"PROFILE_{field.upper()}_MISMATCH")
    try:
        observed_horizon = int(value["horizon_steps"])
        expected_horizon = int(profile["horizon_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise _test_result_failure(result_path, "HORIZON_INVALID") from error
    if observed_horizon != expected_horizon:
        raise _test_result_failure(result_path, "PROFILE_HORIZON_STEPS_MISMATCH")

    for field in ("fit_called", "refit_called", "select_called", "test_rows_used_for_fitting"):
        if value.get(field) is not False:
            raise _test_result_failure(result_path, f"{field.upper()}_NOT_FALSE")
    if value.get("test_accessed") is not True:
        raise _test_result_failure(result_path, "TEST_ACCESS_NOT_RECORDED")
    if value.get("deletion_forbidden") is not True:
        raise _test_result_failure(result_path, "DELETION_FORBIDDEN_NOT_TRUE")
    if value.get("checkpoint_contract") != _TEST_RESULT_CHECKPOINT_CONTRACT:
        raise _test_result_failure(result_path, "CHECKPOINT_CONTRACT_INVALID")
    if value.get("support_contract") != SUPPORT_CONTRACT:
        raise _test_result_failure(result_path, "SUPPORT_CONTRACT_INVALID")

    try:
        evaluation_rows = int(value["evaluation_rows"])
    except (KeyError, TypeError, ValueError) as error:
        raise _test_result_failure(result_path, "EVALUATION_ROWS_INVALID") from error
    if isinstance(value.get("evaluation_rows"), bool) or evaluation_rows <= 0:
        raise _test_result_failure(result_path, "EVALUATION_ROWS_INVALID")
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise _test_result_failure(result_path, "METRICS_MISSING")
    for metric_name in _TEST_RESULT_METRICS:
        try:
            metric_value = float(metrics[metric_name])
        except (KeyError, TypeError, ValueError) as error:
            raise _test_result_failure(
                result_path, f"METRIC_INVALID:{metric_name}"
            ) from error
        if not np.isfinite(metric_value):
            raise _test_result_failure(result_path, f"METRIC_NONFINITE:{metric_name}")
    try:
        residual_error = float(metrics["residual_identity_max_abs_error"])
    except (KeyError, TypeError, ValueError) as error:
        raise _test_result_failure(result_path, "RESIDUAL_IDENTITY_INVALID") from error
    if residual_error > 1e-10:
        raise _test_result_failure(result_path, "RESIDUAL_IDENTITY_INVALID")

    hash_fields = (
        "evaluation_support_hash",
        "prediction_value_sha256",
        "prediction_file_sha256",
        "global_selection_freeze_sha256",
        "candidate_manifest_sha256",
        "selected_checkpoint_manifest_sha256",
    )
    hashes = {
        field: _require_sha256(value.get(field), field=field, result_path=result_path)
        for field in hash_fields
    }
    prediction_path = _resolve_run_path(
        run_root, value.get("prediction_path"), field="prediction_path", result_path=result_path
    )
    try:
        prediction_path.relative_to(run_root)
    except ValueError as error:
        raise _test_result_failure(result_path, "PREDICTION_OUTSIDE_RUN_ROOT") from error
    expected_prediction = (result_path.parent / "test.parquet").resolve()
    if prediction_path != expected_prediction:
        raise _test_result_failure(result_path, "PREDICTION_PATH_NOT_EXPECTED")
    if not prediction_path.is_file():
        raise _test_result_failure(result_path, "PREDICTION_FILE_MISSING")
    if sha256_file(prediction_path) != hashes["prediction_file_sha256"]:
        raise _test_result_failure(result_path, "PREDICTION_FILE_SHA_MISMATCH")

    freeze_path = run_root / "freeze" / "GLOBAL_SELECTION_FREEZE.json"
    candidate_manifest_path = run_root / "checkpoint_cache" / "CANDIDATE_CHECKPOINT_MANIFEST.json"
    selected_manifest_path = run_root / "freeze" / "SELECTED_CHECKPOINT_MANIFEST.json"
    freeze = _require_sealed_result_artifact(
        freeze_path, expected_status="GLOBAL_SELECTION_FROZEN", result_path=result_path
    )
    candidate_manifest = _require_sealed_result_artifact(
        candidate_manifest_path,
        expected_status="CANDIDATE_CHECKPOINTS_SEALED",
        result_path=result_path,
    )
    selected_manifest = _require_sealed_result_artifact(
        selected_manifest_path,
        expected_status="SELECTED_CHECKPOINTS_SEALED",
        result_path=result_path,
    )
    if hashes["global_selection_freeze_sha256"] != sha256_file(freeze_path):
        raise _test_result_failure(result_path, "GLOBAL_FREEZE_SHA_MISMATCH")
    if hashes["candidate_manifest_sha256"] != sha256_file(candidate_manifest_path):
        raise _test_result_failure(result_path, "CANDIDATE_MANIFEST_SHA_MISMATCH")
    if hashes["selected_checkpoint_manifest_sha256"] != sha256_file(selected_manifest_path):
        raise _test_result_failure(result_path, "SELECTED_MANIFEST_SHA_MISMATCH")
    if freeze.get("test_accessed") is not False:
        raise _test_result_failure(result_path, "GLOBAL_FREEZE_TEST_ACCESS_INVALID")
    if selected_manifest.get("global_selection_freeze_sha256") != hashes[
        "global_selection_freeze_sha256"
    ]:
        raise _test_result_failure(result_path, "SELECTED_MANIFEST_GLOBAL_FREEZE_BINDING_INVALID")
    if selected_manifest.get("candidate_checkpoint_manifest_sha256") != hashes[
        "candidate_manifest_sha256"
    ]:
        raise _test_result_failure(result_path, "SELECTED_MANIFEST_CANDIDATE_BINDING_INVALID")

    entries = selected_manifest.get("selected_checkpoints")
    if not isinstance(entries, list) or len(entries) != 81:
        raise _test_result_failure(result_path, "SELECTED_MANIFEST_SCOPE_COUNT_INVALID")
    entry_identity = {
        "dataset": profile["dataset"],
        "task_id": profile["task_id"],
        "head_id": profile["head_id"],
        "information_set": profile["information_set"],
        "availability_scenario": profile["availability_scenario"],
        "proxy_policy": profile["proxy_policy"],
        "direction": profile.get("direction"),
        "model": profile["model"],
    }
    matching = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        try:
            entry_horizon = int(entry.get("horizon_steps", -1))
        except (TypeError, ValueError):
            continue
        if all(
            entry.get(field) == expected for field, expected in entry_identity.items()
        ) and entry_horizon == expected_horizon:
            matching.append(entry)
    if len(matching) != 1:
        raise _test_result_failure(result_path, "SELECTED_PROFILE_NOT_UNIQUE")
    selected = matching[0]
    if value.get("selected_candidate_id") != selected.get("candidate_id"):
        raise _test_result_failure(result_path, "SELECTED_CANDIDATE_ID_MISMATCH")
    try:
        history_steps = int(value["history_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise _test_result_failure(result_path, "HISTORY_STEPS_INVALID") from error
    raw_histories = profile.get("history_steps", [])
    if isinstance(raw_histories, (str, bytes)):
        raw_histories = []
    elif isinstance(raw_histories, (int, float)):
        raw_histories = [raw_histories]
    try:
        profile_histories = {int(item) for item in raw_histories}
    except (TypeError, ValueError) as error:
        raise _test_result_failure(result_path, "PROFILE_HISTORY_STEPS_INVALID") from error
    if history_steps <= 0 or history_steps not in profile_histories:
        raise _test_result_failure(result_path, "HISTORY_STEPS_NOT_IN_PROFILE")
    if config is None:
        config, _ = load_scope()
    try:
        expected_seed = int(config["neural_protocol"]["active_checkpoint_seed"])
        observed_seed = int(value["checkpoint_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise _test_result_failure(result_path, "CHECKPOINT_SEED_INVALID") from error
    try:
        selected_seed = int(selected.get("seed", -1))
    except (TypeError, ValueError) as error:
        raise _test_result_failure(result_path, "SELECTED_CHECKPOINT_SEED_INVALID") from error
    if observed_seed != expected_seed or selected_seed != expected_seed:
        raise _test_result_failure(result_path, "CHECKPOINT_SEED_MISMATCH")
    for field in ("candidate_manifest_sha256", "candidate_record_hash", "config_hash"):
        _require_sha256(selected.get(field), field=f"selected_{field}", result_path=result_path)
    checkpoint_dirs = value.get("checkpoint_dirs")
    if not isinstance(checkpoint_dirs, list) or len(checkpoint_dirs) != 1:
        raise _test_result_failure(result_path, "CHECKPOINT_DIRS_INVALID")
    requested_checkpoint = _resolve_run_path(
        run_root,
        checkpoint_dirs[0],
        field="checkpoint_dirs",
        result_path=result_path,
    )
    selected_checkpoint_value = selected.get("candidate_dir")
    selected_checkpoint = _resolve_run_path(
        run_root,
        selected_checkpoint_value,
        field="selected_candidate_dir",
        result_path=result_path,
    )
    cache_root = (run_root / "checkpoint_cache" / "candidates").resolve()
    for candidate_path in (requested_checkpoint, selected_checkpoint):
        try:
            candidate_path.relative_to(cache_root)
        except ValueError as error:
            raise _test_result_failure(
                result_path, "CHECKPOINT_DIRECTORY_OUTSIDE_CACHE"
            ) from error
    if requested_checkpoint != selected_checkpoint or not requested_checkpoint.is_dir():
        raise _test_result_failure(result_path, "CHECKPOINT_DIRECTORY_MISMATCH")
    candidate_record_path = selected_checkpoint / "CANDIDATE_MANIFEST.json"
    if not candidate_record_path.is_file():
        raise _test_result_failure(result_path, "CANDIDATE_RECORD_MISSING")
    if sha256_file(candidate_record_path) != selected["candidate_manifest_sha256"]:
        raise _test_result_failure(result_path, "SELECTED_CANDIDATE_MANIFEST_SHA_MISMATCH")
    try:
        candidate_record = read_json(candidate_record_path)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise _test_result_failure(result_path, "CANDIDATE_RECORD_UNREADABLE") from error
    if not isinstance(candidate_record, Mapping):
        raise _test_result_failure(result_path, "CANDIDATE_RECORD_INVALID")
    if (
        candidate_record.get("candidate_id") != selected.get("logical_candidate_id")
        or candidate_record.get("record_hash") != selected.get("candidate_record_hash")
        or candidate_record.get("hashes", {}).get("config_hash")
        != selected.get("config_hash")
        or candidate_record.get("sealed") is not True
        or candidate_record.get("deletion_forbidden") is not True
    ):
        raise _test_result_failure(result_path, "CANDIDATE_RECORD_BINDING_INVALID")


def run_tests(
    run_root: Path,
    *,
    dataset: str | None,
    model: str | None,
) -> dict[str, Any]:
    config, matrix = load_scope()
    profiles = [
        dict(profile)
        for profile in matrix.profiles
        if (not dataset or profile["scope"] == dataset or profile["dataset"] == dataset)
        and (not model or profile["model"] == model)
    ]
    if any(profile["scope"] == "tep" for profile in profiles):
        materialize_tep_nowcast_test(run_root)
    # Public-only test runs must not open the private CZ workbook or materialize
    # any target-rod values.  Unlock CZ targets only when this invocation
    # actually includes a CZ profile.
    if any(profile["scope"] == "cz" for profile in profiles):
        materialize_cz_targets(run_root)
    completed = 0
    for profile in profiles:
        result_path = test_output(run_root, profile) / "TEST_RESULT.json"
        if result_path.is_file():
            value = read_json(result_path)
            try:
                _validate_test_result(
                    run_root,
                    profile,
                    value,
                    result_path,
                    config=config,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"STOP_EXISTING_TEST_RESULT_INVALID:{result_path}"
                ) from error
            completed += 1
            continue
        environment = dict(os.environ)
        environment[INFERENCE_ENVIRONMENT] = "1"
        returncode = _run_test_worker_subprocess(
            run_root,
            str(profile["profile_id"]),
            environment=environment,
        )
        if returncode != 0:
            raise RuntimeError(f"STOP_TEST_WORKER_FAILED:{profile['profile_id']}:{returncode}")
        produced = test_output(run_root, profile) / "TEST_RESULT.json"
        if not produced.is_file():
            raise RuntimeError(f"STOP_TEST_WORKER_RESULT_MISSING:{profile['profile_id']}")
        produced_value = read_json(produced)
        _validate_test_result(
            run_root,
            profile,
            produced_value,
            produced,
            config=config,
        )
        completed += 1
        update_status(
            run_root,
            "TEST_INFERENCE_RUNNING",
            scoped_test_completed=completed,
            scoped_test_total=len(profiles),
        )
    result = {
        "status": "PASS",
        "stage": "TEST_INFERENCE_ONLY",
        "scoped_profiles": len(profiles),
        "fit_called": False,
        "refit_called": False,
        "select_called": False,
        "test_accessed": True,
    }
    atomic_write_json(run_root / "logs" / "TEST_RUN.json", result)
    update_status(run_root, "TEST_INFERENCE_COMPLETE")
    return result


def build_report(run_root: Path) -> dict[str, Any]:
    result_paths = sorted((run_root / "test_results").rglob("TEST_RESULT.json"))
    results = [read_json(path) for path in result_paths]
    config, matrix = load_scope()
    if len(results) != len(matrix.profiles):
        raise RuntimeError(
            "STOP_REPORT_REQUIRES_COMPLETE_TEST_RESULTS:"
            f"observed={len(results)} expected={len(matrix.profiles)}"
        )

    # A count alone is insufficient: duplicate or malformed result files could
    # otherwise displace an unrun profile and still enter the leaderboard.
    def profile_key(profile: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            profile["dataset"],
            profile["task_id"],
            profile["head_id"],
            profile["information_set"],
            profile["availability_scenario"],
            profile["proxy_policy"],
            profile.get("direction"),
            int(profile["horizon_steps"]),
            profile["model"],
        )

    expected_profiles = {profile_key(profile) for profile in matrix.profiles}
    profiles_by_key = {profile_key(profile): profile for profile in matrix.profiles}
    if len(profiles_by_key) != len(matrix.profiles):
        raise RuntimeError("STOP_REPORT_EXPECTED_PROFILE_KEYS_NOT_UNIQUE")
    observed_profiles: set[tuple[Any, ...]] = set()
    required_metrics = {
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
    }
    root_resolved = run_root.resolve()
    for result_path, result in zip(result_paths, results, strict=True):
        if result.get("status") != "PASS":
            raise RuntimeError(f"STOP_REPORT_RESULT_STATUS_NOT_PASS:{result_path}")
        if result.get("test_accessed") is not True:
            raise RuntimeError(f"STOP_REPORT_RESULT_NOT_TEST_ACCESS:{result_path}")
        if any(result.get(field) is not False for field in ("fit_called", "refit_called", "select_called")):
            raise RuntimeError(f"STOP_REPORT_RESULT_NOT_INFERENCE_ONLY:{result_path}")
        if result.get("checkpoint_contract") != "CACHED_DEVELOPMENT_BEST_WEIGHT_NO_RETRAIN":
            raise RuntimeError(f"STOP_REPORT_RESULT_CHECKPOINT_CONTRACT_INVALID:{result_path}")
        try:
            key = (
                result["dataset"],
                result["task_id"],
                result["target_head"],
                result["information_set"],
                result["availability_scenario"],
                result["proxy_policy"],
                result.get("direction"),
                int(result["horizon_steps"]),
                result["model"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"STOP_REPORT_RESULT_PROFILE_FIELDS_INVALID:{result_path}") from error
        if key not in expected_profiles:
            raise RuntimeError(f"STOP_REPORT_RESULT_PROFILE_UNEXPECTED:{result_path}:{key}")
        if key in observed_profiles:
            raise RuntimeError(f"STOP_REPORT_RESULT_PROFILE_DUPLICATE:{result_path}:{key}")
        observed_profiles.add(key)
        try:
            _validate_test_result(
                run_root,
                profiles_by_key[key],
                result,
                result_path,
                config=config,
            )
        except RuntimeError as error:
            raise RuntimeError(f"STOP_REPORT_RESULT_INVALID:{result_path}") from error

        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"STOP_REPORT_RESULT_METRICS_MISSING:{result_path}")
        for metric_name in required_metrics:
            value = metrics.get(metric_name)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"STOP_REPORT_RESULT_METRIC_INVALID:{result_path}:{metric_name}"
                ) from error
            if not np.isfinite(numeric):
                raise RuntimeError(
                    f"STOP_REPORT_RESULT_METRIC_NONFINITE:{result_path}:{metric_name}"
                )
        try:
            if int(result["evaluation_rows"]) <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"STOP_REPORT_RESULT_EVALUATION_ROWS_INVALID:{result_path}") from error
        if float(metrics["residual_identity_max_abs_error"]) > 1e-10:
            raise RuntimeError(f"STOP_REPORT_RESULT_RESIDUAL_IDENTITY_INVALID:{result_path}")

        prediction_path_value = result.get("prediction_path")
        prediction_sha = result.get("prediction_file_sha256")
        if not isinstance(prediction_path_value, str) or not isinstance(prediction_sha, str):
            raise RuntimeError(f"STOP_REPORT_RESULT_PREDICTION_BINDING_MISSING:{result_path}")
        prediction_path = Path(prediction_path_value).resolve()
        try:
            prediction_path.relative_to(root_resolved)
        except ValueError as error:
            raise RuntimeError(
                f"STOP_REPORT_RESULT_PREDICTION_OUTSIDE_RUN_ROOT:{result_path}"
            ) from error
        if not prediction_path.is_file() or sha256_file(prediction_path) != prediction_sha:
            raise RuntimeError(f"STOP_REPORT_RESULT_PREDICTION_SHA_MISMATCH:{result_path}")

    if observed_profiles != expected_profiles:
        missing = sorted(expected_profiles - observed_profiles, key=str)
        unexpected = sorted(observed_profiles - expected_profiles, key=str)
        raise RuntimeError(
            f"STOP_REPORT_RESULT_PROFILE_SET_MISMATCH:missing={missing}:unexpected={unexpected}"
        )
    grouping = (
        "dataset",
        "task_id",
        "information_set",
        "availability_scenario",
        "proxy_policy",
        "direction",
        "horizon_steps",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        key = tuple(result.get(name) for name in grouping)
        groups.setdefault(key, []).append(result)
    leaderboards = []
    tep_reference_results = config["active_scope"]["tep"]["reference_results"]
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        ranked = sorted(
            values,
            key=lambda item: float(item["metrics"]["r2_level_reconstructed"]),
            reverse=True,
        )
        leaderboards.append(
            {
                **dict(zip(grouping, key, strict=True)),
                "task_type": (
                    "STRICT_PAST_CURRENT_STATE_NOWCAST"
                    if key[0] == "tep"
                    else "FUTURE_FORECAST"
                ),
                "primary_metric": "R2_LEVEL_RECONSTRUCTED",
                "tep_reference_results": (
                    tep_reference_results if key[0] == "tep" else None
                ),
                "ranking": [
                    {
                        "rank": index,
                        "model": item["model"],
                        "r2_level_reconstructed": item["metrics"]["r2_level_reconstructed"],
                        "r2_delta": item["metrics"]["r2_delta"],
                        "r2_level_persistence": item["metrics"]["r2_level_persistence"],
                        "persistence_skill": item["metrics"]["persistence_skill"],
                        "rmse_level": item["metrics"]["rmse"],
                        "mae_level": item["metrics"]["mae"],
                        "rmse_delta": item["metrics"]["rmse_delta"],
                        "mae_delta": item["metrics"]["mae_delta"],
                        "std_level_target": item["metrics"]["std_level_target"],
                        "std_level_prediction": item["metrics"]["std_level_prediction"],
                        "std_delta_target": item["metrics"]["std_delta_target"],
                        "std_delta_prediction": item["metrics"]["std_delta_prediction"],
                        "variance_ratio_level_prediction_to_target": item["metrics"]["variance_ratio_level_prediction_to_target"],
                        "variance_ratio_delta_prediction_to_target": item["metrics"]["variance_ratio_delta_prediction_to_target"],
                        "residual_identity_max_abs_error": item["metrics"]["residual_identity_max_abs_error"],
                    }
                    for index, item in enumerate(ranked, start=1)
                ],
            }
        )
    report = {
        "status": "PASS",
        "stage": "ACTIVE3_NEURAL3_FINAL_REPORT",
        "created_utc": utc(),
        "result_count": len(results),
        "leaderboard_count": len(leaderboards),
        "leaderboards": leaderboards,
        "primary_metric": "R2_LEVEL_RECONSTRUCTED",
        "secondary_metric": "R2_DELTA",
        "ranking_homogeneity": "HETEROGENEOUS_TEP_NOWCAST_AND_FUTURE_FORECASTS",
        "ranking_interpretation": (
            "TEP estimates the current H0 state from strictly past data; all "
            "other active heads remain future forecasts. This is not a "
            "homogeneous seven-forecast-head leaderboard."
        ),
        "tep_reference_results": tep_reference_results,
        "tep_compatibility_appendix": config["active_scope"]["tep"][
            "compatibility_appendix"
        ],
        "out_of_scope": config["out_of_scope"],
        "candidate_count": 456,
        "selected_checkpoint_reference_count": 81,
        "final_fit_status": "NOT_RUN_RETRAIN_FORBIDDEN_BY_CURRENT_USER_RULE",
        "test_inference_only": True,
        "deletion_forbidden": True,
    }
    atomic_write_json(run_root / "final" / "ACTIVE3_NEURAL3_FINAL_REPORT.json", report)
    atomic_write_json(run_root / "final" / "R2_RANKINGS.json", {"status": "PASS", "leaderboards": leaderboards})
    lines = [
        "# Active-3 Neural-3 GPU extension",
        "",
        "Status: PASS",
        "",
        "Primary metric: R2_LEVEL_RECONSTRUCTED",
        "",
        "TEP is a strict-past H0 current-state nowcast; the other active heads "
        "are future forecasts. These results are not a homogeneous "
        "seven-forecast-head leaderboard.",
        "",
        f"Candidates: 456; selected cached checkpoints: 81; test results: {len(results)}.",
        "",
        "No final fit or candidate retraining was performed.",
    ]
    (run_root / "final" / "ACTIVE3_NEURAL3_FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    update_status(run_root, "REPORT_COMPLETE")
    return report


_PRIVACY_DATA_SUFFIXES = {
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
    ".arrow",
    ".rdata",
    ".rds",
}
_PRIVACY_TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".ps1",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_PRIVACY_CREDENTIAL_SUFFIXES = {
    ".jks",
    ".key",
    ".p12",
    ".pfx",
    ".pem",
    ".ppk",
}
_PRIVACY_CREDENTIAL_NAMES = {
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
_PRIVACY_RUN_ARTIFACT_COMPONENTS = {
    "candidate_locks",
    "candidate_quarantine",
    "candidate_staging",
    "checkpoint_cache",
    "test_results",
}
_PRIVACY_SECRET_PATTERNS = (
    ("private_key_block", re.compile(r"-----BEGIN [A-Z0-9 ]+ PRIVATE KEY-----")),
    ("ssh_public_key", re.compile(r"(?m)^\s*ssh-(?:rsa|ed25519|ecdsa)\s+\S+")),
    ("cloud_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    (
        "api_or_secret_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"]?[A-Za-z0-9+/=_-]{12,}"
        ),
    ),
)


def _privacy_git_paths() -> tuple[list[str], list[str]]:
    tracked = [line.strip() for line in git_output("ls-files").splitlines() if line.strip()]
    untracked = [
        line.strip()
        for line in git_output("ls-files", "--others", "--exclude-standard").splitlines()
        if line.strip()
    ]
    return tracked, untracked


def _privacy_tree_files(root: Path) -> tuple[list[Path], list[str]]:
    """Collect regular files and symlink paths without following symlinks."""

    root = Path(root)
    if root.is_symlink():
        return [], [str(root)]
    if not root.exists():
        return [], []
    files: list[Path] = []
    symlinks: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            symlinks.append(str(path))
        elif path.is_file():
            files.append(path)
    return files, symlinks


def _privacy_path_flags(relative_path: str, *, repository: bool) -> tuple[bool, bool, bool]:
    """Return (CZ data, credential, run artifact) flags for one path."""

    path = Path(relative_path)
    lower = relative_path.lower().replace("\\", "/")
    suffix = path.suffix.lower()
    parts = {part.lower() for part in path.parts}
    cz_tokens = ("cz", "czochralski", "raw_cz", "private_cz")
    if repository:
        is_cz_data = suffix in {".xlsx", ".xls"} or (
            suffix in _PRIVACY_DATA_SUFFIXES
            and any(token in lower for token in cz_tokens)
        )
    else:
        # A run may legitimately contain derived CZ prediction parquet files.
        # Only Excel files or explicitly raw/source/private data paths are
        # treated as forbidden source material in the private run namespace.
        path_tokens = set(re.split(r"[^a-z0-9]+", lower))
        is_cz_data = suffix in {".xlsx", ".xls"} or (
            suffix in _PRIVACY_DATA_SUFFIXES
            and bool(path_tokens.intersection({"cz", "czochralski"}))
            and bool(path_tokens.intersection({"raw", "source", "private"}))
        )
    is_credential = (
        suffix in _PRIVACY_CREDENTIAL_SUFFIXES
        or path.name.lower() in _PRIVACY_CREDENTIAL_NAMES
    )
    is_public_protocol_document = suffix == ".md" and "docs" in parts
    is_run_artifact = repository and bool(
        parts.intersection(_PRIVACY_RUN_ARTIFACT_COMPONENTS)
        or path.name.startswith("GPU_RESOURCE_")
        or (
            path.name.startswith("ACTIVE3_NEURAL3_")
            and not is_public_protocol_document
        )
    )
    return is_cz_data, is_credential, is_run_artifact


def _privacy_content_hits(path: Path, label: str) -> list[dict[str, str]]:
    if path.suffix.lower() not in _PRIVACY_TEXT_SUFFIXES:
        return []
    try:
        payload = path.read_bytes()
    except OSError:
        return []
    # Avoid scanning large binary-like files while retaining all source/config
    # files where credentials are most likely to be accidentally committed.
    if len(payload) > 4 * 1024 * 1024:
        return []
    text = payload.decode("utf-8", errors="ignore")
    return [
        {"path": label, "pattern": name}
        for name, pattern in _PRIVACY_SECRET_PATTERNS
        if pattern.search(text)
    ]


def privacy_audit(run_root: Path) -> dict[str, Any]:
    tracked, untracked = _privacy_git_paths()
    repository_paths = list(dict.fromkeys([*tracked, *untracked]))
    forbidden_cz_data: list[str] = []
    credential_paths: list[str] = []
    repository_run_artifacts: list[str] = []
    content_hits: list[dict[str, str]] = []
    for relative in repository_paths:
        is_cz_data, is_credential, is_run_artifact = _privacy_path_flags(
            relative, repository=True
        )
        if is_cz_data:
            forbidden_cz_data.append(relative)
        if is_credential:
            credential_paths.append(relative)
        if is_run_artifact:
            repository_run_artifacts.append(relative)
        path = _REPOSITORY / relative
        if path.is_file() and not path.is_symlink():
            content_hits.extend(_privacy_content_hits(path, f"repository/{relative}"))

    run_files, run_symlinks = _privacy_tree_files(run_root)
    run_cz_data: list[str] = []
    run_credential_paths: list[str] = []
    for path in run_files:
        relative = path.relative_to(run_root).as_posix()
        is_cz_data, is_credential, _ = _privacy_path_flags(relative, repository=False)
        if is_cz_data:
            run_cz_data.append(relative)
        if is_credential:
            run_credential_paths.append(relative)
        content_hits.extend(_privacy_content_hits(path, f"run_root/{relative}"))

    raw_name_hits: list[str] = []
    raw_names = {path.name for path in RAW_CZ_ROOT.glob("*") if path.is_file()}
    for path in (
        run_root / "final" / "ACTIVE3_NEURAL3_FINAL_REPORT.json",
        run_root / "final" / "ACTIVE3_NEURAL3_FINAL_REPORT.md",
    ):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        raw_name_hits.extend(name for name in raw_names if name in text)

    repository_resolved = _REPOSITORY.resolve()
    run_root_resolved = Path(run_root).resolve()
    try:
        run_root_resolved.relative_to(repository_resolved)
    except ValueError:
        run_root_inside_repository = False
    else:
        run_root_inside_repository = True

    forbidden_data_paths = sorted(set([*forbidden_cz_data, *run_cz_data]))
    credential_path_hits = sorted(set([*credential_paths, *run_credential_paths]))
    content_hits = sorted(
        { (item["path"], item["pattern"]): item for item in content_hits }.values(),
        key=lambda item: (item["path"], item["pattern"]),
    )
    status = (
        "PASS"
        if not forbidden_data_paths
        and not credential_path_hits
        and not repository_run_artifacts
        and not content_hits
        and not raw_name_hits
        and not run_root_inside_repository
        and not run_symlinks
        else "FAILED"
    )
    result = {
        "status": status,
        "stage": "REPORT_AND_REPOSITORY_PRIVACY_AUDIT",
        "repository_inventory_scope": "git_tracked_and_untracked_excluding_ignored",
        "repository_tracked_file_count": len(tracked),
        "repository_untracked_file_count": len(untracked),
        "repository_inventory_file_count": len(repository_paths),
        "forbidden_cz_data_files": forbidden_data_paths,
        "run_root_symlink_paths": sorted(run_symlinks),
        "credential_path_hits": credential_path_hits,
        "high_risk_content_hits": content_hits,
        "repository_run_artifact_paths": sorted(set(repository_run_artifacts)),
        "raw_cz_filename_hits_in_public_summary": sorted(set(raw_name_hits)),
        "cz_raw_upload_forbidden": True,
        "run_root_is_private_and_outside_git": not run_root_inside_repository,
        "run_root_scanned_file_count": len(run_files),
        "test_accessed": True,
    }
    atomic_write_json(run_root / "final" / "REPORT_PRIVACY_AUDIT.json", result)
    if status != "PASS":
        raise RuntimeError("STOP_PRIVACY_AUDIT_FAILED")
    return result


def verify_immutability(run_root: Path) -> dict[str, Any]:
    before_path = run_root / "freeze" / "PRE_TEST_IMMUTABILITY_INVENTORY.json"
    before = read_json(before_path)
    before_files = before.get("files")
    if (
        before.get("status") != "PASS"
        or before.get("deletion_forbidden") is not True
        or before.get("test_results_included") is not True
        or before.get("new_test_results_allowed_after_freeze") is not True
        or not isinstance(before_files, list)
        or before.get("file_count") != len(before_files)
        or before.get("inventory_sha256") != stable_hash(before_files)
    ):
        raise RuntimeError("STOP_PRE_TEST_IMMUTABILITY_INVENTORY_INVALID")
    after = immutable_inventory(run_root)
    after_files = after.get("files")
    if (
        after.get("status") != "PASS"
        or after.get("deletion_forbidden") is not True
        or after.get("test_results_included") is not True
        or not isinstance(after_files, list)
        or after.get("file_count") != len(after_files)
        or after.get("inventory_sha256") != stable_hash(after_files)
    ):
        raise RuntimeError("STOP_POST_TEST_IMMUTABILITY_INVENTORY_INVALID")

    def inventory_map(files: list[Any], error_name: str) -> dict[str, Mapping[str, Any]]:
        entries: dict[str, Mapping[str, Any]] = {}
        for item in files:
            if not isinstance(item, Mapping):
                raise RuntimeError(error_name)
            name = item.get("name")
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or "\\" in name
                or name in entries
            ):
                raise RuntimeError(error_name)
            if any(field not in item for field in ("bytes", "sha256", "mtime_ns")):
                raise RuntimeError(error_name)
            entries[name] = item
        return entries

    before_by_name = inventory_map(before_files, "STOP_PRE_TEST_IMMUTABILITY_INVENTORY_INVALID")
    after_by_name = inventory_map(after_files, "STOP_POST_TEST_IMMUTABILITY_INVENTORY_INVALID")
    changes = []
    for name in sorted(before_by_name):
        left = before_by_name[name]
        right = after_by_name.get(name)
        if right is None:
            changes.append({"name": name, "reason": "FILE_SET_CHANGED"})
            continue
        fields = {
            field: {"before": left[field], "after": right[field]}
            for field in ("bytes", "sha256", "mtime_ns")
            if left[field] != right[field]
        }
        if fields:
            changes.append({"name": name, "reason": "ATTRIBUTES_CHANGED", "fields": fields})
    # Test outputs are materialized after the pre-test inventory is sealed and
    # are therefore the only permitted additions.  Any other new file signals
    # an unexpected mutation of the frozen run namespace.
    for name in sorted(set(after_by_name) - set(before_by_name)):
        if not name.startswith("test_results/"):
            changes.append({"name": name, "reason": "UNEXPECTED_FILE_ADDED"})
    new_test_results = sorted(
        name
        for name in set(after_by_name) - set(before_by_name)
        if name.startswith("test_results/")
    )
    result = {
        "status": "PASS" if not changes else "FAILED",
        "stage": "POST_TEST_CANDIDATE_AND_SELECTED_IMMUTABILITY_VERIFICATION",
        "checked_file_count": len(after["files"]),
        "changes": changes,
        "new_test_result_files": new_test_results,
        "new_test_result_count": len(new_test_results),
        "pre_test_inventory_sha256": before["inventory_sha256"],
        "post_test_inventory_sha256": after["inventory_sha256"],
        "sha256_and_mtime_unchanged": not changes,
        "deletion_forbidden": True,
    }
    atomic_write_json(run_root / "final" / "POST_TEST_IMMUTABILITY_VERIFICATION.json", result)
    if changes:
        raise RuntimeError(f"STOP_POST_TEST_IMMUTABILITY_CHANGED:{len(changes)}")
    update_status(run_root, "COMPLETE", final_status="PASS")
    return result


def run_selftest(run_root: Path) -> dict[str, Any]:
    tests = [
        "tests/test_neural_candidate_cache.py",
        "tests/test_neural_resource_guard.py",
        "tests/test_neural3_cached.py",
        "tests/test_level_reconstruction.py",
        "tests/test_cz_l256_nowcast.py",
        "tests/test_tep_nowcast_c1.py",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_PROJECT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        cwd=_PROJECT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    result = {
        "status": "PASS" if completed.returncode == 0 else "FAILED",
        "stage": "CACHE_RELOAD_LOCK_PROTOCOL_SELFTEST",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
        "tests": tests,
        "test_accessed": False,
    }
    atomic_write_json(run_root / "logs" / "SELFTEST.json", result)
    if completed.returncode != 0:
        raise RuntimeError("STOP_SELFTEST_FAILED")
    update_status(run_root, "SELFTEST_PASS")
    return result


def print_status(run_root: Path) -> dict[str, Any]:
    if not run_root.exists():
        value = {"status": "NOT_INITIALIZED", "run_root": str(run_root)}
    elif (run_root / "STATUS.json").is_file():
        value = read_json(run_root / "STATUS.json")
        try:
            snapshot = collect_resource_snapshot(run_root_data_disk(run_root))
            value["resource_snapshot"] = snapshot.to_json()
        except Exception as error:
            value["resource_snapshot_error"] = f"{type(error).__name__}: {error}"
    else:
        value = {"status": "INITIALIZING", "run_root": str(run_root)}
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        nargs="?",
        choices=(
            "init",
            "selftest",
            "build-cz",
            "pilot",
            "development",
            "freeze",
            "final-fit",
            "seal",
            "test",
            "report",
            "privacy-audit",
            "verify",
            "status",
            "all",
            "worker",
            "test-worker",
        ),
        default="status",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", choices=("tep", "sru", "cz"))
    parser.add_argument("--model", choices=("LSTM", "iTransformer", "TimeMixer"))
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--status", action="store_true", dest="status_flag")
    parser.add_argument("--candidate-id")
    parser.add_argument("--profile-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    if args.status_flag or args.stage == "status":
        print_status(run_root)
        return
    if args.stage == "worker":
        if not args.candidate_id:
            raise ValueError("worker requires --candidate-id")
        print(json.dumps(candidate_worker(run_root, args.candidate_id), ensure_ascii=False))
        return
    if args.stage == "test-worker":
        if not args.profile_id:
            raise ValueError("test-worker requires --profile-id")
        print(json.dumps(test_worker(run_root, args.profile_id), ensure_ascii=False))
        return
    if args.stage == "init":
        print(json.dumps(initialize_run(run_root, resume=args.resume), ensure_ascii=False))
    elif args.stage == "selftest":
        print(json.dumps(run_selftest(run_root), ensure_ascii=False))
    elif args.stage == "build-cz":
        print(json.dumps(build_cz_development(run_root), ensure_ascii=False))
    elif args.stage == "pilot":
        print(json.dumps(run_pilots(run_root), ensure_ascii=False))
    elif args.stage == "development":
        print(json.dumps(run_development(
            run_root,
            dataset=args.dataset,
            model=args.model,
            max_concurrency=args.max_concurrency,
            resume=args.resume,
        ), ensure_ascii=False))
    elif args.stage == "freeze":
        print(json.dumps(freeze_selection(run_root), ensure_ascii=False))
    elif args.stage == "final-fit":
        raise RuntimeError("STOP_FINAL_FIT_RETRAIN_FORBIDDEN_BY_CURRENT_USER_RULE")
    elif args.stage == "seal":
        print(json.dumps(seal_checkpoints(run_root), ensure_ascii=False))
    elif args.stage == "test":
        print(json.dumps(run_tests(run_root, dataset=args.dataset, model=args.model), ensure_ascii=False))
    elif args.stage == "report":
        print(json.dumps(build_report(run_root), ensure_ascii=False))
    elif args.stage == "privacy-audit":
        print(json.dumps(privacy_audit(run_root), ensure_ascii=False))
    elif args.stage == "verify":
        print(json.dumps(verify_immutability(run_root), ensure_ascii=False))
    elif args.stage == "all":
        initialize_run(run_root, resume=args.resume)
        run_selftest(run_root)
        build_cz_development(run_root)
        run_pilots(run_root)
        development = run_development(
            run_root,
            dataset=args.dataset,
            model=args.model,
            max_concurrency=args.max_concurrency,
            resume=args.resume,
        )
        if development["status"] != "PASS":
            print(json.dumps(development, ensure_ascii=False))
            return
        if args.dataset or args.model:
            raise RuntimeError("STOP_SCOPED_DEVELOPMENT_CANNOT_ADVANCE_TO_GLOBAL_FREEZE")
        freeze_selection(run_root)
        seal_checkpoints(run_root)
        run_tests(run_root, dataset=None, model=None)
        build_report(run_root)
        privacy_audit(run_root)
        print(json.dumps(verify_immutability(run_root), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        event(
            "FATAL",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
