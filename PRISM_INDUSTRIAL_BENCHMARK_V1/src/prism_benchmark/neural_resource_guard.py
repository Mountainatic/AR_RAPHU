from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


# User-facing resource limits are stated in decimal GB.  Keep the historical
# exported name for tests/callers, but use the same decimal unit everywhere so
# the 85 GB stopline remains below the observed 90 GiB cgroup hard limit.
GB = 10**9
GIB = GB
MIB = 1024**2


@dataclass(frozen=True)
class ResourceThresholds:
    """Hard dispatch and runtime stoplines for the Neural-3 scheduler."""

    ram_dispatch_budget_bytes: int = 75 * GIB
    ram_pause_dispatch_bytes: int = 82 * GIB
    ram_safe_stop_bytes: int = 85 * GIB
    disk_stopline_bytes: int = 5 * GIB
    vram_dispatch_budget_bytes: int = 27 * GIB

    def __post_init__(self) -> None:
        values = (
            self.ram_dispatch_budget_bytes,
            self.ram_pause_dispatch_bytes,
            self.ram_safe_stop_bytes,
            self.disk_stopline_bytes,
            self.vram_dispatch_budget_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise ValueError("resource thresholds must be finite integer byte counts")
        if any(value <= 0 for value in values):
            raise ValueError("resource thresholds must be positive")
        if not (
            self.ram_dispatch_budget_bytes
            < self.ram_pause_dispatch_bytes
            < self.ram_safe_stop_bytes
        ):
            raise ValueError("RAM thresholds must satisfy dispatch < pause < safe-stop")

    @classmethod
    def from_gib(cls, values: Mapping[str, Any]) -> "ResourceThresholds":
        return cls(
            ram_dispatch_budget_bytes=_gib_bytes(values["ram_dispatch_budget_gib"]),
            ram_pause_dispatch_bytes=_gib_bytes(values["ram_pause_dispatch_gib"]),
            ram_safe_stop_bytes=_gib_bytes(values["ram_safe_stop_gib"]),
            disk_stopline_bytes=_gib_bytes(values["data_disk_stopline_gib"]),
            vram_dispatch_budget_bytes=_gib_bytes(values["vram_dispatch_budget_gib"]),
        )


@dataclass(frozen=True)
class ResourceSnapshot:
    observed_at_utc: str
    memory_current_bytes: int | None
    memory_limit_bytes: int | None
    memory_source: str
    gpu_index: int
    gpu_used_bytes: int | None
    gpu_total_bytes: int | None
    data_disk_path: str
    data_disk_free_bytes: int | None
    data_disk_total_bytes: int | None
    telemetry_errors: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceDecision:
    action: str
    dispatch_allowed: bool
    request_checkpoint: bool
    safe_stop: bool
    reasons: tuple[str, ...]
    projected_ram_bytes: int | None
    projected_vram_bytes: int | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadMatrix:
    profiles: tuple[dict[str, Any], ...]
    selection_candidates: tuple[dict[str, Any], ...]
    selected_checkpoint_references: tuple[dict[str, Any], ...]

    @property
    def total_training_units(self) -> int:
        return len(self.selection_candidates)

    def counts(self) -> dict[str, int]:
        return {
            "profiles": len(self.profiles),
            "selection_candidates": len(self.selection_candidates),
            "selected_checkpoint_references": len(self.selected_checkpoint_references),
            "total_training_units": self.total_training_units,
        }


def _gib_bytes(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("GiB values must be finite positive numbers")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("GiB values must be finite positive numbers") from error
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError("GiB values must be finite positive numbers")
    byte_count = numeric * GIB
    if not math.isfinite(byte_count) or byte_count <= 0:
        raise ValueError("GiB values must fit in a finite positive byte count")
    return int(byte_count)


def _normalize_snapshot_bytes(
    value: Any,
    *,
    invalid_reason: str,
    field_name: str,
) -> tuple[int | None, tuple[str, ...]]:
    """Accept only finite, non-negative integer byte counts from telemetry.

    Dataclass annotations do not enforce runtime types, and test fixtures or
    serialized inputs can therefore contain floats, NaN, or Inf. Invalid
    values are treated as unavailable so malformed snapshots cannot dispatch.
    """

    if value is None:
        return None, ()
    if isinstance(value, bool) or not isinstance(value, Integral):
        if isinstance(value, Real):
            try:
                detail = (
                    f"{field_name}_NON_FINITE"
                    if not math.isfinite(float(value))
                    else f"{field_name}_NON_INTEGER"
                )
            except (TypeError, ValueError, OverflowError):
                detail = f"{field_name}_NON_INTEGER"
        else:
            detail = f"{field_name}_NON_INTEGER"
        return None, (invalid_reason, detail)
    normalized = int(value)
    if normalized < 0:
        return None, (invalid_reason, f"{field_name}_NEGATIVE")
    return normalized, ()


def _validate_expected_increment(value: Any, *, field_name: str) -> int:
    """Validate a predicted per-job byte increment before arithmetic."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be a finite non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be a finite non-negative integer")
    return normalized


def _read_cgroup_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _read_proc_memory(meminfo_path: Path) -> tuple[int | None, int | None]:
    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None
    values: dict[str, int] = {}
    for line in lines:
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.strip().split()
        if not fields:
            continue
        try:
            values[key] = int(fields[0]) * 1024
        except ValueError:
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None, total
    return max(0, total - available), total


def read_effective_memory(
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    meminfo_path: Path = Path("/proc/meminfo"),
    require_cgroup: bool = False,
) -> tuple[int | None, int | None, str]:
    """Read cgroup-v2 usage/limit, optionally refusing host fallback.

    The AutoDL runner treats the cgroup limit as authoritative.  Keeping the
    host ``/proc/meminfo`` fallback available for diagnostics and local unit
    tests is useful, but callers that make dispatch decisions must opt into
    ``require_cgroup`` so a missing cgroup cannot become a fail-open path.
    """

    current_path = cgroup_root / "memory.current"
    limit_path = cgroup_root / "memory.max"
    current = _read_cgroup_integer(current_path)
    if current is not None and limit_path.is_file():
        limit = _read_cgroup_integer(limit_path)
        # A dispatch decision must have a finite, authoritative cgroup limit.
        # ``max`` is valid cgroup syntax, but it does not prove that the
        # configured hard limit is enforced by this container.
        if require_cgroup and limit is None:
            return None, None, "cgroup_unavailable"
        return current, limit, "cgroup_v2"
    if require_cgroup:
        return None, None, "cgroup_unavailable"
    current, limit = _read_proc_memory(meminfo_path)
    return current, limit, "proc_meminfo" if current is not None else "unavailable"


def _run_nvidia_smi(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout


def query_gpu_memory(
    *,
    gpu_index: int = 0,
    command_runner: Callable[[Sequence[str]], str] = _run_nvidia_smi,
) -> tuple[int | None, int | None, str | None]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )
    try:
        output = command_runner(command)
        for raw_line in output.splitlines():
            fields = [field.strip() for field in raw_line.split(",")]
            if len(fields) != 3 or int(fields[0]) != gpu_index:
                continue
            used_mib = float(fields[1])
            total_mib = float(fields[2])
            if (
                not math.isfinite(used_mib)
                or not math.isfinite(total_mib)
                or not used_mib.is_integer()
                or not total_mib.is_integer()
                or used_mib < 0
                or total_mib <= 0
                or used_mib > total_mib
            ):
                return None, None, f"GPU_MEMORY_VALUES_INVALID:{gpu_index}"
            return int(used_mib * MIB), int(total_mib * MIB), None
        return None, None, f"GPU_INDEX_NOT_REPORTED:{gpu_index}"
    except Exception as error:
        return None, None, f"NVIDIA_SMI_FAILED:{type(error).__name__}:{error}"


def collect_resource_snapshot(
    data_disk: Path,
    *,
    gpu_index: int = 0,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    meminfo_path: Path = Path("/proc/meminfo"),
    command_runner: Callable[[Sequence[str]], str] = _run_nvidia_smi,
    require_cgroup: bool = False,
) -> ResourceSnapshot:
    try:
        memory_current, memory_limit, memory_source = read_effective_memory(
            cgroup_root=cgroup_root,
            meminfo_path=meminfo_path,
            require_cgroup=require_cgroup,
        )
    except Exception as error:
        memory_current, memory_limit, memory_source = None, None, "unavailable"
        memory_error = f"MEMORY_TELEMETRY_FAILED:{type(error).__name__}:{error}"
    else:
        memory_error = None
    gpu_used, gpu_total, gpu_error = query_gpu_memory(
        gpu_index=gpu_index,
        command_runner=command_runner,
    )
    errors: list[str] = []
    if memory_current is None:
        errors.append("MEMORY_TELEMETRY_UNAVAILABLE")
    if memory_error is not None:
        errors.append(memory_error)
    if gpu_error is not None:
        errors.append(gpu_error)
    try:
        disk = shutil.disk_usage(data_disk)
        disk_free = int(disk.free)
        disk_total = int(disk.total)
        if disk_free < 0 or disk_total <= 0 or disk_free > disk_total:
            raise ValueError("disk usage values are outside valid bounds")
    except Exception as error:
        disk_free = None
        disk_total = None
        errors.append(f"DISK_TELEMETRY_UNAVAILABLE:{type(error).__name__}")
    return ResourceSnapshot(
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        memory_current_bytes=memory_current,
        memory_limit_bytes=memory_limit,
        memory_source=memory_source,
        gpu_index=gpu_index,
        gpu_used_bytes=gpu_used,
        gpu_total_bytes=gpu_total,
        data_disk_path=str(data_disk.resolve()),
        data_disk_free_bytes=disk_free,
        data_disk_total_bytes=disk_total,
        telemetry_errors=tuple(errors),
    )


def decide_resource_action(
    snapshot: ResourceSnapshot,
    *,
    expected_job_rss_bytes: int = 0,
    expected_job_vram_bytes: int = 0,
    thresholds: ResourceThresholds = ResourceThresholds(),
) -> ResourceDecision:
    """Fail closed when deciding whether another candidate may be dispatched."""

    expected_job_rss_bytes = _validate_expected_increment(
        expected_job_rss_bytes, field_name="expected_job_rss_bytes"
    )
    expected_job_vram_bytes = _validate_expected_increment(
        expected_job_vram_bytes, field_name="expected_job_vram_bytes"
    )
    reasons = list(snapshot.telemetry_errors)
    memory_current, memory_current_issues = _normalize_snapshot_bytes(
        snapshot.memory_current_bytes,
        invalid_reason="MEMORY_TELEMETRY_INVALID",
        field_name="MEMORY_CURRENT",
    )
    memory_limit, memory_limit_issues = _normalize_snapshot_bytes(
        snapshot.memory_limit_bytes,
        invalid_reason="MEMORY_LIMIT_TELEMETRY_INVALID",
        field_name="MEMORY_LIMIT",
    )
    gpu_used, gpu_used_issues = _normalize_snapshot_bytes(
        snapshot.gpu_used_bytes,
        invalid_reason="GPU_TELEMETRY_INVALID",
        field_name="GPU_USED",
    )
    gpu_total, gpu_total_issues = _normalize_snapshot_bytes(
        snapshot.gpu_total_bytes,
        invalid_reason="GPU_TELEMETRY_INVALID",
        field_name="GPU_TOTAL",
    )
    disk_free, disk_free_issues = _normalize_snapshot_bytes(
        snapshot.data_disk_free_bytes,
        invalid_reason="DISK_TELEMETRY_INVALID",
        field_name="DISK_FREE",
    )
    disk_total, disk_total_issues = _normalize_snapshot_bytes(
        snapshot.data_disk_total_bytes,
        invalid_reason="DISK_TELEMETRY_INVALID",
        field_name="DISK_TOTAL",
    )
    for issues in (
        memory_current_issues,
        memory_limit_issues,
        gpu_used_issues,
        gpu_total_issues,
        disk_free_issues,
        disk_total_issues,
    ):
        reasons.extend(issues)
    if snapshot.memory_source != "cgroup_v2":
        reasons.append("MEMORY_CGROUP_NOT_AUTHORITATIVE")
    if memory_current is not None and memory_current < 0:
        reasons.append("MEMORY_TELEMETRY_INVALID")
    if memory_limit is None:
        reasons.append("MEMORY_LIMIT_TELEMETRY_UNAVAILABLE")
    elif memory_limit <= 0:
        reasons.append("MEMORY_LIMIT_TELEMETRY_INVALID")
    if (
        gpu_used is not None
        and gpu_total is not None
        and (
            gpu_used < 0
            or gpu_total <= 0
            or gpu_used > gpu_total
        )
    ):
        reasons.append("GPU_TELEMETRY_INVALID")
    if (
        disk_free is not None
        and disk_total is not None
        and (
            disk_free < 0
            or disk_total <= 0
            or disk_free > disk_total
        )
    ):
        reasons.append("DISK_TELEMETRY_INVALID")
    projected_ram = (
        None
        if memory_current is None
        else memory_current + expected_job_rss_bytes
    )
    projected_vram = (
        None
        if gpu_used is None
        else gpu_used + expected_job_vram_bytes
    )

    safe_stop = bool(
        memory_current is not None
        and memory_current >= thresholds.ram_safe_stop_bytes
    )
    if safe_stop:
        reasons.append("RAM_SAFE_STOP_REACHED")
    elif (
        memory_current is not None
        and memory_current >= thresholds.ram_pause_dispatch_bytes
    ):
        reasons.append("RAM_PAUSE_DISPATCH_REACHED")

    if projected_ram is None:
        if "MEMORY_TELEMETRY_UNAVAILABLE" not in reasons:
            reasons.append("MEMORY_TELEMETRY_UNAVAILABLE")
    elif projected_ram > thresholds.ram_dispatch_budget_bytes:
        reasons.append("PROJECTED_RAM_EXCEEDS_DISPATCH_BUDGET")
    if (
        projected_ram is not None
        and memory_limit is not None
        and projected_ram >= memory_limit
    ):
        reasons.append("PROJECTED_RAM_REACHES_CGROUP_LIMIT")

    if gpu_used is None or gpu_total is None:
        if not any(reason.startswith("NVIDIA_SMI_FAILED") for reason in reasons):
            reasons.append("GPU_TELEMETRY_UNAVAILABLE")
    elif projected_vram > thresholds.vram_dispatch_budget_bytes:
        reasons.append("PROJECTED_VRAM_EXCEEDS_DISPATCH_BUDGET")
    if (
        projected_vram is not None
        and gpu_total is not None
        and projected_vram >= gpu_total
    ):
        reasons.append("PROJECTED_VRAM_REACHES_DEVICE_TOTAL")

    if (
        disk_free is None
        or disk_total is None
    ):
        if not any(reason.startswith("DISK_TELEMETRY_UNAVAILABLE") for reason in reasons):
            reasons.append("DISK_TELEMETRY_UNAVAILABLE")
    elif disk_free < thresholds.disk_stopline_bytes:
        reasons.append("DATA_DISK_BELOW_STOPLINE")

    reasons = list(dict.fromkeys(reasons))
    if safe_stop:
        action = "CHECKPOINT_AND_SAFE_STOP"
    elif reasons:
        action = "STOP_DISPATCH"
    else:
        action = "ALLOW_DISPATCH"
    return ResourceDecision(
        action=action,
        dispatch_allowed=not reasons,
        request_checkpoint=safe_stop,
        safe_stop=safe_stop,
        reasons=tuple(reasons),
        projected_ram_bytes=projected_ram,
        projected_vram_bytes=projected_vram,
    )


def controlled_thread_environment(blas_threads: int = 1) -> dict[str, str]:
    if blas_threads < 1:
        raise ValueError("blas_threads must be positive")
    value = str(int(blas_threads))
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
        "BLIS_NUM_THREADS": value,
    }


def apply_thread_limits(blas_threads: int = 1) -> dict[str, str | None]:
    """Set BLAS/OpenMP limits before workers import numerical libraries."""

    values = controlled_thread_environment(blas_threads)
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    return previous


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _public_views(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dataset_key in ("tep", "sru"):
        scope = config["active_scope"][dataset_key]
        for task in scope["tasks"]:
            for variant in task["view_variants"]:
                result.append(
                    {
                        "scope": dataset_key,
                        "dataset": task["dataset"],
                        "task_id": task["task_id"],
                        "head_id": task["head_id"],
                        "information_set": variant["information_set"],
                        "availability_scenario": variant["availability_scenario"],
                        "proxy_policy": variant["proxy_policy"],
                        "direction": None,
                        "horizon_steps": task["h_steps"],
                        "history_steps": list(scope["history_steps"]),
                        "history_labels": dict(scope["history_labels"]),
                    }
                )
    return result


def _cz_views(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    scope = config["active_scope"]["cz"]
    result: list[dict[str, Any]] = []
    for direction in scope["directions"]:
        for horizon_steps in scope["horizons_steps"]:
            for information_set in scope["information_sets"]:
                task_id = str(scope["task_id_template"]).format(h=horizon_steps)
                head_id = str(scope["head_id_template"]).format(h=horizon_steps)
                result.append(
                    {
                        "scope": "cz",
                        "dataset": scope["dataset"],
                        "task_id": task_id,
                        "head_id": head_id,
                        "information_set": information_set,
                        "availability_scenario": "record_time",
                        "proxy_policy": "primary",
                        "direction": direction,
                        "horizon_steps": horizon_steps,
                        "history_steps": [scope["history_steps"]],
                        "history_labels": dict(scope["history_labels"]),
                    }
                )
    return result


def _view_id(view: Mapping[str, Any]) -> str:
    components = [
        str(view["head_id"]),
        str(view["information_set"]),
        str(view["availability_scenario"]),
        str(view["proxy_policy"]),
    ]
    if view["direction"] is not None:
        components.append(str(view["direction"]))
    return "__".join(components)


def build_workload_matrix(config: Mapping[str, Any]) -> WorkloadMatrix:
    models = list(config["neural_protocol"]["models"])
    capacities = list(config["neural_protocol"]["capacities"])
    learning_rates = list(config["neural_protocol"]["learning_rates"])
    screening_seed = int(config["neural_protocol"]["screening_seed"])
    checkpoint_seed = int(config["neural_protocol"]["active_checkpoint_seed"])
    views = [*_public_views(config), *_cz_views(config)]

    profiles: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    selected_references: list[dict[str, Any]] = []
    for view in views:
        view_id = _view_id(view)
        for model in models:
            profile_id = f"{view_id}__{model}"
            profile = {
                **view,
                "view_id": view_id,
                "profile_id": profile_id,
                "model": model,
            }
            profiles.append(profile)
            for history_steps in view["history_steps"]:
                for capacity in capacities:
                    for learning_rate in learning_rates:
                        history_label = str(
                            view["history_labels"][str(history_steps)]
                        )
                        history_label = (
                            history_label
                            if history_label.startswith("L")
                            else f"L{history_label}"
                        )
                        candidate_label = (
                            f"{model}__{history_label}__{capacity}__lr{learning_rate:g}"
                        )
                        candidates.append(
                            {
                                **profile,
                                "phase": "DEVELOPMENT_SELECTION",
                                "candidate_id": f"{profile_id}__{candidate_label}",
                                "candidate_label": candidate_label,
                                "history_steps": int(history_steps),
                                "capacity": capacity,
                                "learning_rate": float(learning_rate),
                                "seed": screening_seed,
                                "fit_split": "train",
                                "selection_split": "validation",
                                "requires_global_selection_freeze": False,
                            }
                        )
            selected_references.append(
                {
                    **profile,
                    "phase": "SELECTED_CACHED_CHECKPOINT_REFERENCE",
                    "checkpoint_reference_id": f"{profile_id}__SELECTED_CACHED_BEST",
                    "selected_candidate_source": "GLOBAL_SELECTION_FREEZE.json",
                    "seed": checkpoint_seed,
                    "fit_called": False,
                    "refit_called": False,
                    "requires_global_selection_freeze": True,
                }
            )

    matrix = WorkloadMatrix(
        profiles=tuple(profiles),
        selection_candidates=tuple(candidates),
        selected_checkpoint_references=tuple(selected_references),
    )
    expected = {
        key: int(value) for key, value in config["declared_workload_counts"].items()
    }
    if matrix.counts() != expected:
        raise ValueError(
            f"workload count mismatch: observed={matrix.counts()} expected={expected}"
        )
    for label, records, identity in (
        ("profile", matrix.profiles, "profile_id"),
        ("candidate", matrix.selection_candidates, "candidate_id"),
        (
            "selected checkpoint reference",
            matrix.selected_checkpoint_references,
            "checkpoint_reference_id",
        ),
    ):
        identifiers = [str(record[identity]) for record in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"duplicate {label} identifiers")
    return matrix


def load_workload_config(path: Path) -> tuple[dict[str, Any], WorkloadMatrix]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported Active-3 Neural-3 config schema")
    return config, build_workload_matrix(config)
