"""TEP-only CPU/PRISM development for the frozen L256 history extension.

This launcher deliberately has no CZ, SRU, Neural3, Stage-2, test, or OOD
stage.  It constructs a run-local train/validation-only L256 support view and
passes the frozen history override explicitly to every history-selecting job.
The formal freeze/checkpoint/test process is a separate follow-up launcher.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

from prism_benchmark.cpu_data import ViewSpec, input_columns, sha256_file
from prism_benchmark.stage0 import write_json
from prism_benchmark.v211_a import run_a_view
from prism_benchmark.v211_c import run_c_view
from prism_benchmark.v211_config import TEP_CPU_HISTORY_EXTENSION_PROTOCOL
from prism_benchmark.v211_joint_stability import run_joint_stability_view
from prism_benchmark.v211_k import run_k_channel
from prism_benchmark.v211_public_all_baselines import (
    COMPLETED_STATUSES,
    DYNAMIC_MODELS,
    SIMPLE_MODELS,
    STATIC_INPUT_MODELS,
    run_ar_job,
    run_arx_job,
    run_dpls_job,
    run_hammerstein_job,
    run_narx_job,
    run_simple_job,
    run_static_input_job,
)
from prism_benchmark.v211_support import SUPPORT_CONTRACT, support_id_hash
from prism_benchmark.v211_w import run_w_view
from prism_benchmark.v2_views import (
    development_dynamic_views,
    development_input_views,
)


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "configs" / "tep_cpu_history_extension_20260826.json"
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/PRISM_V211_TEP_CPU_HISTORY_L256_20260826_R1"
)
EXPECTED_PROTOCOL = "TEP_CPU_HISTORY_EXTENSION_L256_V1"
EXPECTED_TASK = "TEP_G_REP_H1"
EXPECTED_HEAD = "TEP_G_REP_H1__H1__W2"
EXPECTED_HISTORIES = (2, 4, 8, 128, 256)
COMMON_HISTORY = 256
EXPECTED_K_CHANNEL_JOBS = 47
SUCCESS_STATUSES = {
    "PASS",
    "SOLVER_FAILED_RETAINED",
    "NOT_RUN_PROTOCOL_INCOMPATIBLE",
    "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
    "JOINT_STABILITY_STABILITY_IMPROVED_BUT_NOT_SUPPORTED",
}


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def load_config() -> dict[str, Any]:
    value = _read_json(CONFIG_PATH)
    required = {
        "protocol_id": EXPECTED_PROTOCOL,
        "status": "FROZEN_BY_USER_BEFORE_FORMAL_DEVELOPMENT",
        "active_datasets": ["tep"],
        "active_tasks": [EXPECTED_TASK],
        "active_heads": [EXPECTED_HEAD],
        "information_sets": ["input_only", "dynamic"],
        "history_aware_steps": list(EXPECTED_HISTORIES),
        "common_support_history_steps": COMMON_HISTORY,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeError(f"STOP_TEP_EXTENSION_CONFIG_MISMATCH:{key}")
    resources = value.get("resources", {})
    if resources.get("workers") != 1 or resources.get("blas_threads") != 1:
        raise RuntimeError("STOP_TEP_EXTENSION_SINGLE_WORKER_CONTRACT_CHANGED")
    if resources.get("recommended_memory_limit_gib") != 75:
        raise RuntimeError("STOP_TEP_EXTENSION_SOFT_MEMORY_LIMIT_CHANGED")
    if resources.get("hard_memory_limit_gib") != 90:
        raise RuntimeError("STOP_TEP_EXTENSION_HARD_MEMORY_LIMIT_CHANGED")
    if resources.get("minimum_runtime_free_gib") != 5:
        raise RuntimeError("STOP_TEP_EXTENSION_STORAGE_STOPLINE_CHANGED")
    if set(value.get("registered_protocol_incompatible", ())) != {
        "SEASONAL_PERSISTENCE",
        "N4SID",
    }:
        raise RuntimeError("STOP_TEP_EXTENSION_INCOMPATIBLE_REGISTRY_CHANGED")
    disabled = value.get("disabled", {})
    for name in ("sru", "cz_czochralski", "neural3", "stage2"):
        if disabled.get(name) != "NOT_RUN_BY_USER_SCOPE":
            raise RuntimeError(f"STOP_TEP_EXTENSION_DISABLED_SCOPE_CHANGED:{name}")
    value["config_sha256"] = sha256_file(CONFIG_PATH)
    return value


def _require_uv_runtime() -> None:
    if os.environ.get("AR_RAPHU_RUNTIME_MANAGER") != "uv":
        raise RuntimeError("AR_RAPHU_RUNTIME_MANAGER must be exactly 'uv'")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Python 3.10 required, got {sys.version.split()[0]}")
    virtual_environment = os.environ.get("VIRTUAL_ENV")
    if not virtual_environment or Path(sys.prefix).resolve() != Path(
        virtual_environment
    ).resolve():
        raise RuntimeError("interpreter is not the active uv-managed VIRTUAL_ENV")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT.parent), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_baseline_ancestor(config: dict[str, Any]) -> None:
    for key in ("baseline_commit", "implementation_start_commit"):
        result = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT.parent),
                "merge-base",
                "--is-ancestor",
                str(config[key]),
                "HEAD",
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"STOP_{key.upper()}_NOT_ANCESTOR")


def _free_gib(path: Path) -> float:
    return float(shutil.disk_usage(path).free / (1024**3))


def _storage_guard(run_root: Path, stage: str, config: dict[str, Any]) -> None:
    available = _free_gib(run_root.parent)
    stopline = float(config["resources"]["minimum_runtime_free_gib"])
    audit = {
        "status": "PASS" if available >= stopline else "STOP_LOW_STORAGE",
        "stage": stage,
        "available_gib": available,
        "stopline_gib": stopline,
        "created_utc": _utc(),
    }
    write_json(run_root / "logs" / f"STORAGE_{stage.upper()}.json", audit)
    if available < stopline:
        raise RuntimeError(f"STOP_LOW_STORAGE:{available:.3f}<{stopline:.3f}")


def _tep_views(shared: Path) -> tuple[list[ViewSpec], list[ViewSpec]]:
    inputs = [
        view
        for view in development_input_views(shared)
        if view.head.dataset == "tep" and view.head.task_id == EXPECTED_TASK
    ]
    dynamics = [
        view
        for view in development_dynamic_views(shared)
        if view.head.dataset == "tep" and view.head.task_id == EXPECTED_TASK
    ]
    if len(inputs) != 1 or len(dynamics) != 2:
        raise RuntimeError(
            f"STOP_TEP_VIEW_CARDINALITY:input={len(inputs)}:dynamic={len(dynamics)}"
        )
    for view in [*inputs, *dynamics]:
        if view.head.head_id != EXPECTED_HEAD:
            raise RuntimeError(f"STOP_UNEXPECTED_TEP_HEAD:{view.head.head_id}")
    return inputs, dynamics


def _l256_mask(frame: pd.DataFrame, information_set: str) -> np.ndarray:
    required = {
        "origin",
        "causal_history_floor",
        "sample_support_contract",
        "base_origin_id",
        "view_sample_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"STOP_L256_SUPPORT_COLUMNS_MISSING:{sorted(missing)}")
    if set(frame["sample_support_contract"].astype(str).unique()) != {
        SUPPORT_CONTRACT
    }:
        raise RuntimeError("STOP_L256_NATIVE_SUPPORT_CONTRACT_MISMATCH")
    floor = frame["causal_history_floor"].to_numpy(dtype=np.int64)
    origin = frame["origin"].to_numpy(dtype=np.int64)
    mask = origin - COMMON_HISTORY >= floor
    if information_set == "dynamic":
        if "latest_available_target_index" not in frame.columns:
            raise RuntimeError("STOP_L256_DYNAMIC_TARGET_AVAILABILITY_MISSING")
        latest = frame["latest_available_target_index"].to_numpy(dtype=np.int64)
        mask &= latest - (COMMON_HISTORY - 1) >= floor
    elif information_set != "input_only":
        raise RuntimeError(f"STOP_INFORMATION_SET_OUT_OF_SCOPE:{information_set}")
    return mask


def build_l256_development_shared(
    source_shared: Path, run_root: Path
) -> dict[str, Any]:
    """Create a run-local C1 view with only train/validation L256 rows.

    No test/ood sample parquet is opened or copied.  Non-sample C1 data are
    referenced by read-only symlinks; the authoritative C1 remains untouched.
    """

    destination = run_root / "shared_l256_development"
    sample_root = destination / "sample_ids"
    if destination.exists():
        manifest = run_root / "logs" / "L256_DEVELOPMENT_SUPPORT.json"
        if not manifest.is_file():
            raise RuntimeError("STOP_PARTIAL_L256_SHARED_EXISTS")
        return _read_json(manifest)
    destination.mkdir(parents=True)
    for child in sorted(source_shared.iterdir()):
        if child.name in {"sample_ids", "base_data"}:
            continue
        target = destination / child.name
        os.symlink(
            str(child.resolve()),
            str(target),
            target_is_directory=child.is_dir(),
        )
    base_root = destination / "base_data" / "tep"
    base_root.mkdir(parents=True)
    for split in ("train", "validation"):
        source_path = source_shared / "base_data" / "tep" / f"{split}.parquet"
        os.symlink(str(source_path.resolve()), str(base_root / source_path.name))
    sample_root.mkdir()
    source_inputs, source_dynamics = _tep_views(source_shared)
    records: list[dict[str, Any]] = []
    for view in [*source_inputs, *source_dynamics]:
        view_destination = sample_root / view.relative_root
        view_destination.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation"):
            source_path = source_shared / "sample_ids" / view.relative_root / f"{split}.parquet"
            frame = pd.read_parquet(source_path)
            common = frame.loc[_l256_mask(frame, view.information_set)].copy()
            if common.empty:
                raise RuntimeError(
                    f"STOP_EMPTY_L256_SUPPORT:{view.information_set}:{split}"
                )
            output_path = view_destination / f"{split}.parquet"
            common.to_parquet(output_path, index=False, compression="zstd")
            records.append(
                {
                    "target_head": view.head.head_id,
                    "information_set": view.information_set,
                    "split": split,
                    "source_rows": int(len(frame)),
                    "rows": int(len(common)),
                    "support_hash": support_id_hash(common),
                    "source_path": str(source_path),
                    "source_sha256": sha256_file(source_path),
                    "path": str(output_path.relative_to(run_root)),
                    "sha256": sha256_file(output_path),
                }
            )
    result = {
        "status": "PASS",
        "stage": "L256_DEVELOPMENT_COMMON_SUPPORT",
        "support_contract": SUPPORT_CONTRACT,
        "common_history_steps": COMMON_HISTORY,
        "records": records,
        "base_data_inventory": [
            {
                "split": split,
                "source_path": str(
                    source_shared / "base_data" / "tep" / f"{split}.parquet"
                ),
                "bytes": (
                    source_shared / "base_data" / "tep" / f"{split}.parquet"
                ).stat().st_size,
                "sha256": sha256_file(
                    source_shared / "base_data" / "tep" / f"{split}.parquet"
                ),
            }
            for split in ("train", "validation")
        ],
        "test_sample_files_created": False,
        "ood_sample_files_created": False,
        "test_base_files_exposed": False,
        "ood_base_files_exposed": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(run_root / "logs" / "L256_DEVELOPMENT_SUPPORT.json", result)
    return result


def run_scope(source_shared: Path, run_root: Path) -> dict[str, Any]:
    config = load_config()
    _require_baseline_ancestor(config)
    if _git("status", "--porcelain"):
        raise RuntimeError("STOP_TEP_EXTENSION_WORKTREE_NOT_CLEAN")
    scope_path = run_root / "logs" / "SCOPE.json"
    if scope_path.is_file():
        existing = _read_json(scope_path)
        if existing.get("source_commit") != _git("rev-parse", "HEAD"):
            raise RuntimeError("STOP_SOURCE_COMMIT_CHANGED_AFTER_SCOPE")
        if existing.get("config_sha256") != config["config_sha256"]:
            raise RuntimeError("STOP_CONFIG_CHANGED_AFTER_SCOPE")
        return existing
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"REFUSING_NONEMPTY_NEW_RUN_ROOT:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir()
    _storage_guard(run_root, "scope", config)
    support = build_l256_development_shared(source_shared, run_root)
    derived = run_root / "shared_l256_development"
    inputs, dynamics = _tep_views(derived)
    k_jobs = [
        (view, channel)
        for view in inputs
        for channel in input_columns(
            derived, view.head.task_id, view.proxy_policy
        )
    ]
    if len(k_jobs) != EXPECTED_K_CHANNEL_JOBS:
        raise RuntimeError(
            "STOP_TEP_K_CHANNEL_CARDINALITY:"
            f"{len(k_jobs)}!={EXPECTED_K_CHANNEL_JOBS}"
        )
    source_registry_inventory = [
        {
            "name": name,
            "bytes": (source_shared / name).stat().st_size,
            "sha256": sha256_file(source_shared / name),
        }
        for name in (
            "DATASET_HASHES.json",
            "TASK_REGISTRY.json",
            "SPLIT_REGISTRY.json",
            "SAMPLE_ID_REGISTRY.json",
            "PROTOCOL.json",
            "LOCKBOX.json",
        )
    ]
    result = {
        "status": "PASS",
        "stage": "TEP_CPU_HISTORY_EXTENSION_SCOPE",
        "protocol_id": config["protocol_id"],
        "source_commit": _git("rev-parse", "HEAD"),
        "baseline_commit": config["baseline_commit"],
        "implementation_start_commit": config["implementation_start_commit"],
        "config_sha256": config["config_sha256"],
        "source_shared": str(source_shared.resolve()),
        "development_shared": str(derived.resolve()),
        "active_datasets": ["tep"],
        "active_tasks": [EXPECTED_TASK],
        "input_views": len(inputs),
        "dynamic_views": len(dynamics),
        "k_channel_jobs": len(k_jobs),
        "history_aware_steps": list(EXPECTED_HISTORIES),
        "common_support_history_steps": COMMON_HISTORY,
        "workers": 1,
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "sru_status": "NOT_RUN_BY_USER_SCOPE",
        "cz_status": "NOT_RUN_BY_USER_SCOPE",
        "test_accessed": False,
        "ood_accessed": False,
        "support_manifest_sha256": _stable_hash(support),
        "data_manifest_sha256": _stable_hash(source_registry_inventory),
        "protocol_sha256": sha256_file(source_shared / "PROTOCOL.json"),
        "pyproject_sha256": sha256_file(PROJECT.parent / "pyproject.toml"),
        "uv_lock_sha256": sha256_file(PROJECT.parent / "uv.lock"),
        "source_registry_inventory": source_registry_inventory,
    }
    write_json(scope_path, result)
    return result


def _paths(run_root: Path) -> tuple[Path, Path]:
    return run_root / "shared_l256_development", run_root / "results"


def _result_path(output: Path, stage: str, view: ViewSpec, channel: str | None = None) -> Path:
    if stage == "K":
        assert channel is not None
        return output / "DEVELOPMENT" / "K" / view.head.head_id / view.proxy_policy / channel / "RESULT.json"
    if stage in {"C", "W"}:
        return output / "DEVELOPMENT" / stage / view.head.head_id / view.proxy_policy / "RESULT.json"
    return output / "DEVELOPMENT" / stage / view.head.head_id / view.availability_scenario / view.proxy_policy / "RESULT.json"


def _baseline_result_path(output: Path, family: str, model: str, view: ViewSpec) -> Path:
    return output / "BASELINE_DEVELOPMENT" / family / "PREDICTIONS" / model / view.relative_root / "RESULT.json"


def _completed(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _read_json(path)
    return value if value.get("status") in SUCCESS_STATUSES else None


def _run_one(path: Path, function: Callable[..., dict[str, Any]], *arguments: Any) -> dict[str, Any]:
    previous = _completed(path)
    return previous if previous is not None else dict(function(*arguments))


def _summary(output: Path, stage: str, results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(results)
    statuses = [str(value.get("status")) for value in values]
    result = {
        "status": "PASS" if all(status in SUCCESS_STATUSES for status in statuses) else "FAILED",
        "stage": stage,
        "jobs": len(values),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "history_override_config": str(CONFIG_PATH),
        "history_override_config_sha256": sha256_file(CONFIG_PATH),
        "history_aware_steps": list(EXPECTED_HISTORIES),
        "common_support_history_steps": COMMON_HISTORY,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / stage / "SUMMARY.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"STOP_DEVELOPMENT_STAGE_FAILED:{stage}:{result['status_counts']}")
    return result


def run_k(shared: Path, output: Path) -> dict[str, Any]:
    inputs, _ = _tep_views(shared)
    jobs = [
        (view, channel)
        for view in inputs
        for channel in input_columns(
            shared, view.head.task_id, view.proxy_policy
        )
    ]
    if len(jobs) != EXPECTED_K_CHANNEL_JOBS:
        raise RuntimeError(
            "STOP_TEP_K_CHANNEL_CARDINALITY:"
            f"{len(jobs)}!={EXPECTED_K_CHANNEL_JOBS}"
        )
    results = [
        _run_one(
            _result_path(output, "K", view, channel),
            run_k_channel,
            shared,
            PROJECT,
            output,
            view,
            channel,
            TEP_CPU_HISTORY_EXTENSION_PROTOCOL,
            CONFIG_PATH,
        )
        for view, channel in jobs
    ]
    return _summary(output, "K", results)


def _run_view_stage(
    shared: Path,
    output: Path,
    stage: str,
    function: Callable[..., dict[str, Any]],
    views: list[ViewSpec],
    *,
    history_override: bool = False,
) -> dict[str, Any]:
    results = []
    for view in views:
        arguments: tuple[Any, ...] = (
            shared,
            PROJECT,
            output,
            view,
            TEP_CPU_HISTORY_EXTENSION_PROTOCOL,
        )
        if history_override:
            arguments = (*arguments, CONFIG_PATH)
        results.append(
            _run_one(_result_path(output, stage, view), function, *arguments)
        )
    return _summary(output, stage, results)


def run_joint(shared: Path, output: Path, views: list[ViewSpec]) -> dict[str, Any]:
    results = [
        _run_one(
            _result_path(output, "JOINT", view),
            run_joint_stability_view,
            shared,
            PROJECT,
            output,
            None,
            view,
            TEP_CPU_HISTORY_EXTENSION_PROTOCOL,
            CONFIG_PATH,
        )
        for view in views
    ]
    return _summary(output, "JOINT", results)


def _write_incompatible(output: Path, family: str, model: str, view: ViewSpec) -> dict[str, Any]:
    path = _baseline_result_path(output, family, model, view)
    value = {
        "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE",
        "stage": "TEP_CPU_HISTORY_EXTENSION_BASELINE_DEVELOPMENT",
        "model": model,
        "dataset": "tep",
        "task": EXPECTED_TASK,
        "target_head": EXPECTED_HEAD,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "reason": "REGISTERED_PROTOCOL_INCOMPATIBLE_FOR_TEP_HISTORY_EXTENSION",
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(path, value)
    return value


def run_baselines(shared: Path, output: Path, inputs: list[ViewSpec], dynamics: list[ViewSpec]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for view in [*inputs, *dynamics]:
        for model in ("MEAN", "PERSISTENCE"):
            results.append(_run_one(_baseline_result_path(output, "C2", model, view), run_simple_job, shared, PROJECT, output / "BASELINE_DEVELOPMENT", view, model))
        results.append(_write_incompatible(output, "C2", "SEASONAL_PERSISTENCE", view))
    for view in inputs:
        for model in STATIC_INPUT_MODELS:
            results.append(_run_one(_baseline_result_path(output, "C2", model, view), run_static_input_job, shared, PROJECT, output / "BASELINE_DEVELOPMENT", view, model))
        results.append(_run_one(_baseline_result_path(output, "C2", "DPLS", view), run_dpls_job, shared, PROJECT, output / "BASELINE_DEVELOPMENT", view, CONFIG_PATH))
        for model in ("PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"):
            results.append(_run_one(_baseline_result_path(output, "C3", model, view), run_hammerstein_job, shared, PROJECT, output / "BASELINE_DEVELOPMENT", view, model, CONFIG_PATH))
    for view in dynamics:
        results.append(_run_one(_baseline_result_path(output, "C3", "AR", view), run_ar_job, shared, PROJECT, output / "BASELINE_DEVELOPMENT", view, CONFIG_PATH))
        results.append(_run_one(_baseline_result_path(output, "C3", "ARX", view), run_arx_job, shared, PROJECT, output / "BASELINE_DEVELOPMENT", view))
        results.append(_run_one(_baseline_result_path(output, "C3", "LINEAR_NARX", view), run_narx_job, shared, PROJECT, output / "BASELINE_DEVELOPMENT", view))
        results.append(_write_incompatible(output, "C3", "N4SID", view))
    return _summary(output, "BASELINES", results)


def _finite_fold_loss_histories(
    fold_losses: Any,
    *,
    history_position: int,
) -> set[int]:
    """Return histories backed by an actually evaluated finite four-fold candidate."""
    if not isinstance(fold_losses, dict):
        return set()
    histories: set[int] = set()
    for raw_candidate, raw_losses in fold_losses.items():
        try:
            candidate = ast.literal_eval(str(raw_candidate))
            losses = np.asarray(raw_losses, dtype=np.float64).reshape(-1)
            history = int(candidate[history_position])
        except (IndexError, TypeError, ValueError, SyntaxError):
            continue
        if len(losses) == 4 and bool(np.isfinite(losses).all()):
            histories.add(history)
    return histories


def _finite_four_fold_grid(fold_losses: Any, *, history_position: int) -> bool:
    if not isinstance(fold_losses, dict) or not fold_losses:
        return False
    for raw_candidate, raw_losses in fold_losses.items():
        try:
            candidate = ast.literal_eval(str(raw_candidate))
            int(candidate[history_position])
            losses = np.asarray(raw_losses, dtype=np.float64).reshape(-1)
        except (IndexError, TypeError, ValueError, SyntaxError):
            return False
        if len(losses) != 4 or not bool(np.isfinite(losses).all()):
            return False
    return True


def _history_override_audit(record: dict[str, Any]) -> dict[str, Any]:
    direct = record.get("history_override")
    if isinstance(direct, dict):
        return direct
    nested = record.get("selection", {}).get("history_override")
    return nested if isinstance(nested, dict) else {}


def _prediction_artifact_verified(record: dict[str, Any], root: Path) -> bool:
    relative_path = record.get("prediction_path")
    expected_sha256 = record.get("prediction_sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        return False
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate.is_file() and sha256_file(candidate) == expected_sha256


def _prediction_artifact_entry(
    record: dict[str, Any], root: Path, run_root: Path
) -> dict[str, str] | None:
    """Return a run-root-relative prediction reference for the pilot seal.

    Pilot evidence is later checked from the launcher process.  Keep the
    reference relative to the run root and reject paths that escape it (for
    example through a symlink), so a valid hash cannot be used to bless an
    unrelated artifact.
    """

    relative_path = record.get("prediction_path")
    expected_sha256 = record.get("prediction_sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        return None
    root = root.resolve()
    run_root = run_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        relative_to_run_root = candidate.relative_to(run_root)
    except ValueError:
        return None
    if not candidate.is_file() or sha256_file(candidate) != expected_sha256:
        return None
    return {
        "path": relative_to_run_root.as_posix(),
        "sha256": expected_sha256,
    }


def run_pilot(shared: Path, run_root: Path) -> dict[str, Any]:
    scope_path = run_root / "logs" / "SCOPE.json"
    support_path = run_root / "logs" / "L256_DEVELOPMENT_SUPPORT.json"
    if not scope_path.is_file() or not support_path.is_file():
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_PROVENANCE_MISSING")
    scope = _read_json(scope_path)
    support = _read_json(support_path)
    if scope.get("status") != "PASS" or support.get("status") != "PASS":
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_PROVENANCE_NOT_PASS")
    pilot_output = run_root / "pilot" / "results"
    inputs, dynamics = _tep_views(shared)
    view = inputs[0]
    channels = input_columns(shared, EXPECTED_TASK, view.proxy_policy)
    fast = next((name for name in channels if name.lower().startswith("xmv_")), channels[0])
    medium = next((name for name in channels if name.lower().startswith("xmeas_")), channels[-1])
    records = [
        run_k_channel(shared, PROJECT, pilot_output, view, fast, TEP_CPU_HISTORY_EXTENSION_PROTOCOL, CONFIG_PATH),
        run_k_channel(shared, PROJECT, pilot_output, view, medium, TEP_CPU_HISTORY_EXTENSION_PROTOCOL, CONFIG_PATH),
        run_dpls_job(shared, PROJECT, pilot_output / "BASELINE_DEVELOPMENT", view, CONFIG_PATH),
        run_ar_job(shared, PROJECT, pilot_output / "BASELINE_DEVELOPMENT", dynamics[0], CONFIG_PATH),
    ]
    jobs = {
        f"K:{fast}": {
            "record": records[0],
            "fold_losses": records[0].get("profile_fold_losses"),
            "history_position": 1,
            "prediction_root": pilot_output,
            "expected_grid_entries": None,
            "requires_common_scoring": True,
        },
        f"K:{medium}": {
            "record": records[1],
            "fold_losses": records[1].get("profile_fold_losses"),
            "history_position": 1,
            "prediction_root": pilot_output,
            "expected_grid_entries": None,
            "requires_common_scoring": True,
        },
        "DPLS": {
            "record": records[2],
            "fold_losses": records[2].get("selection", {}).get("fold_losses"),
            "history_position": 0,
            "prediction_root": pilot_output / "BASELINE_DEVELOPMENT",
            "expected_grid_entries": 40,
            "requires_common_scoring": False,
        },
        "AR": {
            "record": records[3],
            "fold_losses": records[3]
            .get("selection", {})
            .get("profile_fold_losses"),
            "history_position": 1,
            "prediction_root": pilot_output / "BASELINE_DEVELOPMENT",
            "expected_grid_entries": None,
            "requires_common_scoring": False,
        },
    }
    expected = set(EXPECTED_HISTORIES)
    expected_config_sha256 = sha256_file(CONFIG_PATH)
    coverage_by_job: dict[str, set[int]] = {}
    job_audits: dict[str, dict[str, Any]] = {}
    prediction_artifacts: dict[str, dict[str, str]] = {}
    for job, specification in jobs.items():
        record = specification["record"]
        fold_losses = specification["fold_losses"]
        history_position = int(specification["history_position"])
        histories = _finite_fold_loss_histories(
            fold_losses, history_position=history_position
        )
        coverage_by_job[job] = histories
        override = _history_override_audit(record)
        grid_entries = len(fold_losses) if isinstance(fold_losses, dict) else 0
        expected_grid_entries = specification["expected_grid_entries"]
        grid_size_valid = (
            expected_grid_entries is None
            or grid_entries == int(expected_grid_entries)
        )
        common_scoring_valid = (
            not bool(specification["requires_common_scoring"])
            or record.get("selected_scoring_history_steps") == COMMON_HISTORY
        )
        artifact = _prediction_artifact_entry(
            record, specification["prediction_root"], run_root
        )
        audit = {
            "status": str(record.get("status")),
            "history_steps": sorted(histories),
            "fold_loss_grid_entries": grid_entries,
            "finite_exact_four_fold_grid": _finite_four_fold_grid(
                fold_losses, history_position=history_position
            ),
            "expected_grid_entries": expected_grid_entries,
            "grid_size_valid": grid_size_valid,
            "common_scoring_valid": common_scoring_valid,
            "override_config_sha256": override.get("config_sha256"),
            "override_protocol_id": override.get("protocol_id"),
            "prediction_artifact_verified": _prediction_artifact_verified(
                record, specification["prediction_root"]
            ),
            "prediction_path": None if artifact is None else artifact["path"],
            "prediction_sha256": None if artifact is None else artifact["sha256"],
        }
        if artifact is not None:
            prediction_artifacts[job] = artifact
        audit["passed"] = bool(
            audit["status"] == "PASS"
            and histories == expected
            and audit["finite_exact_four_fold_grid"]
            and audit["grid_size_valid"]
            and audit["common_scoring_valid"]
            and audit["override_config_sha256"] == expected_config_sha256
            and audit["override_protocol_id"] == EXPECTED_PROTOCOL
            and audit["prediction_artifact_verified"]
            and artifact is not None
        )
        job_audits[job] = audit
    missing_by_job = {
        job: sorted(expected.difference(histories))
        for job, histories in coverage_by_job.items()
        if histories != expected
    }
    histories_seen = set.intersection(*coverage_by_job.values())
    missing = sorted(expected.difference(histories_seen))
    source_registry_inventory = scope.get("source_registry_inventory")
    if not isinstance(source_registry_inventory, list) or not source_registry_inventory:
        raise RuntimeError("STOP_TEP_EXTENSION_DATA_MANIFEST_MISSING")
    if scope.get("support_manifest_sha256") != _stable_hash(support):
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_SUPPORT_MANIFEST_CHANGED")
    data_manifest_sha256 = scope.get("data_manifest_sha256")
    if not isinstance(data_manifest_sha256, str):
        data_manifest_sha256 = _stable_hash(source_registry_inventory)
    protocol_sha256 = scope.get("protocol_sha256")
    if not isinstance(protocol_sha256, str):
        protocol_sha256 = sha256_file(shared / "PROTOCOL.json")
    if protocol_sha256 != sha256_file(shared / "PROTOCOL.json"):
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_PROTOCOL_MANIFEST_CHANGED")
    result = {
        "status": "PASS"
        if not missing_by_job and all(audit["passed"] for audit in job_audits.values())
        else "FAILED",
        "stage": "TEP_CPU_HISTORY_EXTENSION_PILOT",
        "pilot_evidence_version": 1,
        "protocol_id": EXPECTED_PROTOCOL,
        "protocol_sha256": protocol_sha256,
        "source_commit": scope.get("source_commit"),
        "config_sha256": expected_config_sha256,
        "history_override_config_sha256": expected_config_sha256,
        "scope_manifest_path": str(scope_path.relative_to(run_root).as_posix()),
        "scope_manifest_sha256": sha256_file(scope_path),
        "support_manifest_path": str(support_path.relative_to(run_root).as_posix()),
        "support_manifest_sha256": _stable_hash(support),
        "support_manifest_file_sha256": sha256_file(support_path),
        "data_manifest_sha256": data_manifest_sha256,
        "source_registry_inventory": source_registry_inventory,
        "jobs": len(records),
        "history_steps_observed": sorted(histories_seen),
        "history_steps_observed_by_job": {
            job: sorted(histories) for job, histories in coverage_by_job.items()
        },
        "missing_registered_histories": missing,
        "missing_registered_histories_by_job": missing_by_job,
        "job_audits": job_audits,
        "prediction_artifacts": prediction_artifacts,
        "prediction_sha256": {
            job: artifact["sha256"] for job, artifact in prediction_artifacts.items()
        },
        "prediction_artifact_count": len(prediction_artifacts),
        "prediction_artifacts_sha256": _stable_hash(prediction_artifacts),
        "common_support_history_steps": COMMON_HISTORY,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(run_root / "logs" / "PILOT.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_FAILED:{result}")
    return result


def status(run_root: Path) -> dict[str, Any]:
    summaries = {}
    for path in sorted((run_root / "results" / "DEVELOPMENT").glob("*/SUMMARY.json")):
        summaries[path.parent.name] = _read_json(path)
    return {
        "run_root": str(run_root),
        "scope": _read_json(run_root / "logs" / "SCOPE.json") if (run_root / "logs" / "SCOPE.json").is_file() else None,
        "pilot": _read_json(run_root / "logs" / "PILOT.json") if (run_root / "logs" / "PILOT.json").is_file() else None,
        "development": summaries,
        "test_accessed": False,
        "ood_accessed": False,
    }


def main() -> None:
    _require_uv_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("scope", "pilot", "k", "c", "w", "a", "joint", "baselines", "development", "status"),
    )
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    source_shared = args.shared.resolve()
    run_root = args.run_root.resolve()
    scope = run_scope(source_shared, run_root)
    if args.stage == "scope":
        print(json.dumps(scope, ensure_ascii=False, sort_keys=True))
        return
    if args.stage == "status":
        print(json.dumps(status(run_root), ensure_ascii=False, indent=2, sort_keys=True))
        return
    config = load_config()
    _storage_guard(run_root, args.stage, config)
    shared, output = _paths(run_root)
    output.mkdir(parents=True, exist_ok=True)
    inputs, dynamics = _tep_views(shared)
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "pilot": lambda: run_pilot(shared, run_root),
        "k": lambda: run_k(shared, output),
        "c": lambda: _run_view_stage(shared, output, "C", run_c_view, inputs),
        "w": lambda: _run_view_stage(shared, output, "W", run_w_view, inputs),
        "a": lambda: _run_view_stage(
            shared,
            output,
            "A",
            run_a_view,
            dynamics,
            history_override=True,
        ),
        "joint": lambda: run_joint(shared, output, dynamics),
        "baselines": lambda: run_baselines(shared, output, inputs, dynamics),
    }
    stages = ("k", "c", "w", "a", "joint", "baselines") if args.stage == "development" else (args.stage,)
    result = {stage: runners[stage]() for stage in stages}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
