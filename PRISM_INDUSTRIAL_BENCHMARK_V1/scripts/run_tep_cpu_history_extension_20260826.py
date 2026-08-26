"""TEP-only CPU/PRISM development for the frozen L256 history extension.

This launcher deliberately has no CZ, SRU, Neural3, Stage-2, test, or OOD
stage.  It constructs a run-local train/validation-only L256 support view and
passes the frozen history override explicitly to every history-selecting job.
The formal freeze/checkpoint/test process is a separate follow-up launcher.
"""

from __future__ import annotations

import argparse
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
        "pyproject_sha256": sha256_file(PROJECT.parent / "pyproject.toml"),
        "uv_lock_sha256": sha256_file(PROJECT.parent / "uv.lock"),
        "source_registry_inventory": [
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
        ],
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


def run_pilot(shared: Path, run_root: Path) -> dict[str, Any]:
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
    histories_seen: set[int] = set()
    for record in records:
        for audit in record.get("registered_history_support_audit", []):
            if audit.get("available") is True:
                histories_seen.add(int(audit["history_steps"]))
        for key in record.get("profile_fold_losses", {}):
            numbers = [int(token) for token in key.replace("(", "").replace(")", "").split(",") if token.strip().isdigit()]
            if len(numbers) >= 2:
                histories_seen.add(numbers[1])
        for key in record.get("selection", {}).get("fold_losses", {}):
            for expected in EXPECTED_HISTORIES:
                if str(expected) in str(key):
                    histories_seen.add(expected)
    missing = sorted(set(EXPECTED_HISTORIES).difference(histories_seen))
    result = {
        "status": "PASS" if not missing and all(str(item.get("status")) in SUCCESS_STATUSES for item in records) else "FAILED",
        "stage": "TEP_CPU_HISTORY_EXTENSION_PILOT",
        "jobs": len(records),
        "history_steps_observed": sorted(histories_seen),
        "missing_registered_histories": missing,
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
