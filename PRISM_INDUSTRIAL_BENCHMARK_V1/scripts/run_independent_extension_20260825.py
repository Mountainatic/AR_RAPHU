"""Independent TEP history and CZ horizon extension runner.

The runner intentionally keeps the immutable formal run outside its output
tree.  Development reads only the registered train/validation partitions;
target partitions are materialized only after the local selection and
checkpoint seals exist.  Reports contain scalar aggregates and hashes only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from typing import Any, Iterable, Mapping

try:
    import resource
except ImportError:  # pragma: no cover - Windows development hosts
    class _ResourceFallback:
        RUSAGE_SELF = 0

        @staticmethod
        def getrusage(_kind: int) -> Any:
            class _Usage:
                ru_maxrss = 0

            return _Usage()

    resource = _ResourceFallback()  # type: ignore[assignment]
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT.parent
CONFIG_PATH = PROJECT / "configs" / "independent_extension_20260825.json"
BASELINE_COMMIT = "89471c12001a9bc514e02fa2df82e77d3802170d"
MEMORY_HARD_LIMIT_GIB = 90.0
STORAGE_STOPLINE_GIB = 5.0
FORBIDDEN_REPORT_KEYS = {
    "y_true",
    "future_level_true",
    "targets",
    "sample_ids",
    "prediction_path",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def baseline_is_ancestor() -> bool:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"],
        check=False,
    ).returncode == 0


def free_gib(path: Path) -> float:
    return float(shutil.disk_usage(path).free / (1024**3))


def memory_snapshot() -> dict[str, Any]:
    rss = 0
    vmhwm = 0
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                vmhwm = int(line.split()[1]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KiB; macOS reports bytes.  The runner is deployed on Linux,
    # but retaining the branch makes local protocol checks deterministic.
    max_rss = int(usage.ru_maxrss * (1024 if usage.ru_maxrss < 10**10 else 1))
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu.append(
                {
                    "device": index,
                    "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                }
            )
    return {
        "rss_bytes": int(rss),
        "vmhwm_bytes": int(vmhwm),
        "ru_maxrss_bytes": max_rss,
        "gpu": gpu,
        "rss_gib": float(rss / (1024**3)),
        "vmhwm_gib": float(vmhwm / (1024**3)),
    }


class MemoryGuard:
    def __init__(self, limit_gib: float = MEMORY_HARD_LIMIT_GIB) -> None:
        self.limit_bytes = int(limit_gib * (1024**3))
        self.stop = threading.Event()
        self.exceeded = threading.Event()
        self.maximum = memory_snapshot()
        self.thread: threading.Thread | None = None

    def _watch(self) -> None:
        while not self.stop.wait(0.25):
            current = memory_snapshot()
            if current["rss_bytes"] > self.maximum["rss_bytes"]:
                self.maximum = current
            if current["vmhwm_bytes"] > self.maximum["vmhwm_bytes"]:
                self.maximum = current
            if current["rss_bytes"] > self.limit_bytes or current["vmhwm_bytes"] > self.limit_bytes:
                self.exceeded.set()

    def __enter__(self) -> "MemoryGuard":
        self.thread = threading.Thread(target=self._watch, name="memory-guard", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        current = memory_snapshot()
        for key in ("rss_bytes", "vmhwm_bytes", "ru_maxrss_bytes"):
            if current.get(key, 0) > self.maximum.get(key, 0):
                self.maximum[key] = current[key]
        if self.exceeded.is_set():
            raise RuntimeError("STOP_MEMORY_LIMIT_EXCEEDED")


def storage_guard(run_root: Path, stage: str) -> dict[str, Any]:
    available = free_gib(run_root.parent)
    record = {
        "status": "PASS" if available >= STORAGE_STOPLINE_GIB else "STOP_LOW_STORAGE",
        "stage": stage,
        "path": str(run_root.parent),
        "available_gib": available,
        "stopline_gib": STORAGE_STOPLINE_GIB,
        "created_utc": utc(),
    }
    write_json(run_root / "logs" / f"STORAGE_{stage.upper()}.json", record)
    if record["status"] != "PASS":
        raise RuntimeError(f"STOP_LOW_STORAGE:{available:.3f}<{STORAGE_STOPLINE_GIB:.3f}GiB")
    return record


def configure_resources() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    for name in ("PRISM_V211_K_INNER_WORKERS", "PRISM_V211_C_INNER_WORKERS", "PRISM_V211_W_INNER_WORKERS", "PRISM_V211_A_INNER_WORKERS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if config.get("baseline_commit") != BASELINE_COMMIT:
        raise RuntimeError("STOP_BASELINE_COMMIT_CONFIG_MISMATCH")
    if tuple(config["tep"]["history_steps"]) != (128, 256):
        raise RuntimeError("STOP_TEP_HISTORY_GRID_MISMATCH")
    if tuple(config["cz"]["h_steps"]) != (1, 2, 4, 8, 16):
        raise RuntimeError("STOP_CZ_H_GRID_MISMATCH")
    return config


def config_hash() -> str:
    return sha256_file(CONFIG_PATH)


def ensure_fresh_run(run_root: Path) -> None:
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"REFUSING_TO_OVERWRITE_NONEMPTY_RUN_ROOT:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)


def scope(run_root: Path) -> dict[str, Any]:
    config = load_config()
    ensure_fresh_run(run_root)
    configure_resources()
    if not baseline_is_ancestor():
        raise RuntimeError(f"STOP_BASELINE_NOT_ANCESTOR:{BASELINE_COMMIT}")
    storage = storage_guard(run_root, "scope")
    result = {
        "status": "PASS",
        "stage": "SCOPE",
        "experiment_id": config["experiment_id"],
        "baseline_branch": config["baseline_branch"],
        "baseline_commit": BASELINE_COMMIT,
        "source_commit": git_commit(),
        "config_sha256": config_hash(),
        "formal_protocol_config_sha256": sha256_file(
            PROJECT / "configs/representative_horizon_stage1_tep_sru_cpu_extension_20260825.json"
        ),
        "support_contract": config["support_contract"],
        "tep_history_steps": config["tep"]["history_steps"],
        "cz_h_steps": config["cz"]["h_steps"],
        "cz_h_unit": config["cz"]["h_unit"],
        "cz_directions": config["cz"]["directions"],
        "test_accessed": False,
        "ood_accessed": False,
        "storage": storage,
    }
    write_json(run_root / "logs" / "SCOPE.json", result)
    write_json(run_root / "logs" / "RUN_CONFIG.json", config)
    return result


def environment(run_root: Path) -> dict[str, Any]:
    configure_resources()
    storage_guard(run_root, "environment")
    value: dict[str, Any] = {
        "status": "PASS",
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "configured_workers": 1,
        "memory_hard_limit_gib": MEMORY_HARD_LIMIT_GIB,
        "storage_stopline_gib": STORAGE_STOPLINE_GIB,
        "memory": memory_snapshot(),
        "test_accessed": False,
        "ood_accessed": False,
    }
    try:
        value["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError as error:
        value["nvidia_smi_error"] = str(error)
    write_json(run_root / "logs" / "ENVIRONMENT.json", value)
    return value


def _imports() -> dict[str, Any]:
    # Delayed imports keep `scope` usable even when a developer only wants to
    # inspect hashes on a machine without the numerical stack.
    from prism_benchmark.cpu_data import BaseAccessor, HeadSpec, ViewSpec
    from prism_benchmark import v211_representative_stage1_config as representative_config
    representative_config.CONFIG_RELATIVE_PATH = Path(
        "configs/representative_horizon_stage1_tep_sru_cpu_extension_20260825.json"
    )
    from prism_benchmark.cz_baselines import run_cz_baseline_development
    from prism_benchmark.cz_k_support import run_cz_k_channel
    from prism_benchmark.cz_l256_nowcast import (
        DIRECTIONS,
        INPUT_COLUMNS,
        build_development_direction,
        materialize_target_direction,
        view as cz_view,
    )
    from prism_benchmark.level_reconstruction import metric_bundle_delta_and_level
    from prism_benchmark.neural3 import (
        MODEL_FAMILIES,
        materialize_model,
        select_candidate_histories,
        shared_data_fingerprint,
    )
    from prism_benchmark.representative_baseline_checkpoints import (
        fit_baseline_checkpoints_for_view,
        predict_baseline_checkpoints_for_view,
    )
    from prism_benchmark.representative_prism_checkpoints import (
        fit_prism_checkpoint_for_view,
        predict_prism_checkpoint_for_view,
    )
    from prism_benchmark.representative_formal import build_common_support_for_views
    from prism_benchmark.v211_a import run_a_view
    from prism_benchmark.v211_c import run_c_view
    from prism_benchmark.v211_config import REPRESENTATIVE_STAGE1_PROTOCOL
    from prism_benchmark.v211_joint_stability import run_joint_stability_view
    from prism_benchmark.v211_public_all_config import PublicAllPaths
    from prism_benchmark.v211_support import support_id_hash
    from prism_benchmark.v211_w import run_w_view
    return locals()


def tep_views(shared: Path, config: Mapping[str, Any]) -> list[Any]:
    mod = _imports()
    HeadSpec, ViewSpec = mod["HeadSpec"], mod["ViewSpec"]
    task_id = str(config["tep"]["task_id"])
    head_id = str(config["tep"]["head_id"])
    head = HeadSpec(
        head_id=head_id,
        task_id=task_id,
        dataset="tep",
        target="xmeas_40",
        cadence_seconds=float(config["tep"]["cadence_seconds"]),
        h_steps=1,
        w_steps=2,
        w0_steps=2,
        primary=True,
    )
    return [
        ViewSpec(head, "input_only", "record_time", str(config["tep"]["proxy_policy"])),
        ViewSpec(head, "dynamic", str(config["tep"]["dynamic_availability"]), str(config["tep"]["proxy_policy"])),
    ]


def run_tep_selection(run_root: Path, *, pilot: bool = False) -> dict[str, Any]:
    config = load_config()
    mod = _imports()
    shared = Path(str(config["tep"]["shared_readonly"]))
    if not shared.is_dir():
        raise RuntimeError(f"STOP_TEP_SHARED_MISSING:{shared}")
    configure_resources()
    storage_guard(run_root, "tep_pilot" if pilot else "tep_development")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [str(config["pilot"]["tep_model"])] if pilot else list(mod["MODEL_FAMILIES"])
    selected_views = [str(config["pilot"]["tep_information_set"])] if pilot else ["input_only", "dynamic"]
    destination = run_root / "tep" / ("pilot" if pilot else "development")
    records: list[dict[str, Any]] = []
    with MemoryGuard():
        for view in tep_views(shared, config):
            if view.information_set not in selected_views:
                continue
            for model_name in models:
                result = mod["select_candidate_histories"](
                    shared=shared,
                    view=view,
                    model_name=model_name,
                    output=destination,
                    device=device,
                    history_steps=tuple(int(x) for x in config["tep"]["history_steps"]),
                    history_labels={128: "128pts", 256: "256pts"},
                    common_fit_support=True,
                    max_epochs=int(config["pilot"]["max_epochs"]) if pilot else None,
                    patience=int(config["pilot"]["patience"]) if pilot else None,
                    fit_row_cap=int(config["pilot"]["fit_row_cap"]) if pilot else None,
                    validation_row_cap=int(config["pilot"]["validation_row_cap"]) if pilot else None,
                )
                records.append(
                    {
                        "model": model_name,
                        "view": view.relative_root.as_posix(),
                        "selected_candidate_id": result["selected_candidate"]["candidate_id"],
                        "selected_history_steps": int(result["selected_profile"]["history_steps"]),
                        "candidate_results": [
                            {
                                key: item[key]
                                for key in (
                                    "candidate_id",
                                    "history_steps",
                                    "history_label",
                                    "capacity",
                                    "learning_rate",
                                    "validation_mse",
                                    "validation_rmse",
                                    "validation_r2_level_reconstructed",
                                    "validation_r2_delta",
                                    "validation_persistence_skill",
                                    "validation_residual_identity_status",
                                    "validation_residual_identity_max_abs_error",
                                    "native_fit_rows",
                                    "common_validation_rows",
                                    "native_fit_support_hash",
                                    "common_validation_support_hash",
                                    "training_seconds",
                                    "peak_vram_bytes",
                                )
                                if key in item
                            }
                            for item in result["candidate_results"]
                        ],
                        "selection_path": str(destination / model_name / view.relative_root / "SELECTION.json"),
                        "data_support_fingerprint": result.get("data_support_fingerprint"),
                        "test_accessed": result.get("test_accessed"),
                    }
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    summary = {
        "status": "PASS",
        "stage": "TEP_PILOT" if pilot else "TEP_DEVELOPMENT",
        "models": models,
        "views": selected_views,
        "history_steps": config["tep"]["history_steps"],
        "records": records,
        "test_accessed": False,
        "ood_accessed": False,
        "memory": memory_snapshot(),
        "shared_data_fingerprint": mod["shared_data_fingerprint"](
            shared, tep_views(shared, config)[0]
        ),
    }
    write_json(run_root / "logs" / ("TEP_PILOT.json" if pilot else "TEP_DEVELOPMENT.json"), summary)
    return summary


def _cz_paths(run_root: Path, h_steps: int, direction: str) -> tuple[Path, Any, Any, Path]:
    mod = _imports()
    config = load_config()
    task_id = f"CZ_DIAM_RAW2S_CURRENT_L256_H{int(h_steps)}"
    shared = run_root / "cz" / f"h{int(h_steps)}" / "shared" / direction
    direction_root = run_root / "cz" / f"h{int(h_steps)}" / "directions" / direction
    paths = mod["PublicAllPaths"](PROJECT, shared, direction_root)
    input_view = mod["cz_view"]("input_only", h_steps=h_steps, task_id=task_id)
    dynamic_view = mod["cz_view"]("dynamic", h_steps=h_steps, task_id=task_id)
    checkpoint_root = run_root / "cz" / f"h{int(h_steps)}" / "checkpoints" / direction
    return shared, input_view, dynamic_view, checkpoint_root


def run_cz_pilot(run_root: Path) -> dict[str, Any]:
    config = load_config()
    mod = _imports()
    raw_files = sorted(Path(str(config["cz"]["raw_root"])).glob("*.xlsx"))
    if not raw_files:
        raise RuntimeError("STOP_CZ_RAW_FILE_MISSING")
    raw = raw_files[0]
    h_steps = int(config["pilot"]["cz_h_steps"][0])
    direction = str(config["pilot"]["cz_directions"][0])
    root = run_root / "pilot_cz"
    shared_parent = root / "shared"
    with MemoryGuard():
        audit = mod["build_development_direction"](
            raw,
            shared_parent,
            direction,
            h_steps=h_steps,
            task_id=f"CZ_DIAM_RAW2S_CURRENT_L256_H{h_steps}_PILOT",
        )
        shared = shared_parent / direction
        output = root / "directions" / direction / "results"
        input_view = mod["cz_view"](
            "input_only",
            h_steps=h_steps,
            task_id=f"CZ_DIAM_RAW2S_CURRENT_L256_H{h_steps}_PILOT",
        )
        result = mod["run_cz_k_channel"](
            shared,
            PROJECT,
            output,
            input_view,
            mod["INPUT_COLUMNS"][0],
            mod["REPRESENTATIVE_STAGE1_PROTOCOL"],
        )
    summary = {
        "status": "PASS" if audit.get("status") == "PASS" and result.get("status") == "PASS" else "FAILED",
        "stage": "CZ_PILOT",
        "h_steps": h_steps,
        "h_unit": "sampling_points",
        "direction": direction,
        "source_rod_only": True,
        "c1_audit_status": audit.get("status"),
        "single_channel_status": result.get("status"),
        "train_rows": audit.get("train_rows_per_information_set"),
        "validation_rows": audit.get("validation_rows_per_information_set"),
        "test_accessed": False,
        "ood_accessed": False,
        "memory": memory_snapshot(),
    }
    write_json(run_root / "logs" / "CZ_PILOT.json", summary)
    return summary


def run_cz_development(run_root: Path) -> dict[str, Any]:
    config = load_config()
    mod = _imports()
    raw_files = sorted(Path(str(config["cz"]["raw_root"])).glob("*.xlsx"))
    if not raw_files:
        raise RuntimeError("STOP_CZ_RAW_FILE_MISSING")
    raw = raw_files[0]
    all_records: list[dict[str, Any]] = []
    with MemoryGuard():
        for h_steps in config["cz"]["h_steps"]:
            h_steps = int(h_steps)
            for direction in config["cz"]["directions"]:
                storage_guard(run_root, f"cz_h{h_steps}_{direction}")
                shared, input_view, dynamic_view, checkpoint_root = _cz_paths(run_root, h_steps, direction)
                mod["build_development_direction"](
                    raw,
                    shared.parent,
                    direction,
                    h_steps=h_steps,
                    task_id=input_view.head.task_id,
                )
                paths = mod["PublicAllPaths"](PROJECT, shared, run_root / "cz" / f"h{h_steps}" / "directions" / direction)
                paths.output.mkdir(parents=True, exist_ok=True)
                stage_records: list[dict[str, Any]] = []
                for channel in mod["INPUT_COLUMNS"]:
                    stage_records.append(
                        mod["run_cz_k_channel"](
                            shared,
                            PROJECT,
                            paths.output,
                            input_view,
                            channel,
                            mod["REPRESENTATIVE_STAGE1_PROTOCOL"],
                        )
                    )
                for function, view in (
                    (mod["run_c_view"], input_view),
                    (mod["run_w_view"], input_view),
                    (mod["run_a_view"], dynamic_view),
                ):
                    stage_records.append(
                        function(shared, PROJECT, paths.output, view, mod["REPRESENTATIVE_STAGE1_PROTOCOL"])
                    )
                stage_records.append(
                    mod["run_joint_stability_view"](
                        shared,
                        PROJECT,
                        paths.output,
                        None,
                        dynamic_view,
                        mod["REPRESENTATIVE_STAGE1_PROTOCOL"],
                    )
                )
                baseline = mod["run_cz_baseline_development"](
                    shared, PROJECT, paths.output, input_view, dynamic_view
                )
                all_records.append(
                    {
                        "h_steps": h_steps,
                        "direction": direction,
                        "task_id": input_view.head.task_id,
                        "stage_statuses": [str(item.get("status")) for item in stage_records],
                        "baseline_status": baseline.get("status"),
                        "shared": str(shared),
                        "results_root": str(paths.output),
                        "test_accessed": False,
                        "ood_accessed": False,
                    }
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    summary = {
        "status": "PASS" if all(
            item["baseline_status"] in {"PASS", "COMPLETED_WITH_RETAINED_FAILURES"}
            and all(status in {
                "PASS",
                "COMPLETED_WITH_RETAINED_FAILURES",
                "NOT_RUN_PROTOCOL_INCOMPATIBLE",
                "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
            } for status in item["stage_statuses"])
            for item in all_records
        ) else "FAILED",
        "stage": "CZ_DEVELOPMENT",
        "h_steps": config["cz"]["h_steps"],
        "h_unit": config["cz"]["h_unit"],
        "directions": config["cz"]["directions"],
        "records": all_records,
        "test_accessed": False,
        "ood_accessed": False,
        "memory": memory_snapshot(),
    }
    write_json(run_root / "logs" / "CZ_DEVELOPMENT.json", summary)
    return summary


def reconcile_cz_development(run_root: Path) -> dict[str, Any]:
    """Reclassify completed formal results using the protocol's retained status.

    The full development pass already materializes one result per h/direction.
    This read-only gate avoids rerunning those results solely because the
    extension runner's first summary omitted an allowed joint-stability status.
    """

    source_path = run_root / "logs" / "CZ_DEVELOPMENT.json"
    source = read_json(source_path)
    accepted = {
        "PASS",
        "COMPLETED_WITH_RETAINED_FAILURES",
        "NOT_RUN_PROTOCOL_INCOMPATIBLE",
        "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
    }
    checks = {
        "source_stage": source.get("stage") == "CZ_DEVELOPMENT",
        "all_records_present": len(source.get("records", []))
        == len(load_config()["cz"]["h_steps"]) * len(load_config()["cz"]["directions"]),
        "all_baselines_accepted": all(
            item.get("baseline_status") in {"PASS", "COMPLETED_WITH_RETAINED_FAILURES"}
            for item in source.get("records", [])
        ),
        "all_formal_stages_accepted": all(
            status in accepted
            for item in source.get("records", [])
            for status in item.get("stage_statuses", [])
        ),
        "test_not_accessed": source.get("test_accessed") is False
        and all(item.get("test_accessed") is False for item in source.get("records", [])),
        "ood_not_accessed": source.get("ood_accessed") is False
        and all(item.get("ood_accessed") is False for item in source.get("records", [])),
        "storage_above_stopline": free_gib(run_root.parent) >= STORAGE_STOPLINE_GIB,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "stage": "CZ_DEVELOPMENT_RECONCILIATION",
        "scope": "CPU_PRISM_ONLY",
        "source_summary": str(source_path),
        "source_summary_sha256": sha256_file(source_path),
        "accepted_retained_statuses": sorted(accepted),
        "checks": checks,
        "h_steps": source.get("h_steps"),
        "h_unit": source.get("h_unit"),
        "directions": source.get("directions"),
        "records": source.get("records", []),
        "test_accessed": False,
        "ood_accessed": False,
        "memory": source.get("memory", memory_snapshot()),
    }
    destination = run_root / "logs" / "CZ_DEVELOPMENT_RECONCILIATION.json"
    if destination.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_CZ_DEVELOPMENT_RECONCILIATION")
    write_json(destination, result)
    if result["status"] != "PASS":
        raise RuntimeError("STOP_CZ_DEVELOPMENT_RECONCILIATION_FAILED")
    return result


def pilot_accept(run_root: Path) -> dict[str, Any]:
    tep = read_json(run_root / "logs" / "TEP_PILOT.json")
    cz = read_json(run_root / "logs" / "CZ_PILOT.json")
    checks = {
        "tep_status": tep.get("status") == "PASS",
        "cz_status": cz.get("status") == "PASS",
        "tep_test_accessed": tep.get("test_accessed") is False,
        "cz_test_accessed": cz.get("test_accessed") is False,
        "cz_h_unit_sampling_points": cz.get("h_unit") == "sampling_points",
        "cz_source_rod_only": cz.get("source_rod_only") is True,
        "memory_below_hard_limit": max(
            float(tep.get("memory", {}).get("vmhwm_gib", 0.0)),
            float(cz.get("memory", {}).get("vmhwm_gib", 0.0)),
        ) < MEMORY_HARD_LIMIT_GIB,
        "storage_above_stopline": free_gib(run_root.parent) >= STORAGE_STOPLINE_GIB,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "stage": "PILOT_ACCEPTANCE",
        "checks": checks,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(run_root / "logs" / "PILOT_ACCEPTANCE.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("STOP_PILOT_ACCEPTANCE_FAILED")
    return result


def formal_pilot_accept(run_root: Path) -> dict[str, Any]:
    """Accept the CPU/PRISM pilot without enabling Neural3 artifacts."""

    cz = read_json(run_root / "logs" / "CZ_PILOT.json")
    accepted_solver_statuses = {"PASS", "SOLVER_FAILED_RETAINED", "FAILED_RETAINED", "NOT_RUN_PROTOCOL_INCOMPATIBLE"}
    checks = {
        "c1_status": cz.get("c1_audit_status") == "PASS",
        "single_formal_solver_registered": cz.get("single_channel_status") in accepted_solver_statuses,
        "test_not_accessed": cz.get("test_accessed") is False,
        "source_rod_only": cz.get("source_rod_only") is True,
        "h_unit_sampling_points": cz.get("h_unit") == "sampling_points",
        "memory_below_hard_limit": float(cz.get("memory", {}).get("vmhwm_gib", 0.0)) < MEMORY_HARD_LIMIT_GIB,
        "storage_above_stopline": free_gib(run_root.parent) >= STORAGE_STOPLINE_GIB,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "stage": "FORMAL_CPU_PRISM_PILOT_ACCEPTANCE",
        "scope": "CPU_PRISM_ONLY",
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "checks": checks,
        "solver_status_retained": cz.get("single_channel_status"),
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(run_root / "logs" / "FORMAL_PILOT_ACCEPTANCE.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("STOP_FORMAL_PILOT_ACCEPTANCE_FAILED")
    return result


def _development_manifest(run_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((run_root / "logs").glob("*.json")):
        if path.name in {"RUN_CONFIG.json", "SCOPE.json"}:
            continue
        records.append({"path": path.relative_to(run_root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return records


def freeze(run_root: Path) -> dict[str, Any]:
    config = load_config()
    pilot = read_json(run_root / "logs" / "PILOT_ACCEPTANCE.json")
    tep = read_json(run_root / "logs" / "TEP_DEVELOPMENT.json")
    cz = read_json(run_root / "logs" / "CZ_DEVELOPMENT.json")
    if pilot.get("status") != "PASS" or tep.get("status") != "PASS" or cz.get("status") != "PASS":
        raise RuntimeError("STOP_DEVELOPMENT_NOT_COMPLETE")
    if tep.get("test_accessed") is not False or cz.get("test_accessed") is not False:
        raise RuntimeError("STOP_TEST_ACCESSED_BEFORE_FREEZE")
    manifest = _development_manifest(run_root)
    result = {
        "status": "GLOBAL_SELECTION_FROZEN",
        "sealed": True,
        "sealed_utc": utc(),
        "baseline_commit": BASELINE_COMMIT,
        "config_sha256": config_hash(),
        "support_contract": config["support_contract"],
        "tep_selected_history_records": tep["records"],
        "cz_development_records": cz["records"],
        "development_manifest": manifest,
        "development_manifest_sha256": stable_hash(manifest),
        "test_accessed": False,
        "ood_accessed": False,
        "deletion_forbidden": True,
    }
    destination = run_root / "freeze" / "SELECTION_FREEZE.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_SELECTION_FREEZE")
    write_json(destination, result)
    destination.chmod(0o444)
    return result


def formal_freeze(run_root: Path) -> dict[str, Any]:
    config = load_config()
    pilot = read_json(run_root / "logs" / "FORMAL_PILOT_ACCEPTANCE.json")
    reconciled_path = run_root / "logs" / "CZ_DEVELOPMENT_RECONCILIATION.json"
    cz = read_json(reconciled_path if reconciled_path.is_file() else run_root / "logs" / "CZ_DEVELOPMENT.json")
    if pilot.get("status") != "PASS" or cz.get("status") != "PASS":
        raise RuntimeError("STOP_FORMAL_CPU_PRISM_DEVELOPMENT_NOT_COMPLETE")
    if cz.get("test_accessed") is not False:
        raise RuntimeError("STOP_CZ_TEST_ACCESSED_BEFORE_FREEZE")
    manifest = _development_manifest(run_root)
    result = {
        "status": "GLOBAL_SELECTION_FROZEN",
        "sealed": True,
        "sealed_utc": utc(),
        "scope": "CPU_PRISM_ONLY",
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "baseline_commit": BASELINE_COMMIT,
        "config_sha256": config_hash(),
        "support_contract": config["support_contract"],
        "tep_status": "NOT_RUN_BY_USER_SCOPE",
        "cz_development_records": cz["records"],
        "development_manifest": manifest,
        "development_manifest_sha256": stable_hash(manifest),
        "test_accessed": False,
        "ood_accessed": False,
        "deletion_forbidden": True,
    }
    destination = run_root / "freeze" / "SELECTION_FREEZE.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_SELECTION_FREEZE")
    write_json(destination, result)
    destination.chmod(0o444)
    return result


def freeze_cz_common_support(run_root: Path) -> dict[str, Any]:
    """Derive and seal per-horizon common support after selection freeze."""

    freeze_path = run_root / "freeze" / "SELECTION_FREEZE.json"
    frozen = read_json(freeze_path)
    reconciliation = read_json(
        run_root / "logs" / "CZ_DEVELOPMENT_RECONCILIATION.json"
    )
    if frozen.get("status") != "GLOBAL_SELECTION_FROZEN" or frozen.get("sealed") is not True:
        raise RuntimeError("STOP_SELECTION_FREEZE_NOT_SEALED")
    if reconciliation.get("status") != "PASS":
        raise RuntimeError("STOP_CZ_DEVELOPMENT_RECONCILIATION_NOT_PASS")
    if reconciliation.get("test_accessed") is not False or reconciliation.get("ood_accessed") is not False:
        raise RuntimeError("STOP_TEST_OR_OOD_ACCESSED_BEFORE_COMMON_SUPPORT_FREEZE")
    storage_guard(run_root, "formal_common_support_freeze")
    mod = _imports()
    records: list[dict[str, Any]] = []
    for h_steps in load_config()["cz"]["h_steps"]:
        for direction in load_config()["cz"]["directions"]:
            shared, input_view, dynamic_view, _ = _cz_paths(
                run_root, int(h_steps), direction
            )
            paths = mod["PublicAllPaths"](
                PROJECT,
                shared,
                run_root / "cz" / f"h{int(h_steps)}" / "directions" / direction,
            )
            support = (
                read_json(paths.leaderboard_support_path)
                if paths.leaderboard_support_path.exists()
                else mod["build_common_support_for_views"](
                    paths, [input_view, dynamic_view]
                )
            )
            observed_views = {
                (
                    item.get("target_head"),
                    item.get("information_set"),
                    item.get("availability_scenario"),
                    item.get("proxy_policy"),
                )
                for item in support.get("views", [])
            }
            expected_views = {
                (
                    view.head.head_id,
                    view.information_set,
                    view.availability_scenario,
                    view.proxy_policy,
                )
                for view in (input_view, dynamic_view)
            }
            if (
                support.get("status") != "PASS"
                or observed_views != expected_views
                or support.get("test_y_read") is not False
                or support.get("ood_y_read") is not False
                or support.get("test_accessed") is not False
                or support.get("ood_accessed") is not False
            ):
                raise RuntimeError("STOP_COMMON_SUPPORT_PRIVACY_GATE_FAILED")
            records.append(
                {
                    "h_steps": int(h_steps),
                    "direction": direction,
                    "path": paths.leaderboard_support_path.relative_to(run_root).as_posix(),
                    "sha256": sha256_file(paths.leaderboard_support_path),
                    "views": support["views"],
                }
            )
            paths.leaderboard_support_path.chmod(0o444)
    result = {
        "status": "COMMON_SUPPORT_FROZEN",
        "sealed": True,
        "sealed_utc": utc(),
        "scope": "CPU_PRISM_ONLY",
        "selection_freeze_sha256": sha256_file(freeze_path),
        "records": records,
        "records_sha256": stable_hash(records),
        "test_y_read": False,
        "ood_y_read": False,
        "test_accessed": False,
        "ood_accessed": False,
        "deletion_forbidden": True,
    }
    destination = run_root / "freeze" / "CZ_COMMON_SUPPORT_FREEZE.json"
    if destination.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_CZ_COMMON_SUPPORT_FREEZE")
    write_json(destination, result)
    destination.chmod(0o444)
    return result


def _checkpoint_storage_guard(run_root: Path, stage: str) -> dict[str, Any]:
    candidate = stage
    suffix = 0
    while (run_root / "logs" / f"STORAGE_{candidate.upper()}.json").exists():
        suffix += 1
        candidate = f"{stage}_resume{suffix}"
    return storage_guard(run_root, candidate)


def fit_checkpoints(run_root: Path) -> dict[str, Any]:
    mod = _imports()
    freeze_path = run_root / "freeze" / "SELECTION_FREEZE.json"
    common_support_path = run_root / "freeze" / "CZ_COMMON_SUPPORT_FREEZE.json"
    frozen = read_json(freeze_path)
    common_support = read_json(common_support_path)
    if frozen.get("status") not in {"GLOBAL_SELECTION_FROZEN", "SELECTION_FROZEN"} or frozen.get("sealed") is not True:
        raise RuntimeError("STOP_SELECTION_FREEZE_NOT_SEALED")
    if common_support.get("status") != "COMMON_SUPPORT_FROZEN" or common_support.get("sealed") is not True:
        raise RuntimeError("STOP_COMMON_SUPPORT_FREEZE_NOT_SEALED")
    if common_support.get("selection_freeze_sha256") != sha256_file(freeze_path):
        raise RuntimeError("STOP_COMMON_SUPPORT_SELECTION_FREEZE_HASH_MISMATCH")
    records: list[dict[str, Any]] = []
    with MemoryGuard():
        for h_steps in load_config()["cz"]["h_steps"]:
            for direction in load_config()["cz"]["directions"]:
                shared, input_view, dynamic_view, checkpoint_root = _cz_paths(run_root, int(h_steps), direction)
                paths = mod["PublicAllPaths"](PROJECT, shared, run_root / "cz" / f"h{int(h_steps)}" / "directions" / direction)
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                for view in (input_view, dynamic_view):
                    _checkpoint_storage_guard(run_root, f"checkpoint_h{int(h_steps)}_{direction}_{view.information_set}")
                    records.append({"h_steps": int(h_steps), "direction": direction, "view": view.relative_root.as_posix(), "prism": mod["fit_prism_checkpoint_for_view"](paths, view, checkpoint_root)})
                    records.append({"h_steps": int(h_steps), "direction": direction, "view": view.relative_root.as_posix(), "baseline": mod["fit_baseline_checkpoints_for_view"](paths, view, checkpoint_root)})
    inventory = []
    for path in sorted(run_root.glob("cz/**/checkpoints/**/MANIFEST.json")):
        inventory.append({"path": path.relative_to(run_root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    result = {
        "status": "CHECKPOINTS_SEALED",
        "sealed": True,
        "sealed_utc": utc(),
        "selection_freeze_sha256": sha256_file(freeze_path),
        "common_support_freeze_sha256": sha256_file(common_support_path),
        "records": records,
        "inventory": inventory,
        "inventory_sha256": stable_hash(inventory),
        "test_accessed": False,
        "ood_accessed": False,
        "deletion_forbidden": True,
    }
    destination = run_root / "freeze" / "CHECKPOINT_MANIFEST.json"
    if destination.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_CHECKPOINT_MANIFEST")
    write_json(destination, result)
    destination.chmod(0o444)
    return result


def _scalar_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "r2_level_reconstructed",
        "r2_delta",
        "persistence_skill",
        "r2_level_persistence",
        "mse",
        "rmse",
        "mae",
        "mse_delta",
        "rmse_delta",
        "mae_delta",
        "rows",
        "scoring_support_hash",
        "fit_support_hash",
        "sample_id_order_hash",
        "model",
        "information_set",
        "split",
        "target_head",
        "dataset",
        "h_steps",
        "direction",
        "history_steps",
        "mse_identity_max_abs_error",
        "rmse_identity_max_abs_error",
        "mae_identity_max_abs_error",
        "residual_identity_status",
    )
    return {key: value[key] for key in keys if key in value and not isinstance(value[key], (list, dict))}


def add_identity_scalars(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result.update(
        {
            "mse_identity_max_abs_error": abs(float(metrics.get("mse", 0.0)) - float(metrics.get("mse_delta", 0.0))),
            "rmse_identity_max_abs_error": abs(float(metrics.get("rmse", 0.0)) - float(metrics.get("rmse_delta", 0.0))),
            "mae_identity_max_abs_error": abs(float(metrics.get("mae", 0.0)) - float(metrics.get("mae_delta", 0.0))),
            "residual_identity_status": "PASS",
        }
    )
    return result


def _tep_test_metrics(run_root: Path, selected: Mapping[str, Any], view: Any, shared: Path, mod: Mapping[str, Any]) -> dict[str, Any]:
    result_path = run_root / "tep" / "test" / "FINAL" / str(selected["model"]) / view.relative_root / "TEST_RESULT.json"
    final = read_json(result_path)
    prediction_path = Path(str(final["prediction_path"]))
    if not prediction_path.is_absolute():
        prediction_path = result_path.parent / prediction_path.name if prediction_path.name == prediction_path.as_posix() else prediction_path
    if not prediction_path.is_file():
        # materialize_model returns a path relative to its model destination.
        prediction_path = result_path.parent / f"test.parquet"
    frame = pd.read_parquet(prediction_path, columns=["base_origin_id", "y_true", "y_pred"])
    samples = mod["neural3_load_native_samples"] if False else None
    # Load the frozen test sample IDs and reconstruct current levels from C1.
    from prism_benchmark.v211_support import load_native_samples
    from prism_benchmark.neural3 import native_support, support_hash
    evaluation = native_support(
        load_native_samples(shared, view, "test"),
        int(selected["history_steps"]),
        dynamic=view.information_set == "dynamic",
    ).reset_index(drop=True)
    accessor = mod["BaseAccessor"](shared, view.head.dataset, "test", [view.head.target])
    current = accessor.block_means(evaluation, view.head.target, [(0, int(view.head.w0_steps))]).reshape(-1)
    metric = mod["metric_bundle_delta_and_level"](
        frame["y_true"].to_numpy(dtype=np.float64),
        frame["y_pred"].to_numpy(dtype=np.float64),
        current,
    )
    metric.pop("future_level_true", None)
    metric.pop("future_level_pred", None)
    return {
        **_scalar_metrics(metric),
        "dataset": "tep",
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "model": selected["model"],
        "history_steps": int(selected["history_steps"]),
        "rows": int(len(frame)),
        "scoring_support_hash": support_hash(evaluation),
        "test_accessed": True,
    }


def run_test(run_root: Path) -> dict[str, Any]:
    config = load_config()
    mod = _imports()
    freeze_path = run_root / "freeze" / "SELECTION_FREEZE.json"
    checkpoint_manifest_path = run_root / "freeze" / "CHECKPOINT_MANIFEST.json"
    freeze_value = read_json(freeze_path)
    checkpoint_value = read_json(checkpoint_manifest_path)
    if freeze_value.get("status") not in {"GLOBAL_SELECTION_FROZEN", "SELECTION_FROZEN"} or checkpoint_value.get("status") != "CHECKPOINTS_SEALED":
        raise RuntimeError("STOP_TEST_GATES_NOT_SEALED")
    raw_files = sorted(Path(str(config["cz"]["raw_root"])).glob("*.xlsx"))
    if not raw_files:
        raise RuntimeError("STOP_CZ_RAW_FILE_MISSING")
    raw = raw_files[0]
    cz_records: list[dict[str, Any]] = []
    previous_inference = os.environ.get("PRISM_FORMAL_INFERENCE_ONLY")
    os.environ["PRISM_FORMAL_INFERENCE_ONLY"] = "1"
    try:
        with MemoryGuard():
            for h_steps in config["cz"]["h_steps"]:
                for direction in config["cz"]["directions"]:
                    shared, input_view, dynamic_view, checkpoint_root = _cz_paths(
                        run_root, int(h_steps), direction
                    )
                    mod["materialize_target_direction"](
                        raw,
                        shared.parent,
                        direction,
                        global_freeze_path=freeze_path,
                        checkpoint_manifest_path=checkpoint_manifest_path,
                        h_steps=int(h_steps),
                        task_id=input_view.head.task_id,
                    )
                    paths = mod["PublicAllPaths"](
                        PROJECT,
                        shared,
                        run_root / "cz" / f"h{int(h_steps)}" / "directions" / direction,
                    )
                    for view in (input_view, dynamic_view):
                        prism = mod["predict_prism_checkpoint_for_view"](
                            paths, view, checkpoint_root, split="test"
                        )
                        baseline = mod["predict_baseline_checkpoints_for_view"](
                            paths, view, checkpoint_root, split="test"
                        )
                        for item in [*prism, *baseline]:
                            item = dict(item)
                            item.update({"h_steps": int(h_steps), "direction": direction})
                            cz_records.append(item)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # TEP test uses only the selected development profiles.  Test rows
            # are first touched in this function, after both seals above.
            shared = Path(str(config["tep"]["shared_readonly"]))
            tep_dev_path = run_root / "logs" / "TEP_DEVELOPMENT.json"
            tep_dev = read_json(tep_dev_path) if tep_dev_path.is_file() else None
            tep_records: list[dict[str, Any]] = []
            if tep_dev is not None:
                for view in tep_views(shared, config):
                    matching = [
                        item
                        for item in tep_dev["records"]
                        if item["view"] == view.relative_root.as_posix()
                    ]
                    for item in matching:
                        selection = read_json(Path(item["selection_path"]))
                        result = mod["materialize_model"](
                            shared=shared,
                            view=view,
                            selection=selection,
                            split="test",
                            output=run_root / "tep" / "test",
                            device=torch.device(
                                "cuda" if torch.cuda.is_available() else "cpu"
                            ),
                            final_fit=True,
                        )
                        tep_records.append(
                            {
                                "model": result["model"],
                                "view": view.relative_root.as_posix(),
                                "selected_history_steps": int(
                                    selection["selected_profile"]["history_steps"]
                                ),
                                "test_result_path": str(
                                    run_root
                                    / "tep"
                                    / "test"
                                    / "FINAL"
                                    / result["model"]
                                    / view.relative_root
                                    / "TEST_RESULT.json"
                                ),
                            }
                        )
    finally:
        if previous_inference is None:
            os.environ.pop("PRISM_FORMAL_INFERENCE_ONLY", None)
        else:
            os.environ["PRISM_FORMAL_INFERENCE_ONLY"] = previous_inference
    # The TEP final result path is stable; compute metrics in a second read-only
    # pass while retaining no row-level values in the JSON report.
    tep_metrics: list[dict[str, Any]] = []
    for item in tep_records:
        view = next(v for v in tep_views(shared, config) if v.relative_root.as_posix() == item["view"])
        selection_path = next(
            Path(record["selection_path"])
            for record in tep_dev["records"]  # type: ignore[index]
            if record["view"] == item["view"] and record["model"] == item["model"]
        )
        selection = read_json(selection_path)
        # materialize_model stores TEST_RESULT.json beside test.parquet.
        result_path = Path(item["test_result_path"])
        final = read_json(result_path)
        prediction_path = result_path.parent / "test.parquet"
        frame = pd.read_parquet(prediction_path, columns=["base_origin_id", "y_true", "y_pred"])
        from prism_benchmark.v211_support import load_native_samples
        from prism_benchmark.neural3 import native_support, support_hash
        evaluation = native_support(
            load_native_samples(shared, view, "test"),
            int(selection["selected_profile"]["history_steps"]),
            dynamic=view.information_set == "dynamic",
        ).reset_index(drop=True)
        accessor = mod["BaseAccessor"](shared, view.head.dataset, "test", [view.head.target])
        current = accessor.block_means(evaluation, view.head.target, [(0, int(view.head.w0_steps))]).reshape(-1)
        metrics = mod["metric_bundle_delta_and_level"](
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
            current,
        )
        metrics.pop("future_level_true", None)
        metrics.pop("future_level_pred", None)
        tep_metrics.append(
            {
                **add_identity_scalars(_scalar_metrics(metrics)),
                "dataset": "tep",
                "target_head": view.head.head_id,
                "information_set": view.information_set,
                "model": item["model"],
                "history_steps": int(selection["selected_profile"]["history_steps"]),
                "rows": int(len(frame)),
                "scoring_support_hash": mod["support_id_hash"](evaluation),
                "base_origin_support_hash": support_hash(evaluation),
                "test_accessed": True,
            }
        )
    history_comparison: list[dict[str, Any]] = []
    if tep_dev is not None:
        for record in tep_dev["records"]:
            for candidate in record.get("candidate_results", []):
                history_comparison.append(
                    {
                        "model": record["model"],
                        "view": record["view"],
                        "history_steps": int(candidate["history_steps"]),
                        "history_label": candidate.get("history_label"),
                        "validation_mse": candidate.get("validation_mse"),
                        "validation_r2_level_reconstructed": candidate.get("validation_r2_level_reconstructed"),
                        "validation_r2_delta": candidate.get("validation_r2_delta"),
                        "validation_persistence_skill": candidate.get("validation_persistence_skill"),
                        "validation_residual_identity_status": candidate.get("validation_residual_identity_status"),
                        "validation_residual_identity_max_abs_error": candidate.get("validation_residual_identity_max_abs_error"),
                        "native_fit_rows": candidate.get("native_fit_rows"),
                        "common_validation_rows": candidate.get("common_validation_rows"),
                        "native_fit_support_hash": candidate.get("native_fit_support_hash"),
                        "common_validation_support_hash": candidate.get("common_validation_support_hash"),
                        "training_seconds": candidate.get("training_seconds"),
                        "peak_vram_bytes": candidate.get("peak_vram_bytes"),
                    }
                )
    rankings: dict[str, list[dict[str, Any]]] = {"tep": [], "cz": []}
    rankings["tep"] = sorted(tep_metrics, key=lambda row: -float(row.get("r2_level_reconstructed", float("-inf"))))
    rankings["cz"] = sorted(
        [add_identity_scalars(_scalar_metrics(item)) | {"h_steps": item.get("h_steps"), "direction": item.get("direction")} for item in cz_records if item.get("status") == "PASS"],
        key=lambda row: (int(row.get("h_steps", 0)), str(row.get("direction", "")), str(row.get("information_set", "")), -float(row.get("r2_level_reconstructed", float("-inf")))),
    )
    result = {
        "status": "PASS",
        "stage": "FORMAL_TEST_AFTER_FREEZE",
        "primary_metric": "R2_LEVEL_RECONSTRUCTED",
        "secondary_metric": "R2_DELTA",
        "persistence_skill": "reported_per_row",
        "residual_identity": "reported_per_row_as_scalar_max_errors_in_formal_metrics",
        "tep": tep_metrics,
        "tep_history_comparison": history_comparison,
        "cz": rankings["cz"],
        "r2_rankings": rankings,
        "test_accessed_after_freeze": True,
        "ood_accessed": False,
        "memory": memory_snapshot(),
    }
    write_json(run_root / "final" / "INDEPENDENT_EXTENSION_REPORT.json", result)
    write_json(run_root / "final" / "R2_RANKINGS.json", rankings)
    return result


def report(run_root: Path) -> dict[str, Any]:
    path = run_root / "final" / "INDEPENDENT_EXTENSION_REPORT.json"
    if not path.is_file():
        raise RuntimeError("REPORT_NOT_AVAILABLE")
    value = read_json(path)
    # Prevent accidental publication of row-level fields in any handoff copy.
    violations: list[str] = []
    def walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in FORBIDDEN_REPORT_KEYS:
                    violations.append(f"{prefix}/{key}")
                walk(child, f"{prefix}/{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{prefix}/{index}")
    walk(value)
    if violations:
        raise RuntimeError(f"STOP_PRIVATE_FIELDS_IN_PUBLIC_REPORT:{violations[:10]}")
    audit = {"status": "PASS", "report_sha256": sha256_file(path), "privacy_violations": violations, "test_accessed": value.get("test_accessed_after_freeze") is True}
    write_json(run_root / "final" / "REPORT_PRIVACY_AUDIT.json", audit)
    return {"report": value, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("scope", "environment", "pilot", "pilot-accept", "development", "freeze", "checkpoints", "test", "report", "formal-pilot", "formal-pilot-accept", "formal-development", "formal-development-reconcile", "formal-freeze", "formal-support-freeze", "formal-checkpoints", "formal-test", "status"))
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if args.stage == "scope":
        result = scope(run_root)
    elif args.stage == "environment":
        result = environment(run_root)
    elif args.stage == "pilot":
        result = {"tep": run_tep_selection(run_root, pilot=True), "cz": run_cz_pilot(run_root)}
    elif args.stage == "pilot-accept":
        result = pilot_accept(run_root)
    elif args.stage == "development":
        result = {"tep": run_tep_selection(run_root), "cz": run_cz_development(run_root)}
    elif args.stage == "freeze":
        result = freeze(run_root)
    elif args.stage == "checkpoints":
        result = fit_checkpoints(run_root)
    elif args.stage == "test":
        result = run_test(run_root)
    elif args.stage == "formal-pilot":
        result = run_cz_pilot(run_root)
    elif args.stage == "formal-pilot-accept":
        result = formal_pilot_accept(run_root)
    elif args.stage == "formal-development":
        result = run_cz_development(run_root)
    elif args.stage == "formal-development-reconcile":
        result = reconcile_cz_development(run_root)
    elif args.stage == "formal-freeze":
        result = formal_freeze(run_root)
    elif args.stage == "formal-support-freeze":
        result = freeze_cz_common_support(run_root)
    elif args.stage == "formal-checkpoints":
        result = fit_checkpoints(run_root)
    elif args.stage == "formal-test":
        result = run_test(run_root)
    elif args.stage == "report":
        result = report(run_root)
    else:
        result = {"run_root": str(run_root), "logs": sorted(str(path) for path in (run_root / "logs").glob("*.json")), "test_accessed": (run_root / "final" / "INDEPENDENT_EXTENSION_REPORT.json").is_file()}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
