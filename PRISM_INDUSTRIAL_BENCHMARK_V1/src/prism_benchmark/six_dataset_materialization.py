from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .cpu_data import ViewSpec, sha256_file
from .neural3 import MODEL_FAMILIES, materialize_model
from .six_dataset_extension import (
    DIRECTIONS,
    cz_paths,
    cz_view,
    extension_views,
    public5_root,
)
from .stage0 import write_json
from .v211_public_all_baseline_materialization import materialize_baseline_view
from .v211_public_all_closure import LEGAL_JOINT_DEVELOPMENT_STATUSES
from .v211_public_all_materialization import (
    materialize_dynamic_prism_view,
    materialize_input_prism_view,
    preflight_public_all_materialization,
)
from .v211_support import SUPPORT_CONTRACT


ACCESS_AUDIT_NAME = "SIX_DATASET_TEST_OOD_ACCESS_AUDIT.json"
FAILURE_AUDIT_NAME = "LOCKBOX_ACCESSED_RUNTIME_FAILURE.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _selection_root(
    run_root: Path,
    scope: str,
    direction: str | None,
) -> Path:
    root = run_root / "results" / "NEURAL3"
    if scope == "public5":
        return root / "PUBLIC5"
    if direction is None:
        raise ValueError("CZ neural selection needs a direction")
    return root / "CZ" / direction


def _selection_path(
    run_root: Path,
    scope: str,
    direction: str | None,
    model: str,
    view: ViewSpec,
) -> Path:
    return (
        _selection_root(run_root, scope, direction)
        / model
        / view.relative_root
        / "SELECTION.json"
    )


def _registered_splits(shared: Path, view: ViewSpec) -> tuple[str, ...]:
    root = shared / "sample_ids" / view.relative_root
    result = tuple(
        split
        for split in ("test", "ood")
        if (root / f"{split}.parquet").is_file()
    )
    if "test" not in result:
        raise FileNotFoundError(root / "test.parquet")
    return result


def preflight_extension_materialization(
    run_root: Path,
    project: Path,
    *,
    public_root: Path | None = None,
) -> dict[str, Any]:
    public_root = public5_root() if public_root is None else public_root.resolve()
    freeze_path = (
        run_root / "freeze" / "SIX_DATASET_CZ_NEURAL3_DEVELOPMENT_FREEZE.json"
    )
    freeze = _read_json(freeze_path)
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("final materialization requires a frozen extension")
    if freeze.get("test_accessed") is not False:
        raise RuntimeError("extension freeze already records test access")
    if freeze.get("support_contract") != SUPPORT_CONTRACT:
        raise RuntimeError("extension freeze support contract mismatch")

    common_support = (
        run_root / "freeze" / "NEURAL3_EXTENSION_COMMON_SUPPORT.json"
    )
    support = _read_json(common_support)
    if support.get("status") != "PASS":
        raise RuntimeError("extension common support is not PASS")
    if support.get("test_y_read") is not False:
        raise RuntimeError("common support construction read test targets")

    neural_records: list[dict[str, Any]] = []
    for scope, shared, view, direction in extension_views(run_root, public_root):
        splits = _registered_splits(shared, view)
        for model in MODEL_FAMILIES:
            path = _selection_path(
                run_root,
                scope,
                direction,
                model,
                view,
            )
            selection = _read_json(path)
            if selection.get("status") != "PASS":
                raise RuntimeError(f"neural selection is not PASS: {path}")
            if selection.get("support_contract") != SUPPORT_CONTRACT:
                raise RuntimeError(f"neural support contract mismatch: {path}")
            if selection.get("test_accessed") is not False:
                raise RuntimeError(f"neural selection accessed test: {path}")
            neural_records.append(
                {
                    "scope": scope,
                    "direction": direction,
                    "model": model,
                    "view": view.relative_root.as_posix(),
                    "selection_path": str(path),
                    "selection_sha256": sha256_file(path),
                    "splits": list(splits),
                }
            )

    cz_records: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        paths = cz_paths(project, run_root, direction)
        input_view = cz_view(run_root, direction, "input_only")
        dynamic_view = cz_view(run_root, direction, "dynamic")
        prism_preflight = preflight_public_all_materialization(
            paths,
            [dynamic_view],
        )
        baseline_summary_path = (
            paths.output / "BASELINE_DEVELOPMENT" / "SUMMARY.json"
        )
        baseline_summary = _read_json(baseline_summary_path)
        if baseline_summary.get("status") == "FAILED":
            raise RuntimeError(f"CZ baseline development failed: {direction}")
        joint_path = (
            paths.output
            / "DEVELOPMENT"
            / "JOINT"
            / dynamic_view.head.head_id
            / dynamic_view.availability_scenario
            / dynamic_view.proxy_policy
            / "RESULT.json"
        )
        joint = _read_json(joint_path)
        if joint.get("status") not in LEGAL_JOINT_DEVELOPMENT_STATUSES:
            raise RuntimeError(f"illegal CZ Joint status: {direction}")
        _registered_splits(paths.shared, input_view)
        _registered_splits(paths.shared, dynamic_view)
        cz_records.append(
            {
                "direction": direction,
                "prism_preflight": prism_preflight,
                "baseline_summary_path": str(baseline_summary_path),
                "baseline_summary_sha256": sha256_file(baseline_summary_path),
                "joint_status": joint.get("status"),
            }
        )

    result = {
        "status": "PASS",
        "stage": "F3_EXTENSION_MATERIALIZATION_PREFLIGHT",
        "freeze_path": str(freeze_path),
        "freeze_sha256": sha256_file(freeze_path),
        "common_support_path": str(common_support),
        "common_support_sha256": sha256_file(common_support),
        "public5_root": str(public_root),
        "public5_prism_cpu_retraining": False,
        "neural_jobs": len(neural_records),
        "neural_records": neural_records,
        "cz_records": cz_records,
        "test_y_read": False,
        "ood_y_read": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    output = run_root / "freeze" / "MATERIALIZATION_PREFLIGHT.json"
    write_json(output, result)
    return result


def _materialize_cz(
    run_root: Path,
    project: Path,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        paths = cz_paths(project, run_root, direction)
        input_view = cz_view(run_root, direction, "input_only")
        dynamic_view = cz_view(run_root, direction, "dynamic")
        for split in _registered_splits(paths.shared, input_view):
            for item in materialize_input_prism_view(
                paths,
                input_view,
                split=split,
            ):
                audits.append({"direction": direction, **item})
            for item in materialize_baseline_view(
                paths,
                input_view,
                split=split,
            ):
                audits.append({"direction": direction, **item})
        for split in _registered_splits(paths.shared, dynamic_view):
            for item in materialize_dynamic_prism_view(
                paths,
                dynamic_view,
                split=split,
            ):
                audits.append({"direction": direction, **item})
            for item in materialize_baseline_view(
                paths,
                dynamic_view,
                split=split,
            ):
                audits.append({"direction": direction, **item})
        print(
            json.dumps(
                {
                    "event": "CZ_FINAL_MATERIALIZATION_DONE",
                    "direction": direction,
                    "audits": sum(
                        item.get("direction") == direction for item in audits
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        gc.collect()
    return audits


def _materialize_neural(
    run_root: Path,
    public_root: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    jobs = extension_views(run_root, public_root)
    total = len(jobs) * len(MODEL_FAMILIES)
    completed = 0
    for scope, shared, view, direction in jobs:
        destination = _selection_root(run_root, scope, direction)
        for model in MODEL_FAMILIES:
            selection = _read_json(
                _selection_path(
                    run_root,
                    scope,
                    direction,
                    model,
                    view,
                )
            )
            for split in _registered_splits(shared, view):
                result = materialize_model(
                    shared=shared,
                    view=view,
                    selection=selection,
                    split=split,
                    output=destination,
                    device=device,
                    final_fit=True,
                )
                audits.append(
                    {
                        "scope": scope,
                        "direction": direction,
                        **result,
                    }
                )
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "NEURAL3_FINAL_MATERIALIZATION_DONE",
                        "completed": completed,
                        "total": total,
                        "scope": scope,
                        "direction": direction,
                        "model": model,
                        "view": view.relative_root.as_posix(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return audits


def materialize_extension(
    run_root: Path,
    project: Path,
    *,
    public_root: Path | None = None,
) -> dict[str, Any]:
    public_root = public5_root() if public_root is None else public_root.resolve()
    final = run_root / "final"
    final.mkdir(parents=True, exist_ok=True)
    access_path = final / ACCESS_AUDIT_NAME
    if access_path.is_file():
        previous = _read_json(access_path)
        if previous.get("status") == "PASS":
            return previous
        raise RuntimeError("extension lockbox access has already started")

    preflight = preflight_extension_materialization(
        run_root,
        project,
        public_root=public_root,
    )
    started = time.time()
    access_started: dict[str, Any] = {
        "status": "TEST_OOD_ACCESS_STARTED",
        "stage": "T1_SIX_DATASET_EXTENSION_FINAL_ACCESS",
        "first_access_timestamp": started,
        "freeze_sha256": preflight["freeze_sha256"],
        "common_support_sha256": preflight["common_support_sha256"],
        "public5_historical_predictions_reused": True,
        "public5_prism_cpu_retrained": False,
        "test_accessed": True,
        "ood_accessed": False,
        "test_y_read": True,
        "ood_y_read": False,
    }
    write_json(access_path, access_started)

    audits: list[dict[str, Any]] = []
    ood_accessed = False
    try:
        audits.extend(_materialize_cz(run_root, project))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        neural_audits = _materialize_neural(run_root, public_root, device)
        audits.extend(neural_audits)
        ood_accessed = any(
            item.get("split") == "ood" for item in audits
        )
    except Exception as error:
        failure = {
            **access_started,
            "status": "LOCKBOX_ACCESSED_RUNTIME_FAILURE",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_audits": len(audits),
            "ood_accessed": ood_accessed,
            "ood_y_read": ood_accessed,
        }
        write_json(final / FAILURE_AUDIT_NAME, failure)
        write_json(access_path, failure)
        raise

    status_counts: dict[str, int] = {}
    for item in audits:
        status = str(item.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    result = {
        **access_started,
        "status": "PASS",
        "elapsed_seconds": time.time() - started,
        "models": audits,
        "model_audits": len(audits),
        "status_counts": status_counts,
        "ood_accessed": ood_accessed,
        "ood_y_read": ood_accessed,
        "post_test_reselection": False,
        "neural_test_used_for_tuning": False,
        "cz_target_rod_used_for_selection": False,
    }
    write_json(access_path, result)
    return result
