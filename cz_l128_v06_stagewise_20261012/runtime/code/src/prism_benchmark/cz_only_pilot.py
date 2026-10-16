"""Isolated CZ-only preview with the same freeze/checkpoint/test guarantees as formal."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .cpu_data import sha256_file
from .cz_l256_nowcast import (
    DIRECTIONS,
    TASK_ID as CZ_TASK_ID,
    materialize_target_direction,
    view as cz_view,
)
from .portable_checkpoints import (
    INFERENCE_ONLY_ENV,
    activate_inference_fit_guard,
    seal_checkpoint_tree,
)
from .representative_baseline_checkpoints import (
    fit_baseline_checkpoints_for_view,
    predict_baseline_checkpoints_for_view,
    verify_baseline_checkpoint_reload,
)
from .representative_formal import (
    CHECKPOINT_MANIFEST_NAME,
    GLOBAL_FREEZE_NAME,
    _assert_no_out_of_scope_artifacts,
    _checkpoint_inventory,
    _cz_direction_paths,
    _git_commit,
    _manifest_files,
    _rankings,
    _read_json,
    _selection_records,
    _stable_json_hash,
    _support_acceptance,
    _utc,
    _verify_checkpoint_inventory,
    build_common_support_for_views,
    checkpoint_namespace_root,
    free_gib,
    require_checkpoint_manifest,
    require_global_freeze,
)
from .representative_prism_checkpoints import (
    fit_prism_checkpoint_for_view,
    predict_prism_checkpoint_for_view,
    verify_prism_checkpoint_reload,
)
from .stage0 import write_json
from .v211_representative_stage1_config import (
    RESERVED_DATASETS,
    load_representative_stage1_descriptor,
)


PILOT_PROTOCOL_ID = "CZ_ONLY_L256_DELTA_NOWCAST_PILOT_V1"
PILOT_SCOPE_CLASS = "CZ_ONLY_PREVIEW_EXCLUDED_FROM_FORMAL_THREE_DATASET_SELECTION"
PILOT_ACTIVE_DATASETS = ("cz_czochralski",)


def pilot_path_views(project: Path, run_root: Path) -> list[tuple[str, Any, list[Any]]]:
    return [
        (
            f"cz:{direction}",
            _cz_direction_paths(project, run_root, direction),
            [cz_view("input_only"), cz_view("dynamic")],
        )
        for direction in DIRECTIONS
    ]


def write_cz_pilot_scope(project: Path, run_root: Path) -> dict[str, Any]:
    load_representative_stage1_descriptor(project)
    result = {
        "status": "PASS",
        "protocol_id": PILOT_PROTOCOL_ID,
        "scope_class": PILOT_SCOPE_CLASS,
        "source_commit": _git_commit(project),
        "active_datasets": list(PILOT_ACTIVE_DATASETS),
        "primary_tasks": [CZ_TASK_ID],
        "directions": list(DIRECTIONS),
        "information_sets": ["input_only", "dynamic"],
        "reserved_datasets": dict(RESERVED_DATASETS),
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "stage2_status": "NOT_RUN_BY_USER_SCOPE",
        "excluded_from_full_formal_selection": True,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(run_root / "logs" / "CZ_PILOT_SCOPE.json", result)
    return result


def create_cz_pilot_selection_freeze(project: Path, run_root: Path) -> dict[str, Any]:
    status = _read_json(run_root / "logs" / "CZ_DEVELOPMENT_ACCEPTANCE.json")
    if status.get("status") != "PASS":
        raise RuntimeError("STOP_CZ_PILOT_DEVELOPMENT_NOT_COMPLETE")
    if status.get("test_accessed") is not False or status.get("ood_accessed") is not False:
        raise RuntimeError("STOP_CZ_PILOT_TARGET_ACCESSED_BEFORE_FREEZE")
    for _, paths, views in pilot_path_views(project, run_root):
        build_common_support_for_views(paths, views)
    manifest = _manifest_files(
        run_root, ("RESULT.json", "SUMMARY.json", "*ACCEPTANCE*.json")
    )
    if not manifest:
        raise RuntimeError("STOP_CZ_PILOT_NO_DEVELOPMENT_SELECTION_ARTIFACTS")
    descriptor = load_representative_stage1_descriptor(project)
    selections = _selection_records(run_root)
    result = {
        "status": "GLOBAL_SELECTION_FROZEN",
        "sealed": True,
        "sealed_utc": _utc(),
        "protocol_id": PILOT_PROTOCOL_ID,
        "scope_class": PILOT_SCOPE_CLASS,
        "source_commit": _git_commit(project),
        "protocol_config_sha256": descriptor["config_sha256"],
        "active_datasets": list(PILOT_ACTIVE_DATASETS),
        "primary_tasks": [CZ_TASK_ID],
        "directions": list(DIRECTIONS),
        "reserved_datasets": dict(RESERVED_DATASETS),
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "stage2_status": "NOT_RUN_BY_USER_SCOPE",
        "excluded_from_full_formal_selection": True,
        "development_artifacts": manifest,
        "development_manifest_sha256": _stable_json_hash(manifest),
        "selection_records": selections,
        "selection_records_sha256": _stable_json_hash(selections),
        "gate_retained_records": [
            item
            for item in selections
            if item["status"]
            in {
                "PASS",
                "FAILED_RETAINED",
                "NOT_RUN_PROTOCOL_INCOMPATIBLE",
                "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
            }
        ],
        "test_accessed": False,
        "ood_accessed": False,
    }
    destination = run_root / "freeze" / GLOBAL_FREEZE_NAME
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite CZ pilot freeze: {destination}")
    write_json(destination, result)
    destination.chmod(0o444)
    return result


def fit_and_seal_cz_pilot_checkpoints(project: Path, run_root: Path) -> dict[str, Any]:
    freeze = require_global_freeze(run_root)
    if freeze.get("protocol_id") != PILOT_PROTOCOL_ID:
        raise RuntimeError("STOP_CZ_PILOT_FREEZE_PROTOCOL_MISMATCH")
    checkpoint_root = run_root / "checkpoints"
    if checkpoint_root.exists() and any(checkpoint_root.iterdir()):
        raise RuntimeError(f"refusing existing CZ pilot checkpoint root: {checkpoint_root}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    descriptor = load_representative_stage1_descriptor(project)
    records: list[dict[str, Any]] = []
    for namespace, paths, views in pilot_path_views(project, run_root):
        namespace_root = checkpoint_namespace_root(checkpoint_root, namespace)
        if free_gib(run_root.parent) < float(descriptor["minimum_runtime_free_gib"]):
            raise RuntimeError("STOP_LOW_STORAGE_BEFORE_NEXT_CZ_PILOT_CHECKPOINT_VIEW")
        for candidate in views:
            records.append(
                {
                    "namespace": namespace,
                    **fit_prism_checkpoint_for_view(paths, candidate, namespace_root),
                }
            )
            records.extend(
                {"namespace": namespace, **item}
                for item in fit_baseline_checkpoints_for_view(
                    paths, candidate, namespace_root
                )
            )
    previous = os.environ.get(INFERENCE_ONLY_ENV)
    os.environ[INFERENCE_ONLY_ENV] = "1"
    try:
        replay = []
        for path in sorted(checkpoint_root.rglob("checkpoint.json")):
            checkpoint = path.parent
            state = _read_json(path)
            replay.append(
                verify_prism_checkpoint_reload(checkpoint)
                if state.get("family") == "PRISM"
                else verify_baseline_checkpoint_reload(checkpoint)
            )
    finally:
        if previous is None:
            os.environ.pop(INFERENCE_ONLY_ENV, None)
        else:
            os.environ[INFERENCE_ONLY_ENV] = previous
    entries = _checkpoint_inventory(checkpoint_root)
    if not entries:
        raise RuntimeError("STOP_NO_CZ_PILOT_CHECKPOINTS_CREATED")
    result = {
        "status": "CHECKPOINTS_SEALED",
        "sealed": True,
        "sealed_utc": _utc(),
        "deletion_forbidden": True,
        "protocol_id": PILOT_PROTOCOL_ID,
        "scope_class": PILOT_SCOPE_CLASS,
        "source_commit": _git_commit(project),
        "global_selection_freeze_sha256": sha256_file(
            run_root / "freeze" / GLOBAL_FREEZE_NAME
        ),
        "global_selection_freeze_manifest_hash": freeze[
            "development_manifest_sha256"
        ],
        "protocol_config_sha256": descriptor["config_sha256"],
        "entries": entries,
        "entry_count": len(entries),
        "fit_records": records,
        "reload_prediction_replay": replay,
        "reload_prediction_replay_status": "PASS",
        "excluded_from_full_formal_selection": True,
        "test_accessed": False,
        "ood_accessed": False,
    }
    destination = run_root / "freeze" / CHECKPOINT_MANIFEST_NAME
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite CZ pilot checkpoint manifest: {destination}")
    write_json(destination, result)
    destination.chmod(0o444)
    seal_checkpoint_tree(checkpoint_root)
    _verify_checkpoint_inventory(checkpoint_root, entries)
    return result


def run_cz_pilot_test_inference(
    *, project: Path, run_root: Path, raw_cz: Path
) -> dict[str, Any]:
    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError("STOP_CZ_PILOT_TEST_MUST_RUN_IN_INFERENCE_ONLY_PROCESS")
    activate_inference_fit_guard()
    freeze = require_global_freeze(run_root)
    manifest = require_checkpoint_manifest(run_root)
    if freeze.get("protocol_id") != PILOT_PROTOCOL_ID:
        raise RuntimeError("STOP_CZ_PILOT_FREEZE_PROTOCOL_MISMATCH")
    if manifest.get("protocol_id") != PILOT_PROTOCOL_ID:
        raise RuntimeError("STOP_CZ_PILOT_CHECKPOINT_PROTOCOL_MISMATCH")
    descriptor = load_representative_stage1_descriptor(project)
    if free_gib(run_root.parent) < float(descriptor["minimum_runtime_free_gib"]):
        raise RuntimeError("STOP_LOW_STORAGE_BEFORE_CZ_PILOT_TARGET_UNLOCK")
    target_audits = [
        materialize_target_direction(
            raw_cz,
            run_root / "cz" / "shared",
            direction,
            global_freeze_path=run_root / "freeze" / GLOBAL_FREEZE_NAME,
            checkpoint_manifest_path=run_root / "freeze" / CHECKPOINT_MANIFEST_NAME,
        )
        for direction in DIRECTIONS
    ]
    records: list[dict[str, Any]] = []
    for namespace, paths, views in pilot_path_views(project, run_root):
        namespace_root = checkpoint_namespace_root(
            run_root / "checkpoints", namespace
        )
        if free_gib(run_root.parent) < float(descriptor["minimum_runtime_free_gib"]):
            raise RuntimeError("STOP_LOW_STORAGE_DURING_CZ_PILOT_TEST_INFERENCE")
        for candidate in views:
            records.extend(
                {"namespace": namespace, **item}
                for item in predict_prism_checkpoint_for_view(
                    paths, candidate, namespace_root, split="test"
                )
            )
            records.extend(
                {"namespace": namespace, **item}
                for item in predict_baseline_checkpoints_for_view(
                    paths, candidate, namespace_root, split="test"
                )
            )
    support = _support_acceptance(records)
    _verify_checkpoint_inventory(run_root / "checkpoints", manifest["entries"])
    _assert_no_out_of_scope_artifacts(run_root)
    rankings = _rankings(records)
    report = {
        "status": "PASS",
        "stage": "CZ_ONLY_PILOT_TEST_INFERENCE_COMPLETE",
        "protocol_id": PILOT_PROTOCOL_ID,
        "scope_class": PILOT_SCOPE_CLASS,
        "primary_metric": "R2_LEVEL_RECONSTRUCTED",
        "secondary_metric": "R2_DELTA",
        "active_datasets": list(PILOT_ACTIVE_DATASETS),
        "primary_tasks": [CZ_TASK_ID],
        "reserved_datasets": dict(RESERVED_DATASETS),
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "stage2_status": "NOT_RUN_BY_USER_SCOPE",
        "excluded_from_full_formal_selection": True,
        "target_rod_access_audits": target_audits,
        "methods": records,
        "support_acceptance": support,
        "rankings": rankings,
        "checkpoint_entry_count": manifest["entry_count"],
        "checkpoints_unchanged_after_test": True,
        "fit_refit_select_called_in_test": False,
        "test_accessed_after_global_freeze_and_checkpoints": True,
        "ood_accessed": False,
    }
    write_json(run_root / "final" / "CZ_ONLY_PILOT_LEVEL_DELTA_REPORT.json", report)
    write_json(run_root / "final" / "INPUT_ONLY_LEADERBOARD.json", rankings["input_only"])
    write_json(run_root / "final" / "DYNAMIC_LEADERBOARD.json", rankings["dynamic"])
    acceptance = {
        "status": "PASS",
        "protocol_id": PILOT_PROTOCOL_ID,
        "scope_class": PILOT_SCOPE_CLASS,
        "pilot_datasets": list(PILOT_ACTIVE_DATASETS),
        "pilot_dataset_count": 1,
        "pilot_directions": list(DIRECTIONS),
        "excluded_from_full_formal_selection": True,
        "neural3_artifacts_created": False,
        "reserved_datasets_accessed": False,
        "stage2_accessed": False,
        "first_target_access_after_freeze": all(
            item.get("target_rod_first_access_after_freeze") is True
            for item in target_audits
        ),
        "all_predictions_checkpoint_traceable": all(
            item.get("checkpoint_hash")
            for item in records
            if item.get("status") == "PASS"
        ),
        "checkpoint_hash_and_mtime_unchanged": True,
        "level_delta_identity_tolerance": 1e-10,
        "support_acceptance_status": "PASS",
    }
    write_json(run_root / "final" / "CZ_ONLY_PILOT_ACCEPTANCE.json", acceptance)
    return acceptance
