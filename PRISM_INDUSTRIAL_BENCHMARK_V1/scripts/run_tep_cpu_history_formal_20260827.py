"""Formal freeze/checkpoint/test/report lifecycle for the TEP 128/256 CPU run.

Development is produced by ``run_tep_cpu_history_extension_20260826.py``.
This follow-up never reselects a model from test.  It seals the development
choices, derives test support from metadata only, fits portable checkpoints on
the train/validation L256 support, and unlocks TEP test in an inference-only
process after both seals exist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from prism_benchmark.cpu_data import SAMPLE_RUNTIME_COLUMNS, ViewSpec, sha256_file
from prism_benchmark.portable_checkpoints import (
    INFERENCE_ONLY_ENV,
    activate_inference_fit_guard,
    seal_checkpoint_tree,
    stable_hash,
)
from prism_benchmark.representative_formal import (
    _checkpoint_inventory,
    _verify_checkpoint_inventory,
)
from prism_benchmark.stage0 import write_json
from prism_benchmark.v211_public_all_baselines import (
    SupportRequirement,
    apply_common_requirements,
)
from prism_benchmark.v211_public_all_config import PublicAllPaths
from prism_benchmark.v211_support import (
    SUPPORT_COLUMNS,
    SUPPORT_CONTRACT,
    support_id_hash,
)
from prism_benchmark.v2_views import development_dynamic_views, development_input_views


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "configs" / "tep_cpu_history_extension_20260826.json"
PROTOCOL_ID = "TEP_CPU_HISTORY_EXTENSION_L256_V1"
TASK_ID = "TEP_G_REP_H1"
HEAD_ID = "TEP_G_REP_H1__H1__W2"
HISTORIES = (128, 256)
COMMON_HISTORY = 256
SELECTION_FREEZE_NAME = "TEP_SELECTION_FREEZE.json"
SUPPORT_FREEZE_NAME = "TEP_COMMON_SUPPORT_FREEZE.json"
CHECKPOINT_MANIFEST_NAME = "CHECKPOINT_MANIFEST.json"
FORMAL_TEST_NAME = "TEP_FORMAL_TEST_REPORT.json"
REPORT_NAME = "TEP_CPU_PRISM_HISTORY_EXTENSION_REPORT.json"
LEGAL_DEVELOPMENT_STATUSES = {
    "PASS",
    "SOLVER_FAILED_RETAINED",
    "NOT_RUN_PROTOCOL_INCOMPATIBLE",
    "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
    "JOINT_STABILITY_STABILITY_IMPROVED_BUT_NOT_SUPPORTED",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT.parent), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _config() -> dict[str, Any]:
    value = _read_json(CONFIG_PATH)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "active_datasets": ["tep"],
        "active_tasks": [TASK_ID],
        "active_heads": [HEAD_ID],
        "history_aware_steps": list(HISTORIES),
        "common_support_history_steps": COMMON_HISTORY,
    }
    for key, required in expected.items():
        if value.get(key) != required:
            raise RuntimeError(f"STOP_TEP_FORMAL_CONFIG_MISMATCH:{key}")
    value["config_sha256"] = sha256_file(CONFIG_PATH)
    return value


def _storage_guard(run_root: Path, stage: str) -> dict[str, Any]:
    config = _config()
    free = shutil.disk_usage(run_root.parent).free / (1024**3)
    stopline = float(config["resources"]["minimum_runtime_free_gib"])
    payload = {
        "status": "PASS" if free >= stopline else "STOP_LOW_STORAGE",
        "stage": stage,
        "available_gib": free,
        "stopline_gib": stopline,
        "created_utc": _utc(),
    }
    write_json(run_root / "logs" / f"STORAGE_FORMAL_{stage.upper()}.json", payload)
    if free < stopline:
        raise RuntimeError(f"STOP_LOW_STORAGE:{free:.3f}<{stopline:.3f}")
    return payload


def _tep_views(shared: Path) -> tuple[list[ViewSpec], list[ViewSpec]]:
    inputs = [
        view
        for view in development_input_views(shared)
        if view.head.dataset == "tep" and view.head.task_id == TASK_ID
    ]
    dynamics = [
        view
        for view in development_dynamic_views(shared)
        if view.head.dataset == "tep" and view.head.task_id == TASK_ID
    ]
    if len(inputs) != 1 or len(dynamics) != 2:
        raise RuntimeError(
            f"STOP_TEP_FORMAL_VIEW_CARDINALITY:{len(inputs)}:{len(dynamics)}"
        )
    return inputs, dynamics


def _require_development(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scope_path = run_root / "logs" / "SCOPE.json"
    pilot_path = run_root / "logs" / "PILOT.json"
    scope = _read_json(scope_path)
    pilot = _read_json(pilot_path)
    if scope.get("status") != "PASS" or pilot.get("status") != "PASS":
        raise RuntimeError("STOP_TEP_FORMAL_SCOPE_OR_PILOT_NOT_PASS")
    if scope.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise RuntimeError("STOP_TEP_FORMAL_CONFIG_CHANGED_AFTER_DEVELOPMENT")
    if pilot.get("history_steps_observed") != list(HISTORIES):
        raise RuntimeError("STOP_TEP_FORMAL_PILOT_HISTORY_COVERAGE")
    if scope.get("test_accessed") is not False or pilot.get("test_accessed") is not False:
        raise RuntimeError("STOP_TEP_FORMAL_TEST_ACCESSED_BEFORE_FREEZE")
    for stage in ("K", "C", "W", "A", "JOINT", "BASELINES"):
        summary = _read_json(
            run_root / "results" / "DEVELOPMENT" / stage / "SUMMARY.json"
        )
        if summary.get("status") != "PASS":
            raise RuntimeError(f"STOP_TEP_FORMAL_DEVELOPMENT_NOT_PASS:{stage}")
        if summary.get("test_accessed") is not False:
            raise RuntimeError(f"STOP_TEP_FORMAL_TEST_ACCESSED:{stage}")
    return scope, pilot


def _selected_history(value: Mapping[str, Any]) -> int | None:
    for key in ("selected_profile_history_steps", "selected_fit_history_steps"):
        raw = value.get(key)
        if raw is not None:
            return int(raw)
    selection = value.get("selection")
    if isinstance(selection, Mapping):
        raw = selection.get("selected_history")
        if raw is not None:
            return int(raw)
        profile = selection.get("selected_profile")
        if isinstance(profile, (list, tuple)) and len(profile) >= 2:
            return int(profile[1])
    contract = value.get("a_contract")
    if isinstance(contract, Mapping):
        profile = contract.get("profile")
        if isinstance(profile, (list, tuple)) and len(profile) >= 2:
            return int(profile[1])
    profile = value.get("ar_profile")
    if isinstance(profile, (list, tuple)) and len(profile) >= 2:
        return int(profile[1])
    return None


def _development_inventory(run_root: Path) -> list[dict[str, Any]]:
    roots = (
        run_root / "results" / "DEVELOPMENT",
        run_root / "results" / "BASELINE_DEVELOPMENT",
    )
    records: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.rglob("RESULT.json")):
            value = _read_json(path)
            status = str(value.get("status"))
            if status not in LEGAL_DEVELOPMENT_STATUSES:
                raise RuntimeError(f"STOP_TEP_FORMAL_RESULT_STATUS:{path}:{status}")
            if value.get("test_accessed") is True or value.get("ood_accessed") is True:
                raise RuntimeError(f"STOP_TEP_FORMAL_PREMATURE_ACCESS:{path}")
            history = _selected_history(value)
            if history is not None and history not in HISTORIES:
                raise RuntimeError(f"STOP_TEP_FORMAL_SELECTED_HISTORY:{path}:{history}")
            records.append(
                {
                    "path": path.relative_to(run_root).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "status": status,
                    "stage": value.get("stage"),
                    "model": value.get("model"),
                    "channel": value.get("channel"),
                    "information_set": value.get("information_set"),
                    "availability_scenario": value.get("availability_scenario"),
                    "proxy_policy": value.get("proxy_policy"),
                    "selected_history_steps": history,
                    "elapsed_seconds": value.get("elapsed_seconds"),
                    "validation_r2_delta": value.get("r2"),
                    "validation_rows": value.get("rows"),
                }
            )
    if not records:
        raise RuntimeError("STOP_TEP_FORMAL_NO_DEVELOPMENT_RESULTS")
    return records


def freeze_selection(run_root: Path) -> dict[str, Any]:
    destination = run_root / "freeze" / SELECTION_FREEZE_NAME
    if destination.exists():
        raise RuntimeError(f"REFUSING_TO_OVERWRITE_SELECTION_FREEZE:{destination}")
    scope, pilot = _require_development(run_root)
    inventory = _development_inventory(run_root)
    selected = [
        int(item["selected_history_steps"])
        for item in inventory
        if item["selected_history_steps"] is not None
    ]
    if not selected or not set(selected).issubset(HISTORIES):
        raise RuntimeError("STOP_TEP_FORMAL_HISTORY_SELECTION_MISSING")
    result = {
        "status": "SELECTION_FROZEN",
        "sealed": True,
        "sealed_utc": _utc(),
        "deletion_forbidden": True,
        "protocol_id": PROTOCOL_ID,
        "baseline_commit": _config()["baseline_commit"],
        "development_commit": scope["source_commit"],
        "formal_runner_commit": _git("rev-parse", "HEAD"),
        "config_sha256": sha256_file(CONFIG_PATH),
        "pilot_sha256": sha256_file(run_root / "logs" / "PILOT.json"),
        "history_candidates_steps": list(HISTORIES),
        "common_support_history_steps": COMMON_HISTORY,
        "selected_history_counts": {
            str(history): selected.count(history) for history in HISTORIES
        },
        "development_results": inventory,
        "development_results_hash": stable_hash(inventory),
        "development_result_count": len(inventory),
        "test_accessed": False,
        "test_targets_read": False,
        "ood_accessed": False,
        "pilot_manifest_hash": stable_hash(pilot),
    }
    write_json(destination, result)
    destination.chmod(0o444)
    return result


def _require_sealed(path: Path, status: str) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("status") != status or value.get("sealed") is not True:
        raise RuntimeError(f"STOP_TEP_FORMAL_SEAL_INVALID:{path}")
    return value


def _requirement(view: ViewSpec) -> SupportRequirement:
    if view.information_set == "input_only":
        return SupportRequirement(input_history_steps=COMMON_HISTORY)
    return SupportRequirement(
        input_history_steps=COMMON_HISTORY,
        target_delta_steps=1,
        target_history_steps=COMMON_HISTORY,
    )


def _metadata_support(source_shared: Path, view: ViewSpec, split: str) -> pd.DataFrame:
    columns = [
        name
        for name in SAMPLE_RUNTIME_COLUMNS
        if name != "y_true"
    ]
    columns.extend(SUPPORT_COLUMNS)
    path = source_shared / "sample_ids" / view.relative_root / f"{split}.parquet"
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    return apply_common_requirements(frame, [_requirement(view)]).reset_index(drop=True)


def freeze_common_support(source_shared: Path, run_root: Path) -> dict[str, Any]:
    selection_path = run_root / "freeze" / SELECTION_FREEZE_NAME
    selection = _require_sealed(selection_path, "SELECTION_FROZEN")
    destination = run_root / "freeze" / SUPPORT_FREEZE_NAME
    leaderboard = run_root / "freeze" / "TASK_LEADERBOARD_COMMON_SUPPORT.json"
    if destination.exists() or leaderboard.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_TEP_COMMON_SUPPORT_FREEZE")
    inputs, dynamics = _tep_views(source_shared)
    views: list[dict[str, Any]] = []
    for view in [*inputs, *dynamics]:
        requirement = _requirement(view)
        splits: dict[str, Any] = {}
        for split in ("train", "validation", "test"):
            common = _metadata_support(source_shared, view, split)
            splits[split] = {
                "rows": int(len(common)),
                "support_hash": support_id_hash(common),
                "source_rows": int(
                    pd.read_parquet(
                        source_shared
                        / "sample_ids"
                        / view.relative_root
                        / f"{split}.parquet",
                        columns=["origin"],
                    ).shape[0]
                ),
            }
        views.append(
            {
                "target_head": view.head.head_id,
                "dataset": "tep",
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "requirements": [requirement.to_json()],
                "splits": splits,
            }
        )
    support_payload = {
        "status": "PASS",
        "stage": "TEP_L256_TASK_LEADERBOARD_COMMON_SUPPORT",
        "support_contract": SUPPORT_CONTRACT,
        "views": views,
        "test_metadata_read_after_selection_freeze": True,
        "test_y_read": False,
        "test_accessed": False,
        "ood_y_read": False,
        "ood_accessed": False,
    }
    write_json(leaderboard, support_payload)
    result = {
        "status": "COMMON_SUPPORT_FROZEN",
        "sealed": True,
        "sealed_utc": _utc(),
        "deletion_forbidden": True,
        "protocol_id": PROTOCOL_ID,
        "selection_freeze_sha256": sha256_file(selection_path),
        "selection_results_hash": selection["development_results_hash"],
        "leaderboard_support_sha256": sha256_file(leaderboard),
        "support_records_hash": stable_hash(views),
        "views": views,
        "test_metadata_read_after_selection_freeze": True,
        "test_targets_read": False,
        "formal_test_accessed": False,
        "ood_accessed": False,
    }
    write_json(destination, result)
    leaderboard.chmod(0o444)
    destination.chmod(0o444)
    return result


def fit_checkpoints(run_root: Path) -> dict[str, Any]:
    from prism_benchmark.representative_baseline_checkpoints import (
        fit_baseline_checkpoints_for_view,
        verify_baseline_checkpoint_reload,
    )
    from prism_benchmark.representative_prism_checkpoints import (
        fit_prism_checkpoint_for_view,
        verify_prism_checkpoint_reload,
    )

    selection_path = run_root / "freeze" / SELECTION_FREEZE_NAME
    support_path = run_root / "freeze" / SUPPORT_FREEZE_NAME
    _require_sealed(selection_path, "SELECTION_FROZEN")
    _require_sealed(support_path, "COMMON_SUPPORT_FROZEN")
    checkpoint_root = run_root / "checkpoints" / "tep"
    if checkpoint_root.exists():
        raise RuntimeError(f"REFUSING_EXISTING_CHECKPOINT_ROOT:{checkpoint_root}")
    checkpoint_root.mkdir(parents=True)
    shared = run_root / "shared_l256_development"
    paths = PublicAllPaths(PROJECT, shared, run_root)
    inputs, dynamics = _tep_views(shared)
    records: list[dict[str, Any]] = []
    for view in [*inputs, *dynamics]:
        _storage_guard(run_root, "checkpoint_view")
        records.append(fit_prism_checkpoint_for_view(paths, view, checkpoint_root))
        records.extend(fit_baseline_checkpoints_for_view(paths, view, checkpoint_root))
    previous = os.environ.get(INFERENCE_ONLY_ENV)
    os.environ[INFERENCE_ONLY_ENV] = "1"
    try:
        replay: list[dict[str, Any]] = []
        for state_path in sorted(checkpoint_root.rglob("checkpoint.json")):
            state = _read_json(state_path)
            replay.append(
                verify_prism_checkpoint_reload(state_path.parent)
                if state.get("family") == "PRISM"
                else verify_baseline_checkpoint_reload(state_path.parent)
            )
    finally:
        if previous is None:
            os.environ.pop(INFERENCE_ONLY_ENV, None)
        else:
            os.environ[INFERENCE_ONLY_ENV] = previous
    entries = _checkpoint_inventory(checkpoint_root)
    if not entries:
        raise RuntimeError("STOP_TEP_FORMAL_NO_CHECKPOINTS")
    result = {
        "status": "CHECKPOINTS_SEALED",
        "sealed": True,
        "sealed_utc": _utc(),
        "deletion_forbidden": True,
        "protocol_id": PROTOCOL_ID,
        "selection_freeze_sha256": sha256_file(selection_path),
        "support_freeze_sha256": sha256_file(support_path),
        "entries": entries,
        "entry_count": len(entries),
        "fit_records": records,
        "reload_prediction_replay": replay,
        "reload_prediction_replay_status": "PASS",
        "test_accessed": False,
        "ood_accessed": False,
    }
    destination = run_root / "freeze" / CHECKPOINT_MANIFEST_NAME
    if destination.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_CHECKPOINT_MANIFEST")
    write_json(destination, result)
    destination.chmod(0o444)
    seal_checkpoint_tree(checkpoint_root)
    _verify_checkpoint_inventory(checkpoint_root, entries)
    return result


def _require_checkpoints(run_root: Path) -> dict[str, Any]:
    value = _require_sealed(
        run_root / "freeze" / CHECKPOINT_MANIFEST_NAME, "CHECKPOINTS_SEALED"
    )
    _verify_checkpoint_inventory(run_root / "checkpoints" / "tep", value["entries"])
    return value


def _build_test_shared(source_shared: Path, run_root: Path) -> tuple[Path, dict[str, Any]]:
    destination = run_root / "shared_l256_formal"
    if destination.exists():
        raise RuntimeError(f"REFUSING_EXISTING_FORMAL_SHARED:{destination}")
    development = run_root / "shared_l256_development"
    destination.mkdir(parents=True)
    for child in sorted(development.iterdir()):
        if child.name in {"sample_ids", "base_data"}:
            continue
        os.symlink(str(child.resolve()), str(destination / child.name), target_is_directory=child.is_dir())
    base = destination / "base_data" / "tep"
    base.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        source = source_shared / "base_data" / "tep" / f"{split}.parquet"
        os.symlink(str(source.resolve()), str(base / source.name))
    source_inputs, source_dynamics = _tep_views(source_shared)
    dev_inputs, dev_dynamics = _tep_views(development)
    dev_by_key = {
        (view.information_set, view.availability_scenario, view.proxy_policy): view
        for view in [*dev_inputs, *dev_dynamics]
    }
    records: list[dict[str, Any]] = []
    support_freeze = _require_sealed(
        run_root / "freeze" / SUPPORT_FREEZE_NAME, "COMMON_SUPPORT_FROZEN"
    )
    frozen = {
        (item["information_set"], item["availability_scenario"], item["proxy_policy"]): item
        for item in support_freeze["views"]
    }
    for view in [*source_inputs, *source_dynamics]:
        key = (view.information_set, view.availability_scenario, view.proxy_policy)
        target = destination / "sample_ids" / view.relative_root
        target.mkdir(parents=True, exist_ok=True)
        dev_view = dev_by_key[key]
        for split in ("train", "validation"):
            source = development / "sample_ids" / dev_view.relative_root / f"{split}.parquet"
            os.symlink(str(source.resolve()), str(target / source.name))
        source_test = source_shared / "sample_ids" / view.relative_root / "test.parquet"
        test = pd.read_parquet(source_test)
        common = apply_common_requirements(test, [_requirement(view)]).reset_index(drop=True)
        expected = frozen[key]["splits"]["test"]
        if len(common) != int(expected["rows"]) or support_id_hash(common) != expected["support_hash"]:
            raise RuntimeError(f"STOP_TEP_FORMAL_TEST_SUPPORT_DRIFT:{view.relative_root}")
        output = target / "test.parquet"
        common.to_parquet(output, index=False, compression="zstd")
        records.append(
            {
                "view": view.relative_root.as_posix(),
                "rows": int(len(common)),
                "support_hash": support_id_hash(common),
                "source_sha256": sha256_file(source_test),
                "path": output.relative_to(run_root).as_posix(),
                "sha256": sha256_file(output),
            }
        )
    audit = {
        "status": "PASS",
        "stage": "TEP_FORMAL_TEST_UNLOCK",
        "created_utc": _utc(),
        "test_accessed_after_selection_and_checkpoint_seals": True,
        "selection_freeze_sha256": sha256_file(run_root / "freeze" / SELECTION_FREEZE_NAME),
        "checkpoint_manifest_sha256": sha256_file(run_root / "freeze" / CHECKPOINT_MANIFEST_NAME),
        "records": records,
        "ood_accessed": False,
    }
    write_json(run_root / "logs" / "TEP_TEST_UNLOCK_AUDIT.json", audit)
    return destination, audit


def _support_acceptance(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("status") != "PASS":
            continue
        key = (
            str(record["information_set"]),
            str(record["availability_scenario"]),
            str(record["proxy_policy"]),
        )
        groups.setdefault(key, []).append(record)
    audits = []
    for key, items in groups.items():
        rows = {int(item["rows"]) for item in items}
        hashes = {str(item["scoring_support_hash"]) for item in items}
        if len(rows) != 1 or len(hashes) != 1:
            raise RuntimeError(f"STOP_TEP_FORMAL_METHOD_SUPPORT_MISMATCH:{key}")
        audits.append(
            {
                "information_set": key[0],
                "availability_scenario": key[1],
                "proxy_policy": key[2],
                "rows": next(iter(rows)),
                "support_hash": next(iter(hashes)),
                "methods": len(items),
                "status": "PASS",
            }
        )
    return audits


def run_test(source_shared: Path, run_root: Path) -> dict[str, Any]:
    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError("STOP_TEP_FORMAL_TEST_REQUIRES_INFERENCE_ONLY_PROCESS")
    _require_sealed(run_root / "freeze" / SELECTION_FREEZE_NAME, "SELECTION_FROZEN")
    _require_sealed(run_root / "freeze" / SUPPORT_FREEZE_NAME, "COMMON_SUPPORT_FROZEN")
    checkpoint_manifest = _require_checkpoints(run_root)
    destination = run_root / "final" / FORMAL_TEST_NAME
    if destination.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_TEP_FORMAL_TEST")
    _storage_guard(run_root, "test_unlock")
    formal_shared, unlock = _build_test_shared(source_shared, run_root)
    activate_inference_fit_guard()
    from prism_benchmark.representative_baseline_checkpoints import (
        predict_baseline_checkpoints_for_view,
    )
    from prism_benchmark.representative_prism_checkpoints import (
        predict_prism_checkpoint_for_view,
    )

    paths = PublicAllPaths(PROJECT, formal_shared, run_root)
    inputs, dynamics = _tep_views(formal_shared)
    checkpoint_root = run_root / "checkpoints" / "tep"
    records: list[dict[str, Any]] = []
    for view in [*inputs, *dynamics]:
        _storage_guard(run_root, "test_view")
        records.extend(predict_prism_checkpoint_for_view(paths, view, checkpoint_root, split="test"))
        records.extend(predict_baseline_checkpoints_for_view(paths, view, checkpoint_root, split="test"))
    _verify_checkpoint_inventory(checkpoint_root, checkpoint_manifest["entries"])
    passed = [item for item in records if item.get("status") == "PASS"]
    if not passed or any(item.get("fit_called_in_inference") is not False for item in passed):
        raise RuntimeError("STOP_TEP_FORMAL_INFERENCE_AUDIT_FAILED")
    identity_max = max(
        max(
            abs(float(item["mse"]) - float(item["mse_delta"])),
            abs(float(item["rmse"]) - float(item["rmse_delta"])),
            abs(float(item["mae"]) - float(item["mae_delta"])),
        )
        for item in passed
    )
    if identity_max > 1e-10:
        raise RuntimeError(f"STOP_TEP_FORMAL_RESIDUAL_IDENTITY:{identity_max}")
    result = {
        "status": "PASS",
        "stage": "TEP_FORMAL_TEST_INFERENCE_COMPLETE",
        "created_utc": _utc(),
        "protocol_id": PROTOCOL_ID,
        "primary_metric": "R2_LEVEL_RECONSTRUCTED",
        "secondary_metric": "R2_DELTA",
        "prediction_horizon_minutes": 3,
        "history_candidates_steps": list(HISTORIES),
        "common_support_history_steps": COMMON_HISTORY,
        "methods": records,
        "support_acceptance": _support_acceptance(records),
        "residual_identity_status": "PASS",
        "residual_identity_max_abs_error": identity_max,
        "test_unlock_audit": unlock,
        "test_accessed_after_selection_and_checkpoint_seals": True,
        "inference_only": True,
        "ood_accessed": False,
    }
    write_json(destination, result)
    return result


WATCHDOG = re.compile(
    r"cgroup_gib=(?P<cgroup>[0-9.]+).*process_tree_rss_gib=(?P<rss>[0-9.]+).*free_gib=(?P<free>[0-9.]+)"
)


def _resource_summary(run_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((run_root / "launcher_logs").glob("*.log")):
        matches = [WATCHDOG.search(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
        values = [match for match in matches if match]
        if values:
            rows.append(
                {
                    "stage": path.stem,
                    "peak_cgroup_gib": max(float(item.group("cgroup")) for item in values),
                    "peak_process_tree_rss_gib": max(float(item.group("rss")) for item in values),
                    "minimum_free_gib": min(float(item.group("free")) for item in values),
                }
            )
    return {
        "stages": rows,
        "peak_cgroup_gib": max((item["peak_cgroup_gib"] for item in rows), default=None),
        "peak_process_tree_rss_gib": max((item["peak_process_tree_rss_gib"] for item in rows), default=None),
        "minimum_free_gib": min((item["minimum_free_gib"] for item in rows), default=None),
        "hard_memory_limit_gib": 90,
        "recommended_memory_limit_gib": 75,
        "storage_stopline_gib": 5,
    }


def build_report(source_shared: Path, run_root: Path) -> dict[str, Any]:
    del source_shared
    selection = _require_sealed(
        run_root / "freeze" / SELECTION_FREEZE_NAME, "SELECTION_FROZEN"
    )
    support = _require_sealed(
        run_root / "freeze" / SUPPORT_FREEZE_NAME, "COMMON_SUPPORT_FROZEN"
    )
    checkpoints = _require_checkpoints(run_root)
    formal = _read_json(run_root / "final" / FORMAL_TEST_NAME)
    if formal.get("status") != "PASS":
        raise RuntimeError("STOP_TEP_FORMAL_TEST_NOT_PASS")
    ranking_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in formal["methods"]:
        if item.get("status") != "PASS":
            continue
        key = (
            str(item["information_set"]),
            str(item["availability_scenario"]),
            str(item["proxy_policy"]),
        )
        ranking_groups.setdefault(key, []).append(
            {
                "model": item["model"],
                "rows": item["rows"],
                "support_hash": item["scoring_support_hash"],
                "r2_level_reconstructed": item["r2_level_reconstructed"],
                "r2_delta": item["r2_delta"],
                "persistence_skill": item["persistence_skill"],
                "mse": item["mse"],
                "elapsed_seconds": item["elapsed_seconds"],
                "checkpoint_hash": item["checkpoint_hash"],
            }
        )
    rankings = []
    for key, rows in ranking_groups.items():
        ordered = sorted(rows, key=lambda item: float(item["r2_level_reconstructed"]), reverse=True)
        rankings.append(
            {
                "information_set": key[0],
                "availability_scenario": key[1],
                "proxy_policy": key[2],
                "ranking": [{"rank": index + 1, **item} for index, item in enumerate(ordered)],
            }
        )
    development_rows = selection["development_results"]
    history_selected = [
        item for item in development_rows if item.get("selected_history_steps") is not None
    ]
    report = {
        "status": "PASS",
        "stage": "TEP_CPU_PRISM_HISTORY_EXTENSION_REPORT",
        "created_utc": _utc(),
        "repository": "Mountainatic/AR_RAPHU",
        "baseline_commit": _config()["baseline_commit"],
        "development_commit": selection["development_commit"],
        "formal_runner_commit": _git("rev-parse", "HEAD"),
        "protocol_id": PROTOCOL_ID,
        "config_sha256": sha256_file(CONFIG_PATH),
        "selection_freeze_sha256": sha256_file(run_root / "freeze" / SELECTION_FREEZE_NAME),
        "common_support_freeze_sha256": sha256_file(run_root / "freeze" / SUPPORT_FREEZE_NAME),
        "support_records_hash": support["support_records_hash"],
        "checkpoint_manifest_sha256": sha256_file(run_root / "freeze" / CHECKPOINT_MANIFEST_NAME),
        "checkpoint_entry_count": checkpoints["entry_count"],
        "experiment_matrix": {
            "dataset": "TEP",
            "task": TASK_ID,
            "target_head": HEAD_ID,
            "prediction_horizon_minutes": 3,
            "history_candidates_steps": list(HISTORIES),
            "common_support_history_steps": COMMON_HISTORY,
            "information_sets": ["input_only", "dynamic"],
            "methods": "FORMAL_CPU_PRISM_AND_REGISTERED_CPU_BASELINES_ONLY",
            "neural3": "NOT_RUN_BY_USER_SCOPE",
        },
        "history_selection": {
            "selected_history_counts": selection["selected_history_counts"],
            "records": history_selected,
            "sample_loss_contract": "BOTH_128_AND_256_SCORED_ON_IDENTICAL_L256_TARGET_ROWS",
        },
        "rankings": rankings,
        "support_acceptance": formal["support_acceptance"],
        "residual_identity_status": formal["residual_identity_status"],
        "residual_identity_max_abs_error": formal["residual_identity_max_abs_error"],
        "resources": _resource_summary(run_root),
        "test_accessed_after_selection_and_checkpoint_seals": True,
        "test_reselection": False,
        "ood_accessed": False,
        "reproduction_commands": [
            "python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/launch_tep_cpu_history_extension_20260826.py --shared <formal-shared> --run-root <run-root> --pilot-only",
            "python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/launch_tep_cpu_history_extension_20260826.py --shared <formal-shared> --run-root <run-root> --skip-pilot --pilot-evidence <run-root>/logs/PILOT.json",
            "python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/run_tep_cpu_history_formal_20260827.py freeze --shared <formal-shared> --run-root <run-root>",
            "python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/run_tep_cpu_history_formal_20260827.py support-freeze --shared <formal-shared> --run-root <run-root>",
            "python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/run_tep_cpu_history_formal_20260827.py checkpoints --shared <formal-shared> --run-root <run-root>",
            "PRISM_FORMAL_INFERENCE_ONLY=1 python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/run_tep_cpu_history_formal_20260827.py test --shared <formal-shared> --run-root <run-root>",
            "python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/run_tep_cpu_history_formal_20260827.py report --shared <formal-shared> --run-root <run-root>",
        ],
    }
    destination = run_root / "final" / REPORT_NAME
    if destination.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_TEP_FORMAL_REPORT")
    write_json(destination, report)
    privacy = {
        "status": "PASS",
        "stage": "TEP_REPORT_PRIVACY_AUDIT",
        "aggregate_only": True,
        "contains_cz_data": False,
        "contains_row_level_true_sequences": False,
        "contains_sample_identifiers": False,
        "report_sha256": sha256_file(destination),
    }
    write_json(run_root / "final" / "TEP_REPORT_PRIVACY_AUDIT.json", privacy)
    return report


def status(run_root: Path) -> dict[str, Any]:
    paths = {
        "selection": run_root / "freeze" / SELECTION_FREEZE_NAME,
        "support": run_root / "freeze" / SUPPORT_FREEZE_NAME,
        "checkpoints": run_root / "freeze" / CHECKPOINT_MANIFEST_NAME,
        "test": run_root / "final" / FORMAL_TEST_NAME,
        "report": run_root / "final" / REPORT_NAME,
    }
    return {
        "run_root": str(run_root),
        "artifacts": {
            name: None if not path.is_file() else _read_json(path).get("status")
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("freeze", "support-freeze", "checkpoints", "test", "report", "status"),
    )
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    source_shared = args.shared.resolve()
    run_root = args.run_root.resolve()
    if args.stage != "status":
        _storage_guard(run_root, args.stage)
    if args.stage == "freeze":
        result = freeze_selection(run_root)
    elif args.stage == "support-freeze":
        result = freeze_common_support(source_shared, run_root)
    elif args.stage == "checkpoints":
        result = fit_checkpoints(run_root)
    elif args.stage == "test":
        result = run_test(source_shared, run_root)
    elif args.stage == "report":
        result = build_report(source_shared, run_root)
    else:
        result = status(run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
