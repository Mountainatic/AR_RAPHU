"""Formal TEP/SRU/CZ Stage-1 orchestration and freeze guards."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .cpu_data import sha256_file
from .cz_baselines import run_cz_baseline_development
from .cz_k_support import run_cz_k_channel
from .cz_l256_nowcast import (
    DIRECTIONS,
    INPUT_COLUMNS,
    TASK_ID as CZ_TASK_ID,
    build_development_direction,
    materialize_target_direction,
    view as cz_view,
)
from .portable_checkpoints import (
    INFERENCE_ONLY_ENV,
    activate_inference_fit_guard,
    seal_checkpoint_tree,
    stable_hash,
)
from .representative_baseline_checkpoints import (
    fit_baseline_checkpoints_for_view,
    predict_baseline_checkpoints_for_view,
    verify_baseline_checkpoint_reload,
)
from .representative_prism_checkpoints import (
    fit_prism_checkpoint_for_view,
    predict_prism_checkpoint_for_view,
    verify_prism_checkpoint_reload,
)
from .stage0 import write_json
from .v211_a import run_a_view
from .v211_c import run_c_view
from .v211_config import REPRESENTATIVE_STAGE1_PROTOCOL
from .v211_joint_stability import run_joint_stability_view
from .v211_public_all_config import PublicAllPaths
from .v211_public_all_baselines import apply_common_requirements
from .v211_public_all_closure import view_support_requirements
from .v211_representative_stage1_config import (
    ACTIVE_DATASETS,
    PRIMARY_TASKS,
    PROTOCOL_ID,
    RESERVED_DATASETS,
    load_representative_stage1_descriptor,
)
from .v211_w import run_w_view
from .v211_support import load_native_samples, support_id_hash
from .v211_representative_stage1_views import (
    representative_stage1_dynamic_views,
    representative_stage1_input_views,
)


GLOBAL_FREEZE_NAME = "GLOBAL_SELECTION_FREEZE.json"
CHECKPOINT_MANIFEST_NAME = "CHECKPOINT_MANIFEST.json"
ACCEPTABLE_DEVELOPMENT_STATUSES = {
    "PASS",
    "NOT_RUN_PROTOCOL_INCOMPATIBLE",
    "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
    "COMPLETED_WITH_RETAINED_FAILURES",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(project: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(project.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_scope_request(*, datasets: Iterable[str], neural3: bool = False, stage2: bool = False) -> None:
    if neural3:
        raise RuntimeError("STOP_NEURAL3_NOT_RUN_BY_USER_SCOPE")
    if stage2:
        raise RuntimeError("STOP_STAGE2_NOT_RUN_BY_USER_SCOPE")
    requested = set(datasets)
    reserved = requested.intersection(RESERVED_DATASETS)
    if reserved:
        raise RuntimeError(f"STOP_RESERVED_DATASET_NOT_RUN_BY_USER_SCOPE:{sorted(reserved)}")
    unknown = requested.difference(ACTIVE_DATASETS)
    if unknown:
        raise RuntimeError(f"STOP_UNREGISTERED_DATASET:{sorted(unknown)}")


def free_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free / (1024**3))


def storage_preflight(run_parent: Path, minimum_gib: float) -> dict[str, Any]:
    available = free_gib(run_parent)
    if available < float(minimum_gib):
        raise RuntimeError(
            f"STOP_INSUFFICIENT_FREE_STORAGE:{available:.3f}<{float(minimum_gib):.3f}GiB"
        )
    return {
        "status": "PASS",
        "path": str(run_parent),
        "available_gib": available,
        "minimum_gib": float(minimum_gib),
    }


def formal_scope(project: Path, run_root: Path) -> dict[str, Any]:
    descriptor = load_representative_stage1_descriptor(project)
    assert_scope_request(datasets=ACTIVE_DATASETS)
    result = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "source_commit": _git_commit(project),
        "active_datasets": list(ACTIVE_DATASETS),
        "primary_tasks": sorted(PRIMARY_TASKS),
        "reserved_datasets": dict(RESERVED_DATASETS),
        "stage2_status": "NOT_RUN_BY_USER_SCOPE",
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "development_order": list(descriptor["development_order"]),
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(run_root / "logs" / "FORMAL_SCOPE.json", result)
    return result


def _cz_direction_paths(project: Path, run_root: Path, direction: str) -> PublicAllPaths:
    return PublicAllPaths(
        project=project,
        shared=run_root / "cz" / "shared" / direction,
        run_root=run_root / "cz" / "directions" / direction,
    )


def run_cz_direction_development(
    *, project: Path, run_root: Path, raw_path: Path, direction: str
) -> dict[str, Any]:
    if direction not in DIRECTIONS:
        raise KeyError(direction)
    paths = _cz_direction_paths(project, run_root, direction)
    if not (paths.shared / "TASK_REGISTRY.json").is_file():
        build_development_direction(raw_path, run_root / "cz" / "shared", direction)
    input_view = cz_view("input_only")
    dynamic_view = cz_view("dynamic")
    paths.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for channel in INPUT_COLUMNS:
        records.append(
            run_cz_k_channel(
                paths.shared,
                project,
                paths.output,
                input_view,
                channel,
                REPRESENTATIVE_STAGE1_PROTOCOL,
            )
        )
    records.extend(
        [
            run_c_view(paths.shared, project, paths.output, input_view, REPRESENTATIVE_STAGE1_PROTOCOL),
            run_w_view(paths.shared, project, paths.output, input_view, REPRESENTATIVE_STAGE1_PROTOCOL),
            run_a_view(paths.shared, project, paths.output, dynamic_view, REPRESENTATIVE_STAGE1_PROTOCOL),
            run_joint_stability_view(
                paths.shared,
                project,
                paths.output,
                None,
                dynamic_view,
                REPRESENTATIVE_STAGE1_PROTOCOL,
            ),
        ]
    )
    baseline = run_cz_baseline_development(
        paths.shared, project, paths.output, input_view, dynamic_view
    )
    unacceptable = [
        str(item.get("status"))
        for item in records
        if str(item.get("status")) not in ACCEPTABLE_DEVELOPMENT_STATUSES
    ]
    if unacceptable or baseline.get("status") == "FAILED":
        status = "FAILED"
    elif all(item.get("status") == "PASS" for item in records) and baseline.get("status") == "PASS":
        status = "PASS"
    else:
        status = "COMPLETED_WITH_RETAINED_FAILURES"
    summary = {
        "status": status,
        "stage": "CZ_L256_DEVELOPMENT",
        "direction": direction,
        "task_id": CZ_TASK_ID,
        "prism_statuses": [str(item.get("status")) for item in records],
        "baseline_status": baseline.get("status"),
        "test_accessed": False,
        "ood_accessed": False,
        "global_freeze_created": False,
    }
    write_json(paths.logs / "CZ_DEVELOPMENT_SUMMARY.json", summary)
    return summary


def run_all_cz_development(*, project: Path, run_root: Path, raw_path: Path) -> dict[str, Any]:
    summaries = [
        run_cz_direction_development(
            project=project, run_root=run_root, raw_path=raw_path, direction=direction
        )
        for direction in DIRECTIONS
    ]
    result = {
        "status": "PASS"
        if all(item["status"] in {"PASS", "COMPLETED_WITH_RETAINED_FAILURES"} for item in summaries)
        else "FAILED",
        "stage": "CZ_L256_ALL_DIRECTIONS_DEVELOPMENT",
        "directions": summaries,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(run_root / "logs" / "CZ_DEVELOPMENT_ACCEPTANCE.json", result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _manifest_files(root: Path, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in root.rglob(pattern) if path.is_file())
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _selection_records(run_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_root.rglob("RESULT.json")):
        value = _read_json(path)
        selected = {
            key: item
            for key, item in value.items()
            if key == "selection"
            or key == "active"
            or key == "retained"
            or key.startswith("selected_")
            or key.startswith("final_selected")
            or key.endswith("_contract")
        }
        records.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "status": str(value.get("status", "MISSING_STATUS")),
                "dataset": value.get("dataset"),
                "task": value.get("task", value.get("task_id")),
                "target_head": value.get("target_head"),
                "information_set": value.get("information_set"),
                "availability_scenario": value.get("availability_scenario"),
                "proxy_policy": value.get("proxy_policy"),
                "model": value.get("model", path.parts[-4] if len(path.parts) >= 4 else None),
                "selection": selected,
                "selection_hash": _stable_json_hash(selected),
                "artifact_sha256": sha256_file(path),
            }
        )
    return records


def build_common_support_for_views(
    paths: PublicAllPaths, views: Iterable[Any]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for candidate in views:
        requirements = view_support_requirements(paths, candidate)
        split_details: dict[str, Any] = {}
        for split in ("train", "validation", "test", "ood"):
            sample_path = paths.shared / "sample_ids" / candidate.relative_root / f"{split}.parquet"
            if not sample_path.is_file():
                continue
            source = pd.read_parquet(sample_path)
            common = (
                apply_common_requirements(
                    load_native_samples(paths.shared, candidate, split), requirements
                ).reset_index(drop=True)
                if len(source)
                else source
            )
            split_details[split] = {
                "rows": int(len(common)),
                "source_rows": int(len(source)),
                "support_hash": support_id_hash(common),
                "support_contract": str(source["sample_support_contract"].iloc[0])
                if len(source)
                else "NATIVE_K_COMMON_ASSEMBLY_R1",
            }
        records.append(
            {
                "target_head": candidate.head.head_id,
                "dataset": candidate.head.dataset,
                "information_set": candidate.information_set,
                "availability_scenario": candidate.availability_scenario,
                "proxy_policy": candidate.proxy_policy,
                "requirements": [item.to_json() for item in requirements],
                "splits": split_details,
            }
        )
    result = {
        "status": "PASS",
        "stage": "FORMAL_COMMON_SUPPORT_REQUIREMENTS_FREEZE",
        "views": records,
        "test_y_read": False,
        "ood_y_read": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.leaderboard_support_path, result)
    return result


def create_global_selection_freeze(project: Path, run_root: Path) -> dict[str, Any]:
    public_status_path = run_root / "public" / "logs" / "LAUNCH_STATUS.json"
    cz_status_path = run_root / "logs" / "CZ_DEVELOPMENT_ACCEPTANCE.json"
    public_status = _read_json(public_status_path)
    cz_status = _read_json(cz_status_path)
    if public_status.get("status") != "PARTIAL_DEVELOPMENT_CPU_ONLY":
        raise RuntimeError("STOP_TEP_SRU_DEVELOPMENT_NOT_COMPLETE")
    if cz_status.get("status") != "PASS":
        raise RuntimeError("STOP_CZ_DEVELOPMENT_NOT_COMPLETE")
    if public_status.get("test_accessed") is not False or public_status.get("ood_accessed") is not False:
        raise RuntimeError("STOP_PUBLIC_TEST_OR_OOD_ACCESSED_BEFORE_FREEZE")
    if cz_status.get("test_accessed") is not False or cz_status.get("ood_accessed") is not False:
        raise RuntimeError("STOP_CZ_TEST_OR_OOD_ACCESSED_BEFORE_FREEZE")
    public_paths = PublicAllPaths(
        project, run_root / "public" / "shared", run_root / "public"
    )
    public_views = [
        *representative_stage1_input_views(public_paths.shared),
        *representative_stage1_dynamic_views(public_paths.shared),
    ]
    build_common_support_for_views(public_paths, public_views)
    for direction in DIRECTIONS:
        direction_paths = _cz_direction_paths(project, run_root, direction)
        build_common_support_for_views(
            direction_paths, [cz_view("input_only"), cz_view("dynamic")]
        )
    manifest = _manifest_files(run_root, ("RESULT.json", "SUMMARY.json", "*ACCEPTANCE*.json"))
    if not manifest:
        raise RuntimeError("STOP_NO_DEVELOPMENT_SELECTION_ARTIFACTS")
    descriptor = load_representative_stage1_descriptor(project)
    selections = _selection_records(run_root)
    result = {
        "status": "GLOBAL_SELECTION_FROZEN",
        "sealed": True,
        "sealed_utc": _utc(),
        "protocol_id": PROTOCOL_ID,
        "source_commit": _git_commit(project),
        "protocol_config_sha256": descriptor["config_sha256"],
        "active_datasets": list(ACTIVE_DATASETS),
        "primary_tasks": sorted(PRIMARY_TASKS),
        "reserved_datasets": dict(RESERVED_DATASETS),
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "stage2_status": "NOT_RUN_BY_USER_SCOPE",
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
        raise RuntimeError(f"refusing to overwrite global freeze: {destination}")
    write_json(destination, result)
    destination.chmod(0o444)
    return result


def require_global_freeze(run_root: Path) -> dict[str, Any]:
    path = run_root / "freeze" / GLOBAL_FREEZE_NAME
    value = _read_json(path)
    if value.get("status") != "GLOBAL_SELECTION_FROZEN" or value.get("sealed") is not True:
        raise RuntimeError("STOP_GLOBAL_SELECTION_FREEZE_NOT_SEALED")
    for record in value.get("development_artifacts", []):
        artifact = run_root / str(record["path"])
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"STOP_FROZEN_DEVELOPMENT_ARTIFACT_DRIFT:{artifact}")
    return value


def _formal_path_views(project: Path, run_root: Path) -> list[tuple[str, PublicAllPaths, list[Any]]]:
    public = PublicAllPaths(project, run_root / "public" / "shared", run_root / "public")
    result: list[tuple[str, PublicAllPaths, list[Any]]] = [
        (
            "public",
            public,
            [
                *representative_stage1_input_views(public.shared),
                *representative_stage1_dynamic_views(public.shared),
            ],
        )
    ]
    for direction in DIRECTIONS:
        paths = _cz_direction_paths(project, run_root, direction)
        result.append((f"cz:{direction}", paths, [cz_view("input_only"), cz_view("dynamic")]))
    return result


def checkpoint_namespace_root(checkpoint_root: Path, namespace: str) -> Path:
    """Keep public and both cross-rod checkpoint trees physically disjoint."""

    if namespace == "public":
        return checkpoint_root / "public"
    prefix = "cz:"
    if namespace.startswith(prefix):
        direction = namespace[len(prefix) :]
        if direction not in DIRECTIONS:
            raise RuntimeError(f"STOP_UNKNOWN_CZ_CHECKPOINT_DIRECTION:{direction}")
        return checkpoint_root / "cz" / direction
    raise RuntimeError(f"STOP_UNKNOWN_CHECKPOINT_NAMESPACE:{namespace}")


def _checkpoint_inventory(checkpoint_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in sorted(checkpoint_root.rglob("MANIFEST.json")):
        checkpoint = manifest_path.parent
        manifest = _read_json(manifest_path)
        state = _read_json(checkpoint / "checkpoint.json")
        for model in state.get("models", [state.get("model")]):
            if not model:
                continue
            records.append(
                {
                    "dataset": state["dataset"],
                    "task": state["task"],
                    "target_head": state["target_head"],
                    "information_set": state["information_set"],
                    "availability_scenario": state["availability_scenario"],
                    "proxy_policy": state["proxy_policy"],
                    "method": str(model),
                    "family": state["family"],
                    "fit_rows": int(state["fit_rows"]),
                    "fit_support_hash": state["fit_support_hash"],
                    "selection_hash": state["selection_hash"],
                    "checkpoint_dir": checkpoint.relative_to(checkpoint_root).as_posix(),
                    "checkpoint_hash": manifest["checkpoint_hash"],
                    "files": [
                        {
                            **item,
                            "mtime_ns": (checkpoint / str(item["name"])).stat().st_mtime_ns,
                        }
                        for item in manifest["files"]
                    ],
                    "deletion_forbidden": True,
                }
            )
    return records


def _verify_checkpoint_inventory(checkpoint_root: Path, entries: Iterable[Mapping[str, Any]]) -> None:
    for entry in entries:
        checkpoint = checkpoint_root / str(entry["checkpoint_dir"])
        files = list(entry["files"])
        if stable_hash(
            [
                {key: value for key, value in item.items() if key != "mtime_ns"}
                for item in files
            ]
        ) != entry["checkpoint_hash"]:
            raise RuntimeError(f"STOP_CHECKPOINT_ENTRY_HASH_MISMATCH:{checkpoint}")
        for item in files:
            path = checkpoint / str(item["name"])
            if (
                not path.is_file()
                or path.stat().st_size != int(item["bytes"])
                or path.stat().st_mtime_ns != int(item["mtime_ns"])
                or sha256_file(path) != item["sha256"]
            ):
                raise RuntimeError(f"STOP_FROZEN_CHECKPOINT_DRIFT:{path}")


def fit_and_seal_formal_checkpoints(project: Path, run_root: Path) -> dict[str, Any]:
    freeze = require_global_freeze(run_root)
    checkpoint_root = run_root / "checkpoints"
    if checkpoint_root.exists() and any(checkpoint_root.iterdir()):
        raise RuntimeError(f"refusing existing checkpoint root: {checkpoint_root}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    descriptor = load_representative_stage1_descriptor(project)
    records: list[dict[str, Any]] = []
    for namespace, paths, views in _formal_path_views(project, run_root):
        namespace_root = checkpoint_namespace_root(checkpoint_root, namespace)
        if free_gib(run_root.parent) < float(descriptor["minimum_runtime_free_gib"]):
            raise RuntimeError("STOP_LOW_STORAGE_BEFORE_NEXT_CHECKPOINT_VIEW")
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
    # Reload every newly written checkpoint with the same guard used by test.
    import os

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
        raise RuntimeError("STOP_NO_FORMAL_CHECKPOINTS_CREATED")
    result = {
        "status": "CHECKPOINTS_SEALED",
        "sealed": True,
        "sealed_utc": _utc(),
        "deletion_forbidden": True,
        "protocol_id": PROTOCOL_ID,
        "source_commit": _git_commit(project),
        "global_selection_freeze_sha256": sha256_file(
            run_root / "freeze" / GLOBAL_FREEZE_NAME
        ),
        "global_selection_freeze_manifest_hash": freeze["development_manifest_sha256"],
        "protocol_config_sha256": descriptor["config_sha256"],
        "entries": entries,
        "entry_count": len(entries),
        "reload_prediction_replay": replay,
        "reload_prediction_replay_status": "PASS",
        "test_accessed": False,
        "ood_accessed": False,
    }
    destination = run_root / "freeze" / CHECKPOINT_MANIFEST_NAME
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite checkpoint manifest: {destination}")
    write_json(destination, result)
    destination.chmod(0o444)
    seal_checkpoint_tree(checkpoint_root)
    _verify_checkpoint_inventory(checkpoint_root, entries)
    return result


def require_checkpoint_manifest(run_root: Path) -> dict[str, Any]:
    manifest = _read_json(run_root / "freeze" / CHECKPOINT_MANIFEST_NAME)
    if manifest.get("status") != "CHECKPOINTS_SEALED" or manifest.get("sealed") is not True:
        raise RuntimeError("STOP_CHECKPOINT_MANIFEST_NOT_SEALED")
    if manifest.get("deletion_forbidden") is not True:
        raise RuntimeError("STOP_CHECKPOINT_DELETION_GUARD_ABSENT")
    _verify_checkpoint_inventory(run_root / "checkpoints", manifest["entries"])
    return manifest


def _support_acceptance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") != "PASS":
            continue
        key = (
            str(record["namespace"]),
            str(record["dataset"]),
            str(record["target_head"]),
            str(record["information_set"]),
            str(record["availability_scenario"]),
            str(record["proxy_policy"]),
        )
        groups.setdefault(key, []).append(record)
    audits: list[dict[str, Any]] = []
    for key, rows in groups.items():
        counts = {int(item["rows"]) for item in rows}
        support = {str(item["scoring_support_hash"]) for item in rows}
        order = {str(item["sample_id_order_hash"]) for item in rows}
        if len(counts) != 1 or len(support) != 1 or len(order) != 1:
            raise RuntimeError(f"STOP_FORMAL_METHOD_SUPPORT_MISMATCH:{key}")
        audits.append(
            {
                "status": "PASS",
                "view": list(key),
                "methods": len(rows),
                "rows": next(iter(counts)),
                "support_hash": next(iter(support)),
                "sample_id_order_hash": next(iter(order)),
            }
        )
    return audits


def _rankings(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"input_only": [], "dynamic": []}
    for information_set in result:
        selected = [
            item
            for item in records
            if item.get("status") == "PASS"
            and item.get("information_set") == information_set
        ]
        groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for item in selected:
            key = (
                str(item["namespace"]),
                str(item["dataset"]),
                str(item["target_head"]),
                str(item["availability_scenario"]),
                str(item["proxy_policy"]),
            )
            groups.setdefault(key, []).append(item)
        for key, rows in sorted(groups.items()):
            rows.sort(
                key=lambda item: (
                    -float(item["r2_level_reconstructed"]),
                    -float(item["r2_delta"]),
                    str(item["model"]),
                )
            )
            result[information_set].append(
                {
                    "view": {
                        "namespace": key[0],
                        "dataset": key[1],
                        "target_head": key[2],
                        "availability_scenario": key[3],
                        "proxy_policy": key[4],
                        "information_set": information_set,
                    },
                    "leaderboard": [
                        {"rank": index + 1, **item}
                        for index, item in enumerate(rows)
                    ],
                }
            )
    return result


def _assert_no_out_of_scope_artifacts(run_root: Path) -> None:
    forbidden = ("neural3", "debutanizer", "pmsm", "metropt", "stage2")
    violations = [
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if any(token in path.name.lower() for token in forbidden)
    ]
    if violations:
        raise RuntimeError(f"STOP_OUT_OF_SCOPE_ARTIFACT_PRESENT:{violations[:20]}")


def run_formal_test_inference(
    *, project: Path, run_root: Path, raw_cz: Path
) -> dict[str, Any]:
    import os

    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError("STOP_FORMAL_TEST_MUST_RUN_IN_INFERENCE_ONLY_PROCESS")
    activate_inference_fit_guard()
    require_global_freeze(run_root)
    manifest = require_checkpoint_manifest(run_root)
    descriptor = load_representative_stage1_descriptor(project)
    if free_gib(run_root.parent) < float(descriptor["minimum_runtime_free_gib"]):
        raise RuntimeError("STOP_LOW_STORAGE_BEFORE_TEST_UNLOCK")
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
    for namespace, paths, views in _formal_path_views(project, run_root):
        namespace_root = checkpoint_namespace_root(
            run_root / "checkpoints", namespace
        )
        if free_gib(run_root.parent) < float(descriptor["minimum_runtime_free_gib"]):
            raise RuntimeError("STOP_LOW_STORAGE_DURING_TEST_INFERENCE")
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
        "stage": "FORMAL_TEST_INFERENCE_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "primary_metric": "R2_LEVEL_RECONSTRUCTED",
        "secondary_metric": "R2_DELTA",
        "active_datasets": list(ACTIVE_DATASETS),
        "reserved_datasets": dict(RESERVED_DATASETS),
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "stage2_status": "NOT_RUN_BY_USER_SCOPE",
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
    write_json(run_root / "final" / "FORMAL_LEVEL_DELTA_REPORT.json", report)
    write_json(
        run_root / "final" / "INPUT_ONLY_LEADERBOARD.json", rankings["input_only"]
    )
    write_json(
        run_root / "final" / "DYNAMIC_LEADERBOARD.json", rankings["dynamic"]
    )
    acceptance = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "formal_datasets": ["tep", "sru", "cz_czochralski"],
        "formal_dataset_count": 3,
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
    write_json(run_root / "final" / "FINAL_ACCEPTANCE.json", acceptance)
    return acceptance
