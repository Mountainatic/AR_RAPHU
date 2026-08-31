"""Strict-past TEP H0/W1 C1 materialization for the Neural-3 GPU matrix."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK = "TEP_G_NOWCAST_H0"
HEAD = "TEP_G_NOWCAST_H0__H0__W1"
SOURCE_TASK = "TEP_G_REP_H1"
TARGET = "xmeas_40"
SUPPORT_CONTRACT = "NATIVE_K_COMMON_ASSEMBLY_R1"
COMMON_SUPPORT_HISTORY_STEPS = 256
VIEWS = (
    ("input_only", "record_time", 0),
    ("dynamic", "record_time", 0),
    ("dynamic", "analyzer_maturity_5_steps", 5),
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"REFUSING_EXISTING_NOWCAST_C1:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _hardlink_new(source: Path, destination: Path) -> None:
    """Bind immutable source bytes without creating forbidden C1 symlinks."""

    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"REFUSING_EXISTING_NOWCAST_LINK:{destination}")
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"STOP_NOWCAST_SOURCE_FILE_UNSAFE:{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source.resolve(), destination)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_json(path: Path, value: Any) -> None:
    """Atomically refresh mutable stage metadata without touching data artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _refresh_lockbox(destination: Path) -> None:
    locked = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*.parquet")
        if path.name in {"test.parquet", "ood.parquet"}
    )
    _replace_json(
        destination / "LOCKBOX.json",
        {
            "contract": "PRISM_C1_TEST_LOCKBOX_V1",
            "protocol_frozen": False,
            "access_rule": "GLOBAL_SELECTION_AND_SELECTED_CHECKPOINTS_FROZEN",
            "metric_access_before_freeze": False,
            "locked_files": locked,
        },
    )


def _refresh_sample_registry(destination: Path) -> None:
    files = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name == "SAMPLE_ID_REGISTRY.json":
            continue
        entry: dict[str, Any] = {
            "path": path.relative_to(destination).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if path.suffix == ".parquet":
            entry["rows"] = pq.ParquetFile(path).metadata.num_rows
            entry["dtype"] = "schema_in_parquet"
        files.append(entry)
    _replace_json(
        destination / "SAMPLE_ID_REGISTRY.json",
        {
            "contract": "IMMUTABLE_SAMPLE_IDS_V1",
            "sample_support_contract": SUPPORT_CONTRACT,
            "anchor_universe": "TEP_H0_L256_COMMON_TARGET_ROWS",
            "materialized_splits": sorted(
                {
                    Path(item["path"]).stem
                    for item in files
                    if item["path"].startswith(f"sample_ids/{HEAD}/")
                    and item["path"].endswith(".parquet")
                }
            ),
            "protocol_sha256": _sha256_file(destination / "PROTOCOL.json"),
            "files": files,
        },
    )


def _registries(source: Path, destination: Path) -> None:
    source_protocol = _read(source / "PROTOCOL.json")
    tasks = [dict(item) for item in source_protocol["tasks"] if item["task_id"] == SOURCE_TASK]
    if len(tasks) != 1:
        raise RuntimeError("STOP_NOWCAST_SOURCE_PROTOCOL_TASK")
    tasks[0]["task_id"] = TASK
    protocol = dict(source_protocol)
    protocol["tasks"] = tasks
    protocol["nowcast_extension"] = {
        "status": "H0_W1_STRICT_PAST_REGISTERED",
        "history_interval": "[t-L,t)",
        "current_t_excluded": True,
        "anchor": "D[t-1]",
        "target_delta": "D[t]-D[t-1]",
        "level_reconstruction": "D[t-1]+delta_pred",
        "common_support_history_steps": COMMON_SUPPORT_HISTORY_STEPS,
        "test_materialized_after_selection_freeze_only": True,
    }
    _write_new(destination / "PROTOCOL.json", protocol)

    source_views = _read(source / "dataset_views" / "VIEW_REGISTRY.json")
    views = [dict(item) for item in source_views if item["task_id"] == SOURCE_TASK]
    if not views:
        raise RuntimeError("STOP_NOWCAST_SOURCE_INPUT_VIEW")
    for item in views:
        item["task_id"] = TASK
    _write_new(destination / "dataset_views" / "VIEW_REGISTRY.json", views)
    _write_new(
        destination / "TASK_REGISTRY.json",
        {
            "contract_status": "H0_W1_NOWCAST_APPROVED_GPU_REPLACEMENT",
            "sample_support_contract": SUPPORT_CONTRACT,
            "heads": [
                {
                    "cadence_seconds": 180.0,
                    "dataset": "tep",
                    "h_steps": 0,
                    "head_id": HEAD,
                    "horizon_error": 0.0,
                    "primary": True,
                    "requested_horizon_seconds": 0.0,
                    "requested_window_seconds": 180.0,
                    "status": "PASS",
                    "target": TARGET,
                    "task_id": TASK,
                    "w0_steps": 1,
                    "w_steps": 1,
                    "window_error": 0.0,
                }
            ],
        },
    )
    for name in ("DATASET_HASHES.json", "SPLIT_REGISTRY.json"):
        _hardlink_new(source / name, destination / name)
    _refresh_lockbox(destination)


def _sample_frame(
    base: pd.DataFrame,
    split: str,
    information: str,
    availability: str,
    delay: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for entity, group in base.groupby("entity_id", sort=False):
        group = group.sort_values("row_in_entity")
        rows = group["row_in_entity"].to_numpy(dtype=np.int64)
        if not np.array_equal(rows, np.arange(len(group), dtype=np.int64)):
            raise RuntimeError(f"STOP_NOWCAST_NONCONTIGUOUS_ENTITY:{split}:{entity}")
        first_origin = COMMON_SUPPORT_HISTORY_STEPS + int(delay)
        if len(group) <= first_origin:
            continue
        origin = np.arange(first_origin, len(group), dtype=np.int64)
        values = group[TARGET].to_numpy(dtype=np.float64)
        entity_text = str(entity)
        base_ids = [f"NOWCAST:{split}:{entity_text}:{int(item)}" for item in origin]
        view_ids = [
            f"{item}:{information}:{availability}:proxy_excluded" for item in base_ids
        ]
        parts.append(
            pd.DataFrame(
                {
                    "base_origin_id": base_ids,
                    "view_sample_id": view_ids,
                    "dataset": "tep",
                    "entity_id": entity_text,
                    "task_id": TASK,
                    "target_head": HEAD,
                    "split": split,
                    "origin": origin,
                    "current_start": origin - 1,
                    "current_stop_exclusive": origin,
                    "target_start": origin,
                    "target_stop_exclusive": origin + 1,
                    "dependency_start": origin - 1,
                    "dependency_stop_exclusive": origin + 1 + int(delay),
                    "latest_available_target_index": origin - 1 - int(delay),
                    "availability_delay_steps": int(delay),
                    "availability_scenario": availability,
                    "proxy_policy": "proxy_excluded",
                    "information_set": information,
                    "causal_history_floor": 0,
                    "anchor_history_steps": 1,
                    "sample_support_contract": SUPPORT_CONTRACT,
                    "lmax_steps": COMMON_SUPPORT_HISTORY_STEPS,
                    "y_true": values[origin] - values[origin - 1],
                }
            )
        )
    if not parts:
        raise RuntimeError(f"STOP_NOWCAST_EMPTY_SPLIT:{split}")
    frame = pd.concat(parts, ignore_index=True)
    if not (frame["latest_available_target_index"] < frame["origin"]).all():
        raise AssertionError("STOP_NOWCAST_CURRENT_TARGET_ENTERED_DYNAMIC_HISTORY")
    return frame


def _materialize_split(
    source: Path, destination: Path, split: str
) -> dict[str, Any]:
    source_base = source / "base_data" / "tep" / f"{split}.parquet"
    target_base = destination / "base_data" / "tep" / source_base.name
    _hardlink_new(source_base, target_base)
    base = pd.read_parquet(
        source_base, columns=["entity_id", "row_in_entity", TARGET]
    )
    records = []
    for information, availability, delay in VIEWS:
        frame = _sample_frame(base, split, information, availability, delay)
        output = (
            destination
            / "sample_ids"
            / HEAD
            / information
            / availability
            / "proxy_excluded"
            / f"{split}.parquet"
        )
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"REFUSING_EXISTING_NOWCAST_SAMPLES:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False, compression="zstd")
        records.append(
            {
                "view": output.parent.relative_to(destination / "sample_ids").as_posix(),
                "rows": len(frame),
                "minimum_origin": int(frame["origin"].min()),
            }
        )
    return {"split": split, "base_rows_read": len(base), "views": records}


def _validate_audit(
    path: Path, *, stage: str, test_accessed: bool
) -> dict[str, Any]:
    value = _read(path)
    if (
        value.get("status") != "PASS"
        or value.get("stage") != stage
        or value.get("task") != TASK
        or value.get("head") != HEAD
        or value.get("test_accessed") is not test_accessed
        or value.get("common_support_history_steps") != COMMON_SUPPORT_HISTORY_STEPS
    ):
        raise RuntimeError(f"STOP_EXISTING_NOWCAST_AUDIT_INVALID:{path}")
    return value


def build_development(source: Path, destination: Path) -> dict[str, Any]:
    audit_path = destination / "TEP_NOWCAST_C1_DEVELOPMENT_AUDIT.json"
    if audit_path.is_file():
        return _validate_audit(
            audit_path, stage="TEP_NOWCAST_DEVELOPMENT_C1", test_accessed=False
        )
    if destination.exists():
        raise RuntimeError(f"REFUSING_PARTIAL_NOWCAST_SHARED:{destination}")
    destination.mkdir(parents=True)
    _registries(source, destination)
    result = {
        "status": "PASS",
        "stage": "TEP_NOWCAST_DEVELOPMENT_C1",
        "created_utc": _utc(),
        "task": TASK,
        "head": HEAD,
        "h_steps": 0,
        "w_steps": 1,
        "history_interval": "[t-L,t)",
        "current_t_excluded": True,
        "anchor": "D[t-1]",
        "target_delta": "D[t]-D[t-1]",
        "common_support_history_steps": COMMON_SUPPORT_HISTORY_STEPS,
        "test_accessed": False,
        "ood_accessed": False,
        "splits": [
            _materialize_split(source, destination, split)
            for split in ("train", "validation")
        ],
    }
    _write_new(audit_path, result)
    _refresh_sample_registry(destination)
    return result


def build_test(
    source: Path,
    destination: Path,
    global_selection_freeze: Path,
    selected_checkpoint_manifest: Path,
) -> dict[str, Any]:
    audit_path = destination / "TEP_NOWCAST_C1_TEST_UNLOCK_AUDIT.json"
    if audit_path.is_file():
        return _validate_audit(
            audit_path,
            stage="TEP_NOWCAST_TEST_C1_AFTER_SELECTION_AND_CHECKPOINT_FREEZE",
            test_accessed=True,
        )
    freeze = _read(global_selection_freeze)
    selected = _read(selected_checkpoint_manifest)
    if freeze.get("status") != "GLOBAL_SELECTION_FROZEN" or freeze.get("sealed") is not True:
        raise RuntimeError("STOP_NOWCAST_TEST_BEFORE_GLOBAL_SELECTION_FREEZE")
    if selected.get("status") != "SELECTED_CHECKPOINTS_SEALED" or selected.get("sealed") is not True:
        raise RuntimeError("STOP_NOWCAST_TEST_BEFORE_SELECTED_CHECKPOINT_SEAL")
    result = {
        "status": "PASS",
        "stage": "TEP_NOWCAST_TEST_C1_AFTER_SELECTION_AND_CHECKPOINT_FREEZE",
        "created_utc": _utc(),
        "task": TASK,
        "head": HEAD,
        "common_support_history_steps": COMMON_SUPPORT_HISTORY_STEPS,
        "test_accessed": True,
        "ood_accessed": False,
        "split": _materialize_split(source, destination, "test"),
    }
    _write_new(audit_path, result)
    _refresh_lockbox(destination)
    _refresh_sample_registry(destination)
    return result


def protocol_summary() -> Mapping[str, Any]:
    return {
        "task": TASK,
        "head": HEAD,
        "history_interval": "[t-L,t)",
        "current_t_excluded": True,
        "anchor": "D[t-1]",
        "target_delta": "D[t]-D[t-1]",
        "common_support_history_steps": COMMON_SUPPORT_HISTORY_STEPS,
    }


__all__ = [
    "COMMON_SUPPORT_HISTORY_STEPS",
    "HEAD",
    "TASK",
    "build_development",
    "build_test",
    "protocol_summary",
]
