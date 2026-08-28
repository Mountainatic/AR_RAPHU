"""Build an independent TEP H0/W1 C1 without early formal-test access."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK = "TEP_G_NOWCAST_H0"
HEAD = "TEP_G_NOWCAST_H0__H0__W1"
SOURCE_TASK = "TEP_G_REP_H1"
TARGET = "xmeas_40"
SUPPORT_CONTRACT = "NATIVE_K_COMMON_ASSEMBLY_R1"
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
    if path.exists():
        raise RuntimeError(f"REFUSING_EXISTING_NOWCAST_C1:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _link_new(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"REFUSING_EXISTING_NOWCAST_LINK:{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source.resolve()), str(destination))


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
        "anchor": "D[t-1]",
        "target_delta": "D[t]-D[t-1]",
        "level_reconstruction": "D[t-1]+delta_pred",
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
            "contract_status": "H0_W1_NOWCAST_APPROVED_EXTENSION",
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
    if (source / "SPLIT_REGISTRY.json").is_file():
        _link_new(source / "SPLIT_REGISTRY.json", destination / "SPLIT_REGISTRY.json")


def _sample_frame(base: pd.DataFrame, split: str, information: str, availability: str, delay: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for entity, group in base.groupby("entity_id", sort=False):
        group = group.sort_values("row_in_entity")
        rows = group["row_in_entity"].to_numpy(dtype=np.int64)
        if not np.array_equal(rows, np.arange(len(group), dtype=np.int64)):
            raise RuntimeError(f"STOP_NOWCAST_NONCONTIGUOUS_ENTITY:{split}:{entity}")
        if len(group) < 2:
            continue
        origin = np.arange(1, len(group), dtype=np.int64)
        values = group[TARGET].to_numpy(dtype=np.float64)
        entity_text = str(entity)
        base_ids = [f"NOWCAST:{split}:{entity_text}:{int(item)}" for item in origin]
        view_ids = [f"{item}:{information}:{availability}:proxy_excluded" for item in base_ids]
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
                    "lmax_steps": 1,
                    "y_true": values[1:] - values[:-1],
                }
            )
        )
    if not parts:
        raise RuntimeError(f"STOP_NOWCAST_EMPTY_SPLIT:{split}")
    return pd.concat(parts, ignore_index=True)


def _materialize_split(source: Path, destination: Path, split: str) -> dict[str, Any]:
    source_base = source / "base_data" / "tep" / f"{split}.parquet"
    target_base = destination / "base_data" / "tep" / source_base.name
    _link_new(source_base, target_base)
    base = pd.read_parquet(source_base, columns=["entity_id", "row_in_entity", TARGET])
    records = []
    for information, availability, delay in VIEWS:
        frame = _sample_frame(base, split, information, availability, delay)
        output = destination / "sample_ids" / HEAD / information / availability / "proxy_excluded" / f"{split}.parquet"
        if output.exists():
            raise RuntimeError(f"REFUSING_EXISTING_NOWCAST_SAMPLES:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False, compression="zstd")
        records.append({"view": output.parent.relative_to(destination / "sample_ids").as_posix(), "rows": len(frame)})
    return {"split": split, "base_rows_read": len(base), "views": records}


def build_development(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise RuntimeError(f"REFUSING_EXISTING_NOWCAST_SHARED:{destination}")
    destination.mkdir(parents=True)
    _registries(source, destination)
    records = [_materialize_split(source, destination, split) for split in ("train", "validation")]
    result = {
        "status": "PASS",
        "stage": "TEP_NOWCAST_DEVELOPMENT_C1",
        "created_utc": _utc(),
        "task": TASK,
        "head": HEAD,
        "h_steps": 0,
        "w_steps": 1,
        "history_interval": "[t-L,t)",
        "anchor": "D[t-1]",
        "target_delta": "D[t]-D[t-1]",
        "test_accessed": False,
        "ood_accessed": False,
        "splits": records,
    }
    _write_new(destination / "TEP_NOWCAST_C1_DEVELOPMENT_AUDIT.json", result)
    return result


def build_test(source: Path, destination: Path, selection_freeze: Path) -> dict[str, Any]:
    freeze = _read(selection_freeze)
    if freeze.get("status") != "SELECTION_FROZEN" or freeze.get("sealed") is not True:
        raise RuntimeError("STOP_NOWCAST_TEST_BEFORE_SELECTION_FREEZE")
    result = {
        "status": "PASS",
        "stage": "TEP_NOWCAST_TEST_C1_AFTER_SELECTION_FREEZE",
        "created_utc": _utc(),
        "selection_freeze": str(selection_freeze.resolve()),
        "test_accessed": True,
        "ood_accessed": False,
        "split": _materialize_split(source, destination, "test"),
    }
    _write_new(destination / "TEP_NOWCAST_C1_TEST_UNLOCK_AUDIT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "test"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path)
    args = parser.parse_args()
    if args.stage == "development":
        result = build_development(args.source, args.destination)
    else:
        if args.selection_freeze is None:
            raise RuntimeError("--selection-freeze is required for test")
        result = build_test(args.source, args.destination, args.selection_freeze)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
