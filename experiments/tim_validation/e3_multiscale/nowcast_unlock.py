"""Materialize the TEP H0 test split after both E3 arms are sealed.

The frozen all-dataset nowcast C1 is development-only.  This module creates a
new hard-linked derivative, verifies that the E3 selection/checkpoint stage is
complete, and only then reads the immutable native Public-All test split.
The development C1 is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK = "TEP_G_NOWCAST_H0"
HEAD = "TEP_G_NOWCAST_H0__H0__W1"
SOURCE_HEAD = "TEP_G12__H1__W2"
TARGET = "xmeas_40"
PROXY_POLICY = "proxy_excluded"
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


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"REFUSING_EXISTING_E3_TEP_UNLOCK_ARTIFACT:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_bytes(value))
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def _hardlink_copy(source: str, destination: str) -> str:
    source_path = Path(source)
    if source_path.is_symlink() or not source_path.is_file():
        raise RuntimeError(f"STOP_E3_TEP_UNSAFE_DEVELOPMENT_FILE:{source_path}")
    os.link(source_path.resolve(), destination)
    return destination


def _verify_development_and_seals(
    development_shared: Path, native_shared: Path, task_root: Path
) -> dict[str, Any]:
    development_audit = _read(
        development_shared / "ALL_PUBLIC_NOWCAST_H0_DEVELOPMENT_C1_AUDIT.json"
    )
    lockbox = _read(development_shared / "LOCKBOX.json")
    registry = _read(development_shared / "SAMPLE_ID_REGISTRY.json")
    complete = _read(task_root / "DEVELOPMENT_COMPLETE.json")
    if (
        development_audit.get("status") != "PASS"
        or development_audit.get("test_accessed") is not False
        or development_audit.get("test_materialized") is not False
        or lockbox.get("test_materialized") is not False
        or registry.get("test_materialized") is not False
    ):
        raise RuntimeError("STOP_E3_TEP_DEVELOPMENT_C1_NOT_FROZEN_AND_TEST_FREE")
    if (
        complete.get("status") != "READY_FOR_ISOLATED_TEST_INFERENCE"
        or complete.get("task") != "TEP_G12"
        or complete.get("test_accessed") is not False
    ):
        raise RuntimeError("STOP_E3_TEP_DEVELOPMENT_NOT_COMPLETE")

    seal_records: dict[str, Any] = {}
    for arm in ("uniform", "multiscale"):
        selection_path = task_root / arm / "freeze" / "SELECTION_FREEZE.json"
        checkpoint_path = task_root / arm / "freeze" / "CHECKPOINTS_SEALED.json"
        selection = _read(selection_path)
        checkpoint = _read(checkpoint_path)
        if (
            selection.get("status") != "SELECTION_FROZEN"
            or selection.get("test_accessed") is not False
            or checkpoint.get("status") != "CHECKPOINTS_SEALED"
            or checkpoint.get("test_accessed") is not False
        ):
            raise RuntimeError(f"STOP_E3_TEP_ARM_NOT_SEALED:{arm}")
        seal_records[arm] = {
            "selection_freeze_sha256": _sha256_file(selection_path),
            "checkpoint_seal_sha256": _sha256_file(checkpoint_path),
        }

    development_train = development_shared / "base_data" / "tep" / "train.parquet"
    native_train = native_shared / "base_data" / "tep" / "train.parquet"
    development_validation = (
        development_shared / "base_data" / "tep" / "validation.parquet"
    )
    native_validation = native_shared / "base_data" / "tep" / "validation.parquet"
    hashes = {
        "train": (_sha256_file(development_train), _sha256_file(native_train)),
        "validation": (
            _sha256_file(development_validation),
            _sha256_file(native_validation),
        ),
    }
    if any(left != right for left, right in hashes.values()):
        raise RuntimeError(f"STOP_E3_TEP_NATIVE_SOURCE_DRIFT:{hashes}")
    return {
        "development_registry_sha256": _sha256_file(
            development_shared / "SAMPLE_ID_REGISTRY.json"
        ),
        "native_registry_sha256": _sha256_file(
            native_shared / "SAMPLE_ID_REGISTRY.json"
        ),
        "development_native_base_hashes": {
            split: pair[0] for split, pair in hashes.items()
        },
        "arm_seals": seal_records,
    }


def _source_support(native_shared: Path) -> pd.DataFrame:
    path = (
        native_shared
        / "sample_ids"
        / SOURCE_HEAD
        / "input_only"
        / "record_time"
        / PROXY_POLICY
        / "test.parquet"
    )
    frame = pd.read_parquet(
        path, columns=["entity_id", "origin", "causal_history_floor"]
    ).drop_duplicates()
    distance = (
        frame["origin"].to_numpy(dtype=np.int64)
        - frame["causal_history_floor"].to_numpy(dtype=np.int64)
    )
    frame = frame.loc[distance >= COMMON_SUPPORT_HISTORY_STEPS].copy()
    if frame.empty:
        raise RuntimeError("STOP_E3_TEP_EMPTY_NATIVE_TEST_SUPPORT")
    return frame.sort_values(["entity_id", "origin"]).reset_index(drop=True)


def _target_lookup(native_shared: Path) -> pd.Series:
    base_path = native_shared / "base_data" / "tep" / "test.parquet"
    frame = pd.read_parquet(
        base_path, columns=["entity_id", "row_in_entity", TARGET]
    )
    if frame.duplicated(["entity_id", "row_in_entity"]).any():
        raise RuntimeError("STOP_E3_TEP_DUPLICATE_NATIVE_TEST_INDEX")
    index = pd.MultiIndex.from_frame(frame[["entity_id", "row_in_entity"]])
    return pd.Series(frame[TARGET].to_numpy(dtype=np.float64), index=index)


def _sample_frame(
    support: pd.DataFrame,
    lookup: pd.Series,
    information_set: str,
    availability_scenario: str,
    delay_steps: int,
) -> pd.DataFrame:
    if delay_steps:
        distance = (
            support["origin"].to_numpy(dtype=np.int64)
            - support["causal_history_floor"].to_numpy(dtype=np.int64)
        )
        support = support.loc[
            distance >= COMMON_SUPPORT_HISTORY_STEPS + int(delay_steps)
        ].copy()
    current_index = pd.MultiIndex.from_arrays(
        [support["entity_id"], support["origin"]]
    )
    previous_index = pd.MultiIndex.from_arrays(
        [support["entity_id"], support["origin"] - 1]
    )
    current = lookup.reindex(current_index).to_numpy(dtype=np.float64)
    previous = lookup.reindex(previous_index).to_numpy(dtype=np.float64)
    finite = np.isfinite(current) & np.isfinite(previous)
    support = support.loc[finite].reset_index(drop=True)
    current = current[finite]
    previous = previous[finite]
    if support.empty:
        raise RuntimeError("STOP_E3_TEP_NO_FINITE_NATIVE_TEST_TARGET")

    origins = support["origin"].to_numpy(dtype=np.int64)
    entities = support["entity_id"].astype(str).to_numpy()
    base_ids = [
        _stable_id("NOWCAST_H0_BASE", "tep", entity, int(origin))
        for entity, origin in zip(entities, origins, strict=True)
    ]
    view_ids = [
        _stable_id(
            "NOWCAST_H0_VIEW",
            base_id,
            HEAD,
            information_set,
            availability_scenario,
            PROXY_POLICY,
        )
        for base_id in base_ids
    ]
    frame = pd.DataFrame(
        {
            "base_origin_id": base_ids,
            "view_sample_id": view_ids,
            "dataset": "tep",
            "entity_id": entities,
            "task_id": TASK,
            "target_head": HEAD,
            "split": "test",
            "origin": origins,
            "current_start": origins - 1,
            "current_stop_exclusive": origins,
            "target_start": origins,
            "target_stop_exclusive": origins + 1,
            "dependency_start": origins - COMMON_SUPPORT_HISTORY_STEPS,
            "dependency_stop_exclusive": origins + 1 + int(delay_steps),
            "latest_available_target_index": origins - 1 - int(delay_steps),
            "availability_delay_steps": int(delay_steps),
            "availability_scenario": availability_scenario,
            "proxy_policy": PROXY_POLICY,
            "information_set": information_set,
            "causal_history_floor": support["causal_history_floor"].to_numpy(
                dtype=np.int64
            ),
            "anchor_history_steps": 1,
            "sample_support_contract": SUPPORT_CONTRACT,
            "lmax_steps": COMMON_SUPPORT_HISTORY_STEPS,
            "y_true": current - previous,
        }
    )
    if not (frame["latest_available_target_index"] < frame["origin"]).all():
        raise AssertionError("STOP_E3_TEP_CURRENT_TARGET_LEAKAGE")
    return frame


def _refresh_lockbox(destination: Path, locked_files: list[str]) -> None:
    _replace_json(
        destination / "LOCKBOX.json",
        {
            "contract": "PRISM_C1_TEST_LOCKBOX_V1",
            "protocol_frozen": True,
            "access_rule": "E3_BOTH_SELECTIONS_AND_CHECKPOINTS_FROZEN",
            "metric_access_before_freeze": False,
            "locked_files": sorted(locked_files),
            "test_materialized": True,
            "ood_materialized": False,
        },
    )


def _refresh_registry(destination: Path) -> None:
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
            "anchor_universe": "ALL_NOWCAST_H0_L256_COMMON_TARGET_ROWS",
            "materialized_splits": ["train", "validation", "test"],
            "test_materialized": True,
            "ood_materialized": False,
            "protocol_sha256": _sha256_file(destination / "PROTOCOL.json"),
            "files": files,
        },
    )


def materialize(
    development_shared: Path,
    native_shared: Path,
    task_root: Path,
    destination: Path,
) -> dict[str, Any]:
    development_shared = development_shared.resolve(strict=True)
    native_shared = native_shared.resolve(strict=True)
    task_root = task_root.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"REFUSING_EXISTING_E3_TEP_DERIVATIVE:{destination}")
    provenance = _verify_development_and_seals(
        development_shared, native_shared, task_root
    )
    shutil.copytree(development_shared, destination, copy_function=_hardlink_copy)

    source_base = native_shared / "base_data" / "tep" / "test.parquet"
    target_base = destination / "base_data" / "tep" / "test.parquet"
    _hardlink_copy(str(source_base), str(target_base))
    support = _source_support(native_shared)
    lookup = _target_lookup(native_shared)
    views = []
    locked_files = [target_base.relative_to(destination).as_posix()]
    for information, availability, delay in VIEWS:
        frame = _sample_frame(support, lookup, information, availability, delay)
        output = (
            destination
            / "sample_ids"
            / HEAD
            / information
            / availability
            / PROXY_POLICY
            / "test.parquet"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False, compression="zstd")
        relative = output.relative_to(destination).as_posix()
        locked_files.append(relative)
        views.append(
            {
                "information_set": information,
                "availability_scenario": availability,
                "delay_steps": delay,
                "rows": len(frame),
                "minimum_origin": int(frame["origin"].min()),
                "maximum_origin": int(frame["origin"].max()),
                "path": relative,
                "sha256": _sha256_file(output),
            }
        )

    result = {
        "status": "COMPLETED",
        "stage": "E3_TEP_TEST_UNLOCK_AFTER_BOTH_ARM_CHECKPOINT_SEALS",
        "created_utc": _utc(),
        "task": "TEP_G12",
        "derived_head": HEAD,
        "development_shared": str(development_shared),
        "native_shared": str(native_shared),
        "destination": str(destination.resolve()),
        "common_support_history_steps": COMMON_SUPPORT_HISTORY_STEPS,
        "history_interval": "[t-L,t)",
        "current_t_excluded": True,
        "test_accessed": True,
        "selection_used_test": False,
        "checkpoint_fit_used_test": False,
        "development_files_hardlinked": True,
        "native_test_base_sha256": _sha256_file(source_base),
        "views": views,
        **provenance,
    }
    _write_new(destination / "E3_TEP_TEST_UNLOCK_AUDIT.json", result)
    _refresh_lockbox(destination, locked_files)
    _refresh_registry(destination)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--development-shared", type=Path, required=True)
    result.add_argument("--native-shared", type=Path, required=True)
    result.add_argument("--task-root", type=Path, required=True)
    result.add_argument("--destination", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    print(
        json.dumps(
            materialize(
                args.development_shared,
                args.native_shared,
                args.task_root,
                args.destination,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
