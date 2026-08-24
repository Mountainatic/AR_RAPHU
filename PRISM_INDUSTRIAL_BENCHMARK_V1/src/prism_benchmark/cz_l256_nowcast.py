"""Frozen raw-2-second CZ L256 delta-nowcast data contract.

This task intentionally coexists with the historical ``CZ_D20`` task.  An
exclusive origin ``t`` denotes the row being predicted: inputs are the 256
rows ``[t-256, t)``, the registered target is ``D[t] - D[t-1]``, and level
reporting reconstructs ``D[t-1] + delta_pred``.

Development and target-rod materialization are separate entry points.  The
target partition cannot be built until both the global selection freeze and
the immutable checkpoint manifest have been sealed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cpu_data import HeadSpec, ViewSpec
from .v211_support import SUPPORT_CONTRACT, support_id_hash


TASK_ID = "CZ_DIAM_RAW2S_CURRENT_L256"
RAW_PERIOD_SECONDS = 2.0
HISTORY_STEPS = 256
H_STEPS = 0
W_STEPS = 1
W0_STEPS = 1
INPUT_COLUMNS = (
    "main_heater_power",
    "joint_lift",
    "crystal_rotation",
    "crucible_rotation",
)
TARGET_COLUMN = "crystal_diameter"
ROD_SHEETS = {"Rod_1": "Sheet1", "Rod_2": "Sheet2"}
BREAKPOINTS = {"Rod_1": (334, 688, 734), "Rod_2": (745,)}
DIRECTIONS = {
    "Rod_1_to_Rod_2": ("Rod_1", "Rod_2"),
    "Rod_2_to_Rod_1": ("Rod_2", "Rod_1"),
}
RAW_TO_CANONICAL = {
    "主加热功率": "main_heater_power",
    "晶升速度": "crystal_lift",
    "埚升速度": "crucible_lift",
    "晶转速度": "crystal_rotation",
    "埚转速度": "crucible_rotation",
    "晶体直径": TARGET_COLUMN,
}


@dataclass(frozen=True)
class RawSegment:
    rod_id: str
    segment_id: str
    raw_start: int
    raw_stop: int
    frame: pd.DataFrame


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _segment_ranges(length: int, breakpoints: Iterable[int]) -> list[tuple[int, int]]:
    points = [0, *(int(value) for value in breakpoints), int(length)]
    if points != sorted(set(points)) or points[-1] != length:
        raise ValueError(f"invalid breakpoints for length={length}: {points}")
    return [(start, stop) for start, stop in zip(points[:-1], points[1:]) if stop > start]


def _read_rod(raw_path: Path, rod_id: str) -> pd.DataFrame:
    if rod_id not in ROD_SHEETS:
        raise KeyError(rod_id)
    frame = pd.read_excel(raw_path, sheet_name=ROD_SHEETS[rod_id], header=0)
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    missing = sorted(set(RAW_TO_CANONICAL) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{rod_id} missing required columns: {missing}")
    numeric = frame[list(RAW_TO_CANONICAL)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise RuntimeError(f"{rod_id} has nonnumeric or missing required values: {bad}")
    return frame


def _raw_segments(rod_id: str, raw: pd.DataFrame) -> list[RawSegment]:
    selected = raw[list(RAW_TO_CANONICAL)].rename(columns=RAW_TO_CANONICAL)
    segments: list[RawSegment] = []
    for index, (start, stop) in enumerate(_segment_ranges(len(selected), BREAKPOINTS[rod_id])):
        frame = selected.iloc[start:stop].reset_index(drop=True).copy()
        frame["row_in_entity"] = np.arange(len(frame), dtype=np.int64)
        frame["entity_id"] = f"{rod_id}_segment_{index}"
        segments.append(RawSegment(rod_id, f"{rod_id}_segment_{index}", start, stop, frame))
    if not segments:
        raise RuntimeError(f"{rod_id} has no raw segments")
    return segments


def _split_source_origins(
    segments: list[RawSegment],
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Create an 80/20 chronological split with disjoint dependency intervals."""

    train: dict[str, set[int]] = {segment.segment_id: set() for segment in segments}
    validation: dict[str, set[int]] = {segment.segment_id: set() for segment in segments}
    purge = HISTORY_STEPS + 1
    eligible = [
        (segment, list(range(HISTORY_STEPS, len(segment.frame))))
        for segment in segments
        if len(segment.frame) > HISTORY_STEPS
    ]
    total = sum(len(valid) for _, valid in eligible)
    desired_train = int(math.floor(0.8 * total))
    assigned_train = 0
    validation_started = False
    for segment, valid in eligible:
        if validation_started:
            validation[segment.segment_id] = set(valid)
            continue
        needed = desired_train - assigned_train
        if needed >= len(valid):
            train[segment.segment_id] = set(valid)
            assigned_train += len(valid)
            continue
        if needed <= 0:
            validation[segment.segment_id] = set(valid)
            validation_started = True
            continue
        validation_start = needed + purge
        if validation_start >= len(valid):
            # A short breakpoint segment cannot hold both sides and the purge;
            # keep it on the training side and start validation at the next
            # independent segment.
            train[segment.segment_id] = set(valid)
            assigned_train += len(valid)
            continue
        train[segment.segment_id] = set(valid[:needed])
        validation[segment.segment_id] = set(valid[validation_start:])
        assigned_train += needed
        validation_started = True
    if not any(train.values()) or not any(validation.values()):
        raise RuntimeError("CZ source rod cannot realize nonempty purged train/validation")
    return train, validation


def _fit_joint_lift(
    segments: list[RawSegment], train_origins: dict[str, set[int]]
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    support: list[str] = []
    for segment in segments:
        used: set[int] = set()
        for origin in train_origins.get(segment.segment_id, set()):
            used.update(range(origin - HISTORY_STEPS, origin))
        if not used:
            continue
        ordered = sorted(used)
        rows.append(
            segment.frame.iloc[ordered][["crystal_lift", "crucible_lift"]].to_numpy(
                dtype=np.float64
            )
        )
        support.extend(f"{segment.segment_id}:{row}" for row in ordered)
    if not rows:
        raise RuntimeError("no source-training history rows for joint-lift PCA")
    train = np.vstack(rows)
    mean = train.mean(axis=0, dtype=np.float64)
    scale = train.std(axis=0, dtype=np.float64)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    standardized = (train - mean) / scale
    covariance = np.cov(standardized, rowvar=False, ddof=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    loading = eigenvectors[:, order[0]].astype(np.float64)
    if loading[0] < 0:
        loading = -loading
    return {
        "status": "FROZEN_TRAIN_ONLY",
        "fit_scope": "SOURCE_ROD_DEVELOPMENT_TRAIN_HISTORY_ROWS_ONLY",
        "variables": ["crystal_lift", "crucible_lift"],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "pc1_loading": loading.tolist(),
        "eigenvalues": eigenvalues[order].tolist(),
        "sign_rule": "crystal_lift_loading_must_be_nonnegative",
        "source_training_rows": int(len(train)),
        "fit_support_hash": _hash_text("\n".join(support)),
    }


def _apply_joint_lift(
    segments: list[RawSegment], contract: dict[str, Any]
) -> list[RawSegment]:
    mean = np.asarray(contract["mean"], dtype=np.float64)
    scale = np.asarray(contract["scale"], dtype=np.float64)
    loading = np.asarray(contract["pc1_loading"], dtype=np.float64)
    result: list[RawSegment] = []
    for segment in segments:
        frame = segment.frame.copy()
        values = frame[["crystal_lift", "crucible_lift"]].to_numpy(dtype=np.float64)
        frame["joint_lift"] = ((values - mean) / scale) @ loading
        result.append(RawSegment(segment.rod_id, segment.segment_id, segment.raw_start, segment.raw_stop, frame))
    return result


def _sample_rows(
    segments: list[RawSegment],
    *,
    direction: str,
    split: str,
    origin_filter: dict[str, set[int]] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for segment in segments:
        diameter = segment.frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
        for origin in range(HISTORY_STEPS, len(segment.frame)):
            if origin_filter is not None and origin not in origin_filter.get(segment.segment_id, set()):
                continue
            current = float(diameter[origin - 1])
            future = float(diameter[origin])
            base_id = _hash_text(f"{TASK_ID}|{direction}|{segment.segment_id}|{origin}")
            records.append(
                {
                    "base_origin_id": base_id,
                    "view_sample_id": _hash_text(f"input_only|{base_id}"),
                    "dataset": "cz_czochralski",
                    "entity_id": segment.segment_id,
                    "task_id": TASK_ID,
                    "target_head": TASK_ID,
                    "split": split,
                    "origin": origin,
                    "current_start": origin - 1,
                    "current_stop_exclusive": origin,
                    "history_start": origin - HISTORY_STEPS,
                    "history_stop_exclusive": origin,
                    "target_start": origin,
                    "target_stop_exclusive": origin + 1,
                    "dependency_start": origin - HISTORY_STEPS,
                    "dependency_stop_exclusive": origin + 1,
                    "latest_available_target_index": origin - 1,
                    "availability_delay_steps": 0,
                    "availability_scenario": "record_time",
                    "proxy_policy": "primary",
                    "information_set": "input_only",
                    "causal_history_floor": origin - HISTORY_STEPS,
                    "anchor_history_steps": 1,
                    "sample_support_contract": SUPPORT_CONTRACT,
                    "lmax_steps": HISTORY_STEPS,
                    "current_level": current,
                    "future_level_true": future,
                    "y_true": future - current,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    dynamic = frame.copy()
    dynamic["information_set"] = "dynamic"
    dynamic["view_sample_id"] = [
        _hash_text(f"dynamic|{value}") for value in dynamic["base_origin_id"].astype(str)
    ]
    return pd.concat([frame, dynamic], ignore_index=True)


def _base_frame(segments: list[RawSegment]) -> pd.DataFrame:
    columns = [
        "main_heater_power",
        "crystal_lift",
        "crucible_lift",
        "joint_lift",
        "crystal_rotation",
        "crucible_rotation",
        TARGET_COLUMN,
        "entity_id",
        "row_in_entity",
    ]
    return pd.concat([segment.frame[columns] for segment in segments], ignore_index=True)


def _empty_samples() -> pd.DataFrame:
    return _sample_rows([], direction="EMPTY", split="test")


def _write_partition(root: Path, split: str, base: pd.DataFrame, samples: pd.DataFrame) -> None:
    if not base.empty:
        path = root / "base_data" / "cz_czochralski" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        base.to_parquet(path, index=False, compression="zstd")
    for information_set in ("input_only", "dynamic"):
        frame = samples[samples["information_set"] == information_set].copy()
        sample_path = root / "sample_ids" / TASK_ID / information_set / "record_time" / "primary" / f"{split}.parquet"
        target_path = root / "targets" / f"{TASK_ID}__H0__W1" / f"{split}.parquet"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(sample_path, index=False, compression="zstd")
        frame[["base_origin_id", "entity_id", "origin", "current_level", "future_level_true", "y_true"]].to_parquet(
            target_path, index=False, compression="zstd"
        )


def head() -> HeadSpec:
    return HeadSpec(TASK_ID, TASK_ID, "cz_czochralski", TARGET_COLUMN, RAW_PERIOD_SECONDS, H_STEPS, W_STEPS, W0_STEPS, True)


def view(information_set: str) -> ViewSpec:
    if information_set not in {"input_only", "dynamic"}:
        raise ValueError(information_set)
    return ViewSpec(head(), information_set, "record_time", "primary")


def _write_metadata(root: Path, direction: str, source_rod: str, target_rod: str, pca: dict[str, Any]) -> None:
    _write_json(root / "TASK_REGISTRY.json", {"dataset": "cz_czochralski", "heads": [{"head_id": TASK_ID, "task_id": TASK_ID, "dataset": "cz_czochralski", "target": TARGET_COLUMN, "cadence_seconds": RAW_PERIOD_SECONDS, "h_steps": H_STEPS, "w_steps": W_STEPS, "w0_steps": W0_STEPS, "primary": True, "direction": direction}]})
    _write_json(root / "dataset_views" / "VIEW_REGISTRY.json", [{"task_id": TASK_ID, "dataset": "cz_czochralski", "head_id": TASK_ID, "information_set": information_set, "availability_scenario": "record_time", "proxy_policy": "primary", "input_columns": list(INPUT_COLUMNS), "target_history_column": TARGET_COLUMN if information_set == "dynamic" else None} for information_set in ("input_only", "dynamic")])
    _write_json(root / "PROTOCOL.json", {"protocol_id": "REPRESENTATIVE_STAGE1_TEP_SRU_CZ_L256_FORMAL_V1", "support_contract": SUPPORT_CONTRACT, "tasks": [{"task_id": TASK_ID, "proxy_policies": ["primary"]}], "primary_views": ["record_time/primary"], "direction": direction, "source_rod": source_rod, "target_rod": target_rod, "target_rod_used_for_direction_selection": False})
    _write_json(root / "CZ_TASK_REALIZATION.json", {"task_id": TASK_ID, "cadence_seconds": RAW_PERIOD_SECONDS, "aggregation": "NONE_RAW_ROWS", "origin_semantics": "CURRENT_TARGET_AT_EXCLUSIVE_HISTORY_STOP", "history_interval": "[t-256,t)", "lookback_steps": HISTORY_STEPS, "lookback_seconds": HISTORY_STEPS * RAW_PERIOD_SECONDS, "target_representation": "DELTA_FROM_LAST_OBSERVED_LEVEL", "target_formula": "D[t]-D[t-1]", "level_reconstruction": "D[t-1]+delta_pred", "max_input_index": "t-1", "target_index": "t", "h_steps": 0, "w_steps": 1, "w0_steps": 1, "direction": direction})
    _write_json(root / "JOINT_LIFT_PCA_CONTRACT.json", pca)


def build_development_direction(raw_path: Path, output_root: Path, direction: str) -> dict[str, Any]:
    if direction not in DIRECTIONS:
        raise KeyError(direction)
    source_rod, target_rod = DIRECTIONS[direction]
    root = output_root / direction
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing existing CZ L256 direction root: {root}")
    raw = _read_rod(raw_path, source_rod)
    segments = _raw_segments(source_rod, raw)
    train_origins, validation_origins = _split_source_origins(segments)
    pca = _fit_joint_lift(segments, train_origins)
    segments = _apply_joint_lift(segments, pca)
    root.mkdir(parents=True, exist_ok=True)
    _write_metadata(root, direction, source_rod, target_rod, pca)
    train = _sample_rows(segments, direction=direction, split="train", origin_filter=train_origins)
    validation = _sample_rows(segments, direction=direction, split="validation", origin_filter=validation_origins)
    empty = train.iloc[0:0].copy()
    _write_partition(root, "train", _base_frame(segments), train)
    _write_partition(root, "validation", pd.DataFrame(), validation)
    _write_partition(root, "test", pd.DataFrame(), empty.assign(split="test"))
    _write_partition(root, "ood", pd.DataFrame(), empty.assign(split="ood"))
    audit = {"status": "PASS", "stage": "CZ_SOURCE_DEVELOPMENT_C1", "created_utc": _utc(), "direction": direction, "source_rod": source_rod, "target_rod": target_rod, "raw_file_sha256": _sha256_file(raw_path), "raw_rows_read": int(len(raw)), "target_sheet_read_for_this_direction": False, "train_rows_per_information_set": int(len(train) // 2), "validation_rows_per_information_set": int(len(validation) // 2), "test_rows": 0, "test_accessed": False, "support_contract": SUPPORT_CONTRACT}
    _write_json(root / "C1_NATIVE_SUPPORT_AUDIT.json", audit)
    return audit


def _require_sealed(path: Path, expected_status: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != expected_status or value.get("sealed") is not True:
        raise RuntimeError(f"required sealed artifact is invalid: {path}")
    return value


def materialize_target_direction(
    raw_path: Path,
    output_root: Path,
    direction: str,
    *,
    global_freeze_path: Path,
    checkpoint_manifest_path: Path,
) -> dict[str, Any]:
    _require_sealed(global_freeze_path, "GLOBAL_SELECTION_FROZEN")
    _require_sealed(checkpoint_manifest_path, "CHECKPOINTS_SEALED")
    source_rod, target_rod = DIRECTIONS[direction]
    root = output_root / direction
    test_sample = root / "sample_ids" / TASK_ID / "input_only" / "record_time" / "primary" / "test.parquet"
    if test_sample.is_file() and len(pd.read_parquet(test_sample)):
        raise RuntimeError(f"refusing to overwrite materialized CZ target partition: {test_sample}")
    pca = json.loads((root / "JOINT_LIFT_PCA_CONTRACT.json").read_text(encoding="utf-8"))
    raw = _read_rod(raw_path, target_rod)
    segments = _apply_joint_lift(_raw_segments(target_rod, raw), pca)
    test = _sample_rows(segments, direction=direction, split="test")
    _write_partition(root, "test", _base_frame(segments), test)
    for information_set in ("input_only", "dynamic"):
        subset = test[test["information_set"] == information_set]
        if not np.all(subset["history_stop_exclusive"].to_numpy(dtype=np.int64) <= subset["target_start"].to_numpy(dtype=np.int64)):
            raise AssertionError("STOP_CZ_L256_LOOKAHEAD_LEAK")
    audit = {"status": "PASS", "stage": "CZ_TARGET_TEST_C1", "created_utc": _utc(), "direction": direction, "source_rod": source_rod, "target_rod": target_rod, "raw_file_sha256": _sha256_file(raw_path), "raw_rows_read": int(len(raw)), "test_rows_per_information_set": int(len(test) // 2), "input_support_hash": support_id_hash(test[test["information_set"] == "input_only"]), "dynamic_support_hash": support_id_hash(test[test["information_set"] == "dynamic"]), "global_freeze_sha256": _sha256_file(global_freeze_path), "checkpoint_manifest_sha256": _sha256_file(checkpoint_manifest_path), "test_accessed": True, "target_rod_first_access_after_freeze": True}
    _write_json(root / "CZ_TARGET_TEST_ACCESS_AUDIT.json", audit)
    return audit
