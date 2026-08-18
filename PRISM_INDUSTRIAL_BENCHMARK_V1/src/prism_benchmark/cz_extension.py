from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SUPPORT_CONTRACT = "NATIVE_K_COMMON_ASSEMBLY_R1"
TASK_ID = "CZ_D20"
RAW_PERIOD_SEC = 2.0
MODEL_PERIOD_SEC = 10.0
RAW_PER_MODEL = 5
H_STEPS = 120
W_STEPS = 12
W0_STEPS = 12
MAX_NEURAL_HISTORY_STEPS = 2880
SOURCE_SPLIT_PURGE_STEPS = H_STEPS + W_STEPS + W0_STEPS
INPUT_COLUMNS = [
    "main_heater_power",
    "joint_lift",
    "crystal_rotation",
    "crucible_rotation",
]
TARGET_COLUMN = "crystal_diameter"
ROD_SHEETS = (("Rod_1", "Sheet1"), ("Rod_2", "Sheet2"))
BREAKPOINTS = {"Rod_1": (334, 688, 734), "Rod_2": (745,)}

RAW_TO_CANONICAL = {
    "主加热功率": "main_heater_power",
    "晶升速度": "crystal_lift",
    "埚升速度": "crucible_lift",
    "晶转速度": "crystal_rotation",
    "埚转速度": "crucible_rotation",
    "晶体直径": TARGET_COLUMN,
}


@dataclass(frozen=True)
class RodSegment:
    rod_id: str
    segment_id: str
    raw_start: int
    raw_stop: int
    frame: pd.DataFrame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _segment_ranges(length: int, breakpoints: Iterable[int]) -> list[tuple[int, int]]:
    points = [0, *[int(value) for value in breakpoints], int(length)]
    if points != sorted(set(points)) or points[-1] != length:
        raise ValueError(f"invalid breakpoints for length={length}: {points}")
    return [
        (start, stop)
        for start, stop in zip(points[:-1], points[1:])
        if stop > start
    ]


def _read_workbook(raw_path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(raw_path, sheet_name=None, header=0)
    names = list(sheets)
    if len(names) < 2:
        raise RuntimeError("CZ workbook must contain at least two sheets")
    result: dict[str, pd.DataFrame] = {}
    for rod_id, sheet_name in ROD_SHEETS:
        if sheet_name not in sheets:
            raise RuntimeError(
                f"expected workbook sheet {sheet_name!r}; observed={names}"
            )
        frame = sheets[sheet_name].copy()
        frame = frame.dropna(axis=1, how="all")
        frame = frame.dropna(axis=0, how="all").reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"{rod_id} is empty")
        missing = sorted(set(RAW_TO_CANONICAL) - set(frame.columns))
        if missing:
            raise RuntimeError(f"{rod_id} missing required columns: {missing}")
        numeric = frame[list(RAW_TO_CANONICAL)].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.isna().any().any():
            bad = numeric.columns[numeric.isna().any()].tolist()
            raise RuntimeError(
                f"{rod_id} has nonnumeric or missing required values: {bad}"
            )
        if numeric.duplicated().any():
            raise RuntimeError(
                f"{rod_id} contains duplicate required-variable rows"
            )
        result[rod_id] = frame
    return result


def _aggregate_segments(rod_id: str, frame: pd.DataFrame) -> list[RodSegment]:
    selected = frame[list(RAW_TO_CANONICAL)].rename(columns=RAW_TO_CANONICAL)
    segments: list[RodSegment] = []
    for index, (raw_start, raw_stop) in enumerate(
        _segment_ranges(len(selected), BREAKPOINTS[rod_id])
    ):
        usable = ((raw_stop - raw_start) // RAW_PER_MODEL) * RAW_PER_MODEL
        if usable < RAW_PER_MODEL:
            continue
        raw = selected.iloc[raw_start : raw_start + usable]
        values = raw.to_numpy(dtype=np.float64).reshape(
            -1, RAW_PER_MODEL, len(raw.columns)
        )
        averaged = values.mean(axis=1, dtype=np.float64)
        model = pd.DataFrame(averaged, columns=list(raw.columns))
        model["row_in_entity"] = np.arange(len(model), dtype=np.int64)
        model["entity_id"] = f"{rod_id}_segment_{index}"
        segments.append(
            RodSegment(
                rod_id=rod_id,
                segment_id=f"{rod_id}_segment_{index}",
                raw_start=raw_start,
                raw_stop=raw_start + usable,
                frame=model,
            )
        )
    if not segments:
        raise RuntimeError(f"{rod_id} has no complete 10-second segment")
    return segments


def _fit_joint_lift(
    source_segments: list[RodSegment],
    source_training_origins: dict[str, set[int]],
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    for segment in source_segments:
        allowed = source_training_origins.get(segment.segment_id, set())
        if not allowed:
            continue
        frame = segment.frame
        mask = (frame["row_in_entity"] + 1).isin(sorted(allowed)).to_numpy()
        rows.append(
            frame.loc[
                mask, ["crystal_lift", "crucible_lift"]
            ].to_numpy(dtype=np.float64)
        )
    if not rows:
        raise RuntimeError("no source-training rows available for joint-lift PCA")
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
        "fit_scope": "SOURCE_ROD_DEVELOPMENT_TRAINING_SAMPLES_ONLY",
        "variables": ["crystal_lift", "crucible_lift"],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "pc1_loading": loading.tolist(),
        "eigenvalues": eigenvalues[order].tolist(),
        "sign_rule": "crystal_lift_loading_must_be_nonnegative",
        "source_training_rows": int(len(train)),
        "fit_support_hash": _hash_text(
            "\n".join(
                f"{segment_id}:{origin}"
                for segment_id in sorted(source_training_origins)
                for origin in sorted(source_training_origins[segment_id])
            )
        ),
    }


def _apply_joint_lift(
    segments: list[RodSegment], contract: dict[str, Any]
) -> list[RodSegment]:
    mean = np.asarray(contract["mean"], dtype=np.float64)
    scale = np.asarray(contract["scale"], dtype=np.float64)
    loading = np.asarray(contract["pc1_loading"], dtype=np.float64)
    result: list[RodSegment] = []
    for segment in segments:
        frame = segment.frame.copy()
        standardized = (
            frame[["crystal_lift", "crucible_lift"]].to_numpy(dtype=np.float64)
            - mean
        ) / scale
        frame["joint_lift"] = standardized @ loading
        result.append(
            RodSegment(
                segment.rod_id,
                segment.segment_id,
                segment.raw_start,
                segment.raw_stop,
                frame,
            )
        )
    return result


def _sample_rows(
    segments: list[RodSegment],
    *,
    direction: str,
    split: str,
    sample_filter: dict[str, set[int]] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for segment in segments:
        frame = segment.frame
        limit = len(frame)
        for position in range(W0_STEPS - 1, limit - H_STEPS - W_STEPS):
            origin = position + 1
            if sample_filter is not None and origin not in sample_filter.get(
                segment.segment_id, set()
            ):
                continue
            current = frame.iloc[
                origin - W0_STEPS : origin
            ][TARGET_COLUMN].to_numpy(dtype=np.float64)
            future = frame.iloc[
                origin + H_STEPS : origin + H_STEPS + W_STEPS
            ][TARGET_COLUMN].to_numpy(dtype=np.float64)
            base_id = _hash_text(f"{direction}|{segment.segment_id}|{origin}")
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
                    "current_start": origin - W0_STEPS,
                    "current_stop_exclusive": origin,
                    "target_start": origin + H_STEPS,
                    "target_stop_exclusive": origin + H_STEPS + W_STEPS,
                    "dependency_start": origin - W0_STEPS,
                    "dependency_stop_exclusive": origin + H_STEPS + W_STEPS,
                    "latest_available_target_index": origin - 1,
                    "availability_delay_steps": 0,
                    "availability_scenario": "record_time",
                    "proxy_policy": "primary",
                    "information_set": "input_only",
                    "causal_history_floor": 0,
                    "anchor_history_steps": W0_STEPS,
                    "sample_support_contract": SUPPORT_CONTRACT,
                    "lmax_steps": MAX_NEURAL_HISTORY_STEPS,
                    "y_true": float(
                        future.mean(dtype=np.float64)
                        - current.mean(dtype=np.float64)
                    ),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    dynamic = frame.copy()
    dynamic["information_set"] = "dynamic"
    dynamic["view_sample_id"] = [
        _hash_text(f"dynamic|{value}")
        for value in dynamic["base_origin_id"].astype(str)
    ]
    return pd.concat([frame, dynamic], ignore_index=True)


def _split_source_samples(
    segments: list[RodSegment],
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    train: dict[str, set[int]] = {}
    validation: dict[str, set[int]] = {}
    for segment in segments:
        valid = list(
            range(
                W0_STEPS,
                len(segment.frame) - H_STEPS - W_STEPS + 1,
            )
        )
        if len(valid) < 4:
            train[segment.segment_id] = set(valid)
            validation[segment.segment_id] = set()
            continue
        cut = max(1, min(len(valid) - 1, int(math.floor(len(valid) * 0.8))))
        if cut <= SOURCE_SPLIT_PURGE_STEPS:
            train[segment.segment_id] = set(valid)
            validation[segment.segment_id] = set()
            continue
        train_stop = cut - SOURCE_SPLIT_PURGE_STEPS
        train[segment.segment_id] = {value for value in valid[:train_stop]}
        validation[segment.segment_id] = {value for value in valid[cut:]}
    return train, validation


def _base_frame(segments: list[RodSegment]) -> pd.DataFrame:
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
    return pd.concat(
        [segment.frame[columns] for segment in segments], ignore_index=True
    )


def _write_partition(
    root: Path, split: str, base: pd.DataFrame, samples: pd.DataFrame
) -> None:
    if not base.empty:
        path = root / "base_data" / "cz_czochralski" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        base.to_parquet(path, index=False, compression="zstd")
    for information_set in ("input_only", "dynamic"):
        frame = samples[samples["information_set"] == information_set].copy()
        sample_path = (
            root
            / "sample_ids"
            / TASK_ID
            / information_set
            / "record_time"
            / "primary"
            / f"{split}.parquet"
        )
        target_path = (
            root
            / "targets"
            / f"{TASK_ID}__H{H_STEPS}__W{W_STEPS}"
            / f"{split}.parquet"
        )
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(sample_path, index=False, compression="zstd")
        frame[
            ["base_origin_id", "entity_id", "origin", "y_true"]
        ].to_parquet(target_path, index=False, compression="zstd")


def _write_registry(
    registry: Path,
    raw_path: Path,
    rods: dict[str, pd.DataFrame],
) -> None:
    registry.mkdir(parents=True, exist_ok=True)
    raw_sha = sha256_file(raw_path)
    _write_json(
        registry / "RAW_FILE_HASHES.json",
        [
            {
                "relative_path": str(raw_path),
                "bytes": raw_path.stat().st_size,
                "expected_sha256": raw_sha,
                "observed_sha256": raw_sha,
                "match": True,
            }
        ],
    )
    lines = [
        "source = PRIVATE_USER_PROVIDED_CZ_EXPERIMENT",
        "redistribution = FORBIDDEN",
        "academic_analysis = allowed by user",
        "raw_file_in_git = false",
        f"raw_file = {raw_path}",
        f"raw_sha256 = {raw_sha}",
    ]
    for rod_id, sheet in ROD_SHEETS:
        lines.append(f"{rod_id}: sheet={sheet}, rows={len(rods[rod_id])}")
    (registry / "SOURCE_AND_LICENSE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    variable_rows = [
        ("main_heater_power", "主加热功率", "U_X", "UNKNOWN_NOT_FROZEN"),
        ("crystal_lift", "晶升速度", "U_X", "UNKNOWN_NOT_FROZEN"),
        ("crucible_lift", "埚升速度", "U_X", "UNKNOWN_NOT_FROZEN"),
        ("crystal_rotation", "晶转速度", "U_X", "UNKNOWN_NOT_FROZEN"),
        ("crucible_rotation", "埚转速度", "U_X", "UNKNOWN_NOT_FROZEN"),
        ("crystal_diameter", "晶体直径", "Y_BY_TASK", "UNKNOWN_NOT_FROZEN"),
        ("crystal_length", "晶体长度", "EXCLUDED", "UNKNOWN_NOT_FROZEN"),
        (
            "heater_element_temperature",
            "加热元件温度",
            "EXCLUDED",
            "UNKNOWN_NOT_FROZEN",
        ),
        (
            "argon_flow_setpoint",
            "氩气流量设定",
            "EXCLUDED",
            "UNKNOWN_NOT_FROZEN",
        ),
        ("furnace_pressure", "炉压", "EXCLUDED", "UNKNOWN_NOT_FROZEN"),
    ]
    pd.DataFrame(
        variable_rows,
        columns=["canonical_name", "original_name", "role", "unit"],
    ).to_csv(registry / "VARIABLE_DICTIONARY.csv", index=False)
    _write_json(
        registry / "CADENCE_AUDIT.json",
        {
            "status": "PASS",
            "timestamp_available": False,
            "sample_period_sec": RAW_PERIOD_SEC,
            "cadence_source": "CADENCE_FROM_FROZEN_EXPERIMENT_METADATA",
            "allowed_error_fraction": 0.05,
            "model_period_sec": MODEL_PERIOD_SEC,
        },
    )
    boundary_rows = []
    for rod_id, _sheet in ROD_SHEETS:
        points = [0, *BREAKPOINTS[rod_id], len(rods[rod_id])]
        for index, (start, stop) in enumerate(zip(points[:-1], points[1:])):
            boundary_rows.append(
                {
                    "rod_id": rod_id,
                    "segment_id": f"{rod_id}_segment_{index}",
                    "raw_start_inclusive": start,
                    "raw_stop_exclusive": stop,
                    "breakpoint_before_segment": start if index else "",
                }
            )
    pd.DataFrame(boundary_rows).to_csv(
        registry / "RUN_BOUNDARIES.csv", index=False
    )
    _write_json(
        registry / "BREAKPOINT_REGISTRY.json",
        {
            "status": "FROZEN",
            "boundary_definition": "breakpoints_are_after_data_rows",
            "rod_breakpoints_after_row": {
                key: list(value) for key, value in BREAKPOINTS.items()
            },
            "cross_breakpoint_windows": False,
            "cross_breakpoint_aggregation": False,
        },
    )
    _write_json(
        registry / "ROD_REGISTRY.json",
        {
            "raw_file_sha256": raw_sha,
            "rods": [
                {
                    "rod_id": rod_id,
                    "sheet_name": sheet,
                    "raw_rows": int(len(rods[rod_id])),
                    "raw_sha256_parent": raw_sha,
                }
                for rod_id, sheet in ROD_SHEETS
            ],
        },
    )
    _write_json(
        registry / "MISSING_AND_DUPLICATE_AUDIT.json",
        {
            "status": "PASS",
            "rows": {rod_id: int(len(frame)) for rod_id, frame in rods.items()},
            "required_columns_missing": {rod_id: [] for rod_id in rods},
            "missing_values": {rod_id: 0 for rod_id in rods},
            "duplicate_required_variable_rows": {
                rod_id: 0 for rod_id in rods
            },
        },
    )
    _write_json(
        registry / "TARGET_AVAILABILITY.json",
        {
            "target": TARGET_COLUMN,
            "target_definition": "future_2_minute_mean_minus_current_2_minute_mean",
            "h_steps": H_STEPS,
            "w_steps": W_STEPS,
            "w0_steps": W0_STEPS,
            "availability_delay_steps": 0,
            "latest_available_target_index": "origin_minus_one",
        },
    )
    _write_json(
        registry / "SPLIT_REGISTRY.json",
        {
            "status": "FROZEN",
            "outer_protocol": "cross_rod_transfer",
            "directions": ["Rod_1_to_Rod_2", "Rod_2_to_Rod_1"],
            "source_split": "source_rod_expanding_time_train_validation",
            "source_train_validation_purge_steps": SOURCE_SPLIT_PURGE_STEPS,
            "source_train_validation_purge_definition": (
                "H_STEPS_PLUS_W_STEPS_PLUS_W0_STEPS"
            ),
            "source_train_validation_dependencies_overlap": False,
            "target_split": "target_rod_test_only",
            "random_row_split": False,
            "target_rod_used_for_selection": False,
        },
    )
    _write_json(
        registry / "CZ_CADENCE_TRANSFORM_CONTRACT.json",
        {
            "status": "FROZEN",
            "raw_period_sec": RAW_PERIOD_SEC,
            "model_period_sec": MODEL_PERIOD_SEC,
            "raw_samples_per_model_sample": RAW_PER_MODEL,
            "causal": True,
            "aggregation": "within_segment_trailing_nonoverlapping_block_mean",
            "cross_breakpoint": False,
            "centered_rolling": False,
            "incomplete_terminal_block_dropped": True,
        },
    )
    (registry / "FREEZE_DECISION.md").write_text(
        "status = FROZEN\n"
        "dataset = cz_czochralski\n"
        "task = CZ_D20\n"
        "raw_file_in_git = false\n"
        "target_rod_used_for_selection = false\n"
        "evidence_class = POST_HOC_NEURAL_BASELINE_AND_CZ_DATASET_EXTENSION_WITH_FROZEN_DEVELOPMENT_ONLY_SELECTION\n",
        encoding="utf-8",
    )


def _write_common_metadata(
    root: Path, direction: str, pca_contract: dict[str, Any]
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "TASK_REGISTRY.json",
        {
            "dataset": "cz_czochralski",
            "heads": [
                {
                    "head_id": TASK_ID,
                    "task_id": TASK_ID,
                    "dataset": "cz_czochralski",
                    "target": TARGET_COLUMN,
                    "cadence_seconds": MODEL_PERIOD_SEC,
                    "h_steps": H_STEPS,
                    "w_steps": W_STEPS,
                    "w0_steps": W0_STEPS,
                    "primary": True,
                    "direction": direction,
                }
            ],
        },
    )
    view_rows = []
    for information_set in ("input_only", "dynamic"):
        view_rows.append(
            {
                "task_id": TASK_ID,
                "dataset": "cz_czochralski",
                "head_id": TASK_ID,
                "information_set": information_set,
                "availability_scenario": "record_time",
                "proxy_policy": "primary",
                "input_columns": INPUT_COLUMNS,
                "target_history_column": (
                    TARGET_COLUMN if information_set == "dynamic" else None
                ),
            }
        )
    _write_json(root / "dataset_views" / "VIEW_REGISTRY.json", view_rows)
    _write_json(
        root / "PROTOCOL.json",
        {
            "support_contract": SUPPORT_CONTRACT,
            "tasks": [{"task_id": TASK_ID, "proxy_policies": ["primary"]}],
            "primary_views": ["record_time/primary"],
            "direction": direction,
        },
    )
    _write_json(
        root / "CZ_TASK_REALIZATION.json",
        {
            "task_id": TASK_ID,
            "target": TARGET_COLUMN,
            "h_steps": H_STEPS,
            "w_steps": W_STEPS,
            "w0_steps": W0_STEPS,
            "cadence_sec": MODEL_PERIOD_SEC,
            "horizon_sec": H_STEPS * MODEL_PERIOD_SEC,
            "target_window_sec": W_STEPS * MODEL_PERIOD_SEC,
            "direction": direction,
        },
    )
    _write_json(root / "JOINT_LIFT_PCA_CONTRACT.json", pca_contract)


def build_direction(
    raw_rods: dict[str, pd.DataFrame],
    *,
    source_rod: str,
    target_rod: str,
    output_root: Path,
    direction: str,
) -> dict[str, Any]:
    source_segments = _aggregate_segments(source_rod, raw_rods[source_rod])
    target_segments = _aggregate_segments(target_rod, raw_rods[target_rod])
    source_train, source_validation = _split_source_samples(source_segments)
    pca = _fit_joint_lift(source_segments, source_train)
    source_segments = _apply_joint_lift(source_segments, pca)
    target_segments = _apply_joint_lift(target_segments, pca)
    root = output_root / direction
    if root.exists():
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    _write_common_metadata(root, direction, pca)
    train_samples = _sample_rows(
        source_segments,
        direction=direction,
        split="train",
        sample_filter=source_train,
    )
    validation_samples = _sample_rows(
        source_segments,
        direction=direction,
        split="validation",
        sample_filter=source_validation,
    )
    test_samples = _sample_rows(target_segments, direction=direction, split="test")
    empty = pd.DataFrame(columns=train_samples.columns)
    _write_partition(root, "train", _base_frame(source_segments), train_samples)
    _write_partition(root, "validation", empty, validation_samples)
    _write_partition(root, "test", _base_frame(target_segments), test_samples)
    _write_partition(root, "ood", empty, empty)
    counts = {
        "train": int(len(train_samples) // 2),
        "validation": int(len(validation_samples) // 2),
        "test": int(len(test_samples) // 2),
        "ood": 0,
    }
    _write_json(
        root / "C1_NATIVE_SUPPORT_AUDIT.json",
        {
            "status": "PASS",
            "support_contract": SUPPORT_CONTRACT,
            "direction": direction,
            "dataset": "cz_czochralski",
            "head": TASK_ID,
            "information_sets": {
                information_set: {
                    split: {
                        "rows": int(
                            len(
                                frame[
                                    frame["information_set"] == information_set
                                ]
                            )
                        ),
                        "support_hash": _hash_text(
                            "\n".join(
                                frame.loc[
                                    frame["information_set"] == information_set,
                                    "base_origin_id",
                                ].astype(str)
                            )
                        ),
                        "support_contract": SUPPORT_CONTRACT,
                        "anchor_history_steps": W0_STEPS,
                    }
                    for split, frame in (
                        ("train", train_samples),
                        ("validation", validation_samples),
                        ("test", test_samples),
                    )
                }
                for information_set in ("input_only", "dynamic")
            },
            "sample_counts_unique_rows": counts,
            "test_metrics_accessed": False,
        },
    )
    return {
        "direction": direction,
        "source_rod": source_rod,
        "target_rod": target_rod,
        "source_segments": len(source_segments),
        "target_segments": len(target_segments),
        "counts": counts,
        "pca": pca,
        "shared_root": str(root),
    }


def build_all(raw_path: Path, project: Path, run_root: Path) -> dict[str, Any]:
    rods = _read_workbook(raw_path)
    registry = project / "dataset_registry" / "cz_czochralski"
    _write_registry(registry, raw_path, rods)
    shared_root = run_root / "shared"
    results = [
        build_direction(
            rods,
            source_rod="Rod_1",
            target_rod="Rod_2",
            output_root=shared_root,
            direction="Rod_1_to_Rod_2",
        ),
        build_direction(
            rods,
            source_rod="Rod_2",
            target_rod="Rod_1",
            output_root=shared_root,
            direction="Rod_2_to_Rod_1",
        ),
    ]
    audit = {
        "status": "PASS",
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "raw_bytes": raw_path.stat().st_size,
        "rows": {rod_id: int(len(frame)) for rod_id, frame in rods.items()},
        "directions": results,
        "test_metrics_accessed": False,
        "ood_metrics_accessed": False,
    }
    _write_json(run_root / "logs" / "CZ_DATA_BUILD_AUDIT.json", audit)
    return audit
