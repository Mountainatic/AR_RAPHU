"""Build the immutable CPU/GPU shared L6 benchmark views."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    atomic_json,
    atomic_npz,
    environment_snapshot,
    load_json,
    sha256_array,
    sha256_file,
)


def _imports(repo_root: Path):
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from MULTISCALE_PHYSICS_AUDIT_V1.src.baselines import (
        local_trend_prediction,
    )
    from MULTISCALE_PHYSICS_AUDIT_V1.src.data_loader import (
        load_workbook_data,
    )
    from MULTISCALE_PHYSICS_AUDIT_V1.src.multiresolution_lags import (
        expand_lag_blocks,
        lag_block_matrix,
    )
    from MULTISCALE_PHYSICS_AUDIT_V1.src.resampling import PCA1Transform
    from MULTISCALE_PHYSICS_AUDIT_V1.src.segmentation import Segment
    from MULTISCALE_PHYSICS_AUDIT_V1.src.targets import build_target_rows
    from MULTISCALE_PHYSICS_AUDIT_V1.src.timebase import Timebase

    return {
        "local_trend_prediction": local_trend_prediction,
        "load_workbook_data": load_workbook_data,
        "expand_lag_blocks": expand_lag_blocks,
        "lag_block_matrix": lag_block_matrix,
        "PCA1Transform": PCA1Transform,
        "Segment": Segment,
        "build_target_rows": build_target_rows,
        "Timebase": Timebase,
    }


def _causal_mean(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    output = np.empty_like(values)
    for index in range(len(values)):
        left = max(0, index - width + 1)
        output[index] = (
            cumulative[index + 1] - cumulative[left]
        ) / (index + 1 - left)
    return output


def _direction_arrays(
    workbook,
    *,
    train_sheet: str,
    test_sheet: str,
    protocol: dict[str, Any],
    api: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    starts = {
        sheet: max(protocol["breakpoints"][sheet]) + 1
        for sheet in protocol["required_sheets"]
    }
    columns_by_sheet = {
        sheet: {
            name: np.asarray(values[starts[sheet] :], dtype=np.float64)
            for name, values in workbook.rods[sheet].columns.items()
        }
        for sheet in protocol["required_sheets"]
    }
    lift_train = np.column_stack(
        (
            columns_by_sheet[train_sheet]["晶升速度"],
            columns_by_sheet[train_sheet]["埚升速度"],
        )
    )
    pca = api["PCA1Transform"].fit(lift_train)
    timebase = api["Timebase"](float(protocol["sample_period_sec"]))
    blocks = api["expand_lag_blocks"](
        protocol["lag_blocks_min"],
        history_min=float(protocol["history_min"]),
    )
    cadence_raw = timebase.cadence_step(float(protocol["cadence_sec"]))
    sequence_steps = timebase.samples_for_minutes(
        float(protocol["history_min"])
    ) // cadence_raw
    direction_arrays: dict[str, dict[str, np.ndarray]] = {}
    for role, sheet in (("train", train_sheet), ("test", test_sheet)):
        columns = columns_by_sheet[sheet]
        lift_matrix = np.column_stack(
            (columns["晶升速度"], columns["埚升速度"])
        )
        signals = {
            "joint_lift": pca.transform(lift_matrix),
            "heater_power": columns["主加热功率"],
            "crystal_rotation": columns["晶转速度"],
            "crucible_rotation": columns["埚转速度"],
        }
        target = columns["晶体直径"]
        rows = api["build_target_rows"](
            target,
            api["Segment"](0, len(target), "main"),
            timebase=timebase,
            cadence_sec=float(protocol["cadence_sec"]),
            horizon_min=float(protocol["horizon_min"]),
            target_window_min=float(protocol["target_window_min"]),
            history_min=float(protocol["history_min"]),
        )
        multires = np.column_stack(
            [
                api["lag_block_matrix"](
                    signals[channel],
                    rows.origins,
                    blocks,
                    timebase=timebase,
                )
                for channel in protocol["controls"]
            ]
        )
        lag_points = np.arange(sequence_steps, dtype=np.int64) * cadence_raw
        sequence_index = rows.origins[:, None] - lag_points[None, :]
        if int(sequence_index.min()) < 0:
            raise AssertionError("SEQUENCE_HISTORY_CROSSES_BOUNDARY")
        anti_aliased = {
            channel: _causal_mean(values, cadence_raw)
            for channel, values in signals.items()
        }
        sequence_u = np.stack(
            [
                np.column_stack(
                    [anti_aliased[channel][sequence_index] for channel in protocol["controls"]]
                )
                for sequence_index in sequence_index
            ]
        )
        # The loop above yields (samples, sequence_steps, controls).
        sequence_y = target[sequence_index]
        y_centered = sequence_y - rows.current_mean[:, None]
        global_origins = rows.origins + starts[sheet]
        future_left = global_origins + rows.horizon_samples
        future_right = future_left + rows.window_samples
        sample_ids = np.array(
            [
                f"{sheet}:origin={origin}:future={left}:{right}"
                for origin, left, right in zip(
                    global_origins, future_left, future_right
                )
            ],
            dtype="U80",
        )
        maturity_rows = int(
            round(
                float(protocol["residual_ar"]["maturity_min"])
                * 60.0
                / float(protocol["cadence_sec"])
            )
        )
        maximum_residual_history = int(
            round(
                max(protocol["residual_ar"]["history_candidates_min"])
                * 60.0
                / float(protocol["cadence_sec"])
            )
        )
        evaluation_mask = np.arange(len(sample_ids)) >= (
            maturity_rows + maximum_residual_history
        )
        if np.any(global_origins >= future_left):
            raise AssertionError("TARGET_NOT_IN_FUTURE")
        if int(sequence_index.max()) > int(rows.origins.max()):
            raise AssertionError("FUTURE_INPUT_DETECTED")
        direction_arrays[role] = {
            "sample_id": sample_ids,
            "origin_raw_index": global_origins.astype(np.int64),
            "future_left_raw_index": future_left.astype(np.int64),
            "future_right_raw_index": future_right.astype(np.int64),
            "target_z": rows.target.astype(np.float64),
            "current_y_mean": rows.current_mean.astype(np.float64),
            "future_y_mean": rows.future_mean.astype(np.float64),
            "persistence_prediction": np.zeros(len(rows.target), dtype=np.float64),
            "local_trend_prediction": api["local_trend_prediction"](
                target, rows
            ).astype(np.float64),
            "multiresolution_u": multires.astype(np.float64),
            "sequence_u": sequence_u.astype(np.float32),
            "sequence_y": sequence_y.astype(np.float32),
            "sequence_y_centered": y_centered.astype(np.float32),
            "evaluation_mask": evaluation_mask,
        }
    metadata = {
        "train_sheet": train_sheet,
        "test_sheet": test_sheet,
        "stable_starts_raw": starts,
        "pca": {
            "mean": pca.mean.tolist(),
            "scale": pca.scale.tolist(),
            "vector": pca.vector.tolist(),
            "explained_fraction": pca.explained_fraction,
            "sign_rule": protocol["pca_sign_rule"],
        },
        "lag_blocks": [
            {
                "start_min": block.start_min,
                "stop_min": block.stop_min,
                "midpoint_min": block.midpoint_min,
            }
            for block in blocks
        ],
        "multiresolution_columns": [
            f"{channel}:lag={block.start_min:g}-{block.stop_min:g}min"
            for channel in protocol["controls"]
            for block in blocks
        ],
        "sequence_steps": sequence_steps,
        "sequence_control_order": protocol["controls"],
        "maturity_rows": int(
            round(
                float(protocol["residual_ar"]["maturity_min"])
                * 60.0
                / float(protocol["cadence_sec"])
            )
        ),
        "max_residual_history_rows": int(
            round(
                max(protocol["residual_ar"]["history_candidates_min"])
                * 60.0
                / float(protocol["cadence_sec"])
            )
        ),
    }
    return direction_arrays, metadata


def _save_direction(
    shared_root: Path,
    direction_name: str,
    arrays: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    direction_root = shared_root / direction_name
    records: list[dict[str, Any]] = []
    for role, values in arrays.items():
        tabular = {
            key: value
            for key, value in values.items()
            if key
            not in {"sequence_u", "sequence_y", "sequence_y_centered"}
        }
        sequence = {
            "sample_id": values["sample_id"],
            "sequence_u": values["sequence_u"],
            "sequence_y": values["sequence_y"],
            "sequence_y_centered": values["sequence_y_centered"],
            "target_z": values["target_z"],
            "evaluation_mask": values["evaluation_mask"],
        }
        tabular_path = (
            direction_root / "multiresolution_tabular_view" / f"{role}.npz"
        )
        sequence_path = direction_root / "sequence_view" / f"{role}.npz"
        atomic_npz(tabular_path, **tabular)
        atomic_npz(sequence_path, **sequence)
        graph_path = direction_root / "graph_view" / f"{role}.npz"
        adjacency = np.eye(len(metadata["sequence_control_order"]), dtype=np.float32)
        atomic_npz(
            graph_path,
            sample_id=values["sample_id"],
            node_sequence=np.transpose(values["sequence_u"], (0, 2, 1)),
            adjacency=adjacency,
            target_z=values["target_z"],
            evaluation_mask=values["evaluation_mask"],
        )
        for view, path, stored in (
            ("multiresolution_tabular_view", tabular_path, tabular),
            ("sequence_view", sequence_path, sequence),
            (
                "graph_view",
                graph_path,
                {
                    "sample_id": values["sample_id"],
                    "node_sequence": np.transpose(
                        values["sequence_u"], (0, 2, 1)
                    ),
                    "adjacency": adjacency,
                    "target_z": values["target_z"],
                    "evaluation_mask": values["evaluation_mask"],
                },
            ),
        ):
            records.append(
                {
                    "direction": direction_name,
                    "role": role,
                    "view": view,
                    "file": path.relative_to(shared_root).as_posix(),
                    "arrays": {
                        key: {
                            "dtype": str(value.dtype),
                            "shape": list(value.shape),
                            "sha256": sha256_array(value),
                        }
                        for key, value in stored.items()
                    },
                    "file_sha256": sha256_file(path),
                }
            )
    atomic_json(direction_root / "metadata.json", metadata)
    return records


def validate_shared(shared_root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (shared_root / "DATA_AND_SPLIT_HASHES.json").read_text(encoding="utf-8")
    )
    problems: list[str] = []
    for record in manifest["files"]:
        path = shared_root / record["file"]
        if not path.is_file():
            problems.append(f"MISSING:{record['file']}")
            continue
        if sha256_file(path) != record["file_sha256"]:
            problems.append(f"FILE_HASH:{record['file']}")
        with np.load(path) as stored:
            for name, expected in record["arrays"].items():
                if name not in stored.files:
                    problems.append(f"MISSING_ARRAY:{record['file']}:{name}")
                    continue
                value = stored[name]
                if sha256_array(value) != expected["sha256"]:
                    problems.append(f"ARRAY_HASH:{record['file']}:{name}")
    forbidden = [
        path.relative_to(shared_root).as_posix()
        for path in shared_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls"}
    ]
    problems.extend(f"FORBIDDEN:{value}" for value in forbidden)
    return {
        "status": "PASS" if not problems else "FAIL",
        "files_checked": len(manifest["files"]),
        "problems": problems,
    }


def build_shared_dataset(
    *,
    repo_root: Path,
    project_root: Path,
    data_path: Path,
    config_path: Path,
    shared_root: Path,
    package_path: Path,
) -> dict[str, Any]:
    protocol, config_sha = load_json(config_path)
    if sha256_file(data_path) != protocol["data_sha256"]:
        raise RuntimeError("DATA_SHA256_MISMATCH")
    api = _imports(repo_root)
    workbook = api["load_workbook_data"](
        data_path,
        required_sheets=protocol["required_sheets"],
        required_columns=protocol["required_columns"],
    )
    if shared_root.exists():
        shutil.rmtree(shared_root)
    shared_root.mkdir(parents=True)
    atomic_json(
        shared_root / "BENCHMARK_PROTOCOL.json",
        {
            **protocol,
            "config_sha256": config_sha,
            "frozen": True,
            "regeneration_by_gpu_forbidden": True,
        },
    )
    records: list[dict[str, Any]] = []
    direction_summaries = {}
    for train_sheet, test_sheet in protocol["outer_directions"]:
        name = f"{train_sheet.lower()}_to_{test_sheet.lower()}"
        arrays, metadata = _direction_arrays(
            workbook,
            train_sheet=train_sheet,
            test_sheet=test_sheet,
            protocol=protocol,
            api=api,
        )
        records.extend(_save_direction(shared_root, name, arrays, metadata))
        direction_summaries[name] = {
            "train_samples": int(len(arrays["train"]["target_z"])),
            "test_samples": int(len(arrays["test"]["target_z"])),
            "evaluation_samples": int(
                np.sum(arrays["test"]["evaluation_mask"])
            ),
            "train_sample_id_sha256": sha256_array(
                arrays["train"]["sample_id"]
            ),
            "test_sample_id_sha256": sha256_array(
                arrays["test"]["sample_id"]
            ),
            "target_sha256": {
                "train": sha256_array(arrays["train"]["target_z"]),
                "test": sha256_array(arrays["test"]["target_z"]),
            },
        }
    atomic_json(
        shared_root / "DATA_AND_SPLIT_HASHES.json",
        {
            "schema": protocol["schema"],
            "data_sha256": workbook.sha256,
            "config_sha256": config_sha,
            "directions": direction_summaries,
            "files": records,
        },
    )
    atomic_json(
        shared_root / "scaler_and_pca_metadata" / "README.json",
        {
            "status": "EMBEDDED_IN_DIRECTION_METADATA",
            "rule": "PCA and all preprocessing are fit on the training rod only.",
        },
    )
    atomic_json(
        shared_root / "masks" / "README.json",
        {
            "evaluation_mask": (
                "Common final evaluation subset after residual maturity plus "
                "maximum 40-minute residual history."
            )
        },
    )
    atomic_json(
        shared_root / "target_arrays" / "README.json",
        {
            "target": "future 2-minute mean diameter at +20 minutes minus current 2-minute mean",
            "unit": "source diameter unit",
        },
    )
    validation = validate_shared(shared_root)
    atomic_json(shared_root / "VALIDATION_REPORT.json", validation)
    lines = [
        "# Shared L6 Benchmark Dataset Validation",
        "",
        f"- Status: **{validation['status']}**",
        f"- Source data SHA256: `{workbook.sha256}`",
        f"- Protocol SHA256: `{config_sha}`",
        "- Target: L6, 20-minute horizon, 2-minute output window, 40-minute history.",
        "- All histories are causal and remain inside the frozen stable segment.",
        "- PCA is fitted separately on each training rod and frozen for its test rod.",
        "- The shared package contains no raw Excel workbook.",
        "",
        "This package is immutable input for both CPU and GPU benchmark batches.",
        "",
    ]
    (shared_root / "VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if validation["status"] != "PASS":
        raise RuntimeError(f"SHARED_VALIDATION_FAILED:{validation['problems']}")
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.unlink(missing_ok=True)
    with zipfile.ZipFile(
        package_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for path in sorted(shared_root.rglob("*")):
            if path.is_file():
                bundle.write(path, Path("SHARED_BENCHMARK_DATASET") / path.relative_to(shared_root))
    with tempfile.TemporaryDirectory(prefix="shared_l6_verify_") as temporary:
        with zipfile.ZipFile(package_path) as bundle:
            if any(name.lower().endswith((".xlsx", ".xls")) for name in bundle.namelist()):
                raise RuntimeError("RAW_EXCEL_IN_SHARED_PACKAGE")
            bundle.extractall(temporary)
        roundtrip = validate_shared(
            Path(temporary) / "SHARED_BENCHMARK_DATASET"
        )
        if roundtrip["status"] != "PASS":
            raise RuntimeError(f"SHARED_ROUNDTRIP_FAILED:{roundtrip}")
    return {
        "status": "PASS",
        "shared_root": str(shared_root.resolve()),
        "package": str(package_path.resolve()),
        "package_sha256": sha256_file(package_path),
        "package_size": package_path.stat().st_size,
        "protocol_sha256": config_sha,
        "data_sha256": workbook.sha256,
        "directions": direction_summaries,
        "environment": environment_snapshot(repo_root),
    }
