from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyreadr


DATASET_NAMES = ("tep", "debutanizer", "sru", "pmsm", "metropt")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_numeric_text(path: Path, width: int) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    header: list[str] = []
    number = re.compile(r"^[\s]*[+-]?(?:\d|\.\d)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not number.match(line):
            if line.strip():
                header.append(line.strip())
            continue
        try:
            values = [float(token) for token in line.split()]
        except ValueError:
            header.append(line.strip())
            continue
        if len(values) != width:
            raise ValueError(f"{path}: expected {width} columns, got {len(values)}")
        rows.append(values)
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{path}: no valid {width}-column body")
    return array, header


@dataclass(frozen=True)
class AuditResult:
    dataset: str
    raw_files: list[Path]
    variables: list[dict[str, Any]]
    cadence: dict[str, Any]
    boundaries: list[dict[str, Any]]
    data_quality: dict[str, Any]
    target_availability: dict[str, Any]
    split_registry: dict[str, Any]
    source_license_markdown: str
    decision: str
    blockers: list[str]


def _common_quality(frame: pd.DataFrame, duplicate_subset: list[str] | None = None) -> dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number])
    constants = [str(c) for c in numeric.columns if numeric[c].nunique(dropna=False) <= 1]
    duplicate_rows = int(frame.duplicated(subset=duplicate_subset).sum())
    return {
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "missing_by_column": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "duplicate_rows": duplicate_rows,
        "constant_columns": constants,
    }


def audit_tep(raw_root: Path) -> AuditResult:
    paths = [
        raw_root / "tep_rieth/TEP_FaultFree_Training.RData",
        raw_root / "tep_rieth/TEP_FaultFree_Testing.RData",
        raw_root / "tep_rieth/TEP_Faulty_Training.RData",
        raw_root / "tep_rieth/TEP_Faulty_Testing.RData",
    ]
    summaries: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    columns: list[str] | None = None
    for path in paths:
        objects = pyreadr.read_r(path)
        if len(objects) != 1:
            raise ValueError(f"{path}: expected exactly one R object")
        object_name, frame = next(iter(objects.items()))
        cols = [str(c) for c in frame.columns]
        if columns is None:
            columns = cols
        elif columns != cols:
            raise ValueError("TEP RData column mismatch")
        key = ["faultNumber", "simulationRun", "sample"]
        group_sizes = frame.groupby(["faultNumber", "simulationRun"], sort=True).size()
        for (fault, run), size in group_sizes.items():
            boundaries.append(
                {
                    "source_file": path.name,
                    "object": object_name,
                    "fault_number": int(fault),
                    "simulation_run": int(run),
                    "samples": int(size),
                }
            )
        summaries.append(
            {
                "file": path.name,
                "object": object_name,
                "quality": _common_quality(frame, key),
                "fault_numbers": sorted(int(x) for x in frame["faultNumber"].unique()),
                "simulation_runs": int(frame["simulationRun"].nunique()),
                "sample_min": int(frame["sample"].min()),
                "sample_max": int(frame["sample"].max()),
            }
        )
        del frame, objects
    assert columns is not None
    variables = []
    for column in columns:
        if column in {"faultNumber", "simulationRun", "sample"}:
            role, view = "INDEX", "metadata"
        elif column == "xmeas_40":
            role, view = "Y", "target"
        elif column in {f"xmeas_{i}" for i in range(37, 42)}:
            role, view = "PROXY_EXCLUDED", "excluded_primary"
        elif column.startswith("xmv_"):
            role, view = "U", "primary_candidate"
        else:
            role, view = "X", "primary_candidate_pending_upstream_registration"
        variables.append({"name": column, "role": role, "primary_view": view, "unit": "SOURCE_DICTIONARY_REQUIRED"})
    return AuditResult(
        dataset="tep",
        raw_files=paths,
        variables=variables,
        cadence={
            "physical_seconds": 180,
            "source": "Rieth dataset metadata",
            "observed_sample_step": 1,
            "status": "SUPPORTED",
        },
        boundaries=boundaries,
        data_quality={"files": summaries},
        target_availability={
            "target": "xmeas_40",
            "main": "record_time",
            "sensitivity": "analyzer_maturity_15_minutes",
            "maturity_delay_seconds": 900,
        },
        split_registry={
            "unit": ["source_file", "faultNumber", "simulationRun"],
            "rule": "complete_run_nominal_disturbance_stratified",
            "exact_ids": "PENDING_STAGE0_FREEZE",
            "unseen_disturbance_ood": "PENDING_STAGE0_FREEZE",
        },
        source_license_markdown=(
            "# TEP source and license\n\n"
            "Canonical data: Rieth et al. Tennessee Eastman Process Simulation Data, "
            "Harvard Dataverse DOI `10.7910/DVN/6C3JR1`. The local four RData files "
            "are never redistributed by this project.\n"
        ),
        decision="BLOCKED",
        blockers=["TEP_EXACT_RUN_SPLIT_NOT_FROZEN"],
    )


def audit_debutanizer(raw_root: Path) -> AuditResult:
    path = raw_root / "industrial-debutanizer-soft-sensor/data/debutanizer_data.txt"
    array, header = read_numeric_text(path, 8)
    frame = pd.DataFrame(array, columns=[f"u{i}" for i in range(1, 8)] + ["y"])
    variables = [
        {"name": name, "role": "Y" if name == "y" else "U_X", "primary_view": "target" if name == "y" else "primary", "unit": "SOURCE_DICTIONARY_REQUIRED"}
        for name in frame.columns
    ]
    return AuditResult(
        dataset="debutanizer",
        raw_files=[path],
        variables=variables,
        cadence={"physical_seconds": 360, "source": "Fortuna literature", "status": "LITERATURE_SUPPORTED_NO_TIMESTAMP"},
        boundaries=[{"sequence_id": "debutanizer_1", "start_row": 0, "stop_row": len(frame), "samples": len(frame)}],
        data_quality={**_common_quality(frame), "header_lines": header},
        target_availability={
            "target": "y",
            "file_declares_pretranslated_samples": 8,
            "main": "record_time_use_published_pretranslated_target",
            "sensitivity_delay_minutes": 60,
        },
        split_registry={
            "rule": "chronological_60_20_20_floor_boundaries",
            "train": [0, math.floor(0.6 * len(frame))],
            "validation": [math.floor(0.6 * len(frame)), math.floor(0.8 * len(frame))],
            "test": [math.floor(0.8 * len(frame)), len(frame)],
            "inner": "four_fold_expanding",
            "purge_seconds": "max_history_plus_horizon_plus_target_window_plus_label_delay",
        },
        source_license_markdown=(
            "# Debutanizer source and license\n\n"
            "Data originate from the Fortuna et al. industrial soft-sensor material "
            "and associated publication DOI `10.1016/j.conengprac.2004.04.013`. "
            "Treat as copyrighted supplementary material: academic analysis only in "
            "this project and no raw redistribution.\n"
        ),
        decision="PASS",
        blockers=[],
    )


def audit_sru(raw_root: Path) -> AuditResult:
    path = raw_root / "sru/SRU_data.txt"
    array, header = read_numeric_text(path, 7)
    columns = [f"u{i}" for i in range(1, 6)] + ["y1", "y2"]
    frame = pd.DataFrame(array, columns=columns)
    changes = {}
    for target in ("y1", "y2"):
        delta = np.diff(frame[target].to_numpy())
        change_indices = np.flatnonzero(delta != 0) + 1
        changes[target] = {
            "unique_values": int(frame[target].nunique()),
            "nonzero_change_count": int(len(change_indices)),
            "median_change_gap_rows": float(np.median(np.diff(change_indices))) if len(change_indices) > 1 else None,
        }
    variables = [
        {"name": name, "role": "Y" if name.startswith("y") else "U", "primary_view": "target" if name.startswith("y") else "primary", "unit": "scaled"}
        for name in columns
    ]
    return AuditResult(
        dataset="sru",
        raw_files=[path],
        variables=variables,
        cadence={
            "process_seconds": 60,
            "quality_measurement_seconds": 60,
            "source": "published Line 4 SRU description; 30-minute analyzer report retained as provenance sensitivity",
            "dense_target_change_audit": changes,
            "status": "SUPPORTED_PRIMARY_WITH_DOCUMENTED_PROVENANCE_CONFLICT",
        },
        boundaries=[{"sequence_id": "sru_line4", "start_row": 0, "stop_row": len(frame), "samples": len(frame), "line_id": 4}],
        data_quality={**_common_quality(frame), "header_lines": header},
        target_availability={
            "targets": {"SRU_H2S": "y1", "SRU_SO2": "y2"},
            "main": "record_time_dense_one_minute_line4",
            "provenance_sensitivity": "some literature describes 30-minute analyzer measurements",
        },
        split_registry={
            "rule": "chronological_60_20_20_floor_boundaries",
            "train": [0, math.floor(0.6 * len(frame))],
            "validation": [math.floor(0.6 * len(frame)), math.floor(0.8 * len(frame))],
            "test": [math.floor(0.8 * len(frame)), len(frame)],
            "inner": "four_fold_expanding",
            "purge_seconds": "max_history_plus_horizon_plus_target_window",
        },
        source_license_markdown=(
            "# SRU source and license\n\n"
            "Data are from the Fortuna et al. industrial soft-sensor supplementary "
            "material. Treat as copyrighted book material: academic analysis only "
            "and no raw redistribution.\n"
        ),
        decision="PASS",
        blockers=[],
    )


def audit_pmsm(raw_root: Path) -> AuditResult:
    path = raw_root / "pmsm_original/measures_v2.csv"
    frame = pd.read_csv(path)
    expected = {"ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque", "pm", "profile_id"}
    missing_expected = sorted(expected.difference(frame.columns))
    if missing_expected:
        raise ValueError(f"PMSM missing expected columns: {missing_expected}")
    boundaries = [
        {"profile_id": int(profile), "samples": int(size), "duration_seconds": float(size * 0.5)}
        for profile, size in frame.groupby("profile_id", sort=True).size().items()
    ]
    primary = {"ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"}
    secondary = {"stator_winding", "stator_tooth", "stator_yoke"}
    variables = []
    for name in frame.columns:
        if name == "pm": role, view = "Y", "target"
        elif name == "profile_id": role, view = "INDEX", "metadata"
        elif name in primary: role, view = "U_X", "primary"
        elif name in secondary: role, view = "PROXY", "full_sensor_secondary"
        else: role, view = "UNREGISTERED", "excluded"
        variables.append({"name": name, "role": role, "primary_view": view, "unit": "KAGGLE_DATA_DICTIONARY"})
    return AuditResult(
        dataset="pmsm",
        raw_files=[path],
        variables=variables,
        cadence={"physical_seconds": 0.5, "source": "Kaggle dataset card: 2 Hz", "status": "SUPPORTED"},
        boundaries=boundaries,
        data_quality=_common_quality(frame),
        target_availability={"target": "pm", "main": "current_target_available"},
        split_registry={"unit": "profile_id", "exact_ids": "PENDING_STAGE0_DURATION_COMPLETENESS_ONLY_ALLOCATION"},
        source_license_markdown=(
            "# PMSM source and license\n\n"
            "Kaggle `wkirgsn/electric-motor-temperature`, dataset version containing "
            "`measures_v2.csv`; license CC BY-SA 4.0. Raw data is excluded from all "
            "return packages.\n"
        ),
        decision="BLOCKED",
        blockers=["PMSM_EXACT_PROFILE_SPLIT_NOT_FROZEN"],
    )


def audit_metropt(raw_root: Path) -> AuditResult:
    path = raw_root / "metropt/metropt_3.zip"
    member = "MetroPT3(AirCompressor).csv"
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        frame = pd.read_csv(handle)
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    diffs = timestamps.diff().dt.total_seconds().dropna()
    month_counts = timestamps.dt.to_period("M").astype(str).value_counts().sort_index()
    boundaries = [
        {"month": month, "samples": int(size)} for month, size in month_counts.items()
    ]
    primary_p = set(frame.columns) - {"timestamp", "Unnamed: 0", "Reservoirs", "TP3"}
    primary_oil = set(frame.columns) - {"timestamp", "Unnamed: 0", "Oil_temperature"}
    variables = []
    for name in frame.columns:
        if name == "timestamp" or name == "Unnamed: 0": role = "INDEX"
        elif name in {"Reservoirs", "Oil_temperature"}: role = "Y_BY_TASK"
        elif name == "TP3": role = "PROXY_FOR_RESERVOIRS"
        else: role = "U_X"
        variables.append(
            {
                "name": name,
                "role": role,
                "primary_view": "METRO_P60" if name in primary_p else ("METRO_OIL20" if name in primary_oil else "task_dependent"),
                "unit": "UCI_DATA_DICTIONARY",
            }
        )
    quality = _common_quality(frame)
    quality.update(
        {
            "duplicate_timestamps": int(timestamps.duplicated().sum()),
            "timestamp_start": timestamps.min().isoformat(),
            "timestamp_stop": timestamps.max().isoformat(),
        }
    )
    cadence = {
        "official_conflict": ["1 Hz prose", "0.1 Hz feature description"],
        "observed_median_seconds": float(diffs.median()),
        "observed_quantiles_seconds": {str(q): float(diffs.quantile(q)) for q in (0.0, 0.01, 0.5, 0.99, 1.0)},
        "physical_seconds": 10,
        "status": "SUPPORTED_BY_RAW_TIMESTAMPS",
    }
    fault_windows = [
        ["2020-04-18 00:00", "2020-04-18 23:59"],
        ["2020-05-29 23:30", "2020-05-30 06:00"],
        ["2020-06-05 10:00", "2020-06-07 14:30"],
        ["2020-07-15 14:30", "2020-07-15 19:00"],
    ]
    return AuditResult(
        dataset="metropt",
        raw_files=[path],
        variables=variables,
        cadence=cadence,
        boundaries=boundaries,
        data_quality=quality,
        target_availability={"METRO_P60": "Reservoirs", "METRO_OIL20": "Oil_temperature", "main": "current_target_available"},
        split_registry={
            "unit": "calendar_month",
            "train": ["2020-02", "2020-03", "2020-04"],
            "validation": ["2020-05"],
            "test": ["2020-06", "2020-07", "2020-08"],
            "fault_windows": fault_windows,
        },
        source_license_markdown=(
            "# MetroPT-3 source and license\n\n"
            "UCI dataset 791, DOI `10.24432/C5VW3R`, CC BY 4.0. Raw data is "
            "excluded from all return packages.\n"
        ),
        decision="PASS",
        blockers=[],
    )


AUDITORS = {
    "tep": audit_tep,
    "debutanizer": audit_debutanizer,
    "sru": audit_sru,
    "pmsm": audit_pmsm,
    "metropt": audit_metropt,
}


def materialize(result: AuditResult, output_root: Path) -> dict[str, Any]:
    out = output_root / result.dataset
    out.mkdir(parents=True, exist_ok=True)
    raw_hashes = [
        {
            "relative_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in result.raw_files
    ]
    write_json(out / "RAW_FILE_HASHES.json", raw_hashes)
    write_csv(out / "VARIABLE_DICTIONARY.csv", result.variables, ["name", "role", "primary_view", "unit"])
    write_json(out / "CADENCE_AUDIT.json", result.cadence)
    boundary_fields = sorted({key for row in result.boundaries for key in row})
    write_csv(out / "RUN_BOUNDARIES.csv", result.boundaries, boundary_fields)
    write_json(out / "MISSING_AND_DUPLICATE_AUDIT.json", result.data_quality)
    write_json(out / "TARGET_AVAILABILITY.json", result.target_availability)
    write_json(out / "SPLIT_REGISTRY.json", result.split_registry)
    (out / "SOURCE_AND_LICENSE.md").write_text(result.source_license_markdown, encoding="utf-8")
    blocker_lines = [f"- `{blocker}`" for blocker in result.blockers] or ["- None"]
    decision_text = [
        f"# {result.dataset} freeze decision",
        "",
        f"Status: `{result.decision}`",
        "",
        "Blockers:",
        "",
        *blocker_lines,
    ]
    (out / "FREEZE_DECISION.md").write_text("\n".join(decision_text) + "\n", encoding="utf-8")
    return {"dataset": result.dataset, "decision": result.decision, "blockers": result.blockers, "raw_hashes": raw_hashes}


def run_stage0(raw_root: Path, output_root: Path, dataset: str = "all") -> dict[str, Any]:
    raw_root = raw_root.resolve(strict=True)
    output_root = output_root.resolve()
    selected = list(DATASET_NAMES) if dataset == "all" else [dataset]
    unknown = set(selected).difference(AUDITORS)
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    results = []
    for name in selected:
        audit = AUDITORS[name](raw_root)
        results.append(materialize(audit, output_root))
    overall = "PASS" if len(results) == len(DATASET_NAMES) and all(item["decision"] == "PASS" for item in results) else "BLOCKED"
    summary = {"stage": "C0", "status": overall, "datasets": results}
    write_json(output_root / "C0_DECISION.json", summary)
    return summary
