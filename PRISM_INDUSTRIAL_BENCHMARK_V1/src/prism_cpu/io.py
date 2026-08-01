from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DatasetFrame:
    name: str
    frame: pd.DataFrame
    target_columns: list[str]
    input_columns: list[str]
    run_id: np.ndarray
    time_column: str | None
    source_files: list[Path]
    metadata: dict = field(default_factory=dict)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";", engine="python")
    df = df.drop(columns=[c for c in df.columns if str(c).lower().startswith("unnamed")], errors="ignore")
    return df


def _contiguous_runs(df: pd.DataFrame, status_col: str | None = None) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if status_col and status_col in df:
        s = df[status_col].astype(str).fillna("").to_numpy()
        return np.cumsum(np.r_[True, s[1:] != s[:-1]]).astype(np.int64) - 1
    return np.zeros(n, dtype=np.int64)


def load_tep(raw_root: Path) -> DatasetFrame:
    candidates = [raw_root / "new_tep_datasets" / "matlab_data_1year.csv", raw_root / "new_tep_datasets" / "python_data_1year.csv"]
    path = next((p for p in candidates if p.exists() and p.stat().st_size > 10_000), None)
    if path is None:
        raise FileNotFoundError("TEP CSV not found")
    df = _read_csv_auto(path)
    target = "XMEAS(40)"
    if target not in df:
        raise ValueError(f"TEP target missing: {target}")
    numeric = _numeric_columns(df)
    excluded = {target, "STATUS"} | {f"XMEAS({i})" for i in range(37, 42)}
    inputs = [c for c in numeric if c not in excluded]
    return DatasetFrame("TEP", df, [target], inputs, _contiguous_runs(df, "STATUS"), None, [path], {"cadence_seconds": 180, "source_variant": path.name})


def load_debutanizer(raw_root: Path) -> DatasetFrame:
    path = raw_root / "industrial-debutanizer-soft-sensor" / "data" / "debutanizer_data.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=3, engine="python")
    df = df.iloc[:, :8]
    df.columns = [f"u{i}" for i in range(1, 8)] + ["y"]
    inputs = [f"u{i}" for i in range(1, 8)]
    return DatasetFrame("Debutanizer", df, ["y"], inputs, np.zeros(len(df), dtype=np.int64), None, [path], {"cadence_seconds": None, "label_shift_note": "source states y translated by 8 samples"})


def _find_member(zf: zipfile.ZipFile, patterns: Iterable[str]) -> str | None:
    names = zf.namelist()
    for p in patterns:
        for n in names:
            if re.search(p, n, flags=re.I) and not n.endswith("/"):
                return n
    return None


def load_pmsm(raw_root: Path) -> DatasetFrame:
    path = raw_root / "pmsm" / "pmsm_rotor_temp.zip"
    if not path.exists() or path.stat().st_size < 1000:
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zf:
        # The archive also contains four metadata/text files with different schemas;
        # only the registered profile CSVs enter the PMSM task.
        members = [n for n in zf.namelist() if re.search(r"\.csv$", n, flags=re.I) and not n.endswith("/")]
        if not members:
            raise ValueError("PMSM archive contains no tabular file")
        parts = []
        for member in members:
            with zf.open(member) as f:
                part = pd.read_csv(f)
            part["_profile_id"] = Path(member).stem
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
    rename = {str(c).strip(): str(c).strip() for c in df.columns}
    df = df.rename(columns=rename)
    lower = {str(c).lower(): c for c in df.columns}
    target = next((lower[k] for k in ("pm", "pm temperature", "pm_temp", "pm_temperature") if k in lower), None)
    if target is None:
        raise ValueError(f"PMSM pm target missing; columns={list(df.columns)}")
    profile_col = next((c for c in df.columns if str(c).lower() in {"profile_id", "profile", "profileid", "_profile_id"}), None)
    numeric = _numeric_columns(df)
    forbidden = {target, "stator_winding", "stator_tooth", "stator_yoke"}
    inputs = [c for c in numeric if c not in forbidden]
    runs = df[profile_col].astype(str).factorize()[0].astype(np.int64) if profile_col else np.zeros(len(df), dtype=np.int64)
    return DatasetFrame("PMSM", df, [target], inputs, runs, None, [path], {"cadence_seconds": 0.5, "profile_column": profile_col, "archive_members": len(members)})


def load_metro(raw_root: Path) -> DatasetFrame:
    path = raw_root / "metropt" / "metropt_3.zip"
    if not path.exists() or path.stat().st_size < 1000:
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zf:
        member = _find_member(zf, [r"\.csv$"])
        if member is None:
            raise ValueError("MetroPT archive contains no CSV")
        with zf.open(member) as f:
            df = pd.read_csv(f)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    target_res = next((lower[k] for k in lower if k.replace(" ", "_") == "reservoirs"), None)
    target_oil = next((lower[k] for k in lower if k.replace(" ", "_") == "oil_temperature"), None)
    if target_res is None or target_oil is None:
        raise ValueError(f"MetroPT targets missing; columns={list(df.columns)}")
    time_col = next((c for c in df.columns if str(c).lower() in {"timestamp", "datetime", "date"}), None)
    numeric = _numeric_columns(df)
    inputs = [c for c in numeric if c not in {target_res, target_oil, "TP3", "index", "Unnamed: 0"}]
    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        # month is the registered outer grouping; missing timestamps stay in one group.
        runs = t.dt.to_period("M").astype(str).factorize()[0].astype(np.int64)
    else:
        runs = np.zeros(len(df), dtype=np.int64)
    return DatasetFrame("MetroPT-3", df, [target_res, target_oil], inputs, runs, time_col, [path], {"cadence_seconds": None, "archive_member": member, "proxy_excluded": "TP3"})


def load_sru(raw_root: Path) -> DatasetFrame:
    candidates = list((raw_root / "sru").glob("**/*"))
    tables = [p for p in candidates if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".xlsx", ".parquet"}]
    for path in tables:
        try:
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".xlsx":
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, sep=None, engine="python")
            lower = {str(c).lower(): c for c in df.columns}
            h2s = next((lower[k] for k in lower if "h2s" in k), None)
            so2 = next((lower[k] for k in lower if "so2" in k or "sulfur dioxide" in k), None)
            if h2s and so2:
                inputs = [c for c in _numeric_columns(df) if c not in {h2s, so2}]
                return DatasetFrame("SRU", df, [h2s, so2], inputs, np.zeros(len(df), dtype=np.int64), None, [path], {"cadence_seconds": None})
        except Exception:
            continue
    raise FileNotFoundError("SRU raw table with H2S and SO2 not available")


def load_dataset(name: str, raw_root: Path) -> DatasetFrame:
    if name == "TEP":
        return load_tep(raw_root)
    if name == "Debutanizer":
        return load_debutanizer(raw_root)
    if name == "SRU":
        return load_sru(raw_root)
    if name == "PMSM":
        return load_pmsm(raw_root)
    if name == "MetroPT-3":
        return load_metro(raw_root)
    raise KeyError(name)


def audit_frame(ds: DatasetFrame) -> dict:
    df = ds.frame
    numeric = df.select_dtypes(include=[np.number])
    missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    cadence = ds.metadata.get("cadence_seconds")
    if ds.time_column:
        t = pd.to_datetime(df[ds.time_column], errors="coerce")
        dt = t.diff().dt.total_seconds().dropna()
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt):
            cadence = float(dt.median())
            cadence_summary = {"median_seconds": float(dt.median()), "p05": float(dt.quantile(.05)), "p95": float(dt.quantile(.95)), "n": int(len(dt))}
        else:
            cadence_summary = {"median_seconds": None, "n": 0}
    else:
        cadence_summary = {"median_seconds": cadence, "n": 0}
    return {
        "dataset": ds.name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "targets": ds.target_columns,
        "inputs": ds.input_columns,
        "missing_cells": missing,
        "duplicate_rows": duplicate_rows,
        "constant_numeric_columns": [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1],
        "run_count": int(len(np.unique(ds.run_id))) if len(ds.run_id) else 0,
        "cadence": cadence_summary,
        "metadata": ds.metadata,
        "source_files": [str(p) for p in ds.source_files],
    }
