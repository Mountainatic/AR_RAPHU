from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from .cpu_data import (
    SAMPLE_RUNTIME_COLUMNS,
    ViewSpec,
    load_heads,
    sha256_file,
)
from .cpu_selection import regression_metrics
from .six_dataset_extension import (
    DIRECTIONS,
    extension_views,
    public5_root,
)
from .stage0 import write_json
from .v211_public_all_baselines import (
    SupportRequirement,
    apply_common_requirements,
)
from .v211_support import SUPPORT_COLUMNS, SUPPORT_CONTRACT, support_id_hash


BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 20260817
NEURAL_MODELS = ("LSTM", "iTransformer", "TimeMixer")
PRISM_INPUT = "PRISM_V2_1_1_K_C_W"
PRISM_DYNAMIC = "PRISM_V2_1_1_PHYSICS_FIRST"
PRISM_JOINT = "PRISM_V2_1_1_JOINT_KWA"


@dataclass(frozen=True)
class PredictionSpec:
    path: Path
    scope: str
    direction: str | None
    split: str
    model: str
    target_head: str
    information_set: str
    availability_scenario: str
    proxy_policy: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, dict(value))


def _stable_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _iter_parquet_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    seen: set[Path] = set()
    for base, _directories, files in os.walk(root, followlinks=True):
        for name in files:
            if not name.endswith(".parquet"):
                continue
            path = Path(base) / name
            identity = path.resolve()
            if identity in seen:
                continue
            seen.add(identity)
            yield path.absolute()


def _old_specs(
    root: Path,
    *,
    scope: str,
    direction: str | None,
    split: str,
) -> list[PredictionSpec]:
    result: list[PredictionSpec] = []
    for path in _iter_parquet_files(root):
        try:
            relative = path.relative_to(root.absolute())
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) < 5:
            continue
        head, information_set, availability, proxy = parts[-5:-1]
        result.append(
            PredictionSpec(
                path=path,
                scope=scope,
                direction=direction,
                split=split,
                model=path.stem,
                target_head=head,
                information_set=information_set,
                availability_scenario=availability,
                proxy_policy=proxy,
            )
        )
    return result


def _neural_specs(
    root: Path,
    *,
    scope: str,
    direction: str | None,
) -> list[PredictionSpec]:
    result: list[PredictionSpec] = []
    for path in _iter_parquet_files(root):
        try:
            relative = path.relative_to(root.absolute())
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) < 6:
            continue
        model, head, information_set, availability, proxy, split_file = parts[-6:]
        split = Path(split_file).stem
        if split not in {"test", "ood"}:
            continue
        result.append(
            PredictionSpec(
                path=path,
                scope=scope,
                direction=direction,
                split=split,
                model=model,
                target_head=head,
                information_set=information_set,
                availability_scenario=availability,
                proxy_policy=proxy,
            )
        )
    return result


def prediction_specs(
    run_root: Path,
    *,
    public_root: Path | None = None,
) -> list[PredictionSpec]:
    public_root = public5_root() if public_root is None else public_root.resolve()
    result: list[PredictionSpec] = []
    for split in ("test", "ood"):
        result.extend(
            _old_specs(
                public_root / "final" / f"{split}_predictions",
                scope="public5",
                direction=None,
                split=split,
            )
        )
        result.extend(
            _old_specs(
                public_root / "final" / f"baseline_{split}_predictions",
                scope="public5",
                direction=None,
                split=split,
            )
        )
    result.extend(
        _neural_specs(
            run_root / "results" / "NEURAL3" / "PUBLIC5" / "FINAL",
            scope="public5",
            direction=None,
        )
    )
    for direction in DIRECTIONS:
        direction_root = run_root / "directions" / direction
        for split in ("test", "ood"):
            result.extend(
                _old_specs(
                    direction_root / "final" / f"{split}_predictions",
                    scope="cz",
                    direction=direction,
                    split=split,
                )
            )
            result.extend(
                _old_specs(
                    direction_root / "final" / f"baseline_{split}_predictions",
                    scope="cz",
                    direction=direction,
                    split=split,
                )
            )
        result.extend(
            _neural_specs(
                run_root
                / "results"
                / "NEURAL3"
                / "CZ"
                / direction
                / "FINAL",
                scope="cz",
                direction=direction,
            )
        )
    unique: dict[Path, PredictionSpec] = {}
    for item in result:
        unique[item.path] = item
    return sorted(
        unique.values(),
        key=lambda item: (
            item.scope,
            item.direction or "",
            item.target_head,
            item.information_set,
            item.availability_scenario,
            item.proxy_policy,
            item.split,
            item.model,
        ),
    )


def _support_key(
    scope: str,
    direction: str | None,
    target_head: str,
    information_set: str,
    availability_scenario: str,
    proxy_policy: str,
) -> tuple[str, str, str, str, str, str]:
    return (
        scope,
        direction or "",
        target_head,
        information_set,
        availability_scenario,
        proxy_policy,
    )


def _support_records(
    run_root: Path,
) -> dict[tuple[str, str, str, str, str, str], Mapping[str, Any]]:
    payload = _read_json(
        run_root / "freeze" / "NEURAL3_EXTENSION_COMMON_SUPPORT.json"
    )
    result: dict[tuple[str, str, str, str, str, str], Mapping[str, Any]] = {}
    for record in payload.get("records", ()):
        key = _support_key(
            str(record["scope"]),
            record.get("direction"),
            str(record["target_head"]),
            str(record["information_set"]),
            str(record["availability_scenario"]),
            str(record["proxy_policy"]),
        )
        result[key] = record
    return result


def frozen_support_records(
    run_root: Path,
) -> dict[tuple[str, str, str, str, str, str], Mapping[str, Any]]:
    """Return the immutable extension support registry for reporting tools."""

    return _support_records(run_root)


def _shared_for(
    run_root: Path,
    public_root: Path,
    scope: str,
    direction: str | None,
) -> Path:
    if scope == "public5":
        return public_root / "shared"
    if direction is None:
        raise ValueError("CZ support needs a direction")
    return run_root / "shared" / direction


def _allowed_support(
    run_root: Path,
    public_root: Path,
    spec: PredictionSpec,
    records: Mapping[
        tuple[str, str, str, str, str, str], Mapping[str, Any]
    ],
) -> tuple[set[str], str, int]:
    key = _support_key(
        spec.scope,
        spec.direction,
        spec.target_head,
        spec.information_set,
        spec.availability_scenario,
        spec.proxy_policy,
    )
    record = records.get(key)
    if record is None:
        raise KeyError(key)
    split = str(spec.split)
    split_record = record.get("splits", {}).get(split)
    if split_record is None:
        raise FileNotFoundError(f"support split missing: {key}/{split}")
    shared = _shared_for(
        run_root,
        public_root,
        spec.scope,
        spec.direction,
    )
    sample_path = (
        shared
        / "sample_ids"
        / spec.target_head
        / spec.information_set
        / spec.availability_scenario
        / spec.proxy_policy
        / f"{split}.parquet"
    )
    columns = list(
        dict.fromkeys(
            [
                "view_sample_id",
                *SAMPLE_RUNTIME_COLUMNS,
                *SUPPORT_COLUMNS,
            ]
        )
    )
    samples = pd.read_parquet(sample_path, columns=columns)
    requirements = tuple(
        SupportRequirement(**dict(item))
        for item in record.get("requirements", ())
    )
    common = apply_common_requirements(samples, requirements)
    allowed = set(common["view_sample_id"].astype(str))
    expected_hash = str(split_record.get("support_hash", ""))
    observed_hash = support_id_hash(common)
    if expected_hash and expected_hash != observed_hash:
        raise RuntimeError(
            f"support hash mismatch for {key}/{split}: "
            f"{expected_hash} != {observed_hash}"
        )
    return allowed, observed_hash, len(common)


def allowed_support_for_prediction(
    run_root: Path,
    public_root: Path,
    spec: PredictionSpec,
    records: Mapping[
        tuple[str, str, str, str, str, str], Mapping[str, Any]
    ],
) -> tuple[set[str], str, int]:
    """Resolve and verify a prediction's frozen leaderboard support."""

    return _allowed_support(run_root, public_root, spec, records)


def _prediction_columns(path: Path) -> list[str]:
    names = set(parquet.ParquetFile(path).schema.names)
    required = [
        "sample_id",
        "view_sample_id",
        "base_origin_id",
        "profile_id",
        "dataset",
        "entity_id",
        "task_id",
        "task",
        "target_head",
        "split",
        "origin",
        "y_true",
        "y_pred",
        "model",
        "information_set",
        "availability_scenario",
        "proxy_policy",
        "parameter_count",
    ]
    return [name for name in required if name in names]


def _read_prediction(
    spec: PredictionSpec,
    allowed: set[str],
) -> tuple[pd.DataFrame, bool]:
    frame = pd.read_parquet(
        spec.path,
        columns=_prediction_columns(spec.path),
    )
    if "sample_id" not in frame and "view_sample_id" in frame:
        frame = frame.rename(columns={"view_sample_id": "sample_id"})
    if "task_id" not in frame and "task" in frame:
        frame = frame.rename(columns={"task": "task_id"})
    if "sample_id" not in frame:
        raise RuntimeError(f"prediction has no sample_id: {spec.path}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame = frame.loc[frame["sample_id"].isin(allowed)].copy()
    frame["model"] = frame.get(
        "model",
        pd.Series(spec.model, index=frame.index, dtype="string"),
    )
    frame["model"] = frame["model"].fillna(spec.model).astype(str)
    frame["dataset"] = frame.get(
        "dataset",
        pd.Series("", index=frame.index, dtype="string"),
    ).fillna("")
    frame["task_id"] = frame.get(
        "task_id",
        pd.Series(spec.target_head.split("__", 1)[0], index=frame.index),
    ).fillna(spec.target_head.split("__", 1)[0])
    frame["target_head"] = frame.get(
        "target_head",
        pd.Series(spec.target_head, index=frame.index),
    ).fillna(spec.target_head)
    frame["information_set"] = frame.get(
        "information_set",
        pd.Series(spec.information_set, index=frame.index),
    ).fillna(spec.information_set)
    frame["availability_scenario"] = frame.get(
        "availability_scenario",
        pd.Series(spec.availability_scenario, index=frame.index),
    ).fillna(spec.availability_scenario)
    frame["proxy_policy"] = frame.get(
        "proxy_policy",
        pd.Series(spec.proxy_policy, index=frame.index),
    ).fillna(spec.proxy_policy)
    frame["split"] = spec.split
    if "entity_id" not in frame:
        if "profile_id" in frame:
            frame["entity_id"] = frame["profile_id"].astype(str)
        else:
            frame["entity_id"] = ""
    if "origin" not in frame:
        frame["origin"] = np.arange(len(frame), dtype=np.int64)
    if "y_true" not in frame or "y_pred" not in frame:
        raise RuntimeError(f"prediction is missing y columns: {spec.path}")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError(f"duplicate prediction sample IDs: {spec.path}")
    complete = len(frame) == len(allowed)
    return frame.reset_index(drop=True), complete


def _metric_record(
    spec: PredictionSpec,
    frame: pd.DataFrame,
    *,
    support_hash: str,
    support_rows: int,
    coverage_complete: bool,
) -> dict[str, Any]:
    values = regression_metrics(
        frame["y_true"].to_numpy(dtype=np.float64),
        frame["y_pred"].to_numpy(dtype=np.float64),
    )
    parameter_count = (
        int(frame["parameter_count"].iloc[0])
        if "parameter_count" in frame and len(frame)
        and pd.notna(frame["parameter_count"].iloc[0])
        else None
    )
    return {
        "scope": spec.scope,
        "direction": spec.direction or "",
        "dataset": str(frame["dataset"].iloc[0])
        if len(frame)
        else ("cz_czochralski" if spec.scope == "cz" else ""),
        "task_id": str(frame["task_id"].iloc[0])
        if len(frame)
        else spec.target_head.split("__", 1)[0],
        "target_head": spec.target_head,
        "information_set": spec.information_set,
        "availability_scenario": spec.availability_scenario,
        "proxy_policy": spec.proxy_policy,
        "split": spec.split,
        "model": spec.model,
        "model_source": (
            "NEURAL3"
            if spec.model in NEURAL_MODELS
            else "PRISM"
            if spec.model.startswith("PRISM_")
            else "CPU_BASELINE"
        ),
        "status": "PASS" if len(frame) and coverage_complete else (
            "PARTIAL_NATIVE_COVERAGE" if len(frame) else "NO_COMMON_ROWS"
        ),
        "rows": int(len(frame)),
        "primary_support_rows": int(support_rows),
        "coverage_complete": bool(coverage_complete),
        "support_hash": support_hash,
        "prediction_path": str(spec.path),
        "prediction_sha256": sha256_file(spec.path),
        "parameter_count": parameter_count,
        **values,
    }


def _add_skills(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    result["persistence_skill"] = np.nan
    result["dynamic_ar_skill"] = np.nan
    keys = [
        "scope",
        "direction",
        "target_head",
        "information_set",
        "availability_scenario",
        "proxy_policy",
        "split",
    ]
    for _, indices in result.groupby(keys, dropna=False).groups.items():
        group = result.loc[indices]
        persistence = group[
            (group["model"] == "PERSISTENCE") & (group["status"] == "PASS")
        ]
        ar = group[(group["model"] == "AR") & (group["status"] == "PASS")]
        persistence_mse = (
            None if persistence.empty else float(persistence.iloc[0]["mse"])
        )
        ar_mse = None if ar.empty else float(ar.iloc[0]["mse"])
        for index in indices:
            mse_value = result.at[index, "mse"]
            if pd.notna(mse_value) and persistence_mse not in {None, 0.0}:
                result.at[index, "persistence_skill"] = (
                    1.0 - float(mse_value) / persistence_mse
                )
            if (
                result.at[index, "information_set"] == "dynamic"
                and pd.notna(mse_value)
                and ar_mse not in {None, 0.0}
            ):
                result.at[index, "dynamic_ar_skill"] = (
                    1.0 - float(mse_value) / ar_mse
                )
    return result


def _add_ranks(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    result["rank"] = np.nan
    groups = [
        "scope",
        "direction",
        "target_head",
        "information_set",
        "split",
    ]
    for _, indices in result.groupby(groups, dropna=False).groups.items():
        group = result.loc[indices]
        eligible = group[
            (group["status"] == "PASS")
            & group["coverage_complete"].astype(bool)
        ]
        if eligible.empty:
            continue
        result.loc[eligible.index, "rank"] = eligible["mse"].rank(
            method="min",
            ascending=True,
        )
    return result


def collect_extension_metrics(
    run_root: Path,
    *,
    public_root: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    public_root = public5_root() if public_root is None else public_root.resolve()
    records = _support_records(run_root)
    specs = prediction_specs(run_root, public_root=public_root)
    grouped: dict[
        tuple[str, str, str, str, str, str, str], list[PredictionSpec]
    ] = defaultdict(list)
    for spec in specs:
        key = (
            spec.scope,
            spec.direction or "",
            spec.target_head,
            spec.information_set,
            spec.availability_scenario,
            spec.proxy_policy,
            spec.split,
        )
        if _support_key(*key[:-1]) in records:
            grouped[key].append(spec)

    metric_rows: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    for key, group_specs in sorted(grouped.items()):
        first = group_specs[0]
        allowed, support_hash, support_rows = _allowed_support(
            run_root,
            public_root,
            first,
            records,
        )
        for spec in sorted(group_specs, key=lambda item: item.model):
            frame, complete = _read_prediction(spec, allowed)
            metric_rows.append(
                _metric_record(
                    spec,
                    frame,
                    support_hash=support_hash,
                    support_rows=support_rows,
                    coverage_complete=complete,
                )
            )
            catalog.append(
                {
                    "scope": spec.scope,
                    "direction": spec.direction or "",
                    "target_head": spec.target_head,
                    "information_set": spec.information_set,
                    "availability_scenario": spec.availability_scenario,
                    "proxy_policy": spec.proxy_policy,
                    "split": spec.split,
                    "model": spec.model,
                    "path": str(spec.path),
                    "support_hash": support_hash,
                    "support_rows": support_rows,
                    "coverage_complete": complete,
                }
            )
            del frame
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("no extension prediction metrics were materialized")
    metrics = _add_skills(metrics)
    metrics = _add_ranks(metrics)
    metrics.sort_values(
        [
            "scope",
            "direction",
            "target_head",
            "information_set",
            "split",
            "mse",
            "model",
        ],
        na_position="last",
        inplace=True,
    )
    return metrics.reset_index(drop=True), catalog


def _metric_filter(
    metrics: pd.DataFrame,
    *,
    information_set: str,
    scope: str | None = None,
) -> pd.DataFrame:
    result = metrics[
        (metrics["information_set"] == information_set)
        & (metrics["split"] == "test")
        & (metrics["status"] == "PASS")
        & metrics["coverage_complete"].astype(bool)
    ].copy()
    if scope is not None:
        result = result[result["scope"] == scope]
    return result


def write_extension_leaderboards(
    run_root: Path,
    metrics: pd.DataFrame,
) -> dict[str, str]:
    final = run_root / "final"
    final.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    names = {
        "input_only": "SIX_DATASET_INPUT_ONLY_LEADERBOARD.csv",
        "dynamic": "SIX_DATASET_DYNAMIC_LEADERBOARD.csv",
    }
    extension_names = {
        "input_only": "PUBLIC5_PLUS_NEURAL3_INPUT_ONLY_EXTENSION.csv",
        "dynamic": "PUBLIC5_PLUS_NEURAL3_DYNAMIC_EXTENSION.csv",
    }
    for information_set in ("input_only", "dynamic"):
        table = _metric_filter(metrics, information_set=information_set)
        table.sort_values(
            [
                "dataset",
                "direction",
                "target_head",
                "rank",
                "model",
            ],
            na_position="last",
            inplace=True,
        )
        path = final / names[information_set]
        table.to_csv(path, index=False)
        written[information_set] = str(path)
        public_table = _metric_filter(
            metrics,
            information_set=information_set,
            scope="public5",
        )
        public_table.to_csv(final / extension_names[information_set], index=False)
    return written


def _rank_summary(
    table: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, group in table.groupby(
        list(group_columns),
        dropna=False,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for model, values in group.groupby("model"):
            ranks = values["rank"].dropna().to_numpy(dtype=np.float64)
            if not len(ranks):
                continue
            row = {
                column: value
                for column, value in zip(group_columns, group_key, strict=True)
            }
            row.update(
                {
                    "model": model,
                    "groups": int(len(ranks)),
                    "mean_rank": float(np.mean(ranks)),
                    "median_rank": float(np.median(ranks)),
                    "wins": int(np.sum(ranks == 1.0)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_extension_rankings(
    run_root: Path,
    metrics: pd.DataFrame,
) -> dict[str, str]:
    final = run_root / "final"
    outputs: dict[str, str] = {}
    task_rows: list[pd.DataFrame] = []
    dataset_rows: list[pd.DataFrame] = []
    for information_set in ("input_only", "dynamic"):
        table = _metric_filter(metrics, information_set=information_set)
        task = _rank_summary(
            table,
            ["information_set", "task_id"],
        )
        task["ranking_type"] = "TASK_BALANCED"
        task_rows.append(task)
        per_dataset_task = _rank_summary(
            table,
            ["information_set", "dataset", "task_id"],
        )
        dataset = (
            per_dataset_task.groupby(
                ["information_set", "dataset", "model"],
                as_index=False,
            )
            .agg(
                dataset_mean_rank=("mean_rank", "mean"),
                dataset_median_rank=("median_rank", "mean"),
                task_groups=("task_id", "count"),
            )
        )
        dataset.rename(
            columns={"dataset_mean_rank": "mean_rank"},
            inplace=True,
        )
        dataset["ranking_type"] = "DATASET_BALANCED_DESCRIPTIVE_ONLY"
        dataset_rows.append(dataset)
    task_frame = pd.concat(task_rows, ignore_index=True)
    dataset_frame = pd.concat(dataset_rows, ignore_index=True)
    task_path = final / "SIX_DATASET_TASK_BALANCED_RANK.csv"
    dataset_path = final / "SIX_DATASET_DATASET_BALANCED_RANK.csv"
    task_frame.sort_values(
        ["information_set", "task_id", "mean_rank", "model"],
        inplace=True,
    )
    dataset_frame.sort_values(
        ["information_set", "dataset", "mean_rank", "model"],
        inplace=True,
    )
    task_frame.to_csv(task_path, index=False)
    dataset_frame.to_csv(dataset_path, index=False)
    outputs["task"] = str(task_path)
    outputs["dataset"] = str(dataset_path)

    for information_set in ("input_only", "dynamic"):
        cz = _metric_filter(metrics, information_set=information_set, scope="cz")
        macro = (
            cz.groupby(["model"], as_index=False)
            .agg(
                mean_mse=("mse", "mean"),
                mean_rmse=("rmse", "mean"),
                mean_r2=("r2", "mean"),
                mean_rank=("rank", "mean"),
                directions=("direction", "nunique"),
            )
            .sort_values(["mean_rank", "mean_mse", "model"])
        )
        macro.to_csv(
            final / f"CZ_MACRO_{information_set.upper()}.csv",
            index=False,
        )
    return outputs


def _head_steps(target_head: str, scope: str) -> tuple[int, int]:
    match = re.search(r"__H(\d+)__W(\d+)$", target_head)
    if match:
        return int(match.group(1)), int(match.group(2))
    if scope == "cz":
        return 120, 12
    return 1, 1


def _block_bootstrap(
    paired: pd.DataFrame,
    block_length: int,
    *,
    seed: int,
) -> np.ndarray:
    if paired.empty:
        raise ValueError("empty paired bootstrap input")
    blocks: list[tuple[float, int]] = []
    ordered = paired.sort_values(["entity_id", "origin"])
    for _entity, group in ordered.groupby("entity_id", sort=False):
        values = group["loss_difference"].to_numpy(dtype=np.float64)
        for start in range(0, len(values), max(1, block_length)):
            piece = values[start : start + max(1, block_length)]
            if len(piece):
                blocks.append(
                    (float(np.sum(piece, dtype=np.float64)), int(len(piece)))
                )
    if not blocks:
        raise ValueError("no bootstrap blocks")
    sums = np.asarray([item[0] for item in blocks], dtype=np.float64)
    lengths = np.asarray([item[1] for item in blocks], dtype=np.float64)
    draws = len(blocks)
    rng = np.random.default_rng(seed)
    output = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 8):
        count = min(8, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, len(blocks), size=(count, draws))
        output[start : start + count] = (
            sums[indices].sum(axis=1) / lengths[indices].sum(axis=1)
        )
    return output


def _bootstrap_pair(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    block_length: int,
    family: str,
) -> dict[str, Any]:
    left = candidate.sort_values("sample_id").reset_index(drop=True)
    right = comparator.sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(
        left["sample_id"].to_numpy(),
        right["sample_id"].to_numpy(),
    ):
        raise RuntimeError("paired comparison sample_id mismatch")
    if not np.array_equal(
        left["y_true"].to_numpy(dtype=np.float64),
        right["y_true"].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise RuntimeError("paired comparison y_true mismatch")
    paired = pd.DataFrame(
        {
            "entity_id": left["entity_id"].astype(str),
            "origin": left["origin"].to_numpy(dtype=np.int64),
            "loss_difference": (
                np.square(
                    left["y_true"].to_numpy(dtype=np.float64)
                    - left["y_pred"].to_numpy(dtype=np.float64)
                )
                - np.square(
                    right["y_true"].to_numpy(dtype=np.float64)
                    - right["y_pred"].to_numpy(dtype=np.float64)
                )
            ),
        }
    )
    distribution = _block_bootstrap(
        paired,
        block_length,
        seed=int(metadata["seed"]),
    )
    observed = float(np.mean(paired["loss_difference"].to_numpy(dtype=np.float64)))
    comparator_mse = float(
        np.mean(
            np.square(
                right["y_true"].to_numpy(dtype=np.float64)
                - right["y_pred"].to_numpy(dtype=np.float64)
            ),
            dtype=np.float64,
        )
    )
    better = int(np.sum(distribution < 0.0))
    opposite = int(np.sum(distribution > 0.0))
    return {
        **metadata,
        "status": "PASS",
        "paired_rows": int(len(paired)),
        "block_length": int(block_length),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "mse_difference": observed,
        "relative_improvement": (
            None if comparator_mse == 0.0 else -observed / comparator_mse
        ),
        "ci_lower": float(np.quantile(distribution, 0.025)),
        "ci_upper": float(np.quantile(distribution, 0.975)),
        "probability_candidate_better": (
            better + 1
        ) / (BOOTSTRAP_REPLICATES + 1),
        "raw_p_value": min(
            1.0,
            2.0 * min(better + 1, opposite + 1)
            / (BOOTSTRAP_REPLICATES + 1),
        ),
        "holm_p_value": np.nan,
    }


def _holm(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[
            (
                str(row.get("task_id")),
                str(row.get("information_set")),
                str(row.get("comparison_family")),
            )
        ].append(index)
    for indices in groups.values():
        ordered = sorted(
            indices,
            key=lambda index: float(rows[index]["raw_p_value"]),
        )
        previous = 0.0
        for rank, index in enumerate(ordered):
            adjusted = min(
                1.0,
                float(rows[index]["raw_p_value"])
                * (len(ordered) - rank),
            )
            rows[index]["holm_p_value"] = max(previous, adjusted)
            previous = rows[index]["holm_p_value"]


def write_extension_bootstrap(
    run_root: Path,
    metrics: pd.DataFrame,
    catalog: list[dict[str, Any]],
    *,
    public_root: Path | None = None,
) -> str:
    public_root = public5_root() if public_root is None else public_root.resolve()
    records = _support_records(run_root)
    final = run_root / "final"
    by_key = {
        (
            row["scope"],
            row["direction"],
            row["target_head"],
            row["information_set"],
            row["availability_scenario"],
            row["proxy_policy"],
            row["split"],
            row["model"],
        ): row
        for row in catalog
    }
    output: list[dict[str, Any]] = []
    for base in metrics[
        (metrics["split"] == "test")
        & (metrics["status"] == "PASS")
        & metrics["coverage_complete"].astype(bool)
        & metrics["model"].isin(NEURAL_MODELS)
    ].itertuples(index=False):
        comparator = (
            PRISM_INPUT if base.information_set == "input_only" else PRISM_DYNAMIC
        )
        candidate_key = (
            base.scope,
            base.direction,
            base.target_head,
            base.information_set,
            base.availability_scenario,
            base.proxy_policy,
            base.split,
            base.model,
        )
        comparator_key = (*candidate_key[:-1], comparator)
        candidate_meta = by_key.get(candidate_key)
        comparator_meta = by_key.get(comparator_key)
        if candidate_meta is None or comparator_meta is None:
            continue
        candidate_spec = PredictionSpec(
            Path(candidate_meta["path"]),
            base.scope,
            base.direction or None,
            base.split,
            base.model,
            base.target_head,
            base.information_set,
            base.availability_scenario,
            base.proxy_policy,
        )
        comparator_spec = PredictionSpec(
            Path(comparator_meta["path"]),
            base.scope,
            base.direction or None,
            base.split,
            comparator,
            base.target_head,
            base.information_set,
            base.availability_scenario,
            base.proxy_policy,
        )
        allowed, support_hash, _ = _allowed_support(
            run_root,
            public_root,
            candidate_spec,
            records,
        )
        left, left_complete = _read_prediction(candidate_spec, allowed)
        right, right_complete = _read_prediction(comparator_spec, allowed)
        if not left_complete or not right_complete:
            continue
        h_steps, w_steps = _head_steps(base.target_head, base.scope)
        seed_text = "|".join(
            [
                base.scope,
                base.direction,
                base.target_head,
                base.information_set,
                base.model,
                comparator,
            ]
        )
        row = _bootstrap_pair(
            left,
            right,
            metadata={
                "scope": base.scope,
                "direction": base.direction,
                "dataset": base.dataset,
                "task_id": base.task_id,
                "target_head": base.target_head,
                "information_set": base.information_set,
                "model": base.model,
                "candidate": base.model,
                "comparator": comparator,
                "comparison_family": (
                    "NEURAL3_VS_PRISM_INPUT"
                    if base.information_set == "input_only"
                    else "NEURAL3_VS_PRISM_DYNAMIC"
                ),
                "support_hash": support_hash,
                "seed": (
                    BOOTSTRAP_SEED
                    + int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
                )
                % (2**32),
            },
            block_length=max(1, h_steps + w_steps),
            family="NEURAL3_VS_PRISM",
        )
        output.append(row)
        del left, right
    frame = pd.DataFrame(output)
    if output:
        _holm(output)
        frame = pd.DataFrame(output)
    path = final / "SIX_DATASET_NEURAL3_BOOTSTRAP.csv"
    frame.to_csv(path, index=False)
    return str(path)


def write_native_support_audit(
    run_root: Path,
) -> str:
    rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        root = run_root / "directions" / direction / "results" / "DEVELOPMENT"
        for channel in (
            "main_heater_power",
            "joint_lift",
            "crystal_rotation",
            "crucible_rotation",
        ):
            path = root / "K" / "CZ_D20" / "primary" / channel / "RESULT.json"
            if not path.is_file():
                continue
            value = _read_json(path)
            native = value.get("native_support", {})
            rows.append(
                {
                    "dataset": "cz_czochralski",
                    "direction": direction,
                    "channel": channel,
                    "status": value.get("status"),
                    "selected_history_steps": value.get(
                        "selected_profile_history_steps"
                    ),
                    "anchor_train_rows": value.get("anchor_train_rows"),
                    "selected_native_train_rows": value.get(
                        "selected_native_train_rows"
                    ),
                    "local_scoring_rows": value.get(
                        "local_scoring_rows_by_fold"
                    ),
                    "assembly_train_rows": None,
                    "assembly_validation_rows": None,
                    "native_support_hash": native.get(
                        "selected_native_train_support_hash"
                    ),
                    "local_score_hash": native.get(
                        "local_scoring_support_hash"
                    ),
                    "assembly_support_hash": None,
                    "reclaimed_rows": None,
                }
            )
    path = run_root / "final" / "CZ_NATIVE_SUPPORT_AUDIT.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def write_neural_tables(
    run_root: Path,
) -> dict[str, str]:
    final = run_root / "final"
    selection_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for path in sorted(
        (run_root / "results" / "NEURAL3").glob(
            "**/SELECTION.json"
        )
    ):
        value = _read_json(path)
        profile = value.get("selected_profile", {})
        selection_rows.append(
            {
                "model": value.get("model"),
                "dataset": value.get("dataset"),
                "task_id": value.get("task_id"),
                "view": value.get("view"),
                "information_set": value.get("information_set"),
                "selected_lookback_hours": profile.get("lookback_hours"),
                "selected_history_steps": profile.get("history_steps"),
                "selected_capacity": profile.get("capacity"),
                "learning_rate": profile.get("learning_rate"),
                "parameter_count": profile.get("parameter_count"),
                "candidate_count": value.get("candidate_count"),
                "native_fit_rows": value.get("native_support", {}).get(
                    "selected_native_train_rows"
                ),
                "common_validation_rows": value.get("native_support", {}).get(
                    "common_validation_rows"
                ),
                "support_contract": value.get("support_contract"),
            }
        )
        for candidate in value.get("candidate_results", []):
            for seed in (candidate.get("seed"),):
                if seed is not None:
                    seed_rows.append(
                        {
                            "model": value.get("model"),
                            "view": value.get("view"),
                            "candidate_id": candidate.get("candidate_id"),
                            "seed": seed,
                            "validation_mse": candidate.get("validation_mse"),
                        }
                    )
    for path in sorted(
        (run_root / "results" / "NEURAL3").glob(
            "**/*_RESULT.json"
        )
    ):
        value = _read_json(path)
        for seed in value.get("seed_results", []):
            seed_rows.append(
                {
                    "model": value.get("model"),
                    "view": value.get("view"),
                    "split": value.get("split"),
                    "seed": seed.get("seed"),
                    "mse": seed.get("metrics", {}).get("mse"),
                    "rmse": seed.get("metrics", {}).get("rmse"),
                    "mae": seed.get("metrics", {}).get("mae"),
                    "r2": seed.get("metrics", {}).get("r2"),
                }
            )
        resource_rows.append(
            {
                "model": value.get("model"),
                "view": value.get("view"),
                "split": value.get("split"),
                "training_seconds": sum(
                    float(item.get("training_seconds", 0.0))
                    for item in value.get("seed_results", [])
                ),
                "prediction_seconds": sum(
                    float(item.get("prediction_seconds", 0.0))
                    for item in value.get("seed_results", [])
                ),
                "peak_vram_bytes": max(
                    [
                        int(item.get("peak_vram_bytes", 0))
                        for item in value.get("seed_results", [])
                    ]
                    or [0]
                ),
                "parameter_count": value.get("parameter_count"),
            }
        )
    outputs = {
        "hyperparameters": final / "NEURAL3_HYPERPARAMETERS.csv",
        "seed_metrics": final / "NEURAL3_SEED_METRICS.csv",
        "resources": final / "NEURAL3_RESOURCE_METRICS.csv",
    }
    pd.DataFrame(selection_rows).to_csv(outputs["hyperparameters"], index=False)
    pd.DataFrame(seed_rows).to_csv(outputs["seed_metrics"], index=False)
    pd.DataFrame(resource_rows).to_csv(outputs["resources"], index=False)
    return {key: str(value) for key, value in outputs.items()}


def report_extension(
    run_root: Path,
    *,
    public_root: Path | None = None,
) -> dict[str, Any]:
    public_root = public5_root() if public_root is None else public_root.resolve()
    final = run_root / "final"
    final.mkdir(parents=True, exist_ok=True)
    metrics, catalog = collect_extension_metrics(
        run_root,
        public_root=public_root,
    )
    metrics_path = final / "SIX_DATASET_NEURAL3_RESULTS.csv"
    metrics.to_csv(metrics_path, index=False)
    metrics.to_csv(final / "SIX_DATASET_METRICS.csv", index=False)
    catalog_path = final / "EXTENSION_PREDICTION_CATALOG.json"
    _write_json(catalog_path, {"status": "PASS", "records": catalog})
    leaderboards = write_extension_leaderboards(run_root, metrics)
    rankings = write_extension_rankings(run_root, metrics)
    bootstrap = write_extension_bootstrap(
        run_root,
        metrics,
        catalog,
        public_root=public_root,
    )
    native = write_native_support_audit(run_root)
    neural = write_neural_tables(run_root)
    status = {
        "status": "PASS",
        "metrics_rows": len(metrics),
        "catalog_rows": len(catalog),
        "leaderboards": leaderboards,
        "rankings": rankings,
        "bootstrap": bootstrap,
        "native_support": native,
        "neural_tables": neural,
        "test_accessed": True,
        "ood_accessed": bool((metrics["split"] == "ood").any()),
        "post_test_reselection": False,
    }
    _write_json(final / "REPORT_STATUS.json", status)
    return status
