"""Private-data CZ E1--E6 adapter for the frozen raw-2s L256/H4 task.

The executable code is public, but the workbook stays private.  The adapter
uses the authority branch's strict nested-OOF selector, keeps direction-level
development isolated, freezes every route before reading the corresponding
target rod, and emits only derived evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROTOCOL_ID = "PRISM_CZ_RAW2S_L256_H4_PRIVATE_E1_E6_STRICT_NESTED_OOF_R2"
EXPECTED_STRICT_BASE_COMMIT = "2ee6273b8f915cbcdff2f46d56bc80047ddae4a7"
RAW_PERIOD_SECONDS = 2
MODEL_PERIOD_SECONDS = 2
RAW_PER_MODEL = 1
HISTORY_LENGTH = 256
H = 4
W = 1
W0 = 1
# A sample depends on the complete [t-256,t) history and on the target at
# t+H-1.  A row-count purge of L+H makes adjacent fit/evaluation dependency
# intervals disjoint.  W/W0 do not extend this point-target dependency.
PURGE = HISTORY_LENGTH + H
# All multiscale summaries stay inside the frozen 256-point strict-past input.
# The formal input contract remains [t-256,t); shorter windows are feature
# scales, not alternative input supports.
HISTORIES = (64, 128, 256)
ALL_HISTORIES = (32, 64, 128, 192, 256)
DEFAULT_HISTORY = 128
A_HISTORIES = (16, 64, 256)
RIDGES = (1.0, 10.0, 100.0, 1000.0, 10000.0)
SEEDS = tuple(range(10))
PERTURBATION_LEVELS = (0.0, 0.01, 0.025, 0.05, 0.10)
CHANNELS = ("main_heater_power", "joint_lift", "crystal_rotation", "crucible_rotation")
TARGET = "crystal_diameter"
BREAKPOINTS = {"Rod_1": (334, 688, 734), "Rod_2": (745,)}
DIRECTIONS = {
    "Rod_1_to_Rod_2": ("Rod_1", "Rod_2"),
    "Rod_2_to_Rod_1": ("Rod_2", "Rod_1"),
}
# Frozen positional map is required because the source workbook headers are mojibake.
POSITIONAL_COLUMNS = {
    "Rod_1": {"main_heater_power": 1, "crystal_lift": 2, "crystal_rotation": 3,
              "crucible_lift": 4, "crucible_rotation": 5, TARGET: 9},
    "Rod_2": {"main_heater_power": 5, "crystal_lift": 1, "crystal_rotation": 2,
              "crucible_lift": 3, "crucible_rotation": 4, TARGET: 6},
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def support_hash(frame: pd.DataFrame) -> str:
    text = "\n".join(frame["sample_id"].astype(str))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_provenance(project: Path) -> dict[str, Any]:
    repo = project.parent

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            EXPECTED_STRICT_BASE_COMMIT,
            head,
        ],
        check=False,
    ).returncode == 0
    if not ancestor:
        raise RuntimeError("execution commit is not descended from strict OOF authority")
    if status:
        raise RuntimeError("refusing to freeze from a dirty project worktree")
    return {
        "repo": str(repo),
        "head": head,
        "execution_commit": head,
        "strict_base_commit": EXPECTED_STRICT_BASE_COMMIT,
        "strict_base_is_ancestor": ancestor,
        "worktree_clean": True,
    }


def validate_protocol_contract() -> None:
    expected = (2, 2, 1, 256, 4, 1, 1)
    observed = (
        RAW_PERIOD_SECONDS,
        MODEL_PERIOD_SECONDS,
        RAW_PER_MODEL,
        HISTORY_LENGTH,
        H,
        W,
        W0,
    )
    if observed != expected:
        raise RuntimeError(f"raw-2s L256 H4 contract drift: {observed} != {expected}")
    if max(ALL_HISTORIES) > HISTORY_LENGTH or max(A_HISTORIES) > HISTORY_LENGTH:
        raise RuntimeError("a feature scale exceeds the frozen 256-point input")


@dataclass
class Segment:
    rod: str
    segment_id: str
    raw_start: int
    raw_stop: int
    frame: pd.DataFrame


@dataclass
class RidgeContract:
    alpha: float
    mean: list[float]
    scale: list[float]
    coefficient: list[float]
    intercept: float
    columns: list[str]


@dataclass
class RouteSpec:
    stage: str
    candidate_id: str
    alpha: float
    blocks: list[str]
    history_assignment: list[int]


def segment_ranges(length: int, breakpoints: Iterable[int]) -> list[tuple[int, int]]:
    points = [0, *(int(value) for value in breakpoints), int(length)]
    if points != sorted(set(points)):
        raise RuntimeError(f"invalid breakpoints: {points}")
    return [(a, b) for a, b in zip(points[:-1], points[1:]) if b > a]


def read_rod(raw_path: Path, rod: str) -> pd.DataFrame:
    sheet = "Sheet1" if rod == "Rod_1" else "Sheet2"
    raw = pd.read_excel(raw_path, sheet_name=sheet, header=0).dropna(how="all").reset_index(drop=True)
    mapping = POSITIONAL_COLUMNS[rod]
    if raw.shape[1] <= max(mapping.values()):
        raise RuntimeError(f"{rod} has {raw.shape[1]} columns; positional contract cannot be applied")
    frame = pd.DataFrame({name: pd.to_numeric(raw.iloc[:, index], errors="coerce") for name, index in mapping.items()})
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"{rod} contains missing or non-finite required values")
    return frame


def aggregate_rod(rod: str, raw: pd.DataFrame) -> list[Segment]:
    """Segment the raw workbook without temporal aggregation.

    The retained function name is compatibility-only.  RAW_PER_MODEL=1 is a
    frozen protocol assertion: every 2-second source row remains one model row.
    """
    if RAW_PER_MODEL != 1:
        raise RuntimeError("raw-2s protocol forbids temporal aggregation")
    result: list[Segment] = []
    for index, (start, stop) in enumerate(segment_ranges(len(raw), BREAKPOINTS[rod])):
        usable = ((stop - start) // RAW_PER_MODEL) * RAW_PER_MODEL
        if usable < RAW_PER_MODEL:
            continue
        values = raw.iloc[start:start + usable].to_numpy(dtype=np.float64)
        frame = pd.DataFrame(values.reshape(-1, RAW_PER_MODEL, values.shape[1]).mean(axis=1), columns=raw.columns)
        result.append(Segment(rod, f"{rod}_segment_{index}", start, start + usable, frame))
    return result


def fit_joint_lift(segments: Sequence[Segment]) -> dict[str, Any]:
    blocks: list[np.ndarray] = []
    support: list[str] = []
    for segment in segments:
        n = len(segment.frame)
        origins = np.arange(HISTORY_LENGTH, n - H + 1, dtype=np.int64)
        if not len(origins):
            continue
        if len(origins) < 20:
            selected = origins
        else:
            cut = max(1, min(len(origins) - 1, int(math.floor(0.8 * len(origins)))))
            selected = origins[:max(1, cut - PURGE)]
        row_indices = np.unique(
            np.concatenate(
                [np.arange(origin - HISTORY_LENGTH, origin) for origin in selected]
            )
        )
        blocks.append(segment.frame.iloc[row_indices][["crystal_lift", "crucible_lift"]].to_numpy(dtype=np.float64))
        support.extend(f"{segment.segment_id}:{row}" for row in row_indices)
    if not blocks:
        raise RuntimeError("no source-development training rows are available for joint-lift PCA")
    values = np.vstack(blocks)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    standardized = (values - mean) / scale
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(standardized, rowvar=False, ddof=0))
    order = np.argsort(eigenvalues)[::-1]
    loading = eigenvectors[:, order[0]]
    if loading[0] < 0:
        loading = -loading
    return {
        "fit_scope": "SOURCE_ROD_DEVELOPMENT_TRAIN_ROWS_ONLY",
        "variables": ["crystal_lift", "crucible_lift"],
        "mean": mean.tolist(), "scale": scale.tolist(), "loading": loading.tolist(),
        "eigenvalues": eigenvalues[order].tolist(),
        "fit_rows": len(values),
        "fit_support_hash": hashlib.sha256("\n".join(support).encode("utf-8")).hexdigest(),
    }


def apply_joint_lift(segments: Sequence[Segment], pca: Mapping[str, Any]) -> list[Segment]:
    mean = np.asarray(pca["mean"], dtype=np.float64)
    scale = np.asarray(pca["scale"], dtype=np.float64)
    loading = np.asarray(pca["loading"], dtype=np.float64)
    output = []
    for segment in segments:
        frame = segment.frame.copy()
        values = frame[["crystal_lift", "crucible_lift"]].to_numpy(dtype=np.float64)
        frame["joint_lift"] = ((values - mean) / scale) @ loading
        output.append(Segment(segment.rod, segment.segment_id, segment.raw_start, segment.raw_stop, frame))
    return output


def rolling_stats(values: np.ndarray, origins: np.ndarray, history: int) -> np.ndarray:
    prefix = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    prefix2 = np.r_[0.0, np.cumsum(np.square(values), dtype=np.float64)]
    starts = origins - history
    sums = prefix[origins] - prefix[starts]
    sums2 = prefix2[origins] - prefix2[starts]
    means = sums / history
    variance = np.maximum(sums2 / history - np.square(means), 0.0)
    last = values[origins - 1]
    first = values[starts]
    return np.column_stack([last, means, np.sqrt(variance), last - first])


def make_samples(
    segments: Sequence[Segment],
    direction: str,
    outcome_segments: Sequence[Segment] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    cache_rows: dict[str, list[np.ndarray]] = {}
    outcomes = {
        segment.segment_id: segment
        for segment in (segments if outcome_segments is None else outcome_segments)
    }
    for segment in segments:
        outcome_segment = outcomes.get(segment.segment_id)
        if outcome_segment is None or len(outcome_segment.frame) != len(segment.frame):
            raise RuntimeError(f"feature/outcome segment mismatch: {segment.segment_id}")
        n = len(segment.frame)
        origins = np.arange(HISTORY_LENGTH, n - H + 1, dtype=np.int64)
        if not len(origins):
            continue
        feature_diameter = segment.frame[TARGET].to_numpy(dtype=np.float64)
        outcome_diameter = outcome_segment.frame[TARGET].to_numpy(dtype=np.float64)
        current = outcome_diameter[origins - 1]
        future = outcome_diameter[origins + H - 1]
        for local, origin in enumerate(origins):
            absolute_origin = segment.raw_start + int(origin) * RAW_PER_MODEL
            rows.append({
                "sample_id": hashlib.sha256(f"{direction}|{segment.segment_id}|{origin}".encode()).hexdigest(),
                "entity_id": segment.segment_id, "origin": int(origin),
                "absolute_origin": absolute_origin,
                "dependency_start": int(origin - HISTORY_LENGTH),
                "dependency_stop_exclusive": int(origin + H),
                "absolute_dependency_start": absolute_origin - HISTORY_LENGTH * RAW_PER_MODEL,
                "absolute_dependency_stop_exclusive": absolute_origin + H * RAW_PER_MODEL,
                "current_level": float(current[local]), "future_level": float(future[local]),
                "y_true": float(future[local] - current[local]),
            })
        for channel in CHANNELS:
            values = segment.frame[channel].to_numpy(dtype=np.float64)
            for history in ALL_HISTORIES:
                cache_rows.setdefault(f"K:{channel}:{history}", []).append(rolling_stats(values, origins, history))
                short = max(16, history // 4)
                long_stats = rolling_stats(values, origins, history)
                short_stats = rolling_stats(values, origins, short)
                cache_rows.setdefault(f"W:{channel}:{history}", []).append(
                    np.column_stack([short_stats[:, 1] - long_stats[:, 1], short_stats[:, 2] - long_stats[:, 2]])
                )
        last_matrix = np.column_stack([segment.frame[channel].to_numpy(dtype=np.float64)[origins - 1] for channel in CHANNELS])
        cross = np.column_stack([last_matrix[:, i] * last_matrix[:, j] for i in range(4) for j in range(i + 1, 4)])
        cache_rows.setdefault("C:pairwise", []).append(cross)
        a_blocks = [rolling_stats(feature_diameter, origins, history) for history in A_HISTORIES]
        cache_rows.setdefault("A:diameter", []).append(np.column_stack(a_blocks))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"{direction} has no raw-2s CZ samples at the frozen L256 history")
    cache = {name: np.vstack(blocks) for name, blocks in cache_rows.items()}
    return frame, cache


def chronological_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    order = frame.sort_values(
        ["absolute_origin", "entity_id", "origin"], kind="stable"
    ).index.to_numpy(dtype=np.int64)
    cut = max(1, min(len(order) - 1, int(math.floor(0.8 * len(order)))))
    train_stop = max(1, cut - PURGE)
    train = order[:train_stop]
    validation = order[cut:]
    if not len(train) or not len(validation):
        raise RuntimeError("nonempty purged train and validation partitions are required")
    if frame.loc[train, "absolute_origin"].max() >= frame.loc[validation, "absolute_origin"].min():
        raise RuntimeError("chronological split ordering failure")
    if (
        frame.loc[train, "absolute_dependency_stop_exclusive"].max()
        > frame.loc[validation, "absolute_dependency_start"].min()
    ):
        raise RuntimeError("chronological split has overlapping dependency intervals")
    return train, validation


def rolling_folds(frame: pd.DataFrame, indices: np.ndarray, count: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = frame.loc[indices].sort_values(
        ["absolute_origin", "entity_id", "origin"], kind="stable"
    ).index.to_numpy(dtype=np.int64)
    start = max(PURGE + 50, int(0.40 * len(ordered)))
    remaining = len(ordered) - start
    if remaining < count * 20:
        raise RuntimeError("too few source-development rows for four strict rolling folds")
    bounds = np.linspace(start, len(ordered), count + 1, dtype=int)
    folds = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        fit_stop = max(1, left - PURGE)
        fit = ordered[:fit_stop]
        evaluation = ordered[left:right]
        if (
            frame.loc[fit, "absolute_dependency_stop_exclusive"].max()
            > frame.loc[evaluation, "absolute_dependency_start"].min()
        ):
            raise RuntimeError("rolling fold has overlapping dependency intervals")
        folds.append((fit, evaluation))
    return folds


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, columns: Sequence[str]) -> RidgeContract:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    xs = (x - mean) / scale
    intercept = float(y.mean())
    gram = xs.T @ xs + float(alpha) * np.eye(xs.shape[1])
    coefficient = np.linalg.solve(gram, xs.T @ (y - intercept))
    return RidgeContract(float(alpha), mean.tolist(), scale.tolist(), coefficient.tolist(), intercept, list(columns))


def predict_ridge(contract: RidgeContract, x: np.ndarray) -> np.ndarray:
    mean = np.asarray(contract.mean)
    scale = np.asarray(contract.scale)
    coefficient = np.asarray(contract.coefficient)
    return ((np.asarray(x) - mean) / scale) @ coefficient + contract.intercept


def metrics(y: np.ndarray, prediction: np.ndarray, current: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    residual = y - prediction
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum(np.square(y - y.mean())))
    r2 = float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0 else float("nan")
    true_level = current + y
    pred_level = current + prediction
    level_denominator = float(np.sum(np.square(true_level - true_level.mean())))
    level_r2 = float(1.0 - np.sum(np.square(true_level - pred_level)) / level_denominator) if level_denominator > 0 else float("nan")
    persistence_mse = float(np.mean(np.square(y)))
    persistence_rmse = float(np.sqrt(persistence_mse))
    model_mse = float(np.mean(np.square(residual)))
    return {"rmse_delta": rmse, "mae_delta": mae, "r2_delta": r2,
            "r2_level_reconstructed": level_r2,
            "persistence_skill": float(1.0 - model_mse / persistence_mse) if persistence_mse else float("nan"),
            "persistence_skill_mse": float(1.0 - model_mse / persistence_mse) if persistence_mse else float("nan"),
            "persistence_skill_rmse": float(1.0 - rmse / persistence_rmse) if persistence_rmse else float("nan")}


def block_design(cache: Mapping[str, np.ndarray], blocks: Sequence[str], assignment: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    pieces: list[np.ndarray] = []
    names: list[str] = []
    if "K" in blocks:
        for channel, history in zip(CHANNELS, assignment):
            value = cache[f"K:{channel}:{history}"]
            pieces.append(value)
            names.extend(f"K:{channel}:{history}:{stat}" for stat in ("last", "mean", "std", "delta"))
    if "C" in blocks:
        pieces.append(cache["C:pairwise"])
        names.extend(f"C:pair:{i}" for i in range(cache["C:pairwise"].shape[1]))
    if "W" in blocks:
        for channel, history in zip(CHANNELS, assignment):
            pieces.append(cache[f"W:{channel}:{history}"])
            names.extend((f"W:{channel}:{history}:mean_delta", f"W:{channel}:{history}:std_delta"))
    if "A" in blocks:
        pieces.append(cache["A:diameter"])
        names.extend(f"A:diameter:{i}" for i in range(cache["A:diameter"].shape[1]))
    if not pieces:
        return np.empty((len(next(iter(cache.values()))), 0)), []
    return np.column_stack(pieces), names


def candidate_specs(stage: str, parent: RouteSpec | None, dynamic: bool) -> list[RouteSpec]:
    parent_blocks = [] if parent is None else list(parent.blocks)
    parent_assignment = [DEFAULT_HISTORY] * len(CHANNELS) if parent is None else list(parent.history_assignment)
    specs: list[RouteSpec] = []
    if stage == "K":
        assignments = ([64] * 4, [128] * 4, [256] * 4,
                       [64, 128, 256, 128], [256, 64, 128, 256], [128, 256, 64, 128])
        for assignment in assignments:
            for alpha in RIDGES:
                specs.append(RouteSpec(stage, f"K:h={','.join(map(str, assignment))}|a={alpha:g}", alpha, ["K"], list(assignment)))
    elif stage in {"C", "W", "A"}:
        if stage == "A" and not dynamic:
            return []
        blocks = parent_blocks + [stage]
        for alpha in RIDGES:
            specs.append(RouteSpec(stage, f"{stage}:a={alpha:g}", alpha, blocks, parent_assignment))
    else:
        raise ValueError(stage)
    return specs


def fold_losses(frame: pd.DataFrame, cache: Mapping[str, np.ndarray], folds: Sequence[tuple[np.ndarray, np.ndarray]], spec: RouteSpec | None) -> list[float]:
    target = frame["y_true"].to_numpy(dtype=np.float64)
    result = []
    for fit_index, eval_index in folds:
        if spec is None:
            prediction = np.full(len(eval_index), target[fit_index].mean())
        else:
            design, columns = block_design(cache, spec.blocks, spec.history_assignment)
            contract = fit_ridge(design[fit_index], target[fit_index], spec.alpha, columns)
            prediction = predict_ridge(contract, design[eval_index])
        result.append(float(np.mean(np.square(target[eval_index] - prediction))))
    return result


def select_chain(frame: pd.DataFrame, cache: Mapping[str, np.ndarray], train_index: np.ndarray,
                 dynamic: bool, selector: Callable[..., Any]) -> tuple[list[dict[str, Any]], RouteSpec | None]:
    folds = rolling_folds(frame, train_index)
    route: RouteSpec | None = None
    records: list[dict[str, Any]] = []
    for stage in ("K", "C", "W", "A"):
        candidates = candidate_specs(stage, route, dynamic)
        if not candidates:
            records.append({"stage": stage, "routing_status": "ZERO_IDENTITY", "active": False,
                            "reason": "NOT_APPLICABLE_INPUT_ONLY", "selected_candidate": "IDENTITY"})
            continue
        parent = fold_losses(frame, cache, folds, route)
        losses = {spec.candidate_id: fold_losses(frame, cache, folds, spec) for spec in candidates}
        selection = selector(losses, parent, identity=f"{stage}_ZERO_IDENTITY",
                             fold_weights=[len(evaluation) for _, evaluation in folds])
        by_id = {spec.candidate_id: spec for spec in candidates}
        if selection.active:
            route = by_id[str(selection.tuned_nonzero_candidate)]
        report = selection.to_json()
        report["stage"] = stage
        report["selected_candidate"] = str(selection.final_selected_candidate)
        records.append(report)
    return records, route


def fit_route(frame: pd.DataFrame, cache: Mapping[str, np.ndarray], indices: np.ndarray,
              route: RouteSpec | None) -> tuple[RidgeContract | None, np.ndarray, list[str]]:
    if route is None:
        return None, np.empty((len(frame), 0)), []
    design, columns = block_design(cache, route.blocks, route.history_assignment)
    contract = fit_ridge(design[indices], frame.loc[indices, "y_true"].to_numpy(dtype=np.float64), route.alpha, columns)
    return contract, design, columns


def evaluate_prefixes(frame: pd.DataFrame, cache: Mapping[str, np.ndarray], fit_index: np.ndarray,
                      eval_index: np.ndarray, stage_records: Sequence[Mapping[str, Any]], dynamic: bool) -> list[dict[str, Any]]:
    route: RouteSpec | None = None
    output = []
    y = frame["y_true"].to_numpy(dtype=np.float64)
    current = frame["current_level"].to_numpy(dtype=np.float64)
    output.append({"prefix": "IDENTITY", **metrics(y[eval_index], np.full(len(eval_index), y[fit_index].mean()), current[eval_index]), "parameter_count": 1})
    for record in stage_records:
        stage = str(record["stage"])
        candidates = candidate_specs(stage, route, dynamic)
        if bool(record.get("active")):
            route = {spec.candidate_id: spec for spec in candidates}[str(record["tuned_nonzero_candidate"])]
        if route is None:
            prediction = np.full(len(eval_index), y[fit_index].mean())
            params = 1
        else:
            contract, design, _ = fit_route(frame, cache, fit_index, route)
            prediction = predict_ridge(contract, design[eval_index]) if contract else np.full(len(eval_index), y[fit_index].mean())
            params = len(contract.coefficient) + 1 if contract else 1
        prefix = {"K": "K", "C": "KC", "W": "KCW", "A": "KCWA"}[stage]
        output.append({"prefix": prefix, **metrics(y[eval_index], prediction, current[eval_index]), "parameter_count": params})
    joint = RouteSpec("J", "J_FULL", 1e-2, ["K", "C", "W"] + (["A"] if dynamic else []), [DEFAULT_HISTORY] * 4)
    contract, design, _ = fit_route(frame, cache, fit_index, joint)
    prediction = predict_ridge(contract, design[eval_index])
    output.append({"prefix": "J", **metrics(y[eval_index], prediction, current[eval_index]), "parameter_count": len(contract.coefficient) + 1})
    return output


def freeze_direction(raw_path: Path, direction: str, output: Path, selector: Callable[..., Any]) -> dict[str, Any]:
    source, target = DIRECTIONS[direction]
    source_raw = read_rod(raw_path, source)
    source_segments = aggregate_rod(source, source_raw)
    pca = fit_joint_lift(source_segments)
    source_segments = apply_joint_lift(source_segments, pca)
    frame, cache = make_samples(source_segments, direction)
    train, validation = chronological_split(frame)
    directions_root = output / "development" / direction
    directions_root.mkdir(parents=True, exist_ok=True)
    stage_records: dict[str, list[dict[str, Any]]] = {}
    e1_rows = []
    for information_set, dynamic in (("input_only", False), ("dynamic", True)):
        records, route = select_chain(frame, cache, train, dynamic, selector)
        stage_records[information_set] = records
        for row in evaluate_prefixes(frame, cache, train, validation, records, dynamic):
            e1_rows.append({"direction": direction, "partition": "development_validation", "information_set": information_set,
                            "rows": len(validation), "support_hash": support_hash(frame.loc[validation]), **row})
        pd.DataFrame(records).to_json(directions_root / f"{information_set}_stage_routes.json", orient="records", indent=2)
    pd.DataFrame(e1_rows).to_csv(directions_root / "e1_stagewise_development.csv", index=False)
    np.savez_compressed(directions_root / "development_arrays.npz", **{f"cache_{key.replace(':', '__')}": value for key, value in cache.items()},
                        train=train, validation=validation,
                        y_true=frame["y_true"].to_numpy(), current_level=frame["current_level"].to_numpy())
    frame.to_parquet(directions_root / "development_samples.parquet", index=False)
    state = {
        "direction": direction, "source_rod": source, "target_rod": target,
        "target_rod_read_before_direction_freeze": False,
        "pca": pca, "sample_rows": len(frame), "train_rows": len(train), "validation_rows": len(validation),
        "support_hash": support_hash(frame), "stage_records": stage_records,
        "protocol": {
            "task": "CZ_DIAM_RAW2S_CURRENT_L256_H4",
            "cadence_seconds": MODEL_PERIOD_SECONDS,
            "history_points": HISTORY_LENGTH,
            "input": "[t-256,t)",
            "anchor": "D[t-1]",
            "target_delta": "D[t+h-1]-D[t-1]",
            "h": H,
            "w": W,
            "w0": W0,
            "feature_scales": list(HISTORIES),
            "strict_past": True,
            "temporal_aggregation": "NONE",
        },
    }
    write_json(directions_root / "DIRECTION_FREEZE.json", {**state, "sealed": True, "status": "DIRECTION_SELECTION_FROZEN"})
    return state


def route_from_records(records: Sequence[Mapping[str, Any]], dynamic: bool) -> RouteSpec | None:
    route = None
    for record in records:
        candidates = candidate_specs(str(record["stage"]), route, dynamic)
        if bool(record.get("active")):
            route = {spec.candidate_id: spec for spec in candidates}[str(record["tuned_nonzero_candidate"])]
    return route


def fit_checkpoints(raw_path: Path, freezes: Sequence[Mapping[str, Any]], output: Path) -> list[dict[str, Any]]:
    records = []
    for freeze in freezes:
        direction = str(freeze["direction"])
        source = str(freeze["source_rod"])
        segments = apply_joint_lift(aggregate_rod(source, read_rod(raw_path, source)), freeze["pca"])
        frame, cache = make_samples(segments, direction)
        all_index = np.arange(len(frame))
        for information_set, dynamic in (("input_only", False), ("dynamic", True)):
            stage_records = freeze["stage_records"][information_set]
            route = route_from_records(stage_records, dynamic)
            contract, _, columns = fit_route(frame, cache, all_index, route)
            payload = {
                "direction": direction, "information_set": information_set,
                "route": asdict(route) if route else None,
                "contract": asdict(contract) if contract else {"intercept": float(frame["y_true"].mean()), "columns": []},
                "fit_rows": len(frame), "fit_support_hash": support_hash(frame),
                "selection_freeze_sha256": sha256_file(output / "development" / direction / "DIRECTION_FREEZE.json"),
            }
            path = output / "checkpoints" / direction / information_set / "checkpoint.json"
            write_json(path, payload)
            records.append({"path": str(path.relative_to(output)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {"status": "CHECKPOINTS_SEALED", "sealed": True, "created_utc": utc(), "entries": records}
    write_json(output / "freeze" / "CHECKPOINT_MANIFEST.json", manifest)
    return records


def predict_checkpoint(frame: pd.DataFrame, cache: Mapping[str, np.ndarray], checkpoint: Mapping[str, Any]) -> np.ndarray:
    route_payload = checkpoint.get("route")
    if route_payload is None:
        return np.full(len(frame), float(checkpoint["contract"]["intercept"]))
    route = RouteSpec(**route_payload)
    design, columns = block_design(cache, route.blocks, route.history_assignment)
    contract = RidgeContract(**checkpoint["contract"])
    if columns != contract.columns:
        raise RuntimeError("checkpoint feature contract drift")
    return predict_ridge(contract, design)


def run_formal(raw_path: Path, freezes: Sequence[Mapping[str, Any]], output: Path) -> pd.DataFrame:
    manifest = json.loads((output / "freeze" / "CHECKPOINT_MANIFEST.json").read_text())
    if manifest.get("status") != "CHECKPOINTS_SEALED" or manifest.get("sealed") is not True:
        raise RuntimeError("formal target access requires sealed checkpoints")
    rows = []
    for freeze in freezes:
        direction = str(freeze["direction"])
        target = str(freeze["target_rod"])
        target_segments = apply_joint_lift(aggregate_rod(target, read_rod(raw_path, target)), freeze["pca"])
        frame, cache = make_samples(target_segments, direction)
        for information_set in ("input_only", "dynamic"):
            checkpoint_path = output / "checkpoints" / direction / information_set / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            prediction = predict_checkpoint(frame, cache, checkpoint)
            metric = metrics(frame["y_true"].to_numpy(), prediction, frame["current_level"].to_numpy())
            rows.append({"direction": direction, "partition": "formal_target_rod", "information_set": information_set,
                         "rows": len(frame), "support_hash": support_hash(frame), **metric})
            formal_root = output / "formal" / direction
            formal_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"sample_id": frame["sample_id"], "y_true": frame["y_true"], "y_pred": prediction,
                          "current_level": frame["current_level"]}).to_parquet(
                              formal_root / f"{information_set}_predictions.parquet", index=False)
        write_json(output / "formal" / direction / "TARGET_ACCESS_AUDIT.json",
                   {"status": "PASS", "direction": direction, "target_rod": target,
                    "target_partition_for_this_direction_used_before_direction_freeze": False,
                    "target_partition_for_this_direction_used_before_checkpoint_seal": False,
                    "opposite_direction_source_use_isolated_from_this_direction_selection": True,
                    "rows": len(frame), "support_hash": support_hash(frame)})
    result = pd.DataFrame(rows)
    e1_root = output / "E1_STAGEWISE"
    e1_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(e1_root / "formal_metrics.csv", index=False)
    return result


def run_e2(raw_path: Path, freezes: Sequence[Mapping[str, Any]], output: Path, selector: Callable[..., Any]) -> pd.DataFrame:
    rows = []
    regimes = {
        "NULL": set(), "K_ONLY": {"K"}, "KCW": {"K", "C", "W"}, "FULL": {"K", "C", "W", "A"},
    }
    for freeze in freezes:
        direction = str(freeze["direction"])
        source = str(freeze["source_rod"])
        segments = apply_joint_lift(aggregate_rod(source, read_rod(raw_path, source)), freeze["pca"])
        frame, cache = make_samples(segments, direction)
        train, _ = chronological_split(frame)
        base_specs = {
            "K": RouteSpec("K", "truth_K", 1e-2, ["K"], [DEFAULT_HISTORY] * 4),
            "C": RouteSpec("C", "truth_C", 1e-2, ["C"], [DEFAULT_HISTORY] * 4),
            "W": RouteSpec("W", "truth_W", 1e-2, ["W"], [DEFAULT_HISTORY] * 4),
            "A": RouteSpec("A", "truth_A", 1e-2, ["A"], [DEFAULT_HISTORY] * 4),
        }
        standardized_blocks = {}
        for stage, spec in base_specs.items():
            x, _ = block_design(cache, spec.blocks, spec.history_assignment)
            values = x[:, :min(4, x.shape[1])]
            values = (values - values[train].mean(axis=0)) / np.maximum(values[train].std(axis=0), 1e-12)
            standardized_blocks[stage] = values.mean(axis=1)
        signal_scale = np.std(frame.loc[train, "y_true"].to_numpy())
        for regime, truth in regimes.items():
            for seed in SEEDS:
                rng = np.random.default_rng(seed + 1009 * (1 + list(regimes).index(regime)))
                synthetic = sum((0.35 * signal_scale * standardized_blocks[stage] for stage in truth), start=np.zeros(len(frame)))
                synthetic += rng.normal(0.0, 0.20 * max(signal_scale, 1e-9), len(frame))
                synthetic_frame = frame.copy()
                synthetic_frame["y_true"] = synthetic
                records, _ = select_chain(synthetic_frame, cache, train, True, selector)
                for record in records:
                    stage = str(record["stage"])
                    selected = bool(record.get("active"))
                    expected = stage in truth
                    rows.append({"direction": direction, "regime": regime, "seed": seed, "stage": stage,
                                 "truth_active": expected, "selected_active": selected,
                                 "correct": selected == expected, "false_admission": selected and not expected,
                                 "missed_admission": expected and not selected,
                                 "relative_admission_margin": record.get("relative_admission_margin")})
    result = pd.DataFrame(rows)
    root = output / "E2_CZ_INPUT_SEMISYNTHETIC_RECOVERY"
    root.mkdir(parents=True, exist_ok=True)
    result.to_csv(root / "per_seed_stage_recovery.csv", index=False)
    result.groupby(["regime", "stage"]).agg(recovery_rate=("correct", "mean"), false_admission_rate=("false_admission", "mean"),
                                                missed_admission_rate=("missed_admission", "mean"), rows=("correct", "size")).reset_index().to_csv(root / "aggregate.csv", index=False)
    return result


def evaluate_spec(frame: pd.DataFrame, cache: Mapping[str, np.ndarray], train: np.ndarray,
                  validation: np.ndarray, spec: RouteSpec) -> dict[str, Any]:
    design, columns = block_design(cache, spec.blocks, spec.history_assignment)
    contract = fit_ridge(design[train], frame.loc[train, "y_true"].to_numpy(), spec.alpha, columns)
    prediction = predict_ridge(contract, design[validation])
    return {**metrics(frame.loc[validation, "y_true"].to_numpy(), prediction,
                      frame.loc[validation, "current_level"].to_numpy()),
            "parameter_count": len(columns) + 1,
            "prediction_hash": hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest()}


def run_e3_e4(raw_path: Path, freezes: Sequence[Mapping[str, Any]], output: Path, selector: Callable[..., Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    e3_rows, e4_rows = [], []
    universes = {
        "COARSE": ((64, 256), (100.0, 10000.0)),
        "STANDARD": (HISTORIES, RIDGES),
        "EXPANDED": (ALL_HISTORIES, RIDGES),
    }
    for freeze in freezes:
        direction = str(freeze["direction"]); source = str(freeze["source_rod"])
        segments = apply_joint_lift(aggregate_rod(source, read_rod(raw_path, source)), freeze["pca"])
        frame, cache = make_samples(segments, direction); train, validation = chronological_split(frame)
        folds = rolling_folds(frame, train)
        for seed in SEEDS:
            uniform_histories = list(np.random.default_rng(seed).permutation(HISTORIES))
            uniform_specs = [RouteSpec("E3", f"U:h={h}|a={a:g}", a, ["K", "C", "W"], [int(h)] * 4)
                             for h in uniform_histories for a in (100.0, 10000.0)]
            assignment_pool = [values for values in itertools.product(HISTORIES, repeat=len(CHANNELS)) if len(set(values)) > 1]
            selected_indices = np.random.default_rng(seed + 31).choice(len(assignment_pool), size=3, replace=False)
            assignments = [assignment_pool[int(index)] for index in selected_indices]
            multi_specs = [RouteSpec("E3", f"M:h={','.join(map(str,h))}|a={a:g}", a, ["K", "C", "W"], list(h))
                           for h in assignments for a in (100.0, 10000.0)]
            for arm, specs in (("UNIFORM_SCALE", uniform_specs), ("CHANNEL_SPECIFIC_MULTISCALE", multi_specs)):
                losses = {spec.candidate_id: fold_losses(frame, cache, folds, spec) for spec in specs}
                selected = min(specs, key=lambda spec: np.average(losses[spec.candidate_id], weights=[len(v) for _, v in folds]))
                metric = evaluate_spec(frame, cache, train, validation, selected)
                e3_rows.append({"direction": direction, "seed": seed, "arm": arm, "candidate_fit_budget": len(specs),
                                "selected_candidate": selected.candidate_id, "support_hash": support_hash(frame.loc[validation]), **metric})
        for universe, (histories, alphas) in universes.items():
            specs = [RouteSpec("E4", f"{universe}:h={h}|a={a:g}", a, ["K", "C", "W", "A"], [int(h)] * 4)
                     for h in histories for a in alphas]
            losses = {spec.candidate_id: fold_losses(frame, cache, folds, spec) for spec in specs}
            selected = min(specs, key=lambda spec: np.average(losses[spec.candidate_id], weights=[len(v) for _, v in folds]))
            e4_rows.append({"direction": direction, "experiment": "candidate_universe", "variant": universe,
                            "candidate_count": len(specs), "selected_candidate": selected.candidate_id,
                            "support_hash": support_hash(frame.loc[validation]), **evaluate_spec(frame, cache, train, validation, selected)})
        for removed in ("NONE", "C", "W", "A"):
            blocks = [block for block in ("K", "C", "W", "A") if block != removed]
            spec = RouteSpec("E4B", f"REMOVE_{removed}", 1000.0, blocks, [DEFAULT_HISTORY] * 4)
            e4_rows.append({"direction": direction, "experiment": "family_ablation", "variant": f"REMOVE_{removed}",
                            "candidate_count": 1, "selected_candidate": spec.candidate_id,
                            "support_hash": support_hash(frame.loc[validation]), **evaluate_spec(frame, cache, train, validation, spec)})
    e3 = pd.DataFrame(e3_rows); e4 = pd.DataFrame(e4_rows)
    (output / "E3_MULTISCALE").mkdir(parents=True, exist_ok=True)
    (output / "E4_CANDIDATE_SENSITIVITY").mkdir(parents=True, exist_ok=True)
    e3.to_csv(output / "E3_MULTISCALE" / "equal_budget_multiscale.csv", index=False)
    paired = e3.pivot(index=["direction", "seed"], columns="arm", values="rmse_delta").reset_index()
    paired["G_MS"] = (paired["UNIFORM_SCALE"] - paired["CHANNEL_SPECIFIC_MULTISCALE"]) / paired["UNIFORM_SCALE"]
    paired.to_csv(output / "E3_MULTISCALE" / "paired_gain.csv", index=False)
    e4.to_csv(output / "E4_CANDIDATE_SENSITIVITY" / "all_variants.csv", index=False)
    return e3, e4


def run_e5(freezes: Sequence[Mapping[str, Any]], e2: pd.DataFrame, e3: pd.DataFrame, e4: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for freeze in freezes:
        direction = str(freeze["direction"])
        for information_set, records in freeze["stage_records"].items():
            active = [str(record["stage"]) for record in records if bool(record.get("active"))]
            rows.append({"source": "E1", "direction": direction, "condition": information_set,
                         "active_signature": "+".join(active) or "ZERO", "stability_value": 1.0})
    for (direction, regime, seed), group in e2.groupby(["direction", "regime", "seed"]):
        active = group.loc[group["selected_active"], "stage"].astype(str).tolist()
        rows.append({"source": "E2", "direction": direction, "condition": f"{regime}:seed={seed}",
                     "active_signature": "+".join(active) or "ZERO", "stability_value": float(group["correct"].mean())})
    for (direction, seed), group in e3.groupby(["direction", "seed"]):
        gain = (group.loc[group["arm"] == "UNIFORM_SCALE", "rmse_delta"].iloc[0] -
                group.loc[group["arm"] == "CHANNEL_SPECIFIC_MULTISCALE", "rmse_delta"].iloc[0])
        rows.append({"source": "E3", "direction": direction, "condition": f"seed={seed}",
                     "active_signature": "MULTISCALE" if gain > 0 else "UNIFORM", "stability_value": float(gain)})
    for _, row in e4.iterrows():
        rows.append({"source": "E4", "direction": row["direction"], "condition": f"{row['experiment']}:{row['variant']}",
                     "active_signature": str(row["selected_candidate"]), "stability_value": float(row["rmse_delta"])})
    result = pd.DataFrame(rows)
    root = output / "E5_STRUCTURAL_STABILITY"; root.mkdir(parents=True, exist_ok=True)
    result.to_csv(root / "stability_evidence.csv", index=False)
    summary = result.groupby(["source", "direction"]).agg(conditions=("condition", "size"),
                                                            unique_structures=("active_signature", "nunique"),
                                                            stability_mean=("stability_value", "mean"),
                                                            stability_std=("stability_value", "std")).reset_index()
    summary.to_csv(root / "aggregate.csv", index=False)
    return result


RAW_MEASUREMENT_COLUMNS = (
    "main_heater_power",
    "crystal_lift",
    "crucible_lift",
    "crystal_rotation",
    "crucible_rotation",
    TARGET,
)


def training_measurement_scales(
    raw_segments: Sequence[Segment],
    samples: pd.DataFrame,
    train_index: np.ndarray,
) -> dict[str, float]:
    """Fit E6 noise scales on unique raw outer-train measurements only."""
    selected = samples.loc[train_index]
    blocks: list[pd.DataFrame] = []
    for segment in raw_segments:
        segment_samples = selected.loc[selected["entity_id"] == segment.segment_id]
        if segment_samples.empty:
            continue
        mask = np.zeros(len(segment.frame), dtype=bool)
        for start, stop in segment_samples[["dependency_start", "origin"]].itertuples(index=False, name=None):
            mask[int(start):int(stop)] = True
        if mask.any():
            blocks.append(segment.frame.loc[mask, RAW_MEASUREMENT_COLUMNS])
    if not blocks:
        raise RuntimeError("no unique outer-train raw measurements for E6 scaling")
    values = pd.concat(blocks, ignore_index=True)
    return {
        column: max(float(values[column].std(ddof=0)), 1e-12)
        for column in RAW_MEASUREMENT_COLUMNS
    }


def perturb_raw_segments(
    raw_segments: Sequence[Segment],
    scales: Mapping[str, float],
    level: float,
    seed: int,
) -> list[Segment]:
    """Inject deterministic Gaussian noise before PCA/history fusion."""
    output: list[Segment] = []
    for segment_index, segment in enumerate(raw_segments):
        frame = segment.frame.copy()
        rng = np.random.default_rng(seed + 1000003 * segment_index)
        for column in RAW_MEASUREMENT_COLUMNS:
            frame[column] = (
                frame[column].to_numpy(dtype=np.float64)
                + rng.normal(0.0, level * float(scales[column]), len(frame))
            )
        output.append(
            Segment(
                segment.rod,
                segment.segment_id,
                segment.raw_start,
                segment.raw_stop,
                frame,
            )
        )
    return output


def run_e6(raw_path: Path, freezes: Sequence[Mapping[str, Any]], output: Path, selector: Callable[..., Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n1_rows, n2_rows = [], []
    for freeze in freezes:
        direction = str(freeze["direction"]); source = str(freeze["source_rod"]); target = str(freeze["target_rod"])
        source_raw_segments = aggregate_rod(source, read_rod(raw_path, source))
        source_segments = apply_joint_lift(source_raw_segments, freeze["pca"])
        source_frame, source_cache = make_samples(source_segments, direction)
        train, validation = chronological_split(source_frame)
        noise_scales = training_measurement_scales(source_raw_segments, source_frame, train)
        target_raw_segments = aggregate_rod(target, read_rod(raw_path, target))
        target_segments = apply_joint_lift(target_raw_segments, freeze["pca"])
        target_frame, target_cache = make_samples(target_segments, direction)
        for information_set, dynamic in (("input_only", False), ("dynamic", True)):
            checkpoint = json.loads((output / "checkpoints" / direction / information_set / "checkpoint.json").read_text())
            for level in PERTURBATION_LEVELS:
                for seed in SEEDS:
                    noisy_target_raw = perturb_raw_segments(target_raw_segments, noise_scales, level, seed)
                    noisy_target_segments = apply_joint_lift(noisy_target_raw, freeze["pca"])
                    noisy_target_frame, noisy_target_cache = make_samples(
                        noisy_target_segments, direction, outcome_segments=target_segments
                    )
                    prediction = predict_checkpoint(noisy_target_frame, noisy_target_cache, checkpoint)
                    metric = metrics(target_frame["y_true"].to_numpy(), prediction, target_frame["current_level"].to_numpy())
                    n1_rows.append({"direction": direction, "information_set": information_set, "alpha": level, "seed": seed,
                                    "mode": "N1_FROZEN_CHECKPOINT_RAW_PRE_FUSION", "noise_scale_scope": "SOURCE_OUTER_TRAIN_UNIQUE_RAW", **metric})
                    noisy_source_raw = perturb_raw_segments(source_raw_segments, noise_scales, level, seed)
                    noisy_source_segments = apply_joint_lift(noisy_source_raw, freeze["pca"])
                    noisy_source_frame, noisy_source_cache = make_samples(
                        noisy_source_segments, direction, outcome_segments=source_segments
                    )
                    records, route = select_chain(noisy_source_frame, noisy_source_cache, train, dynamic, selector)
                    contract, design, _ = fit_route(noisy_source_frame, noisy_source_cache, train, route)
                    if contract is None:
                        pred = np.full(len(validation), source_frame.loc[train, "y_true"].mean())
                    else:
                        pred = predict_ridge(contract, design[validation])
                    metric2 = metrics(source_frame.loc[validation, "y_true"].to_numpy(), pred,
                                      source_frame.loc[validation, "current_level"].to_numpy())
                    signature = "+".join(str(record["stage"]) for record in records if bool(record.get("active"))) or "ZERO"
                    n2_rows.append({"direction": direction, "information_set": information_set, "alpha": level, "seed": seed,
                                    "mode": "N2_REIDENTIFICATION_RAW_PRE_FUSION", "noise_scale_scope": "SOURCE_OUTER_TRAIN_UNIQUE_RAW", "active_signature": signature, **metric2})
    n1, n2 = pd.DataFrame(n1_rows), pd.DataFrame(n2_rows)
    root = output / "E6_MEASUREMENT_ROBUSTNESS"; root.mkdir(parents=True, exist_ok=True)
    n1.to_csv(root / "N1_frozen_per_seed.csv", index=False)
    n2.to_csv(root / "N2_reidentified_per_seed.csv", index=False)
    n1.groupby(["direction", "information_set", "alpha"]).agg(rmse_mean=("rmse_delta", "mean"), rmse_std=("rmse_delta", "std"),
                                                                  level_r2_mean=("r2_level_reconstructed", "mean")).reset_index().to_csv(root / "N1_aggregate.csv", index=False)
    n2.groupby(["direction", "information_set", "alpha"]).agg(rmse_mean=("rmse_delta", "mean"), rmse_std=("rmse_delta", "std"),
                                                                  unique_structures=("active_signature", "nunique")).reset_index().to_csv(root / "N2_aggregate.csv", index=False)
    return n1, n2


def write_summary(output: Path, freezes: Sequence[Mapping[str, Any]], formal: pd.DataFrame,
                  e2: pd.DataFrame, e3: pd.DataFrame, e4: pd.DataFrame, e5: pd.DataFrame,
                  n1: pd.DataFrame, n2: pd.DataFrame) -> None:
    paired = e3.pivot(index=["direction", "seed"], columns="arm", values="rmse_delta").reset_index()
    paired["gain"] = (paired["UNIFORM_SCALE"] - paired["CHANNEL_SPECIFIC_MULTISCALE"]) / paired["UNIFORM_SCALE"]
    summary = {
        "status": "COMPLETED", "protocol_id": PROTOCOL_ID, "completed_utc": utc(),
        "privacy": {"github_upload": False, "raw_workbook_in_results": False, "server_namespace_private": True},
        "formal_metrics": formal.to_dict(orient="records"),
        "E1_active_routes": {f"{item['direction']}:{view}": [r["stage"] for r in records if r.get("active")]
                             for item in freezes for view, records in item["stage_records"].items()},
        "E2_overall_recovery_rate": float(e2["correct"].mean()),
        "E2_false_admission_rate": float(e2["false_admission"].mean()),
        "E3_mean_multiscale_gain": float(paired["gain"].mean()),
        "E4_best_variants": e4.loc[e4.groupby(["direction", "experiment"])["rmse_delta"].idxmin(),
                                     ["direction", "experiment", "variant", "rmse_delta"]].to_dict(orient="records"),
        "E5_conditions": len(e5),
        "E6_N1_rows": len(n1), "E6_N2_rows": len(n2),
        "limitations": [
            "E2 uses observed CZ input trajectories with generated outcomes, so its structural truth is semisynthetic.",
            "The private CZ adapter uses explicit auditable feature blocks and the canonical strict nested-OOF routing rule; it is not a byte-for-byte rerun of the public multi-dataset E1-E6 task registry.",
            "Two transfer directions are selected independently; macro summaries are reporting-only.",
        ],
    }
    write_json(output / "SUMMARY.json", summary)
    table_columns = ["direction", "information_set", "rows", "rmse_delta", "mae_delta",
                     "r2_delta", "r2_level_reconstructed", "persistence_skill"]
    header = "| " + " | ".join(table_columns) + " |"
    separator = "|" + "|".join("---" for _ in table_columns) + "|"
    table_rows = [
        "| " + " | ".join(
            f"{row[column]:.6g}" if isinstance(row[column], (float, np.floating)) else str(row[column])
            for column in table_columns
        ) + " |"
        for _, row in formal[table_columns].iterrows()
    ]
    markdown = ["# Private CZ raw-2s L256 H4 E1-E6 Result Summary", "", f"Protocol: `{PROTOCOL_ID}`", "",
                "Formal task: 2-second samples, input `[t-256,t)`, anchor `D[t-1]`, target `D[t+3]-D[t-1]` (h=4, 8 seconds), W/W0=1/1.", "",
                "The workbook is not included in this result directory and no artifact is uploaded to GitHub.", "",
                "## Formal target-rod metrics", "", header, separator, *table_rows, "",
                "## Main readouts", "",
                f"- E2 semisynthetic recovery rate: {summary['E2_overall_recovery_rate']:.3f}",
                f"- E2 false-admission rate: {summary['E2_false_admission_rate']:.3f}",
                f"- E3 mean paired multiscale gain: {summary['E3_mean_multiscale_gain']:.3%}",
                f"- E5 evidence rows: {len(e5)}", f"- E6 N1/N2 rows: {len(n1)}/{len(n2)}", ""]
    (output / "SUMMARY.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.resolve(); output = args.output.resolve(); project = args.project.resolve()
    validate_protocol_contract()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing nonempty output namespace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project / "src"))
    from prism_benchmark.strict_oof_selection import strict_nested_oof_select

    manifest = {
        "status": "INPUTS_FROZEN", "created_utc": utc(), "protocol_id": PROTOCOL_ID,
        "git": git_provenance(project),
        "raw": {"path": str(raw), "bytes": raw.stat().st_size, "sha256": sha256_file(raw)},
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "selector": {"path": str(project / "src/prism_benchmark/strict_oof_selection.py"),
                     "sha256": sha256_file(project / "src/prism_benchmark/strict_oof_selection.py")},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
        "protocol": {
            "task": "CZ_DIAM_RAW2S_CURRENT_L256_H4",
            "directions": DIRECTIONS,
            "breakpoints": BREAKPOINTS,
            "positional_columns": POSITIONAL_COLUMNS,
            "sampling_interval_seconds": RAW_PERIOD_SECONDS,
            "model_interval_seconds": MODEL_PERIOD_SECONDS,
            "history_points": HISTORY_LENGTH,
            "input": "[t-256,t)",
            "anchor": "D[t-1]",
            "target_delta": "D[t+h-1]-D[t-1]",
            "H": H,
            "W": W,
            "W0": W0,
            "formal_span_seconds": H * RAW_PERIOD_SECONDS,
            "feature_scales": HISTORIES,
            "expanded_feature_scales": ALL_HISTORIES,
            "temporal_aggregation": "NONE",
            "seeds": SEEDS,
            "selection_rule": "STRICT_NESTED_OOF_EMPIRICAL_RISK",
            "E6": {
                "gaussian_std_fractions": PERTURBATION_LEVELS,
                "injection_point": "RAW_MEASUREMENTS_BEFORE_PCA_AND_HISTORY_FUSION",
                "noise_scale_scope": "UNIQUE_SOURCE_OUTER_TRAIN_RAW_MEASUREMENTS_ONLY",
                "clean_outcome_policy": "PERTURB_INPUT_MEASUREMENTS_ONLY",
            },
        },
        "github_upload": False,
        "dependency_purge_steps": PURGE,
        "dependency_purge_formula": "history_steps + h_steps",
    }
    write_json(output / "freeze" / "INPUT_MANIFEST.json", manifest)
    freezes = [freeze_direction(raw, direction, output, strict_nested_oof_select) for direction in DIRECTIONS]
    write_json(output / "freeze" / "GLOBAL_SELECTION_FREEZE.json",
               {"status": "GLOBAL_SELECTION_FROZEN", "sealed": True, "created_utc": utc(),
                "direction_freezes": [{"direction": item["direction"],
                                       "sha256": sha256_file(output / "development" / item["direction"] / "DIRECTION_FREEZE.json")} for item in freezes],
                "target_metrics_accessed": False})
    e2 = run_e2(raw, freezes, output, strict_nested_oof_select)
    e3, e4 = run_e3_e4(raw, freezes, output, strict_nested_oof_select)
    e5 = run_e5(freezes, e2, e3, e4, output)
    fit_checkpoints(raw, freezes, output)
    formal = run_formal(raw, freezes, output)
    n1, n2 = run_e6(raw, freezes, output, strict_nested_oof_select)
    write_summary(output, freezes, formal, e2, e3, e4, e5, n1, n2)
    write_json(output / "RUN_STATUS.json", {"status": "COMPLETED", "completed_utc": utc(),
                                             "raw_workbook_copied_to_results": False, "github_upload": False})
    print(json.dumps({"status": "COMPLETED", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
