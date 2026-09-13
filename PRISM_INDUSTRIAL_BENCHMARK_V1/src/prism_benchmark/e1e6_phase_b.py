"""Frozen Phase-B experiments for the strict nested-OOF revalidation.

The routines in this module never open formal-test or OOD partitions.  E3 is
an equal-fit-budget K-family experiment on the registered development data.
E4 changes only the pre-registered synthetic candidate universe (E4a) or one
hypothesis family at a time (E4b).  These experiments are deliberately kept
separate so search-volume and family-capability effects cannot be conflated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, deterministic_subsample, inner_folds
from .e1e6_revalidation import (
    EXACT_ZERO,
    STAGES,
    _constant_oof,
    _folds,
    _generate_identifiable,
    _lag_block,
    _metrics,
    _ridge_oof,
    _select_increment,
    support_hash,
    write_json,
)
from .strict_oof_selection import ACTIVE, strict_nested_oof_select, tune_best_nonzero
from .v2_views import registered_views
from .v211_support import load_native_samples


E3_SEEDS = tuple(range(10))
E4_SEEDS = tuple(range(10))
E3_RIDGES = (1e-3, 1e-1)
E4_UNIVERSES: dict[str, dict[str, tuple[float, ...] | tuple[int, ...]]] = {
    "COARSE": {"histories": (1, 4), "ridges": (1e-3, 1.0)},
    "STANDARD": {"histories": (1, 4, 8), "ridges": (1e-6, 1e-3, 1.0)},
    "EXPANDED": {
        "histories": (1, 2, 4, 8, 12),
        "ridges": (1e-8, 1e-6, 1e-3, 1e-1, 1.0),
    },
}


@dataclass(frozen=True)
class E3Task:
    name: str
    shared_name: str
    head: str
    information_set: str
    availability: str
    proxy: str
    dataset: str
    target: str
    histories: tuple[int, ...]
    w0: int


E3_TASKS = (
    E3Task(
        "Debutanizer",
        "public3_shared",
        "DEB_C4__H5__W1",
        "dynamic",
        "record_time",
        "primary",
        "debutanizer",
        "y",
        (10, 20, 40),
        1,
    ),
    E3Task(
        "TEP input-only",
        "tep_shared",
        "TEP_G_NOWCAST_H0__H0__W1",
        "input_only",
        "record_time",
        "proxy_excluded",
        "tep",
        "xmeas_40",
        (128, 256),
        1,
    ),
)


def _fit_ridge(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_evaluation: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x = np.asarray(x_fit, dtype=np.float64)
    y = np.asarray(y_fit, dtype=np.float64)
    z = np.asarray(x_evaluation, dtype=np.float64)
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale * scale < 1e-12] = 1.0
    xs = (x - mean) / scale
    zs = (z - mean) / scale
    target_mean = float(y.mean(dtype=np.float64))
    gram = xs.T @ xs + float(alpha) * np.eye(xs.shape[1], dtype=np.float64)
    coefficient = np.linalg.solve(gram, xs.T @ (y - target_mean))
    return zs @ coefficient + target_mean


def _e3_view(shared: Path, task: E3Task) -> Any:
    matches = [
        view
        for view in registered_views(shared, task.information_set)
        if view.head.head_id == task.head
        and view.availability_scenario == task.availability
        and view.proxy_policy == task.proxy
    ]
    if len(matches) != 1:
        raise RuntimeError(f"E3 requires one registered view for {task.name}; found {len(matches)}")
    return matches[0]


def _e3_candidates(
    channels: Sequence[str], histories: Sequence[int], seed: int, arm: str
) -> list[tuple[tuple[int, ...], float]]:
    if arm == "UNIFORM_SCALE":
        assignments = [tuple([int(history)] * len(channels)) for history in histories]
    elif arm == "CHANNEL_SPECIFIC_MULTISCALE":
        rng = np.random.default_rng(seed)
        assignments = []
        attempts = 0
        while len(assignments) < len(histories):
            index = len(assignments)
            values = tuple(int(value) for value in rng.choice(histories, size=len(channels)))
            if len(histories) > 1 and len(set(values)) == 1:
                values = tuple(
                    int(histories[(index + channel) % len(histories)])
                    for channel in range(len(channels))
                )
            if values not in assignments:
                assignments.append(values)
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("E3 could not register enough unique multiscale assignments")
    else:
        raise ValueError(arm)
    return [
        (assignment, float(alpha))
        for assignment in assignments
        for alpha in E3_RIDGES
    ]


def _candidate_design(
    cache: Mapping[tuple[str, int], np.ndarray],
    channels: Sequence[str],
    assignment: Sequence[int],
) -> np.ndarray:
    blocks = [cache[(channel, int(history))] for channel, history in zip(channels, assignment, strict=True)]
    raw = np.column_stack(blocks)
    return np.column_stack([raw, np.square(raw)])


def _history_supported(
    samples: pd.DataFrame, accessor: BaseAccessor, history: int
) -> pd.DataFrame:
    """Restrict rows to exact entity-local raw support for a strict-past lag."""

    minimum = {
        str(entity): int(rows.min())
        for entity, (rows, _) in accessor.entities.items()
    }
    entities = samples["entity_id"].astype(str).to_numpy()
    origins = samples["origin"].to_numpy(dtype=np.int64)
    keep = np.fromiter(
        (
            entity in minimum and origin - int(history) >= minimum[entity]
            for entity, origin in zip(entities, origins, strict=True)
        ),
        dtype=bool,
        count=len(samples),
    )
    result = samples.loc[keep].reset_index(drop=True)
    if result.empty:
        raise RuntimeError("candidate history support filtering removed every row")
    return result


def _run_e3_task(
    task: E3Task,
    shared_root: Path,
    baseline_run: Path,
) -> list[dict[str, Any]]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    shared = shared_root / task.shared_name
    view = _e3_view(shared, task)
    c_path = (
        baseline_run
        / "results/DEVELOPMENT/C"
        / task.head
        / task.proxy
        / "RESULT.json"
    )
    c_result = json.loads(c_path.read_text(encoding="utf-8"))
    channels = tuple(str(value) for value in c_result["active_channels"])
    if not channels:
        raise RuntimeError(f"E3 {task.name} has no frozen active input channels")
    train_accessor = BaseAccessor(shared, task.dataset, "train", [*channels, task.target])
    validation_accessor = BaseAccessor(shared, task.dataset, "validation", [*channels, task.target])
    train_unfiltered = load_native_samples(shared, view, "train")
    validation_unfiltered = load_native_samples(shared, view, "validation")
    train_full = _history_supported(
        train_unfiltered, train_accessor, max(task.histories)
    )
    validation_full = _history_supported(
        validation_unfiltered, validation_accessor, max(task.histories)
    )
    train = train_full.iloc[deterministic_subsample(train_full, 20000)].reset_index(drop=True)
    validation = validation_full.iloc[
        deterministic_subsample(validation_full, 20000)
    ].reset_index(drop=True)
    folds = inner_folds(train, count=4)
    train_cache: dict[tuple[str, int], np.ndarray] = {}
    validation_cache: dict[tuple[str, int], np.ndarray] = {}
    for channel in channels:
        for history in task.histories:
            train_cache[(channel, history)] = train_accessor.input_regular_lags(
                train, [channel], 1, int(history), 8
            )
            validation_cache[(channel, history)] = validation_accessor.input_regular_lags(
                validation, [channel], 1, int(history), 8
            )
    target = train["y_true"].to_numpy(dtype=np.float64)
    validation_target = validation["y_true"].to_numpy(dtype=np.float64)
    current = validation_accessor.block_means(validation, task.target, [(0, task.w0)])[:, 0]
    rows: list[dict[str, Any]] = []
    for seed in E3_SEEDS:
        for arm in ("UNIFORM_SCALE", "CHANNEL_SPECIFIC_MULTISCALE"):
            candidates = _e3_candidates(channels, task.histories, seed, arm)
            # Both spaces are explicitly truncated to the same legal fit budget.
            budget = len(task.histories) * len(E3_RIDGES)
            if len(candidates) != budget:
                raise RuntimeError("E3 candidate budget mismatch")
            fold_losses: dict[str, list[float]] = {}
            designs: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, ...], float]] = {}
            parent_losses: list[float] = []
            fold_weights: list[int] = []
            for fold_index, (fit_index, evaluation_index) in enumerate(folds):
                y_fit = target[fit_index]
                y_evaluation = target[evaluation_index]
                if fold_index == 0:
                    parent_losses = []
                    fold_weights = []
                parent = np.full(len(evaluation_index), y_fit.mean(dtype=np.float64))
                parent_losses.append(float(np.mean(np.square(y_evaluation - parent), dtype=np.float64)))
                fold_weights.append(len(evaluation_index))
            for assignment, alpha in candidates:
                candidate_id = f"history={','.join(map(str, assignment))}|alpha={alpha:.17g}"
                design_train = _candidate_design(train_cache, channels, assignment)
                design_validation = _candidate_design(validation_cache, channels, assignment)
                losses = []
                for fit_index, evaluation_index in folds:
                    prediction = _fit_ridge(
                        design_train[fit_index],
                        target[fit_index],
                        design_train[evaluation_index],
                        alpha,
                    )
                    losses.append(
                        float(
                            np.mean(
                                np.square(target[evaluation_index] - prediction),
                                dtype=np.float64,
                            )
                        )
                    )
                fold_losses[candidate_id] = losses
                designs[candidate_id] = (design_train, design_validation, assignment, alpha)
            selection = strict_nested_oof_select(
                fold_losses,
                parent_losses,
                identity="K_ZERO_IDENTITY",
                fold_weights=fold_weights,
            )
            tuned = str(selection.tuned_nonzero_candidate)
            design_train, design_validation, assignment, alpha = designs[tuned]
            if selection.active:
                prediction = _fit_ridge(
                    design_train, target, design_validation, alpha
                )
            else:
                prediction = np.full(len(validation), target.mean(dtype=np.float64))
            metric = _metrics(validation_target, prediction, current)
            rows.append(
                {
                    "task": task.name,
                    "seed": seed,
                    "arm": arm,
                    "candidate_fit_budget": budget,
                    "candidate_fits": budget,
                    "ridge_fits_including_outer_folds_and_refit": budget * len(folds) + 1,
                    "candidate_space_status": "COMPLETED",
                    "route": selection.routing_status,
                    "absolute_oof_gain": selection.incremental_gain,
                    "relative_admission_margin": selection.incremental_gain
                    / max(selection.parent_oof_risk, np.finfo(np.float64).eps),
                    "selected_candidate": tuned,
                    "selected_scale_assignment": json.dumps(list(assignment)),
                    "distinct_selected_scales": len(set(assignment)),
                    "channels": len(channels),
                    "train_rows_before_history_support": len(train_unfiltered),
                    "train_rows_after_history_support": len(train_full),
                    "validation_rows_before_history_support": len(validation_unfiltered),
                    "validation_rows_after_history_support": len(validation_full),
                    "parameter_count": int(design_train.shape[1] + 1),
                    "Delta_RMSE": metric["rmse_delta"],
                    "Delta_MAE": metric["mae_delta"],
                    "Delta_R2": metric["r2_delta"],
                    "Level_R2": metric["r2_level_reconstructed"],
                    "support_hash": support_hash(validation["base_origin_id"].astype(str)),
                    "prediction_hash": hashlib.sha256(
                        np.ascontiguousarray(prediction, dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "test_accessed": False,
                    "ood_accessed": False,
                }
            )
    return rows


def _e3_worker(spec: tuple[E3Task, str, str]) -> list[dict[str, Any]]:
    task, shared_root, baseline_run = spec
    return _run_e3_task(task, Path(shared_root), Path(baseline_run))


def run_e3(
    output: Path,
    shared_root: Path,
    baseline_run: Path,
    *,
    workers: int = 2,
) -> pd.DataFrame:
    destination = output / "E3_MULTISCALE"
    destination.mkdir(parents=True, exist_ok=True)
    specs = [(task, str(shared_root), str(baseline_run)) for task in E3_TASKS]
    if workers <= 1:
        nested = [_e3_worker(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
            nested = list(pool.map(_e3_worker, specs))
    frame = pd.DataFrame([row for rows in nested for row in rows])
    frame.to_csv(destination / "equal_budget_multiscale.csv", index=False)
    paired = frame.pivot(index=["task", "seed"], columns="arm", values="Delta_RMSE").reset_index()
    paired["G_MS"] = (
        paired["UNIFORM_SCALE"] - paired["CHANNEL_SPECIFIC_MULTISCALE"]
    ) / paired["UNIFORM_SCALE"]
    paired.to_csv(destination / "paired_multiscale_gain.csv", index=False)
    write_json(
        destination / "STATUS.json",
        {
            "status": "COMPLETED",
            "tasks": [task.name for task in E3_TASKS],
            "seeds": list(E3_SEEDS),
            "budget_definition": "number of registered nonzero candidate configurations",
            "exact_budget_equal": bool(
                (frame.groupby(["task", "seed"])["candidate_fits"].nunique() == 1).all()
            ),
            "same_support_within_pair": bool(
                (frame.groupby(["task", "seed"])["support_hash"].nunique() == 1).all()
            ),
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    return frame


def _zero_report(identity: str) -> dict[str, Any]:
    return {
        "routing_status": "ZERO_IDENTITY",
        "relative_admission_margin": 0.0,
        "final_selected_candidate": identity,
    }


def run_synthetic_variant(
    seed: int,
    regime: str,
    *,
    histories: Sequence[int],
    ridges: Sequence[float],
    include_nonlinear_k: bool = True,
    include_c: bool = True,
    include_w: bool = True,
    include_a: bool = True,
    n: int = 2048,
    x_override: np.ndarray | None = None,
    y_override: np.ndarray | None = None,
    return_prediction: bool = False,
    precomputed_k: bool = False,
) -> dict[str, Any]:
    if (x_override is None) != (y_override is None):
        raise ValueError("x_override and y_override must be supplied together")
    if x_override is None:
        generated = _generate_identifiable(seed, n, regime)
        x = np.asarray(generated["x"], dtype=np.float64)
        y = np.asarray(generated["y"], dtype=np.float64)
    else:
        x = np.asarray(x_override, dtype=np.float64)
        y = np.asarray(y_override, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x),):
            raise ValueError("array variant requires aligned 2-D inputs and 1-D target")
    folds = _folds(len(y))
    parent = _constant_oof(y, folds)
    channel_predictions: dict[int, np.ndarray] = {}
    selected_histories: dict[int, int] = {}
    for channel in range(x.shape[1]):
        candidates: dict[str, np.ndarray] = {}
        for history in histories:
            if precomputed_k:
                lagged = x[:, channel, None]
            else:
                offsets = np.unique(
                    np.rint(np.linspace(1, int(history), min(8, int(history)))).astype(np.int64)
                )
                lagged = _lag_block(x[:, channel], offsets.tolist())
            features = (
                np.column_stack([lagged, np.square(lagged)])
                if include_nonlinear_k
                else lagged
            )
            for alpha in ridges:
                candidates[f"channel={channel}|history={history}|alpha={alpha}"] = features
        routed, report, _ = _select_increment(
            y, parent, candidates, folds, identity=f"K_ZERO_CHANNEL_{channel}"
        )
        if report["routing_status"] == ACTIVE:
            channel_predictions[channel] = routed
            tuned = str(report["tuned_nonzero_candidate"])
            selected_histories[channel] = int(tuned.split("history=")[1].split("|")[0])
    k = (
        _ridge_oof(np.column_stack(list(channel_predictions.values())), y, folds, 1e-3)
        if channel_predictions
        else parent
    )
    if include_c:
        # The registered interaction screen is capped before family fitting;
        # this keeps E6's two-scale TEP summary representation tractable while
        # leaving the six-channel identifiable E2/E4 universe unchanged.
        interaction_columns = min(8, x.shape[1])
        pair_columns = [
            x[:, left] * x[:, right]
            for left in range(interaction_columns)
            for right in range(left + 1, interaction_columns)
        ]
        true_pair = (
            x[:, 0] * x[:, 2]
            if precomputed_k
            else _lag_block(x[:, 0], [1])[:, 0]
            * _lag_block(x[:, 2], [2])[:, 0]
        )
        c_candidates = {}
        for alpha in ridges:
            c_candidates[f"C_TRUE_PAIR_02|alpha={alpha}"] = true_pair[:, None]
            c_candidates[f"C_ALL_PAIRS|alpha={alpha}"] = np.column_stack(pair_columns)
        kc, c_report, _ = _select_increment(y, k, c_candidates, folds, identity="C_ZERO_IDENTITY")
    else:
        kc, c_report = k.copy(), _zero_report("C_FAMILY_REMOVED")
    if include_w:
        square = np.square(kc) - np.mean(np.square(kc))
        cube = np.power(kc, 3) - np.mean(np.power(kc, 3))
        w_candidates = {}
        for alpha in ridges:
            w_candidates[f"W_QUADRATIC|alpha={alpha}"] = square[:, None]
            w_candidates[f"W_SMOOTH_POLY|alpha={alpha}"] = np.column_stack([square, cube, np.tanh(kc)])
        kcw, w_report, _ = _select_increment(y, kc, w_candidates, folds, identity="W_ZERO_IDENTITY")
    else:
        kcw, w_report = kc.copy(), _zero_report("W_FAMILY_REMOVED")
    if include_a:
        residual = y - kcw
        a_candidates = {}
        for lags in ((1,), (1, 2), (1, 2, 4)):
            block = _lag_block(residual, lags)
            for alpha in ridges:
                a_candidates[f"A_LAGS_{'_'.join(map(str, lags))}|alpha={alpha}"] = block
        prediction, a_report, _ = _select_increment(y, kcw, a_candidates, folds, identity="A_ZERO_IDENTITY")
    else:
        prediction, a_report = kcw.copy(), _zero_report("A_FAMILY_REMOVED")
    reports = {"C": c_report, "W": w_report, "A": a_report}
    result = {
        "seed": seed,
        "regime": regime,
        "C_active": c_report["routing_status"] == ACTIVE,
        "W_active": w_report["routing_status"] == ACTIVE,
        "A_active": a_report["routing_status"] == ACTIVE,
        "C_margin": float(c_report.get("relative_admission_margin", 0.0)),
        "W_margin": float(w_report.get("relative_admission_margin", 0.0)),
        "A_margin": float(a_report.get("relative_admission_margin", 0.0)),
        "rmse": float(np.sqrt(np.mean(np.square(y - prediction), dtype=np.float64))),
        "active_channels": json.dumps(sorted(channel_predictions)),
        "selected_histories": json.dumps(selected_histories, sort_keys=True),
        "stage_vector": "".join("1" if reports[stage]["routing_status"] == ACTIVE else "0" for stage in STAGES),
        "prediction_hash": hashlib.sha256(
            np.ascontiguousarray(prediction, dtype=np.float64).tobytes()
        ).hexdigest(),
    }
    if return_prediction:
        result["prediction"] = prediction
    return result


def _e4_worker(spec: tuple[str, int, str, tuple[int, ...], tuple[float, ...], bool, bool, bool, bool]) -> dict[str, Any]:
    label, seed, regime, histories, ridges, nonlinear, c, w, a = spec
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    row = run_synthetic_variant(
        seed,
        regime,
        histories=histories,
        ridges=ridges,
        include_nonlinear_k=nonlinear,
        include_c=c,
        include_w=w,
        include_a=a,
    )
    row["condition"] = label
    row["candidate_universe_size"] = (
        6 * len(histories) * len(ridges)
        + (2 * len(ridges) if c else 0)
        + (2 * len(ridges) if w else 0)
        + (3 * len(ridges) if a else 0)
    )
    return row


def _parallel_e4(specs: Sequence[tuple[Any, ...]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_e4_worker(spec) for spec in specs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_e4_worker, specs, chunksize=1))


def run_e4(output: Path, *, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    regimes = ("S1", "S2", "S3", "S4")
    e4a_specs = []
    for label, universe in E4_UNIVERSES.items():
        histories = tuple(int(value) for value in universe["histories"])
        ridges = tuple(float(value) for value in universe["ridges"])
        for seed in E4_SEEDS:
            for regime in regimes:
                e4a_specs.append((label, seed, regime, histories, ridges, True, True, True, True))
    e4a = pd.DataFrame(_parallel_e4(e4a_specs, workers))
    destination_a = output / "E4A_GRID_SENSITIVITY"
    destination_a.mkdir(parents=True, exist_ok=True)
    e4a.to_csv(destination_a / "grid_density_runs.csv", index=False)
    summary_a = e4a.groupby(["condition", "candidate_universe_size"], as_index=False).agg(
        P_C_ACTIVE=("C_active", "mean"),
        P_W_ACTIVE=("W_active", "mean"),
        P_A_ACTIVE=("A_active", "mean"),
        RMSE_mean=("rmse", "mean"),
        unique_stage_vectors=("stage_vector", "nunique"),
    )
    summary_a.to_csv(destination_a / "grid_density_summary.csv", index=False)
    write_json(
        destination_a / "STATUS.json",
        {
            "status": "COMPLETED",
            "seeds": list(E4_SEEDS),
            "nested_universes": True,
            "same_families": True,
            "test_accessed": False,
            "ood_accessed": False,
        },
    )

    standard = E4_UNIVERSES["STANDARD"]
    h = tuple(int(value) for value in standard["histories"])
    r = tuple(float(value) for value in standard["ridges"])
    arms = {
        "STANDARD_FULL_FAMILY": (True, True, True, True),
        "REMOVE_NONLINEAR_K": (False, True, True, True),
        "REMOVE_C": (True, False, True, True),
        "REMOVE_W": (True, True, False, True),
        "REMOVE_A": (True, True, True, False),
    }
    e4b_specs = [
        (label, seed, regime, h, r, *flags)
        for label, flags in arms.items()
        for seed in E4_SEEDS
        for regime in regimes
    ]
    e4b = pd.DataFrame(_parallel_e4(e4b_specs, workers))
    destination_b = output / "E4B_FAMILY_ABLATION"
    destination_b.mkdir(parents=True, exist_ok=True)
    e4b.to_csv(destination_b / "family_ablation_runs.csv", index=False)
    reference = e4b.loc[e4b["condition"] == "STANDARD_FULL_FAMILY", ["seed", "regime", "rmse"]].rename(columns={"rmse": "reference_rmse"})
    paired = e4b.merge(reference, on=["seed", "regime"], validate="many_to_one")
    paired["relative_rmse_change"] = (paired["rmse"] - paired["reference_rmse"]) / paired["reference_rmse"]
    paired.to_csv(destination_b / "family_ablation_paired.csv", index=False)
    write_json(
        destination_b / "STATUS.json",
        {
            "status": "COMPLETED",
            "reference": "STANDARD_FULL_FAMILY",
            "one_family_removed_per_arm": True,
            "seeds": list(E4_SEEDS),
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    return e4a, e4b


def run_e5_final(output: Path, e3: pd.DataFrame, e4a: pd.DataFrame, e4b: pd.DataFrame) -> dict[str, Any]:
    destination = output / "E5_STRUCTURAL_STABILITY"
    destination.mkdir(parents=True, exist_ok=True)
    sources = []
    for source, frame in (("candidate_universe", e4a), ("family_ablation", e4b)):
        for condition, group in frame.groupby("condition"):
            for stage in STAGES:
                values = group[f"{stage}_active"].astype(int)
                probability = float(values.mean())
                entropy = 0.0
                for p in (probability, 1.0 - probability):
                    if p > 0:
                        entropy -= p * math.log2(p)
                sources.append(
                    {
                        "source": source,
                        "condition": condition,
                        "stage": stage,
                        "admission_probability": probability,
                        "selection_entropy_bits": entropy,
                        "median_margin": float(group[f"{stage}_margin"].median()),
                        "margin_iqr": float(group[f"{stage}_margin"].quantile(0.75) - group[f"{stage}_margin"].quantile(0.25)),
                    }
                )
    pd.DataFrame(sources).to_csv(destination / "final_stage_stability.csv", index=False)
    scale_rows = []
    for task, group in e3.groupby("task"):
        multis = group[group["arm"] == "CHANNEL_SPECIFIC_MULTISCALE"]
        scale_rows.append(
            {
                "task": task,
                "runs": len(multis),
                "conditional_scale_agreement": float(multis["selected_scale_assignment"].value_counts(normalize=True).max()),
                "selection_entropy_bits": float(
                    -(multis["selected_scale_assignment"].value_counts(normalize=True).map(lambda p: p * math.log2(p))).sum()
                ),
                "prediction_unique_hashes": int(multis["prediction_hash"].nunique()),
            }
        )
    pd.DataFrame(scale_rows).to_csv(destination / "final_scale_stability.csv", index=False)
    rashomon_rows = []
    for regime, group in e4a.groupby("regime"):
        best = float(group["rmse"].min())
        near = group[group["rmse"] <= 1.01 * best]
        rashomon_rows.append(
            {
                "regime": regime,
                "best_rmse": best,
                "near_optimal_runs": len(near),
                "near_optimal_stage_vectors": int(near["stage_vector"].nunique()),
                "performance_stable_structure_unstable": bool(near["stage_vector"].nunique() > 1),
            }
        )
    pd.DataFrame(rashomon_rows).to_csv(destination / "rashomon_analysis.csv", index=False)
    result = {
        "status": "COMPLETED",
        "sources": ["seed", "candidate-universe", "family-ablation", "E3 task"],
        "rod_status": "PROTOCOL_BLOCKED_NO_LEGALLY_COMPARABLE_CZ_RODS",
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(destination / "FINAL_STATUS.json", result)
    return result
