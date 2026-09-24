"""Authority-aligned E3--E6 development audits for private raw-2s CZ.

This module is an explicit private-data extension.  It does not change or
masquerade as the public authority branch.  The statistical operations are
delegated to the frozen authority E3/E4/E6 helpers wherever possible, while
the data adapter enforces the CZ H/W/W0=4/1/1 anchor convention.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, deterministic_subsample, inner_folds, input_columns
from .e1e6_phase_b import (
    E3_RIDGES,
    E3_SEEDS,
    _candidate_design,
    _e3_candidates,
    _fit_ridge,
    _history_supported,
    run_e4,
)
from .e1e6_revalidation import _metrics, support_hash, write_json
from .e1e6_robustness import (
    GAUSSIAN_LEVELS,
    N1_SEEDS,
    N2_SEEDS,
    _anchor_noise,
    _apply_contract,
    _frozen_ridge_contract,
    _level_metrics,
    _n1_design,
    _n2_worker,
    _outer_train_process_sigmas,
    _outer_train_target_sigma,
    _perturb_tep_features,
    perturbation_conditions,
)
from .strict_oof_selection import strict_nested_oof_select
from .v2_views import registered_views
from .v211_support import load_native_samples


AUTHORITY_COMMIT = "2ee6273b8f915cbcdff2f46d56bc80047ddae4a7"
HEAD = "CZ_DIAM_RAW2S_CURRENT_L256_H4"
DIRECTIONS = ("Rod_1_to_Rod_2", "Rod_2_to_Rod_1")
CZ_HISTORIES = (8, 16, 32)
CZ_TARGET = "crystal_diameter"
CZ_DATASET = "cz_czochralski"
BLOCK_ROWS = 512
BLOCK_EMBARGO = 260


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_authority_sources(project: Path) -> dict[str, Any]:
    expected = {
        "src/prism_benchmark/e1e6_phase_b.py": "b7a07f4466dbb39fcfaeb77b5733e4783dfffa5ecddbfe8ddac4e1c70ceb4efa",
        "src/prism_benchmark/e1e6_robustness.py": "ec3da27cf67c4fde629e46f3467ffb2bf8b6493d8d56b97c3197177cec4e9ab9",
        "src/prism_benchmark/strict_oof_selection.py": "cac126f10ae015d8b4771816671257a637de668fdaafec66a8bebad926cccfa9",
    }
    actual = {relative: _sha256(project / relative) for relative in expected}
    if actual != expected:
        raise RuntimeError(f"STOP_AUTHORITY_SOURCE_HASH_MISMATCH:{actual}")
    return {
        "status": "PASS",
        "authority_commit": AUTHORITY_COMMIT,
        "sha256": actual,
    }


def _view(shared: Path, information_set: str) -> Any:
    matches = [
        view
        for view in registered_views(shared, information_set)
        if view.head.head_id == HEAD
        and view.availability_scenario == "record_time"
        and view.proxy_policy == "primary"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"CZ extension requires one {information_set} H4 view; found {len(matches)}"
        )
    return matches[0]


def _active_channels(direction_root: Path) -> tuple[str, ...]:
    path = direction_root / f"results/DEVELOPMENT/C/{HEAD}/primary/RESULT.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    channels = tuple(str(item) for item in value["active_channels"])
    if not channels:
        raise RuntimeError("CZ E3 requires at least one frozen active K channel")
    return channels


def _run_e3_direction(anchor: Path, direction: str) -> list[dict[str, Any]]:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    shared = anchor / "cz/h4/shared" / direction
    direction_root = anchor / "cz/h4/directions" / direction
    view = _view(shared, "input_only")
    channels = _active_channels(direction_root)
    train_accessor = BaseAccessor(shared, CZ_DATASET, "train", [*channels, CZ_TARGET])
    validation_accessor = BaseAccessor(
        shared, CZ_DATASET, "validation", [*channels, CZ_TARGET]
    )
    train_unfiltered = load_native_samples(shared, view, "train")
    validation_unfiltered = load_native_samples(shared, view, "validation")
    train_full = _history_supported(train_unfiltered, train_accessor, max(CZ_HISTORIES))
    validation_full = _history_supported(
        validation_unfiltered, validation_accessor, max(CZ_HISTORIES)
    )
    train = train_full.iloc[
        deterministic_subsample(train_full, 20000)
    ].reset_index(drop=True)
    validation = validation_full.iloc[
        deterministic_subsample(validation_full, 20000)
    ].reset_index(drop=True)
    folds = inner_folds(train, count=4)
    train_cache: dict[tuple[str, int], np.ndarray] = {}
    validation_cache: dict[tuple[str, int], np.ndarray] = {}
    for channel in channels:
        for history in CZ_HISTORIES:
            train_cache[(channel, history)] = train_accessor.input_regular_lags(
                train, [channel], 1, history, 8
            )
            validation_cache[(channel, history)] = validation_accessor.input_regular_lags(
                validation, [channel], 1, history, 8
            )
    target = train["y_true"].to_numpy(dtype=np.float64)
    validation_target = validation["y_true"].to_numpy(dtype=np.float64)
    current = validation_accessor.gather(
        validation,
        [CZ_TARGET],
        validation["origin"].to_numpy(dtype=np.int64) - 1,
    )[:, 0]
    rows: list[dict[str, Any]] = []
    for seed in E3_SEEDS:
        for arm in ("UNIFORM_SCALE", "CHANNEL_SPECIFIC_MULTISCALE"):
            started = time.perf_counter()
            candidates = _e3_candidates(channels, CZ_HISTORIES, seed, arm)
            budget = len(CZ_HISTORIES) * len(E3_RIDGES)
            if len(candidates) != budget:
                raise RuntimeError("CZ E3 candidate budget mismatch")
            fold_losses: dict[str, list[float]] = {}
            designs: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, ...], float]] = {}
            parent_losses: list[float] = []
            fold_weights: list[int] = []
            for fit_index, evaluation_index in folds:
                y_fit = target[fit_index]
                y_evaluation = target[evaluation_index]
                parent = np.full(len(evaluation_index), y_fit.mean(dtype=np.float64))
                parent_losses.append(
                    float(np.mean(np.square(y_evaluation - parent), dtype=np.float64))
                )
                fold_weights.append(len(evaluation_index))
            for assignment, alpha in candidates:
                candidate_id = (
                    f"history={','.join(map(str, assignment))}|alpha={alpha:.17g}"
                )
                design_train = _candidate_design(train_cache, channels, assignment)
                design_validation = _candidate_design(
                    validation_cache, channels, assignment
                )
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
                designs[candidate_id] = (
                    design_train,
                    design_validation,
                    assignment,
                    alpha,
                )
            selection = strict_nested_oof_select(
                fold_losses,
                parent_losses,
                identity="K_ZERO_IDENTITY",
                fold_weights=fold_weights,
            )
            tuned = str(selection.tuned_nonzero_candidate)
            design_train, design_validation, assignment, alpha = designs[tuned]
            prediction = (
                _fit_ridge(design_train, target, design_validation, alpha)
                if selection.active
                else np.full(len(validation), target.mean(dtype=np.float64))
            )
            metric = _metrics(validation_target, prediction, current)
            rows.append(
                {
                    "task": direction,
                    "direction": direction,
                    "seed": seed,
                    "arm": arm,
                    "candidate_fit_budget": budget,
                    "candidate_fits": budget,
                    "route": selection.routing_status,
                    "absolute_oof_gain": selection.incremental_gain,
                    "relative_admission_margin": selection.incremental_gain
                    / max(selection.parent_oof_risk, np.finfo(np.float64).eps),
                    "selected_candidate": tuned,
                    "selected_scale_assignment": json.dumps(list(assignment)),
                    "distinct_selected_scales": len(set(assignment)),
                    "channels": len(channels),
                    "active_channels": json.dumps(channels),
                    "candidate_space_degenerate_single_channel": len(channels) == 1,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "parameter_count": int(design_train.shape[1] + 1),
                    "wall_time_seconds": float(time.perf_counter() - started),
                    "Delta_RMSE": metric["rmse_delta"],
                    "Delta_MAE": metric["mae_delta"],
                    "Delta_R2": metric["r2_delta"],
                    "Level_R2": metric["r2_level_reconstructed"],
                    "support_hash": support_hash(
                        validation["base_origin_id"].astype(str)
                    ),
                    "prediction_hash": hashlib.sha256(
                        np.ascontiguousarray(prediction, dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "test_accessed": False,
                    "ood_accessed": False,
                }
            )
    return rows


def _e3_worker(spec: tuple[str, str]) -> list[dict[str, Any]]:
    anchor, direction = spec
    return _run_e3_direction(Path(anchor), direction)


def run_cz_e3(output: Path, anchor: Path, *, workers: int = 2) -> pd.DataFrame:
    destination = output / "E3_MULTISCALE"
    destination.mkdir(parents=True, exist_ok=True)
    specs = [(str(anchor), direction) for direction in DIRECTIONS]
    if workers <= 1:
        nested = [_e3_worker(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
            nested = list(pool.map(_e3_worker, specs))
    frame = pd.DataFrame([row for rows in nested for row in rows])
    frame.to_csv(destination / "equal_budget_multiscale.csv", index=False)
    paired = frame.pivot(
        index=["direction", "seed"], columns="arm", values="Delta_RMSE"
    ).reset_index()
    paired["G_MS"] = (
        paired["UNIFORM_SCALE"] - paired["CHANNEL_SPECIFIC_MULTISCALE"]
    ) / paired["UNIFORM_SCALE"]
    paired.to_csv(destination / "paired_multiscale_gain.csv", index=False)
    same_candidates = bool(
        (
            frame.groupby(["direction", "seed"])["selected_candidate"].nunique()
            == 1
        ).all()
    )
    status = {
        "status": "COMPLETED",
        "scope": "PRIVATE_CZ_AUTHORITY_ALIGNED_EXTENSION_NOT_AUTHORITY_BRANCH_OUTPUT",
        "directions": list(DIRECTIONS),
        "seeds": list(E3_SEEDS),
        "histories": list(CZ_HISTORIES),
        "exact_budget_equal": bool(
            (frame.groupby(["direction", "seed"])["candidate_fits"].nunique() == 1).all()
        ),
        "same_support_within_pair": bool(
            (frame.groupby(["direction", "seed"])["support_hash"].nunique() == 1).all()
        ),
        "single_active_channel_degeneracy": bool(
            frame["candidate_space_degenerate_single_channel"].all()
        ),
        "same_selected_candidate_within_pair": same_candidates,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(destination / "STATUS.json", status)
    return frame


def _four_embargoed_blocks(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    ordered = frame.sort_values("origin").reset_index(drop=True)
    required = 4 * BLOCK_ROWS + 3 * BLOCK_EMBARGO
    if len(ordered) < required:
        raise RuntimeError(f"CZ E6 needs {required} rows in one entity; found {len(ordered)}")
    slack = len(ordered) - required
    start = slack // 2
    blocks = []
    labels = []
    for group in range(4):
        left = start + group * (BLOCK_ROWS + BLOCK_EMBARGO)
        right = left + BLOCK_ROWS
        blocks.append(ordered.iloc[left:right])
        labels.extend([group] * BLOCK_ROWS)
    selected = pd.concat(blocks, ignore_index=True)
    for group in range(3):
        left_max = int(selected.loc[np.asarray(labels) == group, "origin"].max())
        right_min = int(selected.loc[np.asarray(labels) == group + 1, "origin"].min())
        if right_min - left_max <= BLOCK_EMBARGO:
            raise RuntimeError("CZ E6 block embargo certificate failed")
    return selected, np.asarray(labels, dtype=np.int64)


def _registered_cz_arrays(
    anchor: Path, direction: str, information_set: str
) -> dict[str, Any]:
    shared = anchor / "cz/h4/shared" / direction
    view = _view(shared, information_set)
    samples = load_native_samples(shared, view, "train")
    entity_sizes = samples.groupby("entity_id").size().sort_values(ascending=False)
    entity = str(entity_sizes.index[0])
    frame, groups = _four_embargoed_blocks(
        samples[samples["entity_id"].astype(str) == entity]
    )
    channels = tuple(
        str(value)
        for value in input_columns(shared, view.head.task_id, view.proxy_policy)
    )
    accessor = BaseAccessor(shared, CZ_DATASET, "train", [*channels, CZ_TARGET])
    origins = frame["origin"].to_numpy(dtype=np.int64)
    raw_process_blocks: list[np.ndarray] = []
    raw_process_indices: list[np.ndarray] = []
    raw_process_channel_keys: list[int] = []
    for channel_index, channel in enumerate(channels):
        for history in CZ_HISTORIES:
            offsets = np.unique(
                np.rint(np.linspace(1, history, 8)).astype(np.int64)
            )
            indices = origins[:, None] - offsets[None, :]
            raw_process_blocks.append(accessor.gather(frame, [channel], indices))
            raw_process_indices.append(indices)
            raw_process_channel_keys.append(channel_index)
    process = np.column_stack(
        [block.mean(axis=1, dtype=np.float64) for block in raw_process_blocks]
    )
    latest = frame["latest_available_target_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(latest, origins - 1):
        raise RuntimeError("CZ E6 dynamic history is not anchored at t-1")
    if information_set == "dynamic":
        historical = accessor.gather(frame, [CZ_TARGET], latest)
        x = np.column_stack([process, historical])
    else:
        x = process
    data: dict[str, Any] = {
        "x": x,
        "y": frame["y_true"].to_numpy(dtype=np.float64),
        "anchor": accessor.gather(frame, [CZ_TARGET], origins - 1)[:, 0],
        "origins": origins,
        "groups": groups,
        "latest_target": latest,
        "process_columns": process.shape[1],
        "channels": channels,
        "raw_process_blocks": raw_process_blocks,
        "raw_process_indices": raw_process_indices,
        "raw_process_channel_keys": raw_process_channel_keys,
        "availability": "record_time",
        "a_lags": (1, 2, 4),
        "base_origin_ids": frame["base_origin_id"].astype(str).to_numpy(),
        "support_hash": support_hash(frame["base_origin_id"].astype(str)),
        "direction": direction,
        "information_set": information_set,
        "entity": entity,
    }
    data["outer_process_sigmas"] = {
        group: _outer_train_process_sigmas(data, groups != group)
        for group in range(4)
    }
    data["outer_target_sigmas"] = {
        group: _outer_train_target_sigma(data, groups != group)
        for group in range(4)
    }
    return data


def _n2_cz_worker(spec: tuple[Any, ...]) -> list[dict[str, Any]]:
    direction, view_name, mode, data, seed, outer_group = spec
    rows = _n2_worker((view_name, mode, data, seed, outer_group))
    for row in rows:
        row["direction"] = direction
    return rows


def run_cz_e6(
    output: Path, anchor: Path, *, workers: int = 16
) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = output / "E6_MEASUREMENT_ROBUSTNESS"
    destination.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("input_only/record_time", "input_only", ("process_only",)),
        ("dynamic/record_time", "dynamic", ("process_only", "realistic")),
    )
    arrays = {
        (direction, name): _registered_cz_arrays(anchor, direction, information)
        for direction in DIRECTIONS
        for name, information, _ in definitions
    }
    n1_rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for view_name, _, modes in definitions:
            data = arrays[(direction, view_name)]
            clean_design = _n1_design(np.asarray(data["x"]))
            fit_mask = np.asarray(data["groups"]) != 3
            evaluation_mask = ~fit_mask
            process_sigmas = np.asarray(data["outer_process_sigmas"][3])
            target_sigma = float(data["outer_target_sigmas"][3])
            contract = _frozen_ridge_contract(
                clean_design[fit_mask], np.asarray(data["y"])[fit_mask]
            )
            contract_hash = hashlib.sha256(
                np.ascontiguousarray(contract["coefficient"], dtype=np.float64).tobytes()
            ).hexdigest()
            for mode in modes:
                for magnitude in GAUSSIAN_LEVELS:
                    for seed in N1_SEEDS:
                        perturbed = _perturb_tep_features(
                            data,
                            mode=mode,
                            condition="GAUSSIAN",
                            magnitude=magnitude,
                            seed=seed,
                            process_sigmas=process_sigmas,
                            target_sigma=target_sigma,
                        )
                        prediction = _apply_contract(
                            contract, _n1_design(perturbed)[evaluation_mask]
                        )
                        anchor_prediction = (
                            _anchor_noise(
                                np.asarray(data["anchor"])[evaluation_mask],
                                np.asarray(data["origins"])[evaluation_mask],
                                np.asarray(data["groups"])[evaluation_mask],
                                sigma=target_sigma,
                                condition="GAUSSIAN",
                                magnitude=magnitude,
                                seed=seed,
                            )
                            if mode == "realistic"
                            else np.asarray(data["anchor"])[evaluation_mask]
                        )
                        n1_rows.append(
                            {
                                "phase": "N1_FROZEN_MODEL",
                                "direction": direction,
                                "view": view_name,
                                "mode": mode,
                                "condition": "GAUSSIAN",
                                "magnitude": magnitude,
                                "seed": seed,
                                **_level_metrics(
                                    np.asarray(data["y"])[evaluation_mask],
                                    prediction,
                                    np.asarray(data["anchor"])[evaluation_mask],
                                    anchor_prediction,
                                ),
                                "frozen_contract_hash": contract_hash,
                                "support_hash": support_hash(
                                    np.asarray(data["base_origin_ids"])[evaluation_mask]
                                ),
                                "test_accessed": False,
                                "ood_accessed": False,
                            }
                        )
    n1 = pd.DataFrame(n1_rows)
    if not np.isfinite(n1.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("CZ E6 N1 produced non-finite numeric evidence")
    clean_n1 = n1[n1["magnitude"] == 0.0]
    n1_groups = clean_n1.groupby(["direction", "view", "mode"], dropna=False)
    for column in ("Delta_RMSE", "Delta_R2", "Level_R2", "persistence_skill"):
        if not (n1_groups[column].nunique() == 1).all():
            raise RuntimeError(f"CZ E6 N1 alpha=0 seed invariance failed for {column}")
    n1.to_csv(destination / "n1_frozen_model.csv", index=False)
    specs = []
    for direction in DIRECTIONS:
        for view_name, _, modes in definitions:
            data = arrays[(direction, view_name)]
            for mode in modes:
                for seed in N2_SEEDS:
                    for outer_group in range(4):
                        specs.append(
                            (direction, view_name, mode, data, seed, outer_group)
                        )
    if workers <= 1:
        nested = [_n2_cz_worker(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            nested = list(pool.map(_n2_cz_worker, specs, chunksize=1))
    n2 = pd.DataFrame([row for rows in nested for row in rows])
    expected_n1 = len(DIRECTIONS) * 3 * len(GAUSSIAN_LEVELS) * len(N1_SEEDS)
    expected_n2 = (
        len(DIRECTIONS)
        * 3
        * len(perturbation_conditions())
        * len(N2_SEEDS)
        * 4
    )
    if len(n1) != expected_n1 or len(n2) != expected_n2:
        raise RuntimeError(
            f"CZ E6 row-count certificate failed: N1={len(n1)}/{expected_n1}, "
            f"N2={len(n2)}/{expected_n2}"
        )
    if not np.isfinite(n2.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("CZ E6 produced non-finite numeric evidence")
    clean = n2[(n2["condition"] == "GAUSSIAN") & (n2["magnitude"] == 0.0)]
    clean_groups = clean.groupby(
        ["direction", "view", "mode", "outer_fold"], dropna=False
    )
    for column in ("prediction_hash", "stage_vector", "active_channels"):
        if not (clean_groups[column].nunique() == 1).all():
            raise RuntimeError(f"CZ E6 alpha=0 seed invariance failed for {column}")
    n2.to_csv(destination / "n2_reidentification.csv", index=False)
    summary = n2.groupby(
        ["direction", "view", "mode", "condition", "magnitude"], as_index=False
    ).agg(
        P_C_ACTIVE=("C_active", "mean"),
        P_W_ACTIVE=("W_active", "mean"),
        P_A_ACTIVE=("A_active", "mean"),
        median_m_C=("C_margin", "median"),
        median_m_W=("W_margin", "median"),
        median_m_A=("A_margin", "median"),
        Delta_RMSE=("Delta_RMSE", "mean"),
        Delta_R2=("Delta_R2", "mean"),
        Level_R2=("Level_R2", "mean"),
        persistence_skill=("persistence_skill", "mean"),
    )
    summary.to_csv(destination / "n2_robustness_summary.csv", index=False)
    write_json(
        destination / "STATUS.json",
        {
            "status": "COMPLETED",
            "scope": "PRIVATE_CZ_AUTHORITY_ALIGNED_EXTENSION_NOT_AUTHORITY_BRANCH_OUTPUT",
            "protocol": "CZ_RAW2S_CURRENT_L256_H4_W1_W0_1",
            "directions": list(DIRECTIONS),
            "histories": list(CZ_HISTORIES),
            "block_rows": BLOCK_ROWS,
            "block_embargo": BLOCK_EMBARGO,
            "n1_rows": len(n1),
            "n2_rows": len(n2),
            "row_count_certificate": True,
            "finite_numeric_certificate": True,
            "n1_alpha_zero_seed_invariance_certificate": True,
            "alpha_zero_seed_invariance_certificate": True,
            "future_truth_clean": True,
            "formal_test_used_as_evaluation": False,
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    return n1, n2


def run_cz_e5(output: Path, e3: pd.DataFrame) -> dict[str, Any]:
    destination = output / "E5_STRUCTURAL_STABILITY"
    destination.mkdir(parents=True, exist_ok=True)
    e4a = pd.read_csv(output / "E4A_GRID_SENSITIVITY/grid_density_runs.csv")
    e4b = pd.read_csv(output / "E4B_FAMILY_ABLATION/family_ablation_runs.csv")
    rows = []
    for source, frame in (("candidate_universe", e4a), ("family_ablation", e4b)):
        for condition, group in frame.groupby("condition"):
            for stage in ("C", "W", "A"):
                probability = float(group[f"{stage}_active"].astype(int).mean())
                entropy = -sum(
                    p * math.log2(p)
                    for p in (probability, 1.0 - probability)
                    if p > 0
                )
                rows.append(
                    {
                        "source": source,
                        "condition": condition,
                        "stage": stage,
                        "admission_probability": probability,
                        "selection_entropy_bits": entropy,
                        "median_margin": float(group[f"{stage}_margin"].median()),
                    }
                )
    pd.DataFrame(rows).to_csv(destination / "final_stage_stability.csv", index=False)
    scale_rows = []
    for direction, group in e3.groupby("direction"):
        multiscale = group[group["arm"] == "CHANNEL_SPECIFIC_MULTISCALE"]
        frequencies = multiscale["selected_scale_assignment"].value_counts(normalize=True)
        scale_rows.append(
            {
                "direction": direction,
                "runs": len(multiscale),
                "conditional_scale_agreement": float(frequencies.max()),
                "selection_entropy_bits": float(
                    -sum(p * math.log2(p) for p in frequencies if p > 0)
                ),
                "single_active_channel_degeneracy": bool(
                    multiscale["candidate_space_degenerate_single_channel"].all()
                ),
            }
        )
    pd.DataFrame(scale_rows).to_csv(destination / "final_scale_stability.csv", index=False)
    result = {
        "status": "COMPLETED",
        "scope": "PRIVATE_CZ_E3_PLUS_AUTHORITY_SYNTHETIC_E4_STABILITY_SUMMARY",
        "directions_kept_separate": True,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(destination / "FINAL_STATUS.json", result)
    return result


def write_master_status(output: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    statuses = {}
    for name, relative in {
        "E3": "E3_MULTISCALE/STATUS.json",
        "E4A": "E4A_GRID_SENSITIVITY/STATUS.json",
        "E4B": "E4B_FAMILY_ABLATION/STATUS.json",
        "E5": "E5_STRUCTURAL_STABILITY/FINAL_STATUS.json",
        "E6": "E6_MEASUREMENT_ROBUSTNESS/STATUS.json",
    }.items():
        path = output / relative
        statuses[name] = (
            json.loads(path.read_text(encoding="utf-8")).get("status")
            if path.is_file()
            else "NOT_RUN"
        )
    completed = all(value == "COMPLETED" for value in statuses.values())
    value = {
        "status": "COMPLETED" if completed else "PARTIAL",
        "scope": "PRIVATE_CZ_AUTHORITY_ALIGNED_EXTENSION_NOT_AUTHORITY_BRANCH_OUTPUT",
        "authority": dict(authority),
        "protocol": "CZ_RAW2S_CURRENT_L256_H4_W1_W0_1",
        "sampling_seconds": 2,
        "forecast_seconds": 8,
        "formal_target_or_ood_accessed": False,
        "stages": statuses,
    }
    write_json(output / "RUN_STATUS.json", value)
    return value
