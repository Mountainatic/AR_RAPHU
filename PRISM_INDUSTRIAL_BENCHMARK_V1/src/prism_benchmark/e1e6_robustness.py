"""Development-only measurement perturbation revalidation for TEP H0/W1.

N1 freezes a compact, pre-registered ridge predictor on clean development
rows and perturbs only its held-out development evaluation rows.  N2 perturbs
the development evidence and reruns the external-zero K/C/W/A routing in a
fixed compact candidate universe.  No formal-test or OOD partition is read.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor
from .e1e6_phase_b import _fit_ridge
from .e1e6_revalidation import (
    _constant_oof,
    _ridge_oof,
    _select_increment,
    support_hash,
    write_json,
)
from .v2_views import registered_views
from .v211_support import load_native_samples


GAUSSIAN_LEVELS = (0.0, 0.01, 0.025, 0.05, 0.10)
BIAS_LEVELS = (-0.10, -0.05, -0.025, -0.01, 0.01, 0.025, 0.05, 0.10)
QUANTIZATION_LEVELS = (0.01, 0.025, 0.05, 0.10)
N1_SEEDS = tuple(range(30))
N2_SEEDS = tuple(range(10))
TEP_HISTORIES = (128, 256)
TEP_RIDGES = (1e-3,)


def perturbation_conditions() -> tuple[tuple[str, float], ...]:
    return tuple(
        [("GAUSSIAN", value) for value in GAUSSIAN_LEVELS]
        + [("BIAS", value) for value in BIAS_LEVELS]
        + [("LINEAR_DRIFT", 0.05), ("RANDOM_WALK_DRIFT", 0.05)]
        + [("QUANTIZATION", value) for value in QUANTIZATION_LEVELS]
    )


def _keyed_normal(indices: np.ndarray, key: int, seed: int) -> np.ndarray:
    """Deterministic N(0,1) keyed by seed/channel/absolute source timestamp."""

    mask = (1 << 64) - 1

    def mix(values: np.ndarray, salt: int) -> np.ndarray:
        x = np.asarray(values, dtype=np.uint64)
        with np.errstate(over="ignore"):
            x = x ^ np.uint64((seed + 1) * 0x9E3779B1 & mask)
            x = x ^ np.uint64((key + 1009) * 0x85EBCA77 & mask)
            x = x ^ np.uint64(salt)
            x = x + np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            return x ^ (x >> np.uint64(31))

    first = mix(indices, 0x243F6A8885A308D3)
    second = mix(indices, 0x13198A2E03707344)
    scale = float(1 << 53)
    u1 = ((first >> np.uint64(11)).astype(np.float64) + 0.5) / scale
    u2 = ((second >> np.uint64(11)).astype(np.float64) + 0.5) / scale
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)


def _perturb_matrix(
    clean: np.ndarray,
    sigmas: np.ndarray,
    origins: np.ndarray,
    latest_target: np.ndarray,
    groups: np.ndarray,
    *,
    process_columns: int,
    mode: str,
    condition: str,
    magnitude: float,
    seed: int,
) -> np.ndarray:
    values = np.asarray(clean, dtype=np.float64).copy()
    affected = process_columns if mode == "process_only" else values.shape[1]
    for column in range(affected):
        source_indices = (
            origins - 1 if column < process_columns else latest_target
        ) + np.asarray(groups, dtype=np.int64) * 10_000_000
        key = column if column < process_columns else 1000003
        sigma = float(sigmas[column])
        if condition == "GAUSSIAN":
            values[:, column] += magnitude * sigma * _keyed_normal(source_indices, key, seed)
        elif condition == "BIAS":
            values[:, column] += magnitude * sigma
        elif condition == "LINEAR_DRIFT":
            direction = -1.0 if int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:2], 16) % 2 else 1.0
            drift = np.empty(len(values), dtype=np.float64)
            for group in pd.unique(groups):
                index = np.flatnonzero(groups == group)
                drift[index] = np.linspace(-1.0, 1.0, len(index))
            values[:, column] += direction * magnitude * sigma * drift
        elif condition == "RANDOM_WALK_DRIFT":
            innovations = _keyed_normal(source_indices, key, seed)
            walk = np.empty(len(values), dtype=np.float64)
            for group in pd.unique(groups):
                index = np.flatnonzero(groups == group)
                part = np.cumsum(innovations[index], dtype=np.float64)
                part -= part.mean(dtype=np.float64)
                walk_scale = part.std(dtype=np.float64)
                if walk_scale > 0:
                    part /= walk_scale
                walk[index] = part
            values[:, column] += magnitude * sigma * walk
        elif condition == "QUANTIZATION":
            resolution = max(magnitude * sigma, np.finfo(np.float64).eps)
            values[:, column] = np.rint(values[:, column] / resolution) * resolution
        else:
            raise ValueError(condition)
    return values


def _anchor_noise(
    anchor: np.ndarray,
    origins: np.ndarray,
    groups: np.ndarray,
    *,
    sigma: float | None = None,
    condition: str,
    magnitude: float,
    seed: int,
) -> np.ndarray:
    clean = np.asarray(anchor, dtype=np.float64)
    sigma = float(clean.std(dtype=np.float64)) if sigma is None else float(sigma)
    if condition == "GAUSSIAN":
        keys = origins - 1 + np.asarray(groups, dtype=np.int64) * 10_000_000
        return clean + magnitude * sigma * _keyed_normal(keys, 1000003, seed)
    if condition == "BIAS":
        return clean + magnitude * sigma
    if condition == "LINEAR_DRIFT":
        drift = np.empty(len(clean), dtype=np.float64)
        for group in pd.unique(groups):
            index = np.flatnonzero(groups == group)
            drift[index] = np.linspace(-1.0, 1.0, len(index))
        return clean + magnitude * sigma * drift
    if condition == "RANDOM_WALK_DRIFT":
        keys = origins - 1 + np.asarray(groups, dtype=np.int64) * 10_000_000
        innovations = _keyed_normal(keys, 1000003, seed)
        walk = np.empty(len(clean), dtype=np.float64)
        for group in pd.unique(groups):
            index = np.flatnonzero(groups == group)
            part = np.cumsum(innovations[index], dtype=np.float64)
            part -= part.mean(dtype=np.float64)
            if part.std(dtype=np.float64) > 0:
                part /= part.std(dtype=np.float64)
            walk[index] = part
        return clean + magnitude * sigma * walk
    if condition == "QUANTIZATION":
        resolution = max(magnitude * sigma, np.finfo(np.float64).eps)
        return np.rint(clean / resolution) * resolution
    raise ValueError(condition)


def _level_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    clean_anchor: np.ndarray,
    prediction_anchor: np.ndarray,
) -> dict[str, float]:
    delta_error = np.asarray(y) - np.asarray(prediction)
    true_level = np.asarray(clean_anchor) + np.asarray(y)
    predicted_level = np.asarray(prediction_anchor) + np.asarray(prediction)
    error = true_level - predicted_level
    persistence_error = true_level - np.asarray(prediction_anchor)
    delta_tss = float(np.sum(np.square(y - np.mean(y)), dtype=np.float64))
    level_tss = float(np.sum(np.square(true_level - np.mean(true_level)), dtype=np.float64))
    mse = float(np.mean(np.square(error), dtype=np.float64))
    persistence_mse = float(np.mean(np.square(persistence_error), dtype=np.float64))
    return {
        "Delta_RMSE": float(np.sqrt(np.mean(np.square(delta_error), dtype=np.float64))),
        "Delta_R2": 1.0 - float(np.sum(np.square(delta_error), dtype=np.float64)) / delta_tss,
        "Level_RMSE": math.sqrt(mse),
        "Level_R2": 1.0 - float(np.sum(np.square(error), dtype=np.float64)) / level_tss,
        "persistence_skill": 1.0 - mse / persistence_mse,
    }


def _registered_tep_arrays(
    shared: Path,
    baseline_run: Path,
    information_set: str,
    availability: str,
) -> dict[str, Any]:
    head = "TEP_G_NOWCAST_H0__H0__W1"
    proxy = "proxy_excluded"
    views = [
        view
        for view in registered_views(shared, information_set)
        if view.head.head_id == head
        and view.availability_scenario == availability
        and view.proxy_policy == proxy
    ]
    if len(views) != 1:
        raise RuntimeError(f"E6 requires exactly one TEP {information_set}/{availability} view")
    view = views[0]
    c_result = json.loads(
        (
            baseline_run
            / f"results/DEVELOPMENT/C/{head}/{proxy}/RESULT.json"
        ).read_text(encoding="utf-8")
    )
    channels = tuple(str(value) for value in c_result["active_channels"])
    samples = load_native_samples(shared, view, "train")
    groups = [
        group.sort_values("origin").reset_index(drop=True)
        for _, group in samples.groupby("entity_id", sort=True)
        if len(group) >= 239
    ]
    if len(groups) < 4:
        raise RuntimeError("E6 TEP requires four development entities with >=239 rows")
    selected_groups = sorted(groups, key=lambda frame: str(frame["entity_id"].iloc[0]))[:4]
    frame = pd.concat(
        [group.iloc[:239] for group in selected_groups], ignore_index=True
    )
    group_labels = np.repeat(np.arange(4, dtype=np.int64), 239)
    accessor = BaseAccessor(shared, "tep", "train", [*channels, "xmeas_40"])
    # Each original process channel is represented at both frozen TEP history
    # scales.  The eight deterministic lags within a scale are fused before
    # the compact re-identification screen, so the physical history remains
    # {128,256} even though the downstream array runner consumes ready-made
    # scale summaries.
    raw_process_blocks: list[np.ndarray] = []
    raw_process_indices: list[np.ndarray] = []
    raw_process_channel_keys: list[int] = []
    origins = frame["origin"].to_numpy(dtype=np.int64)
    for channel_index, channel in enumerate(channels):
        for history in TEP_HISTORIES:
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
    raw_channel_sigmas = np.asarray(
        [
            np.std(
                np.concatenate(
                    [
                        raw_process_blocks[index].ravel()
                        for index, key in enumerate(raw_process_channel_keys)
                        if key == channel_index
                    ]
                ),
                dtype=np.float64,
            )
            for channel_index in range(len(channels))
        ],
        dtype=np.float64,
    )
    raw_channel_sigmas[raw_channel_sigmas == 0] = 1.0
    latest = frame["latest_available_target_index"].to_numpy(dtype=np.int64)
    if information_set == "dynamic":
        historical = accessor.gather(frame, ["xmeas_40"], latest)
        x = np.column_stack([process, historical])
    else:
        x = process
    anchor = accessor.block_means(frame, "xmeas_40", [(0, 1)])[:, 0]
    return {
        "x": x,
        "y": frame["y_true"].to_numpy(dtype=np.float64),
        "anchor": anchor,
        "origins": origins,
        "groups": group_labels,
        "latest_target": latest,
        "process_columns": process.shape[1],
        "channels": channels,
        "raw_process_blocks": raw_process_blocks,
        "raw_process_indices": raw_process_indices,
        "raw_process_channel_keys": raw_process_channel_keys,
        "raw_channel_sigmas": raw_channel_sigmas,
        "availability": availability,
        "a_lags": (1, 2, 4) if availability == "record_time" else (5, 6, 8),
        "base_origin_ids": frame["base_origin_id"].astype(str).to_numpy(),
        "support_hash": support_hash(frame["base_origin_id"].astype(str)),
    }


def _outer_train_process_sigmas(
    data: Mapping[str, Any], fit_mask: np.ndarray
) -> np.ndarray:
    """Estimate each physical channel scale from unique outer-train readings."""

    result = []
    groups = np.asarray(data["groups"], dtype=np.int64)
    for channel_key in range(len(data["channels"])):
        keyed_values: dict[int, float] = {}
        for block, indices, key in zip(
            data["raw_process_blocks"],
            data["raw_process_indices"],
            data["raw_process_channel_keys"],
            strict=True,
        ):
            if int(key) != channel_key:
                continue
            selected_indices = np.asarray(indices)[fit_mask]
            selected_values = np.asarray(block)[fit_mask]
            selected_groups = np.broadcast_to(groups[fit_mask, None], selected_indices.shape)
            composite = selected_indices + selected_groups * 10_000_000
            for timestamp, value in zip(
                composite.ravel(), selected_values.ravel(), strict=True
            ):
                keyed_values.setdefault(int(timestamp), float(value))
        sigma = float(np.std(tuple(keyed_values.values()), dtype=np.float64))
        result.append(sigma if sigma > 0.0 else 1.0)
    return np.asarray(result, dtype=np.float64)


def _outer_train_target_sigma(
    data: Mapping[str, Any], fit_mask: np.ndarray
) -> float:
    """Estimate the target-channel scale without reading the held-out entity."""

    groups = np.asarray(data["groups"], dtype=np.int64)
    keyed_values: dict[int, float] = {}
    anchor_indices = np.asarray(data["origins"], dtype=np.int64) - 1
    for timestamp, group, value in zip(
        anchor_indices[fit_mask],
        groups[fit_mask],
        np.asarray(data["anchor"])[fit_mask],
        strict=True,
    ):
        keyed_values.setdefault(int(timestamp + group * 10_000_000), float(value))
    if data["x"].shape[1] > data["process_columns"]:
        historical = np.asarray(data["x"][:, data["process_columns"]], dtype=np.float64)
        latest = np.asarray(data["latest_target"], dtype=np.int64)
        for timestamp, group, value in zip(
            latest[fit_mask], groups[fit_mask], historical[fit_mask], strict=True
        ):
            keyed_values.setdefault(int(timestamp + group * 10_000_000), float(value))
    sigma = float(np.std(tuple(keyed_values.values()), dtype=np.float64))
    return sigma if sigma > 0.0 else 1.0


def _perturb_raw_process_block(
    clean: np.ndarray,
    indices: np.ndarray,
    groups: np.ndarray,
    *,
    sigma: float,
    channel_key: int,
    condition: str,
    magnitude: float,
    seed: int,
) -> np.ndarray:
    values = np.asarray(clean, dtype=np.float64).copy()
    group_matrix = np.broadcast_to(np.asarray(groups)[:, None], indices.shape)
    composite = np.asarray(indices, dtype=np.int64) + group_matrix * 10_000_000
    if condition == "GAUSSIAN":
        values += magnitude * sigma * _keyed_normal(
            composite.ravel(), channel_key, seed
        ).reshape(values.shape)
    elif condition == "BIAS":
        values += magnitude * sigma
    elif condition == "LINEAR_DRIFT":
        drift = np.empty_like(values)
        for group in pd.unique(groups):
            mask = groups == group
            source = indices[mask]
            low = float(source.min())
            high = float(source.max())
            drift[mask] = 0.0 if high == low else 2.0 * (source - low) / (high - low) - 1.0
        values += magnitude * sigma * drift
    elif condition == "RANDOM_WALK_DRIFT":
        drift = np.empty_like(values)
        for group in pd.unique(groups):
            mask = groups == group
            source = indices[mask]
            unique = np.unique(source)
            composite_unique = unique + int(group) * 10_000_000
            walk = np.cumsum(
                _keyed_normal(composite_unique, channel_key, seed), dtype=np.float64
            )
            walk -= walk.mean(dtype=np.float64)
            if walk.std(dtype=np.float64) > 0:
                walk /= walk.std(dtype=np.float64)
            drift[mask] = walk[np.searchsorted(unique, source)]
        values += magnitude * sigma * drift
    elif condition == "QUANTIZATION":
        resolution = max(magnitude * sigma, np.finfo(np.float64).eps)
        values = np.rint(values / resolution) * resolution
    else:
        raise ValueError(condition)
    return values


def _perturb_tep_features(
    data: Mapping[str, Any],
    *,
    mode: str,
    condition: str,
    magnitude: float,
    seed: int,
    process_sigmas: np.ndarray | None = None,
    target_sigma: float | None = None,
) -> np.ndarray:
    if process_sigmas is None:
        process_sigmas = np.asarray(data["raw_channel_sigmas"], dtype=np.float64)
    summaries = []
    for block, indices, channel_key in zip(
        data["raw_process_blocks"],
        data["raw_process_indices"],
        data["raw_process_channel_keys"],
        strict=True,
    ):
        perturbed = _perturb_raw_process_block(
            block,
            indices,
            data["groups"],
            sigma=float(process_sigmas[channel_key]),
            channel_key=int(channel_key),
            condition=condition,
            magnitude=magnitude,
            seed=seed,
        )
        summaries.append(perturbed.mean(axis=1, dtype=np.float64))
    result = np.column_stack(summaries)
    if data["x"].shape[1] > data["process_columns"]:
        historical = data["x"][:, data["process_columns"] :]
        if mode == "realistic":
            historical_sigma = np.asarray(
                [
                    float(target_sigma)
                    if target_sigma is not None
                    else float(np.std(historical, dtype=np.float64))
                ],
                dtype=np.float64,
            )
            historical_sigma[historical_sigma == 0] = 1.0
            historical = _perturb_matrix(
                historical,
                historical_sigma,
                data["latest_target"] + 1,
                data["latest_target"],
                data["groups"],
                process_columns=historical.shape[1],
                mode="process_only",
                condition=condition,
                magnitude=magnitude,
                seed=seed,
            )
        result = np.column_stack([result, historical])
    return result


def _route_outer_stage(
    y_train: np.ndarray,
    parent_train: np.ndarray,
    parent_evaluation: np.ndarray,
    train_candidates: Mapping[str, np.ndarray],
    evaluation_candidates: Mapping[str, np.ndarray],
    folds: Sequence[np.ndarray],
    *,
    identity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select on inner OOF evidence, then refit the selected increment once."""

    routed_train, report, _ = _select_increment(
        y_train, parent_train, train_candidates, folds, identity=identity
    )
    routed_evaluation = np.asarray(parent_evaluation, dtype=np.float64).copy()
    if report["routing_status"] == "ACTIVE":
        candidate = str(report["tuned_nonzero_candidate"])
        alpha = float(candidate.rsplit("alpha=", maxsplit=1)[1])
        increment = _fit_ridge(
            np.asarray(train_candidates[candidate]),
            y_train - parent_train,
            np.asarray(evaluation_candidates[candidate]),
            alpha,
        )
        routed_evaluation += increment
    return routed_train, routed_evaluation, report


def _causal_group_lag_block(
    values: np.ndarray, lags: Sequence[int], groups: np.ndarray | None = None
) -> np.ndarray:
    """Build strict-past residual lags with a non-informative zero boundary."""

    raw = np.asarray(values, dtype=np.float64)
    labels = np.zeros(len(raw), dtype=np.int64) if groups is None else np.asarray(groups)
    if labels.shape != (len(raw),):
        raise ValueError("causal lag groups must align with values")
    result = np.zeros((len(raw), len(lags)), dtype=np.float64)
    for label in pd.unique(labels):
        index = np.flatnonzero(labels == label)
        part = raw[index]
        for column, lag in enumerate(lags):
            lag = int(lag)
            if lag <= 0:
                raise ValueError("causal residual lags must be positive")
            if lag < len(part):
                result[index[lag:], column] = part[:-lag]
    return result


def _run_outer_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    x_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    *,
    include_a: bool,
    a_lags: Sequence[int],
) -> dict[str, Any]:
    """Fully refit the compact PRISM route inside one held-out entity fold."""

    labels = pd.unique(train_groups)
    if len(labels) != 3:
        raise ValueError("E6 outer refit requires exactly three training entities")
    folds = [np.flatnonzero(train_groups == label) for label in labels]
    parent_train = _constant_oof(y_train, folds)
    parent_evaluation = np.full(len(y_evaluation), y_train.mean(dtype=np.float64))
    train_channel_predictions: list[np.ndarray] = []
    evaluation_channel_predictions: list[np.ndarray] = []
    active_channels: list[int] = []
    for channel in range(x_train.shape[1]):
        candidate = f"channel={channel}|history=1|alpha={TEP_RIDGES[0]}"
        train_features = np.column_stack(
            [x_train[:, channel], np.square(x_train[:, channel])]
        )
        evaluation_features = np.column_stack(
            [x_evaluation[:, channel], np.square(x_evaluation[:, channel])]
        )
        routed_train, routed_evaluation, report = _route_outer_stage(
            y_train,
            parent_train,
            parent_evaluation,
            {candidate: train_features},
            {candidate: evaluation_features},
            folds,
            identity=f"K_ZERO_CHANNEL_{channel}",
        )
        if report["routing_status"] == "ACTIVE":
            active_channels.append(channel)
            train_channel_predictions.append(routed_train)
            evaluation_channel_predictions.append(routed_evaluation)
    if train_channel_predictions:
        train_stack = np.column_stack(train_channel_predictions)
        evaluation_stack = np.column_stack(evaluation_channel_predictions)
        k_train = _ridge_oof(train_stack, y_train, folds, TEP_RIDGES[0])
        k_evaluation = _fit_ridge(
            train_stack, y_train, evaluation_stack, TEP_RIDGES[0]
        )
    else:
        k_train, k_evaluation = parent_train, parent_evaluation

    interaction_columns = min(8, x_train.shape[1])
    train_pairs = [
        x_train[:, left] * x_train[:, right]
        for left in range(interaction_columns)
        for right in range(left + 1, interaction_columns)
    ]
    evaluation_pairs = [
        x_evaluation[:, left] * x_evaluation[:, right]
        for left in range(interaction_columns)
        for right in range(left + 1, interaction_columns)
    ]
    alpha = TEP_RIDGES[0]
    c_train = {
        f"C_TRUE_PAIR_02|alpha={alpha}": (x_train[:, 0] * x_train[:, 2])[:, None],
        f"C_ALL_PAIRS|alpha={alpha}": np.column_stack(train_pairs),
    }
    c_evaluation = {
        f"C_TRUE_PAIR_02|alpha={alpha}": (
            x_evaluation[:, 0] * x_evaluation[:, 2]
        )[:, None],
        f"C_ALL_PAIRS|alpha={alpha}": np.column_stack(evaluation_pairs),
    }
    kc_train, kc_evaluation, c_report = _route_outer_stage(
        y_train,
        k_train,
        k_evaluation,
        c_train,
        c_evaluation,
        folds,
        identity="C_ZERO_IDENTITY",
    )

    train_square = np.square(kc_train) - np.mean(np.square(kc_train))
    train_cube = np.power(kc_train, 3) - np.mean(np.power(kc_train, 3))
    evaluation_square = np.square(kc_evaluation) - np.mean(np.square(kc_train))
    evaluation_cube = np.power(kc_evaluation, 3) - np.mean(np.power(kc_train, 3))
    w_train = {
        f"W_QUADRATIC|alpha={alpha}": train_square[:, None],
        f"W_SMOOTH_POLY|alpha={alpha}": np.column_stack(
            [train_square, train_cube, np.tanh(kc_train)]
        ),
    }
    w_evaluation = {
        f"W_QUADRATIC|alpha={alpha}": evaluation_square[:, None],
        f"W_SMOOTH_POLY|alpha={alpha}": np.column_stack(
            [evaluation_square, evaluation_cube, np.tanh(kc_evaluation)]
        ),
    }
    kcw_train, kcw_evaluation, w_report = _route_outer_stage(
        y_train,
        kc_train,
        kc_evaluation,
        w_train,
        w_evaluation,
        folds,
        identity="W_ZERO_IDENTITY",
    )

    if include_a:
        train_residual = y_train - kcw_train
        evaluation_residual = y_evaluation - kcw_evaluation
        lag_sets = (
            (int(a_lags[0]),),
            (int(a_lags[0]), int(a_lags[1])),
            tuple(int(value) for value in a_lags),
        )
        a_train = {}
        a_evaluation = {}
        for lags in lag_sets:
            candidate = f"A_LAGS_{'_'.join(map(str, lags))}|alpha={alpha}"
            a_train[candidate] = _causal_group_lag_block(
                train_residual, lags, train_groups
            )
            a_evaluation[candidate] = _causal_group_lag_block(
                evaluation_residual, lags
            )
        prediction_train, prediction, a_report = _route_outer_stage(
            y_train,
            kcw_train,
            kcw_evaluation,
            a_train,
            a_evaluation,
            folds,
            identity="A_ZERO_IDENTITY",
        )
    else:
        prediction_train, prediction = kcw_train, kcw_evaluation
        a_report = {
            "routing_status": "ZERO_IDENTITY",
            "absolute_oof_gain": 0.0,
            "relative_admission_margin": 0.0,
            "outer_fold_margins": [],
            "margin_signs": [],
            "final_selected_candidate": "A_NOT_AVAILABLE_INPUT_ONLY",
            "parent_candidate_id": "A_NOT_AVAILABLE_INPUT_ONLY",
        }
    reports = {"C": c_report, "W": w_report, "A": a_report}
    return {
        "prediction": prediction,
        "train_prediction_hash": hashlib.sha256(
            np.ascontiguousarray(prediction_train, dtype=np.float64).tobytes()
        ).hexdigest(),
        "active_channels": json.dumps(active_channels),
        "stage_vector": "".join(
            "1" if reports[stage]["routing_status"] == "ACTIVE" else "0"
            for stage in ("C", "W", "A")
        ),
        "reports": reports,
    }


def _n2_worker(spec: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        view_name,
        mode,
        data,
        seed,
        outer_group,
    ) = spec
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    rows = []
    fit_mask = np.asarray(data["groups"]) != outer_group
    evaluation_mask = ~fit_mask
    process_sigmas = np.asarray(data["outer_process_sigmas"][outer_group])
    target_sigma = float(data["outer_target_sigmas"][outer_group])
    for condition, magnitude in perturbation_conditions():
        perturbed = _perturb_tep_features(
            data,
            mode=mode,
            condition=condition,
            magnitude=magnitude,
            seed=seed,
            process_sigmas=process_sigmas,
            target_sigma=target_sigma,
        )
        result = _run_outer_pipeline(
            perturbed[fit_mask],
            np.asarray(data["y"])[fit_mask],
            np.asarray(data["groups"])[fit_mask],
            perturbed[evaluation_mask],
            np.asarray(data["y"])[evaluation_mask],
            include_a=("input_only" not in view_name),
            a_lags=data["a_lags"],
        )
        prediction = np.asarray(result["prediction"], dtype=np.float64)
        prediction_anchor = (
            _anchor_noise(
                np.asarray(data["anchor"])[evaluation_mask],
                np.asarray(data["origins"])[evaluation_mask],
                np.asarray(data["groups"])[evaluation_mask],
                sigma=target_sigma,
                condition=condition,
                magnitude=magnitude,
                seed=seed,
            )
            if mode == "realistic"
            else np.asarray(data["anchor"])[evaluation_mask]
        )
        reports = result["reports"]
        rows.append(
            {
                "phase": "N2_REIDENTIFICATION",
                "view": view_name,
                "mode": mode,
                "condition": condition,
                "magnitude": magnitude,
                "seed": seed,
                "outer_fold": int(outer_group),
                "C_active": reports["C"]["routing_status"] == "ACTIVE",
                "W_active": reports["W"]["routing_status"] == "ACTIVE",
                "A_active": reports["A"]["routing_status"] == "ACTIVE",
                "C_margin": reports["C"]["relative_admission_margin"],
                "W_margin": reports["W"]["relative_admission_margin"],
                "A_margin": reports["A"]["relative_admission_margin"],
                "C_absolute_oof_gain": reports["C"].get("absolute_oof_gain", 0.0),
                "W_absolute_oof_gain": reports["W"].get("absolute_oof_gain", 0.0),
                "A_absolute_oof_gain": reports["A"].get("absolute_oof_gain", 0.0),
                "C_outer_fold_margins": json.dumps(reports["C"].get("outer_fold_margins", [])),
                "W_outer_fold_margins": json.dumps(reports["W"].get("outer_fold_margins", [])),
                "A_outer_fold_margins": json.dumps(reports["A"].get("outer_fold_margins", [])),
                "C_candidate": reports["C"].get("final_selected_candidate"),
                "W_candidate": reports["W"].get("final_selected_candidate"),
                "A_candidate": reports["A"].get("final_selected_candidate"),
                "stage_vector": result["stage_vector"],
                "active_channels": result["active_channels"],
                **_level_metrics(
                    np.asarray(data["y"])[evaluation_mask],
                    prediction,
                    np.asarray(data["anchor"])[evaluation_mask],
                    prediction_anchor,
                ),
                "support_hash": support_hash(
                    np.asarray(data["base_origin_ids"])[evaluation_mask]
                ),
                "prediction_hash": hashlib.sha256(
                    np.ascontiguousarray(prediction, dtype=np.float64).tobytes()
                ).hexdigest(),
                "outer_train_process_sigma_hash": hashlib.sha256(
                    np.ascontiguousarray(process_sigmas, dtype=np.float64).tobytes()
                ).hexdigest(),
                "outer_train_target_sigma": target_sigma,
                "test_accessed": False,
                "ood_accessed": False,
            }
        )
    return rows


def _n1_design(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    return np.column_stack([raw, np.square(raw)])


def _frozen_ridge_contract(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> dict[str, np.ndarray | float]:
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale * scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    target_mean = float(y.mean(dtype=np.float64))
    coefficient = np.linalg.solve(
        standardized.T @ standardized + alpha * np.eye(x.shape[1]),
        standardized.T @ (y - target_mean),
    )
    return {"mean": mean, "scale": scale, "target_mean": target_mean, "coefficient": coefficient}


def _apply_contract(contract: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    return (
        (x - np.asarray(contract["mean"])) / np.asarray(contract["scale"])
    ) @ np.asarray(contract["coefficient"]) + float(contract["target_mean"])


def run_e6(
    output: Path,
    shared_root: Path,
    baseline_run: Path,
    *,
    workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = output / "E6_MEASUREMENT_ROBUSTNESS"
    destination.mkdir(parents=True, exist_ok=True)
    shared = shared_root / "tep_shared"
    definitions = (
        ("input_only/record_time", "input_only", "record_time", ("process_only",)),
        ("dynamic/record_time", "dynamic", "record_time", ("process_only", "realistic")),
        ("dynamic/analyzer_maturity_5_steps", "dynamic", "analyzer_maturity_5_steps", ("process_only", "realistic")),
    )
    arrays = {
        name: _registered_tep_arrays(shared, baseline_run, information, availability)
        for name, information, availability, _ in definitions
    }
    for data in arrays.values():
        data["outer_process_sigmas"] = {
            group: _outer_train_process_sigmas(
                data, np.asarray(data["groups"]) != group
            )
            for group in range(4)
        }
        data["outer_target_sigmas"] = {
            group: _outer_train_target_sigma(
                data, np.asarray(data["groups"]) != group
            )
            for group in range(4)
        }
    conditions = perturbation_conditions()

    n1_rows = []
    for view_name, _, _, modes in definitions:
        data = arrays[view_name]
        clean_design = _n1_design(data["x"])
        fit_mask = data["groups"] != 3
        evaluation_mask = data["groups"] == 3
        process_sigmas = np.asarray(data["outer_process_sigmas"][3])
        target_sigma = float(data["outer_target_sigmas"][3])
        contract = _frozen_ridge_contract(clean_design[fit_mask], data["y"][fit_mask])
        contract_hash = hashlib.sha256(
            np.ascontiguousarray(contract["coefficient"], dtype=np.float64).tobytes()
        ).hexdigest()
        for mode in modes:
            for condition, magnitude in conditions:
                if condition != "GAUSSIAN":
                    continue
                for seed in N1_SEEDS:
                    perturbed = _perturb_tep_features(
                        data,
                        mode=mode,
                        condition=condition,
                        magnitude=magnitude,
                        seed=seed,
                        process_sigmas=process_sigmas,
                        target_sigma=target_sigma,
                    )
                    prediction = _apply_contract(contract, _n1_design(perturbed)[evaluation_mask])
                    anchor = data["anchor"][evaluation_mask]
                    prediction_anchor = (
                        _anchor_noise(
                            anchor,
                            data["origins"][evaluation_mask],
                            data["groups"][evaluation_mask],
                            sigma=target_sigma,
                            condition=condition,
                            magnitude=magnitude,
                            seed=seed,
                        )
                        if mode == "realistic"
                        else anchor
                    )
                    n1_rows.append(
                        {
                            "phase": "N1_FROZEN_MODEL",
                            "view": view_name,
                            "mode": mode,
                            "condition": condition,
                            "magnitude": magnitude,
                            "seed": seed,
                            **_level_metrics(
                                data["y"][evaluation_mask], prediction, anchor, prediction_anchor
                            ),
                            "frozen_contract_hash": contract_hash,
                            "support_hash": support_hash(
                                np.asarray(data["base_origin_ids"])[evaluation_mask]
                            ),
                            "outer_train_process_sigma_hash": hashlib.sha256(
                                np.ascontiguousarray(process_sigmas, dtype=np.float64).tobytes()
                            ).hexdigest(),
                            "outer_train_target_sigma": target_sigma,
                            "test_accessed": False,
                            "ood_accessed": False,
                        }
                    )
    n1 = pd.DataFrame(n1_rows)
    n1.to_csv(destination / "n1_frozen_model.csv", index=False)

    specs = []
    for view_name, _, _, modes in definitions:
        data = arrays[view_name]
        for mode in modes:
            for seed in N2_SEEDS:
                for outer_group in range(4):
                    specs.append((view_name, mode, data, seed, outer_group))
    if workers <= 1:
        nested_n2_rows = [_n2_worker(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            nested_n2_rows = list(pool.map(_n2_worker, specs, chunksize=1))
    n2_rows = [row for rows in nested_n2_rows for row in rows]
    n2 = pd.DataFrame(n2_rows)
    expected_n1 = sum(len(modes) for _, _, _, modes in definitions) * len(GAUSSIAN_LEVELS) * len(N1_SEEDS)
    expected_n2 = sum(len(modes) for _, _, _, modes in definitions) * len(conditions) * len(N2_SEEDS) * 4
    if len(n1) != expected_n1 or len(n2) != expected_n2:
        raise RuntimeError(
            f"E6 row-count certificate failed: N1={len(n1)}/{expected_n1}, "
            f"N2={len(n2)}/{expected_n2}"
        )
    numeric = n2.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("E6 produced non-finite numeric evidence")
    clean = n2[(n2["condition"] == "GAUSSIAN") & (n2["magnitude"] == 0.0)]
    clean_groups = clean.groupby(["view", "mode", "outer_fold"], dropna=False)
    for column in ("prediction_hash", "stage_vector", "active_channels"):
        if not (clean_groups[column].nunique() == 1).all():
            raise RuntimeError(f"E6 alpha=0 seed invariance failed for {column}")
    n2.to_csv(destination / "n2_reidentification.csv", index=False)
    summary = n2.groupby(["view", "mode", "condition", "magnitude"], as_index=False).agg(
        P_C_ACTIVE=("C_active", "mean"),
        P_W_ACTIVE=("W_active", "mean"),
        P_A_ACTIVE=("A_active", "mean"),
        median_m_C=("C_margin", "median"),
        median_m_W=("W_margin", "median"),
        median_m_A=("A_margin", "median"),
        Delta_RMSE=("Delta_RMSE", "mean"),
        Level_R2=("Level_R2", "mean"),
        persistence_skill=("persistence_skill", "mean"),
    )
    summary.to_csv(destination / "n2_robustness_summary.csv", index=False)
    write_json(
        destination / "STATUS.json",
        {
            "status": "COMPLETED",
            "scope": "TEP H0/W1 development-only compact reidentification universe",
            "views": [item[0] for item in definitions],
            "histories": list(TEP_HISTORIES),
            "n1_seeds": list(N1_SEEDS),
            "n2_seeds": list(N2_SEEDS),
            "injection_point": "raw aligned strict-past lag measurement before {128,256} fusion, normalization, and PRISM-like stage routing",
            "same_realization_across_magnitudes": True,
            "formal_test_used_as_evaluation": False,
            "development_validation_policy": "N1 fourth entity held out; N2 four fully refit entity-held-out development outer folds with three entity inner folds",
            "noise_scale_policy": "per-outer-fold std from unique outer-train raw measurements only",
            "maturity_a_lags": {"record_time": [1, 2, 4], "analyzer_maturity_5_steps": [5, 6, 8]},
            "n1_rows": len(n1),
            "n2_rows": len(n2),
            "row_count_certificate": True,
            "finite_numeric_certificate": True,
            "alpha_zero_seed_invariance_certificate": True,
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    return n1, n2
