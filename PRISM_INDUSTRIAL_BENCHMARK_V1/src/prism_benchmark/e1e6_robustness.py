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
from .e1e6_phase_b import _fit_ridge, run_synthetic_variant
from .e1e6_revalidation import support_hash, write_json
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
        source_indices = origins - 1 if column < process_columns else latest_target
        key = column if column < process_columns else 1000003
        sigma = float(sigmas[column])
        if condition == "GAUSSIAN":
            values[:, column] += magnitude * sigma * _keyed_normal(source_indices, key, seed)
        elif condition == "BIAS":
            values[:, column] += magnitude * sigma
        elif condition == "LINEAR_DRIFT":
            direction = -1.0 if int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:2], 16) % 2 else 1.0
            values[:, column] += direction * magnitude * sigma * np.linspace(-1.0, 1.0, len(values))
        elif condition == "RANDOM_WALK_DRIFT":
            innovations = _keyed_normal(source_indices, key, seed)
            walk = np.cumsum(innovations, dtype=np.float64)
            walk -= walk.mean(dtype=np.float64)
            walk_scale = walk.std(dtype=np.float64)
            if walk_scale > 0:
                walk /= walk_scale
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
    *,
    condition: str,
    magnitude: float,
    seed: int,
) -> np.ndarray:
    clean = np.asarray(anchor, dtype=np.float64)
    sigma = float(clean.std(dtype=np.float64))
    if condition == "GAUSSIAN":
        return clean + magnitude * sigma * _keyed_normal(origins - 1, 1000003, seed)
    if condition == "BIAS":
        return clean + magnitude * sigma
    if condition == "LINEAR_DRIFT":
        return clean + magnitude * sigma * np.linspace(-1.0, 1.0, len(clean))
    if condition == "RANDOM_WALK_DRIFT":
        walk = np.cumsum(_keyed_normal(origins - 1, 1000003, seed), dtype=np.float64)
        walk -= walk.mean(dtype=np.float64)
        if walk.std(dtype=np.float64) > 0:
            walk /= walk.std(dtype=np.float64)
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
    if not groups:
        raise RuntimeError("E6 TEP requires a contiguous development entity with >=239 rows")
    frame = max(groups, key=len).iloc[:239].reset_index(drop=True)
    accessor = BaseAccessor(shared, "tep", "train", [*channels, "xmeas_40"])
    # Each original process channel is represented at both frozen TEP history
    # scales.  The eight deterministic lags within a scale are fused before
    # the compact re-identification screen, so the physical history remains
    # {128,256} even though the downstream array runner consumes ready-made
    # scale summaries.
    process = np.column_stack(
        [
            accessor.input_regular_lags(frame, [channel], 1, history, 8).mean(
                axis=1, dtype=np.float64
            )
            for channel in channels
            for history in TEP_HISTORIES
        ]
    )
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
        "origins": frame["origin"].to_numpy(dtype=np.int64),
        "latest_target": latest,
        "process_columns": process.shape[1],
        "channels": channels,
        "support_hash": support_hash(frame["base_origin_id"].astype(str)),
    }


def _n2_worker(spec: tuple[Any, ...]) -> dict[str, Any]:
    (
        view_name,
        mode,
        x,
        y,
        anchor,
        origins,
        latest,
        process_columns,
        support,
        condition,
        magnitude,
        seed,
    ) = spec
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sigmas = np.asarray(x, dtype=np.float64).std(axis=0, dtype=np.float64)
    sigmas[sigmas == 0] = 1.0
    perturbed = _perturb_matrix(
        x,
        sigmas,
        origins,
        latest,
        process_columns=process_columns,
        mode=mode,
        condition=condition,
        magnitude=magnitude,
        seed=seed,
    )
    result = run_synthetic_variant(
        seed,
        "REAL_TEP",
        histories=(1,),
        ridges=TEP_RIDGES,
        include_a=("input_only" not in view_name),
        x_override=perturbed,
        y_override=y,
        return_prediction=True,
        precomputed_k=True,
    )
    prediction = np.asarray(result.pop("prediction"), dtype=np.float64)
    prediction_anchor = (
        _anchor_noise(anchor, origins, condition=condition, magnitude=magnitude, seed=seed)
        if mode == "realistic"
        else anchor
    )
    return {
        "phase": "N2_REIDENTIFICATION",
        "view": view_name,
        "mode": mode,
        "condition": condition,
        "magnitude": magnitude,
        "seed": seed,
        "C_active": result["C_active"],
        "W_active": result["W_active"],
        "A_active": result["A_active"],
        "C_margin": result["C_margin"],
        "W_margin": result["W_margin"],
        "A_margin": result["A_margin"],
        "stage_vector": result["stage_vector"],
        **_level_metrics(y, prediction, anchor, prediction_anchor),
        "support_hash": support,
        "prediction_hash": result["prediction_hash"],
        "test_accessed": False,
        "ood_accessed": False,
    }


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
    conditions = perturbation_conditions()

    n1_rows = []
    for view_name, _, _, modes in definitions:
        data = arrays[view_name]
        split = 179
        clean_design = _n1_design(data["x"])
        contract = _frozen_ridge_contract(clean_design[:split], data["y"][:split])
        contract_hash = hashlib.sha256(
            np.ascontiguousarray(contract["coefficient"], dtype=np.float64).tobytes()
        ).hexdigest()
        sigmas = data["x"][:split].std(axis=0, dtype=np.float64)
        sigmas[sigmas == 0] = 1.0
        for mode in modes:
            for condition, magnitude in conditions:
                if condition != "GAUSSIAN":
                    continue
                for seed in N1_SEEDS:
                    perturbed = _perturb_matrix(
                        data["x"],
                        sigmas,
                        data["origins"],
                        data["latest_target"],
                        process_columns=data["process_columns"],
                        mode=mode,
                        condition=condition,
                        magnitude=magnitude,
                        seed=seed,
                    )
                    prediction = _apply_contract(contract, _n1_design(perturbed)[split:])
                    anchor = data["anchor"][split:]
                    prediction_anchor = (
                        _anchor_noise(
                            anchor,
                            data["origins"][split:],
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
                                data["y"][split:], prediction, anchor, prediction_anchor
                            ),
                            "frozen_contract_hash": contract_hash,
                            "support_hash": data["support_hash"],
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
            for condition, magnitude in conditions:
                for seed in N2_SEEDS:
                    specs.append(
                        (
                            view_name,
                            mode,
                            data["x"],
                            data["y"],
                            data["anchor"],
                            data["origins"],
                            data["latest_target"],
                            data["process_columns"],
                            data["support_hash"],
                            condition,
                            magnitude,
                            seed,
                        )
                    )
    if workers <= 1:
        n2_rows = [_n2_worker(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            n2_rows = list(pool.map(_n2_worker, specs, chunksize=1))
    n2 = pd.DataFrame(n2_rows)
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
            "injection_point": "registered strict-past {128,256} scale summaries before normalization and PRISM-like stage routing",
            "same_realization_across_magnitudes": True,
            "formal_test_used_as_evaluation": False,
            "development_validation_policy": "N1 held-out tail; N2 four disjoint development OOF evidence blocks",
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    return n1, n2
