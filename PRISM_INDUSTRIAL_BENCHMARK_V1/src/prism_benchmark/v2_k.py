from __future__ import annotations

import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .c4_prism import channel_class
from .cpu_data import BaseAccessor, ViewSpec, inner_folds, input_columns, load_samples, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_basis import causal_geometric_intervals
from .v2_config import load_frozen_config
from .v2_numerics import deterministic_subsample
from .v2_selection import one_se_select, practical_activation
from .v2_urysohn import fit_contract, predict_contract
from .v2_views import development_input_views


FAMILY_ORDER = [
    "EXACT_ZERO", "LINEAR_DISTRIBUTED_LAG", "RANK_1_URYSOHN", "RANK_2_URYSOHN",
    "RANK_3_URYSOHN", "FULL_FINITE_URYSOHN",
]


def channel_profiles(view: ViewSpec, channel: str, config: dict[str, Any]) -> list[tuple[int, int]]:
    category = channel_class(view.head.dataset, channel)
    deltas = [int(value) for value in config["time_profile_grid"]["class_delta_ratio_steps"][category]]
    if view.head.h_steps > 0:
        histories = [int(value) * view.head.h_steps for value in config["time_profile_grid"]["positive_h_history_multipliers"]]
    else:
        histories = sorted({delta * int(value) for delta in deltas for value in config["time_profile_grid"]["zero_h_history_in_delta_multipliers"]})
    return sorted({(delta, history) for delta in deltas for history in histories if delta <= history}, key=lambda value: (value[1], -value[0]))


def profile_intervals(delta: int, history: int, m_tau: int) -> list[tuple[int, int]]:
    units = max(1, int(math.ceil(history / delta)))
    blocks = causal_geometric_intervals(units, min(m_tau, units), 2.0)
    intervals = [(near * delta, min(far * delta, history)) for near, far in blocks]
    intervals[-1] = (intervals[-1][0], history)
    return [(left, right) for left, right in intervals if right > left]


def profile_values(accessor: BaseAccessor, samples: pd.DataFrame, channel: str, profile: tuple[int, int], m_tau: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    intervals = profile_intervals(*profile, m_tau)
    return accessor.block_means(samples, channel, intervals), intervals


def _cap(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    return frame.iloc[deterministic_subsample(frame["base_origin_id"].astype(str), cap)]


def _als_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    als = config["numerical_certification"]["als"]
    return {
        "als_seeds": tuple(int(value) for value in config["randomness"]["als_initialization_seeds"]),
        "als_max_iterations": int(als["max_iterations"]),
        "als_tolerance": float(als["relative_objective_tolerance"]),
        "als_max_increases": int(als["max_consecutive_objective_increases"]),
        "als_divergence_factor": float(als["divergence_relative_objective_factor"]),
    }


def _candidate_valid(contract: dict[str, Any], rows: int, config: dict[str, Any]) -> bool:
    if contract["family"] == "EXACT_ZERO":
        return True
    if rows < int(config["row_caps"]["minimum_rows_for_any_fitted_candidate"]):
        return False
    if rows < int(config["row_caps"]["minimum_rows_per_free_parameter"]) * int(contract["parameter_count"]):
        return False
    certificate = contract.get("certificate", {})
    if certificate.get("status") == "NUMERICALLY_INVALID":
        return False
    direct = certificate.get("status")
    if direct in {"PASS", "PASS_WITH_WARNING", None}:
        return True
    return direct not in {"NUMERICALLY_INVALID", "SOLVER_FAILED_RETAINED"}


def evaluate_candidate(
    accessor: BaseAccessor,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    channel: str,
    profile: tuple[int, int],
    m_tau: int,
    family: str,
    m_x: int,
    lambdas: tuple[float, float, float],
    config: dict[str, Any],
) -> list[float]:
    losses = []
    for fit_index, validation_index in folds:
        fit = _cap(train.iloc[fit_index], int(config["row_caps"]["single_channel_k_fit"]))
        validation = _cap(train.iloc[validation_index], int(config["row_caps"]["validation_selection_per_fold"]))
        if family == "EXACT_ZERO":
            prediction = np.zeros(len(validation), dtype=np.float64)
        else:
            train_values, _ = profile_values(accessor, fit, channel, profile, m_tau)
            validation_values, _ = profile_values(accessor, validation, channel, profile, m_tau)
            contract = fit_contract(train_values, fit["y_true"].to_numpy(dtype=np.float64), family, m_x, lambdas, **_als_kwargs(config))
            if not _candidate_valid(contract, len(fit), config):
                losses.append(float("inf"))
                continue
            prediction = predict_contract(validation_values, contract)
        losses.append(mse(validation["y_true"].to_numpy(dtype=np.float64), prediction))
    return losses


def _selection(config: dict[str, Any], losses: dict[Any, list[float]], complexity, neutral=None):
    one_se = config["folds_and_selection"]["one_se"]
    return one_se_select(
        losses, complexity, neutral=neutral,
        minimum_usable_folds=int(config["folds_and_selection"]["minimum_usable_folds"]),
        rtol=float(one_se["floating_comparison_relative_tolerance"]),
        atol=float(one_se["floating_comparison_absolute_tolerance"]),
    )


def run_channel(shared: Path, project: Path, output: Path, view: ViewSpec, channel: str) -> dict[str, Any]:
    started = time.time()
    destination = output / "DEVELOPMENT" / "CHANNEL_AUDIT" / view.head.head_id / view.proxy_policy / channel
    destination.mkdir(parents=True, exist_ok=True)
    try:
        config = load_frozen_config(project)
        k_config = config["K_module"]
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        accessor = BaseAccessor(shared, view.head.dataset, "train", [channel])
        validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", [channel])
        folds = inner_folds(train, int(config["folds_and_selection"]["inner_folds"]))
        profiles = channel_profiles(view, channel, config)
        pilot = k_config["penalties"]["pilot"]
        pilot_lambdas = (float(pilot["lambda_0"]), float(pilot["lambda_tau"]), float(pilot["lambda_x"]))
        pilot_m_tau = 8
        profile_losses = {
            profile: evaluate_candidate(accessor, train, folds, channel, profile, pilot_m_tau, "LINEAR_DISTRIBUTED_LAG", 1, pilot_lambdas, config)
            for profile in profiles
        }
        profile_selection = _selection(config, profile_losses, lambda value: (value[1], -value[0]))
        ordered_profiles = sorted(profile_selection.acceptable, key=lambda value: (value[1], -value[0]))
        retained_profiles = ordered_profiles[: int(config["time_profile_grid"]["maximum_profiles_retained_per_channel"])]
        selected_profile = tuple(profile_selection.selected)
        zero_losses = []
        for _, validation_index in folds:
            subset = _cap(train.iloc[validation_index], int(config["row_caps"]["validation_selection_per_fold"]))
            y = subset["y_true"].to_numpy(dtype=np.float64)
            zero_losses.append(float(np.mean(y * y, dtype=np.float64)))
        activation_selection = _selection(
            config, {"EXACT_ZERO": zero_losses, "LINEAR_DISTRIBUTED_LAG": profile_losses[selected_profile]},
            lambda value: (0 if value == "EXACT_ZERO" else 1,), neutral="EXACT_ZERO",
        )
        activation = practical_activation(
            zero_losses, profile_losses[selected_profile],
            minimum_relative_improvement=float(k_config["active_channel_gate"]["relative_validation_mse_improvement_min"]),
            minimum_positive_fraction=float(k_config["active_channel_gate"]["positive_fold_fraction_min"]),
            denominator_floor_scale=float(config["folds_and_selection"]["relative_mse_denominator_floor_scale"]),
        )
        if activation_selection.selected == "EXACT_ZERO" or not activation["pass"]:
            selected_family, selected_m_tau, selected_m_x, selected_lambdas = "EXACT_ZERO", pilot_m_tau, 1, pilot_lambdas
            resolution_selection = ladder_selection = None
            penalty_audit = {}
        else:
            resolution_losses = {
                (int(m_tau), int(m_x)): evaluate_candidate(accessor, train, folds, channel, selected_profile, int(m_tau), "FULL_FINITE_URYSOHN", int(m_x), pilot_lambdas, config)
                for m_tau in k_config["lag_basis"]["candidate_m_tau"] for m_x in k_config["amplitude_basis"]["candidate_m_x"]
            }
            resolution_selection = _selection(config, resolution_losses, lambda value: (value[0], value[1]))
            selected_m_tau, selected_m_x = map(int, resolution_selection.selected)
            ladder_losses = {"EXACT_ZERO": zero_losses}
            for family in FAMILY_ORDER[1:]:
                ladder_losses[family] = evaluate_candidate(accessor, train, folds, channel, selected_profile, selected_m_tau, family, selected_m_x, pilot_lambdas, config)
            initial_ladder = _selection(config, ladder_losses, lambda value: (FAMILY_ORDER.index(value),), neutral="EXACT_ZERO")
            eligible_losses = dict(ladder_losses)
            low_rank = {"RANK_1_URYSOHN", "RANK_2_URYSOHN", "RANK_3_URYSOHN"}
            if low_rank.intersection(initial_ladder.acceptable):
                eligible_losses.pop("FULL_FINITE_URYSOHN", None)
            ladder_selection = _selection(config, eligible_losses, lambda value: (FAMILY_ORDER.index(value),), neutral="EXACT_ZERO")
            selected_family = str(ladder_selection.selected)
            if selected_family != "EXACT_ZERO":
                ladder_activation = practical_activation(
                    zero_losses, eligible_losses[selected_family],
                    minimum_relative_improvement=float(config["folds_and_selection"]["practical_activation_relative_mse_min"]["K"]),
                    minimum_positive_fraction=float(config["folds_and_selection"]["required_positive_fold_fraction"]),
                    denominator_floor_scale=float(config["folds_and_selection"]["relative_mse_denominator_floor_scale"]),
                )
                if not ladder_activation["pass"]:
                    selected_family = "EXACT_ZERO"
            lambdas = list(pilot_lambdas)
            penalty_audit = {}
            if selected_family != "EXACT_ZERO":
                for position, name in enumerate(("lambda_0", "lambda_tau", "lambda_x")):
                    scan = {}
                    for value in k_config["penalties"][name]:
                        candidate = list(lambdas)
                        candidate[position] = float(value)
                        scan[float(value)] = evaluate_candidate(accessor, train, folds, channel, selected_profile, selected_m_tau, selected_family, selected_m_x, tuple(candidate), config)
                    chosen = _selection(config, scan, lambda value: (-float(value),))
                    lambdas[position] = float(chosen.selected)
                    penalty_audit[name] = {"one_se": chosen.to_json(), "fold_losses": {str(key): value for key, value in scan.items()}}
            selected_lambdas = tuple(lambdas)
        final_train = _cap(train, int(config["row_caps"]["single_channel_k_fit"]))
        train_values, intervals = profile_values(accessor, final_train, channel, selected_profile, selected_m_tau)
        validation_values, _ = profile_values(validation_accessor, validation, channel, selected_profile, selected_m_tau)
        contract = fit_contract(train_values, final_train["y_true"].to_numpy(dtype=np.float64), selected_family, selected_m_x, selected_lambdas, **_als_kwargs(config))
        prediction = predict_contract(validation_values, contract)
        frame = validation[["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]].copy()
        frame["y_pred"] = prediction
        frame["model"] = f"PRISM_V2_K_{channel}"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        result = {
            "status": "PASS", "stage": "V2_CHANNEL_E_K", "dataset": view.head.dataset,
            "task": view.head.task_id, "target_head": view.head.head_id, "proxy_policy": view.proxy_policy,
            "channel": channel, "channel_class": channel_class(view.head.dataset, channel),
            "selected_profile": list(selected_profile), "retained_profiles": [list(value) for value in retained_profiles],
            "selected_intervals": [list(value) for value in intervals], "selected_family": selected_family,
            "selected_m_tau": selected_m_tau, "selected_m_x": selected_m_x, "selected_lambdas": list(selected_lambdas),
            "active": selected_family != "EXACT_ZERO" and _candidate_valid(contract, len(final_train), config),
            "profile_selection": profile_selection.to_json(), "profile_fold_losses": {str(key): value for key, value in profile_losses.items()},
            "activation_selection": activation_selection.to_json(), "activation": activation,
            "resolution_selection": None if resolution_selection is None else resolution_selection.to_json(),
            "ladder_selection": None if ladder_selection is None else ladder_selection.to_json(), "penalty_audit": penalty_audit,
            "contract": contract, "prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": sha256_file(prediction_path),
            "test_accessed": False, "elapsed_seconds": time.time() - started,
            **regression_metrics(frame["y_true"].to_numpy(), prediction),
        }
    except Exception as error:
        result = {"status": "SOLVER_FAILED_RETAINED", "stage": "V2_CHANNEL_E_K", "target_head": view.head.head_id,
                  "proxy_policy": view.proxy_policy, "channel": channel, "test_accessed": False,
                  "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                  "elapsed_seconds": time.time() - started}
    write_json(destination / "RESULT.json", result)
    return result


def run_v2_channels(shared: Path, project: Path, output: Path, n_jobs: int) -> dict[str, Any]:
    jobs = [(view, channel) for view in development_input_views(shared) for channel in input_columns(shared, view.head.task_id, view.proxy_policy)]
    results, pending = [], []
    for view, channel in jobs:
        path = output / "DEVELOPMENT" / "CHANNEL_AUDIT" / view.head.head_id / view.proxy_policy / channel / "RESULT.json"
        if path.is_file():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior.get("status") in {"PASS", "SOLVER_FAILED_RETAINED"}:
                results.append(prior)
                continue
        pending.append((view, channel))
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(run_channel, shared, project, output, view, channel) for view, channel in pending]
        for future in as_completed(futures):
            results.append(future.result())
    summary = {"status": "PASS" if all(item["status"] == "PASS" for item in results) else "COMPLETED_WITH_RETAINED_FAILURES",
               "stage": "V2_CHANNEL_E_K", "jobs": len(results), "pass": sum(item["status"] == "PASS" for item in results),
               "active": sum(bool(item.get("active")) for item in results), "test_accessed": False}
    write_json(output / "DEVELOPMENT" / "CHANNEL_AUDIT" / "SUMMARY.json", summary)
    return summary

