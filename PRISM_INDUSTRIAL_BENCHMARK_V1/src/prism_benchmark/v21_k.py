from __future__ import annotations

import itertools
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, ViewSpec, inner_folds, input_columns, load_samples, sha256_file
from .cpu_selection import regression_metrics
from .stage0 import write_json
from .v2_k import (
    CHANNEL_SAMPLE_COLUMNS,
    FAMILY_ORDER,
    _als_kwargs,
    _candidate_valid,
    _cap,
    evaluate_candidate,
    profile_values,
    channel_profiles,
)
from .v2_selection import one_se_select
from .v2_urysohn import fit_contract, predict_contract
from .v21_config import load_v21_and_v2_config
from .v21_selection import guarded_local_one_se_select
from .v21_views import sru_input_views


def _one_se(
    losses: dict[Any, list[float]],
    complexity,
    minimum_usable_folds: int,
):
    return one_se_select(
        losses,
        complexity,
        minimum_usable_folds=minimum_usable_folds,
    )


def _weak_refit_candidates(
    selected: tuple[float, float, float],
    v2_config: dict[str, Any],
) -> list[tuple[float, float, float]]:
    grids = [
        sorted(float(value) for value in v2_config["K_module"]["penalties"][name])
        for name in ("lambda_0", "lambda_tau", "lambda_x")
    ]
    weaker = [
        tuple(float(value) for value in candidate)
        for candidate in itertools.product(*grids)
        if all(value <= limit for value, limit in zip(candidate, selected, strict=True))
    ]
    weaker.append(tuple(float(value) for value in selected))
    return sorted(
        set(weaker),
        key=lambda value: (sum(value), np.prod(np.asarray(value) + 1e-30), value),
    )


def run_k_channel(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    channel: str,
) -> dict[str, Any]:
    started = time.time()
    destination = output / "DEVELOPMENT" / "K" / view.head.head_id / view.proxy_policy / channel
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v21, v2 = load_v21_and_v2_config(project)
        minimum_folds = int(v21["selection"]["minimum_usable_folds"])
        train = load_samples(shared, view, "train", columns=CHANNEL_SAMPLE_COLUMNS)
        validation = load_samples(shared, view, "validation", columns=CHANNEL_SAMPLE_COLUMNS)
        accessor = BaseAccessor(shared, view.head.dataset, "validation", [channel])
        folds = inner_folds(train, int(v21["selection"]["inner_folds"]))
        profiles = channel_profiles(view, channel, v2)
        pilot = v2["K_module"]["penalties"]["pilot"]
        pilot_lambdas = (
            float(pilot["lambda_0"]),
            float(pilot["lambda_tau"]),
            float(pilot["lambda_x"]),
        )
        pilot_m_tau = 8
        profile_losses = {
            profile: evaluate_candidate(
                accessor,
                train,
                folds,
                channel,
                profile,
                pilot_m_tau,
                "LINEAR_DISTRIBUTED_LAG",
                1,
                pilot_lambdas,
                v2,
            )
            for profile in profiles
        }
        profile_selection = _one_se(
            profile_losses,
            lambda value: (value[1], -value[0]),
            minimum_folds,
        )
        selected_profile = tuple(int(value) for value in profile_selection.selected)
        retained_profiles = sorted(
            profile_selection.acceptable,
            key=lambda value: (value[1], -value[0]),
        )[: int(v2["time_profile_grid"]["maximum_profiles_retained_per_channel"])]
        zero_losses = []
        for _, evaluation_index in folds:
            subset = _cap(
                train.iloc[evaluation_index],
                int(v2["row_caps"]["validation_selection_per_fold"]),
            )
            target = subset["y_true"].to_numpy(dtype=np.float64)
            zero_losses.append(float(np.mean(target * target, dtype=np.float64)))
        activation_losses = {
            "EXACT_ZERO": zero_losses,
            "LINEAR_DISTRIBUTED_LAG": profile_losses[selected_profile],
        }
        activation = guarded_local_one_se_select(
            activation_losses,
            lambda value: (0 if value == "EXACT_ZERO" else 1,),
            neutral="EXACT_ZERO",
            minimum_relative_improvement=float(v21["selection"]["minimum_relative_improvement"]["K"]),
            minimum_positive_fraction=float(v21["selection"]["minimum_positive_fold_fraction"]),
            minimum_usable_folds=minimum_folds,
        )
        resolution_selection = None
        ladder_selection = None
        penalty_audit: dict[str, Any] = {}
        if activation.final_selected_candidate == "EXACT_ZERO":
            selected_family = "EXACT_ZERO"
            selected_m_tau = pilot_m_tau
            selected_m_x = 1
            selected_lambdas = pilot_lambdas
        else:
            resolution_losses = {
                (int(m_tau), int(m_x)): evaluate_candidate(
                    accessor,
                    train,
                    folds,
                    channel,
                    selected_profile,
                    int(m_tau),
                    "FULL_FINITE_URYSOHN",
                    int(m_x),
                    pilot_lambdas,
                    v2,
                )
                for m_tau in v21["K_C"]["m_tau"]
                for m_x in v21["K_C"]["m_x"]
            }
            resolution_selection = _one_se(
                resolution_losses,
                lambda value: (value[0] * value[1], value[0], value[1]),
                minimum_folds,
            )
            selected_m_tau, selected_m_x = map(int, resolution_selection.selected)
            ladder_losses = {"EXACT_ZERO": zero_losses}
            for family in FAMILY_ORDER[1:]:
                ladder_losses[family] = evaluate_candidate(
                    accessor,
                    train,
                    folds,
                    channel,
                    selected_profile,
                    selected_m_tau,
                    family,
                    selected_m_x,
                    pilot_lambdas,
                    v2,
                )
            ladder_selection = guarded_local_one_se_select(
                ladder_losses,
                lambda value: (FAMILY_ORDER.index(value),),
                neutral="EXACT_ZERO",
                minimum_relative_improvement=float(v21["selection"]["minimum_relative_improvement"]["K"]),
                minimum_positive_fraction=float(v21["selection"]["minimum_positive_fold_fraction"]),
                minimum_usable_folds=minimum_folds,
            )
            selected_family = str(ladder_selection.final_selected_candidate)
            lambdas = list(pilot_lambdas)
            if selected_family != "EXACT_ZERO":
                for position, name in enumerate(("lambda_0", "lambda_tau", "lambda_x")):
                    scan: dict[float, list[float]] = {}
                    for value in v2["K_module"]["penalties"][name]:
                        candidate = list(lambdas)
                        candidate[position] = float(value)
                        scan[float(value)] = evaluate_candidate(
                            accessor,
                            train,
                            folds,
                            channel,
                            selected_profile,
                            selected_m_tau,
                            selected_family,
                            selected_m_x,
                            tuple(candidate),
                            v2,
                        )
                    selection = _one_se(scan, lambda value: (-float(value),), minimum_folds)
                    lambdas[position] = float(selection.selected)
                    penalty_audit[name] = {
                        "one_se": selection.to_json(),
                        "fold_losses": {str(key): value for key, value in scan.items()},
                    }
            selected_lambdas = tuple(lambdas)
        final_train = _cap(train, int(v2["row_caps"]["single_channel_k_fit"]))
        train_values, intervals = profile_values(
            accessor,
            final_train,
            channel,
            selected_profile,
            selected_m_tau,
        )
        validation_values, _ = profile_values(
            accessor,
            validation,
            channel,
            selected_profile,
            selected_m_tau,
        )
        selection_contract = fit_contract(
            train_values,
            final_train["y_true"].to_numpy(dtype=np.float64),
            selected_family,
            selected_m_x,
            selected_lambdas,
            **_als_kwargs(v2),
        )
        refit_lambdas = selected_lambdas
        refit_contract = selection_contract
        if selected_family != "EXACT_ZERO" and v21["K_C"]["fixed_support_weak_ridge_oof_refit"]:
            for candidate_lambdas in _weak_refit_candidates(selected_lambdas, v2):
                candidate_contract = fit_contract(
                    train_values,
                    final_train["y_true"].to_numpy(dtype=np.float64),
                    selected_family,
                    selected_m_x,
                    candidate_lambdas,
                    **_als_kwargs(v2),
                )
                if _candidate_valid(candidate_contract, len(final_train), v2):
                    refit_lambdas = candidate_lambdas
                    refit_contract = candidate_contract
                    break
        prediction = predict_contract(validation_values, refit_contract)
        frame = validation[
            ["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]
        ].copy()
        frame["y_pred"] = prediction
        frame["model"] = f"PRISM_V2_1_K_{channel}"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        active = selected_family != "EXACT_ZERO" and _candidate_valid(
            refit_contract, len(final_train), v2
        )
        result = {
            "status": "PASS",
            "stage": "E2_K",
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "channel": channel,
            "selected_profile": list(selected_profile),
            "retained_profiles": [list(value) for value in retained_profiles],
            "selected_intervals": [list(value) for value in intervals],
            "selected_family": selected_family,
            "selected_m_tau": selected_m_tau,
            "selected_m_x": selected_m_x,
            "selection_lambdas": list(selected_lambdas),
            "selected_lambdas": list(refit_lambdas),
            "fixed_support_weak_refit": True,
            "active": active,
            "profile_selection": profile_selection.to_json(),
            "profile_fold_losses": {str(key): value for key, value in profile_losses.items()},
            "activation_selection": activation.to_json(),
            "resolution_selection": None if resolution_selection is None else resolution_selection.to_json(),
            "ladder_selection": None if ladder_selection is None else ladder_selection.to_json(),
            "penalty_audit": penalty_audit,
            "selection_contract": selection_contract,
            "contract": refit_contract,
            "final_selected_candidate": selected_family,
            "final_selected_fold_losses": list(
                activation.final_selected_fold_losses
                if ladder_selection is None
                else ladder_selection.final_selected_fold_losses
            ),
            "final_selected_prediction_path": str(prediction_path.relative_to(output)),
            "final_selected_contract": refit_contract,
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(frame["y_true"].to_numpy(dtype=np.float64), prediction),
        }
    except Exception as error:
        result = {
            "status": "SOLVER_FAILED_RETAINED",
            "stage": "E2_K",
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "channel": channel,
            "test_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    write_json(destination / "RESULT.json", result)
    return result


def load_active_channels(output: Path, view: ViewSpec) -> list[dict[str, Any]]:
    root = output / "DEVELOPMENT" / "K" / view.head.head_id / view.proxy_policy
    results = []
    for path in sorted(root.glob("*/RESULT.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "PASS" and value.get("active"):
            results.append(value)
    return results


def run_e2_k(shared: Path, project: Path, output: Path) -> dict[str, Any]:
    results = []
    for view in sru_input_views(shared):
        for channel in input_columns(shared, view.head.task_id, view.proxy_policy):
            results.append(run_k_channel(shared, project, output, view, channel))
    summary = {
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": "E2_K",
        "jobs": len(results),
        "pass": sum(item["status"] == "PASS" for item in results),
        "active": sum(bool(item.get("active")) for item in results),
        "test_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / "K" / "SUMMARY.json", summary)
    return summary
