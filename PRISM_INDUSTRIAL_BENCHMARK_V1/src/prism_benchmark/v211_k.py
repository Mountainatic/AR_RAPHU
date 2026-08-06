from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    inner_folds,
    input_columns,
    load_samples,
    sha256_file,
)
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_k import (
    CHANNEL_SAMPLE_COLUMNS,
    FAMILY_ORDER,
    _als_kwargs,
    _candidate_valid,
    _cap,
    channel_profiles,
    evaluate_candidate,
    profile_values,
)
from .v2_selection import one_se_select
from .v2_runtime import run_parallel
from .v2_urysohn import fit_contract, predict_contract
from .v21_selection import guarded_local_one_se_select
from .v21_views import sru_input_views
from .v211_config import load_v211_configs
from .v211_selection import profile_one_se_regret_guard, select_smallest_stable


EXACT_ZERO = "EXACT_ZERO"


def _profile_complexity(profile: tuple[int, int]) -> tuple[int, int]:
    return int(profile[1]), -int(profile[0])


def _structural_complexity(candidate: Any) -> tuple[Any, ...]:
    if candidate == EXACT_ZERO:
        return (0,)
    profile, m_tau, family, m_x = candidate
    width = int(m_tau) * (1 if family == "LINEAR_DISTRIBUTED_LAG" else int(m_x))
    return (
        1,
        FAMILY_ORDER.index(str(family)),
        width,
        *_profile_complexity(tuple(profile)),
        int(m_tau),
        int(m_x),
    )


def _smoothness_selection(
    *,
    accessor: BaseAccessor,
    train: Any,
    folds: list[tuple[np.ndarray, np.ndarray]],
    channel: str,
    profile: tuple[int, int],
    m_tau: int,
    family: str,
    m_x: int,
    v2: dict[str, Any],
    minimum_folds: int,
) -> tuple[float, float, dict[str, Any]]:
    lambda_0 = 0.0
    lambda_tau = float(v2["K_module"]["penalties"]["pilot"]["lambda_tau"])
    lambda_x = float(v2["K_module"]["penalties"]["pilot"]["lambda_x"])
    audit: dict[str, Any] = {}
    for name in ("lambda_tau", "lambda_x"):
        if name == "lambda_x" and family == "LINEAR_DISTRIBUTED_LAG":
            lambda_x = 0.0
            audit[name] = {"status": "NOT_APPLICABLE_LINEAR_FAMILY", "selected": 0.0}
            continue
        scan: dict[float, list[float]] = {}
        for value in v2["K_module"]["penalties"][name]:
            candidate_tau = float(value) if name == "lambda_tau" else lambda_tau
            candidate_x = float(value) if name == "lambda_x" else lambda_x
            scan[float(value)] = evaluate_candidate(
                accessor,
                train,
                folds,
                channel,
                profile,
                m_tau,
                family,
                m_x,
                (lambda_0, candidate_tau, candidate_x),
                v2,
            )
        selection = one_se_select(
            scan,
            lambda value: (-float(value),),
            minimum_usable_folds=minimum_folds,
        )
        if name == "lambda_tau":
            lambda_tau = float(selection.selected)
        else:
            lambda_x = float(selection.selected)
        audit[name] = {
            "selection": selection.to_json(),
            "fold_losses": {str(key): value for key, value in scan.items()},
        }
    return lambda_tau, lambda_x, audit


def run_k_channel(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    channel: str,
) -> dict[str, Any]:
    started = time.time()
    destination = (
        output / "DEVELOPMENT" / "K" / view.head.head_id / view.proxy_policy / channel
    )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v211, v21, v2 = load_v211_configs(project)
        minimum_folds = int(v21["selection"]["minimum_usable_folds"])
        train = load_samples(shared, view, "train", columns=CHANNEL_SAMPLE_COLUMNS)
        validation = load_samples(
            shared, view, "validation", columns=CHANNEL_SAMPLE_COLUMNS
        )
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
        profile_rule = v211["K"]["profile_selection"]
        profile_selection = profile_one_se_regret_guard(
            profile_losses,
            _profile_complexity,
            maximum_relative_regret=float(
                profile_rule["maximum_relative_regret_vs_best"]
            ),
            maximum_retained_profiles=int(profile_rule["maximum_retained_profiles"]),
            minimum_usable_folds=minimum_folds,
        )
        retained_profiles = [tuple(value) for value in profile_selection.retained_profiles]

        zero_losses = []
        for _, evaluation_index in folds:
            subset = _cap(
                train.iloc[evaluation_index],
                int(v2["row_caps"]["validation_selection_per_fold"]),
            )
            target = subset["y_true"].to_numpy(dtype=np.float64)
            zero_losses.append(float(np.mean(target * target, dtype=np.float64)))

        linear_activation_losses: dict[Any, list[float]] = {EXACT_ZERO: zero_losses}
        for profile in retained_profiles:
            linear_activation_losses[(profile, "LINEAR_DISTRIBUTED_LAG")] = (
                profile_losses[profile]
            )
        activation = guarded_local_one_se_select(
            linear_activation_losses,
            lambda value: (0,)
            if value == EXACT_ZERO
            else (1, *_profile_complexity(value[0])),
            neutral=EXACT_ZERO,
            minimum_relative_improvement=float(
                v21["selection"]["minimum_relative_improvement"]["K"]
            ),
            minimum_positive_fraction=float(
                v21["selection"]["minimum_positive_fold_fraction"]
            ),
            minimum_usable_folds=minimum_folds,
        )

        structural_selection = None
        structural_losses: dict[Any, list[float]] = {EXACT_ZERO: zero_losses}
        if activation.final_selected_candidate == EXACT_ZERO:
            selected_profile = tuple(profile_selection.best_profile)
            selected_m_tau = pilot_m_tau
            selected_m_x = 1
            selected_family = EXACT_ZERO
            lambda_tau = float(pilot["lambda_tau"])
            lambda_x = float(pilot["lambda_x"])
            smoothness_audit: dict[str, Any] = {}
        else:
            for profile in retained_profiles:
                for m_tau in v21["K_C"]["m_tau"]:
                    structural_losses[(profile, int(m_tau), "LINEAR_DISTRIBUTED_LAG", 1)] = evaluate_candidate(
                        accessor,
                        train,
                        folds,
                        channel,
                        profile,
                        int(m_tau),
                        "LINEAR_DISTRIBUTED_LAG",
                        1,
                        pilot_lambdas,
                        v2,
                    )
                    for m_x in v21["K_C"]["m_x"]:
                        for family in FAMILY_ORDER[2:]:
                            structural_losses[(profile, int(m_tau), family, int(m_x))] = evaluate_candidate(
                                accessor,
                                train,
                                folds,
                                channel,
                                profile,
                                int(m_tau),
                                family,
                                int(m_x),
                                pilot_lambdas,
                                v2,
                            )
            structural_selection = guarded_local_one_se_select(
                structural_losses,
                _structural_complexity,
                neutral=EXACT_ZERO,
                minimum_relative_improvement=float(
                    v21["selection"]["minimum_relative_improvement"]["K"]
                ),
                minimum_positive_fraction=float(
                    v21["selection"]["minimum_positive_fold_fraction"]
                ),
                minimum_usable_folds=minimum_folds,
            )
            selected = structural_selection.final_selected_candidate
            if selected == EXACT_ZERO:
                selected_profile = tuple(profile_selection.best_profile)
                selected_m_tau = pilot_m_tau
                selected_m_x = 1
                selected_family = EXACT_ZERO
                lambda_tau = float(pilot["lambda_tau"])
                lambda_x = float(pilot["lambda_x"])
                smoothness_audit = {}
            else:
                selected_profile, selected_m_tau, selected_family, selected_m_x = selected
                selected_profile = tuple(selected_profile)
                selected_m_tau = int(selected_m_tau)
                selected_m_x = int(selected_m_x)
                selected_family = str(selected_family)
                lambda_tau, lambda_x, smoothness_audit = _smoothness_selection(
                    accessor=accessor,
                    train=train,
                    folds=folds,
                    channel=channel,
                    profile=selected_profile,
                    m_tau=selected_m_tau,
                    family=selected_family,
                    m_x=selected_m_x,
                    v2=v2,
                    minimum_folds=minimum_folds,
                )

        final_train = _cap(train, int(v2["row_caps"]["single_channel_k_fit"]))
        train_values, intervals = profile_values(
            accessor, final_train, channel, selected_profile, selected_m_tau
        )
        validation_values, _ = profile_values(
            accessor, validation, channel, selected_profile, selected_m_tau
        )
        ridge_audit: list[dict[str, Any]] = []
        if selected_family == EXACT_ZERO:
            selected_lambdas = pilot_lambdas
            refit_contract = fit_contract(
                train_values,
                final_train["y_true"].to_numpy(dtype=np.float64),
                EXACT_ZERO,
                selected_m_x,
                selected_lambdas,
                **_als_kwargs(v2),
            )
        else:
            ridge_values = sorted(
                {float(value) for value in v2["K_module"]["penalties"]["lambda_0"]}
            )

            def fit_ridge(lambda_0: float) -> dict[str, Any]:
                return fit_contract(
                    train_values,
                    final_train["y_true"].to_numpy(dtype=np.float64),
                    selected_family,
                    selected_m_x,
                    (float(lambda_0), lambda_tau, lambda_x),
                    **_als_kwargs(v2),
                )

            lambda_0, refit_contract, ridge_audit = select_smallest_stable(
                ridge_values,
                fit_ridge,
                valid_candidate=lambda contract: _candidate_valid(
                    dict(contract), len(final_train), v2
                ),
            )
            selected_lambdas = (float(lambda_0), lambda_tau, lambda_x)

        prediction = predict_contract(validation_values, dict(refit_contract))
        frame = validation[
            ["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]
        ].copy()
        frame["y_pred"] = prediction
        frame["model"] = f"PRISM_V2_1_1_K_{channel}"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        active = selected_family != EXACT_ZERO and _candidate_valid(
            dict(refit_contract), len(final_train), v2
        )
        if selected_family == EXACT_ZERO:
            final_fold_losses = zero_losses
        else:
            final_fold_losses = evaluate_candidate(
                accessor,
                train,
                folds,
                channel,
                selected_profile,
                selected_m_tau,
                selected_family,
                selected_m_x,
                tuple(float(value) for value in selected_lambdas),
                v2,
            )
        result = {
            "status": "PASS",
            "stage": "E2R_K",
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
            "selected_lambdas": list(selected_lambdas),
            "ridge_semantics": "NUMERICAL_STABILITY_ONLY",
            "active": active,
            "profile_selection": profile_selection.to_json(),
            "profile_fold_losses": {
                str(key): value for key, value in profile_losses.items()
            },
            "linear_activation_selection": activation.to_json(),
            "structural_selection": None
            if structural_selection is None
            else structural_selection.to_json(),
            "structural_fold_losses": {
                str(key): value for key, value in structural_losses.items()
            },
            "smoothness_audit": smoothness_audit,
            "minimal_stabilizing_ridge_audit": ridge_audit,
            "contract": refit_contract,
            "final_selected_candidate": str(
                (selected_profile, selected_m_tau, selected_family, selected_m_x)
            ),
            "final_selected_fold_losses": list(final_fold_losses),
            "final_selected_prediction_path": str(
                prediction_path.relative_to(output)
            ),
            "final_selected_contract": refit_contract,
            "final_prediction_loss": mse(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
        }
    except Exception as error:
        result = {
            "status": "SOLVER_FAILED_RETAINED",
            "stage": "E2R_K",
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


def run_e2r_k(shared: Path, project: Path, output: Path) -> dict[str, Any]:
    jobs = []
    for view in sru_input_views(shared):
        for channel in input_columns(shared, view.head.task_id, view.proxy_policy):
            jobs.append((shared, project, output, view, channel))
    results = run_parallel(
        run_k_channel,
        jobs,
        int(os.environ.get("PRISM_V211_WORKERS", "8")),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "4")),
        label="PRISM_V211_E2R_K",
    )
    summary = {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in results)
        else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": "E2R_K",
        "jobs": len(results),
        "pass": sum(item["status"] == "PASS" for item in results),
        "active": sum(bool(item.get("active")) for item in results),
        "test_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / "K" / "SUMMARY.json", summary)
    return summary
