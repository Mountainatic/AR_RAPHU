from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, ViewSpec, inner_folds, load_samples, realized_state_profiles, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_config import load_frozen_config
from .v2_numerics import deterministic_subsample, solve_certified
from .v2_selection import one_se_select, practical_activation
from .v2_views import state_development_views


def _subsample(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    return frame.iloc[deterministic_subsample(frame["base_origin_id"].astype(str), cap)]


def _quadratic(features: np.ndarray, maximum_linear: int) -> np.ndarray:
    selected = features[:, : min(features.shape[1], maximum_linear)]
    return np.concatenate([features, selected * selected], axis=1)


def _feasible_state_rows(samples:pd.DataFrame,accessor:BaseAccessor,profile:tuple[int,int])->np.ndarray:
    delta,history=profile;count=max(1,history//max(delta,1));oldest=samples["latest_available_target_index"].to_numpy(dtype=np.int64)-(count-1)*max(delta,1)
    entities=samples["entity_id"].astype(str).to_numpy();minimum=np.asarray([accessor.entities[entity][0].min() for entity in entities],dtype=np.int64)
    return oldest>=minimum


def _standardized_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evaluation: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = x_train.mean(axis=0, dtype=np.float64)
    scale = x_train.std(axis=0, dtype=np.float64)
    scale[scale * scale < 1e-12] = 1.0
    train = (x_train - mean) / scale
    evaluation = (x_evaluation - mean) / scale
    y_mean = float(np.mean(y_train, dtype=np.float64))
    coefficient, certificate = solve_certified(train, y_train - y_mean, float(alpha))
    prediction = evaluation @ coefficient + y_mean
    return prediction, {
        "mean": mean.tolist(), "scale": scale.tolist(), "coefficient": coefficient.tolist(),
        "intercept": y_mean, "alpha": alpha, "numerical_certificate": certificate.to_json(),
    }


def predict_state(features: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    if contract["family"] == "EXACT_ZERO":
        return np.zeros(len(features), dtype=np.float64)
    x = features
    if contract["family"] == "NAR_TARGET_QUADRATIC":
        x = _quadratic(x, int(contract["maximum_linear_state_features_before_expansion"]))
    mean = np.asarray(contract["mean"], dtype=np.float64)
    scale = np.asarray(contract["scale"], dtype=np.float64)
    coefficient = np.asarray(contract["coefficient"], dtype=np.float64)
    return ((x - mean) / scale) @ coefficient + float(contract["intercept"])


def run_state_view(shared: Path, project: Path, output: Path, view: ViewSpec) -> dict[str, Any]:
    started = time.time()
    destination = output / "DEVELOPMENT" / "STATE_ONLY" / view.head.head_id / view.availability_scenario
    destination.mkdir(parents=True, exist_ok=True)
    try:
        config = load_frozen_config(project)
        module = config["A_module"]
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        folds = inner_folds(train, int(config["folds_and_selection"]["inner_folds"]))
        accessor = BaseAccessor(shared, view.head.dataset, "train", [view.head.target])
        profiles = realized_state_profiles(view.head)
        alpha_grid = [float(value) for value in module["ridge_alpha_grid"]]
        losses: dict[Any, list[float]] = {"EXACT_ZERO": []}
        candidates: list[Any] = ["EXACT_ZERO"]
        for family in ("AR_LINEAR", "NAR_TARGET_QUADRATIC"):
            candidates.extend((family, profile, alpha) for profile in profiles for alpha in alpha_grid)
        losses.update({candidate: [] for candidate in candidates[1:]})
        for fit_index, evaluation_index in folds:
            fit = _subsample(train.iloc[fit_index], int(config["row_caps"]["state_fit"]))
            evaluation = _subsample(train.iloc[evaluation_index], int(config["row_caps"]["validation_selection_per_fold"]))
            y_fit = fit["y_true"].to_numpy(dtype=np.float64)
            y_evaluation = evaluation["y_true"].to_numpy(dtype=np.float64)
            losses["EXACT_ZERO"].append(float(np.mean(y_evaluation * y_evaluation, dtype=np.float64)))
            for profile in profiles:
                try:
                    feasible_fit=_feasible_state_rows(fit,accessor,profile);feasible_evaluation=_feasible_state_rows(evaluation,accessor,profile)
                    if not bool(np.all(feasible_evaluation)) or int(feasible_fit.sum())<200:raise ValueError("state profile unavailable on immutable rows")
                    profile_fit=fit.iloc[np.flatnonzero(feasible_fit)]
                    x_fit = accessor.target_state(profile_fit, view.head.target, *profile)
                    x_evaluation = accessor.target_state(evaluation, view.head.target, *profile)
                except ValueError:
                    for family in ("AR_LINEAR", "NAR_TARGET_QUADRATIC"):
                        for alpha in alpha_grid:
                            losses[(family, profile, alpha)].append(float("inf"))
                    continue
                for family in ("AR_LINEAR", "NAR_TARGET_QUADRATIC"):
                    current_fit = x_fit if family == "AR_LINEAR" else _quadratic(x_fit, int(module["target_only_quadratic"]["maximum_linear_state_features_before_expansion"]))
                    current_evaluation = x_evaluation if family == "AR_LINEAR" else _quadratic(x_evaluation, int(module["target_only_quadratic"]["maximum_linear_state_features_before_expansion"]))
                    for alpha in alpha_grid:
                        prediction, _ = _standardized_fit(current_fit, profile_fit["y_true"].to_numpy(dtype=np.float64), current_evaluation, alpha)
                        losses[(family, profile, alpha)].append(mse(y_evaluation, prediction))
        def complexity(candidate: Any) -> tuple[Any, ...]:
            if candidate == "EXACT_ZERO":
                return (0,)
            family, profile, alpha = candidate
            return (1 if family == "AR_LINEAR" else 2, -alpha, profile[1], -profile[0])
        selection = one_se_select(
            losses, complexity, neutral="EXACT_ZERO",
            minimum_usable_folds=int(config["folds_and_selection"]["minimum_usable_folds"]),
            rtol=float(config["folds_and_selection"]["one_se"]["floating_comparison_relative_tolerance"]),
            atol=float(config["folds_and_selection"]["one_se"]["floating_comparison_absolute_tolerance"]),
        )
        selected = selection.selected
        activation = {"pass": False, "reason": "NEUTRAL_SELECTED"}
        if selected != "EXACT_ZERO":
            activation = practical_activation(
                losses["EXACT_ZERO"], losses[selected],
                minimum_relative_improvement=float(config["folds_and_selection"]["practical_activation_relative_mse_min"]["A"]),
                minimum_positive_fraction=float(config["folds_and_selection"]["required_positive_fold_fraction"]),
                denominator_floor_scale=float(config["folds_and_selection"]["relative_mse_denominator_floor_scale"]),
            )
            if not activation["pass"]:
                selected = "EXACT_ZERO"
        final_train = _subsample(train, int(config["row_caps"]["state_fit"]))
        final_accessor = BaseAccessor(shared, view.head.dataset, "validation", [view.head.target])
        if selected == "EXACT_ZERO":
            contract = {"family": "EXACT_ZERO", "parameter_count": 0, "numerical_certificate": {"status": "EXACT_ZERO"}}
            prediction = np.zeros(len(validation), dtype=np.float64)
        else:
            family, profile, alpha = selected
            feasible_final=_feasible_state_rows(final_train,accessor,profile);final_train=final_train.iloc[np.flatnonzero(feasible_final)]
            x_train = accessor.target_state(final_train, view.head.target, *profile)
            if family == "NAR_TARGET_QUADRATIC":
                maximum = int(module["target_only_quadratic"]["maximum_linear_state_features_before_expansion"])
                x_train = _quadratic(x_train, maximum)
            _, fitted = _standardized_fit(x_train, final_train["y_true"].to_numpy(dtype=np.float64), x_train[:1], alpha)
            contract = {"family": family, "profile": list(profile), "parameter_count": len(fitted["coefficient"]) + 1, **fitted}
            if family == "NAR_TARGET_QUADRATIC":
                contract["maximum_linear_state_features_before_expansion"] = maximum
            chunks=[]
            for start in range(0,len(validation),100000):
                current=validation.iloc[start:start+100000];features=final_accessor.target_state(current,view.head.target,*profile);chunks.append(predict_state(features,contract))
            prediction=np.concatenate(chunks) if chunks else np.empty(0,dtype=np.float64)
        frame = validation[["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]].copy()
        frame["y_pred"] = prediction
        frame["model"] = "PRISM_V2_STATE_ONLY"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        result = {
            "status": "PASS", "stage": "V1_STATE_ONLY", "target_head": view.head.head_id,
            "availability_scenario": view.availability_scenario, "selected_candidate": str(selected),
            "contract": contract, "one_se": selection.to_json(), "activation": activation,
            "fold_losses": {str(key): value for key, value in losses.items()},
            "prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": sha256_file(prediction_path),
            "test_accessed": False, "elapsed_seconds": time.time() - started,
            **regression_metrics(frame["y_true"].to_numpy(), prediction),
        }
    except Exception as error:
        result = {"status": "SOLVER_FAILED_RETAINED", "stage": "V1_STATE_ONLY", "target_head": view.head.head_id,
                  "availability_scenario": view.availability_scenario, "test_accessed": False,
                  "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                  "elapsed_seconds": time.time() - started}
    write_json(destination / "RESULT.json", result)
    return result


def run_v1_state(shared: Path, project: Path, output: Path, n_jobs: int) -> dict[str, Any]:
    views = state_development_views(shared)
    results = []
    pending = []
    for view in views:
        path = output / "DEVELOPMENT" / "STATE_ONLY" / view.head.head_id / view.availability_scenario / "RESULT.json"
        if path.is_file():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior.get("status") == "PASS":
                results.append(prior)
                continue
        pending.append(view)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(run_state_view, shared, project, output, view) for view in pending]
        for future in as_completed(futures):
            results.append(future.result())
    summary = {"status": "PASS" if all(item["status"] == "PASS" for item in results) else "COMPLETED_WITH_RETAINED_FAILURES",
               "stage": "V1_STATE_ONLY", "views": len(results), "pass": sum(item["status"] == "PASS" for item in results),
               "test_accessed": False}
    write_json(output / "DEVELOPMENT" / "STATE_ONLY" / "SUMMARY.json", summary)
    return summary
