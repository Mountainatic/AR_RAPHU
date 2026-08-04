from __future__ import annotations

import itertools
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from .cpu_data import BaseAccessor, ViewSpec, inner_folds, load_samples, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_basis import natural_cubic_columns
from .v2_config import load_frozen_config
from .v2_k import _als_kwargs, _cap, profile_values
from .v2_numerics import residualize, solve_certified
from .v2_selection import one_se_select, practical_activation
from .v2_runtime import run_parallel
from .v2_urysohn import basis_from_metadata, fit_contract, predict_contract
from .v2_views import development_input_views


def _load_active_channels(output: Path, view: ViewSpec) -> list[dict[str, Any]]:
    root = output / "DEVELOPMENT" / "CHANNEL_AUDIT" / view.head.head_id / view.proxy_policy
    results = []
    for path in sorted(root.glob("*/RESULT.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "PASS" and value.get("active"):
            results.append(value)
    return results


def _standardize(train: np.ndarray, evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    scale = train.std(axis=0, dtype=np.float64)
    scale[scale * scale < 1e-12] = 1.0
    return (train - mean) / scale, (evaluation - mean) / scale, mean, scale


def _ridge_fit(train: np.ndarray, target: np.ndarray, evaluation: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    x, z, mean, scale = _standardize(train, evaluation)
    target_mean = float(np.mean(target, dtype=np.float64))
    coefficient, certificate = solve_certified(x, target - target_mean, alpha)
    return z @ coefficient + target_mean, {
        "mean": mean.tolist(), "scale": scale.tolist(), "coefficient": coefficient.tolist(),
        "intercept": target_mean, "alpha": alpha, "numerical_certificate": certificate.to_json(),
    }


def _ridge_predict(features: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    return ((features - np.asarray(contract["mean"])) / np.asarray(contract["scale"])) @ np.asarray(contract["coefficient"]) + float(contract["intercept"])


def _qr_columns(design: np.ndarray, maximum: int, floor: float) -> np.ndarray:
    centered = design - design.mean(axis=0, dtype=np.float64)
    _, r, piv = linalg.qr(centered, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    rank = int(np.sum(diagonal > diagonal.max(initial=0.0) * floor))
    return np.asarray(piv[: max(1, min(maximum, rank))], dtype=np.int64)


def fit_physical_features(
    shared: Path,
    view: ViewSpec,
    fit_samples: pd.DataFrame,
    evaluation_samples: pd.DataFrame,
    active: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    fit_split: str,
    evaluation_split: str,
) -> dict[str, Any]:
    if not active:
        return {"channels": [], "compressed_train": np.empty((len(fit_samples), 0)), "compressed_evaluation": np.empty((len(evaluation_samples), 0)),
                "joint_train": np.empty((len(fit_samples), 0)), "joint_evaluation": np.empty((len(evaluation_samples), 0)), "channel_contracts": []}
    channels = [item["channel"] for item in active]
    evaluation_accessor = BaseAccessor(shared, view.head.dataset, evaluation_split, channels)
    # evaluation_split is always a superset of fit_split (train <= validation <= test),
    # so one accessor serves both frames without duplicating base data.
    fit_accessor = evaluation_accessor
    compressed_train, compressed_evaluation = [], []
    joint_train, joint_evaluation, contracts = [], [], []
    per_channel_max = int(config["C_module"]["joint_basis"]["per_channel_max_columns"])
    qr_floor = float(config["C_module"]["joint_basis"]["per_channel_qr_relative_diagonal_floor"])
    for item in active:
        channel = item["channel"]
        profile = tuple(item["selected_profile"])
        m_tau = int(item["selected_m_tau"])
        m_x = int(item["selected_m_x"])
        lambdas = tuple(float(value) for value in item["selected_lambdas"])
        train_values, intervals = profile_values(fit_accessor, fit_samples, channel, profile, m_tau)
        evaluation_values, _ = profile_values(evaluation_accessor, evaluation_samples, channel, profile, m_tau)
        contract = fit_contract(train_values, fit_samples["y_true"].to_numpy(dtype=np.float64), item["selected_family"], m_x, lambdas, **_als_kwargs(config))
        train_contribution = predict_contract(train_values, contract)
        evaluation_contribution = predict_contract(evaluation_values, contract)
        compressed_train.append(train_contribution)
        compressed_evaluation.append(evaluation_contribution)
        basis = basis_from_metadata(contract["basis"])
        train_raw = basis.transform(train_values).reshape(len(train_values), -1)
        evaluation_raw = basis.transform(evaluation_values).reshape(len(evaluation_values), -1)
        selected = _qr_columns(train_raw, per_channel_max, qr_floor)
        joint_train.append(train_raw[:, selected])
        joint_evaluation.append(evaluation_raw[:, selected])
        contracts.append({"channel": channel, "profile": list(profile), "m_tau": m_tau, "m_x": m_x, "lambdas": list(lambdas),
                          "intervals": [list(value) for value in intervals], "k_contract": contract, "joint_columns": selected.tolist()})
    compressed_fit = np.column_stack(compressed_train)
    compressed_eval = np.column_stack(compressed_evaluation)
    joint_fit = np.concatenate(joint_train, axis=1)
    joint_eval = np.concatenate(joint_evaluation, axis=1)
    global_max = int(config["C_module"]["joint_basis"]["global_max_columns"])
    global_columns = np.arange(joint_fit.shape[1], dtype=np.int64)
    if joint_fit.shape[1] > global_max:
        global_columns = _qr_columns(joint_fit, global_max, qr_floor)
        joint_fit = joint_fit[:, global_columns]
        joint_eval = joint_eval[:, global_columns]
    return {"channels": channels, "compressed_train": compressed_fit, "compressed_evaluation": compressed_eval,
            "joint_train": joint_fit, "joint_evaluation": joint_eval, "channel_contracts": contracts,
            "global_joint_columns": global_columns.tolist()}


def _pair_columns(train_scalar: np.ndarray, evaluation_scalar: np.ndarray, pair: tuple[int, int], knot_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    left, right = pair
    train_axes, evaluation_axes, knots_meta = [], [], []
    for index in (left, right):
        values = train_scalar[:, index]
        standardized = (values - values.mean()) / (values.std() or 1.0)
        evaluation_standardized = (evaluation_scalar[:, index] - values.mean()) / (values.std() or 1.0)
        knots = np.unique(np.quantile(standardized, np.arange(1, knot_count + 1) / (knot_count + 1)))
        train_axes.append(natural_cubic_columns(standardized, knots))
        evaluation_axes.append(natural_cubic_columns(evaluation_standardized, knots))
        knots_meta.append(knots.tolist())
    raw_train = np.einsum("ti,tj->tij", train_axes[0], train_axes[1]).reshape(len(train_scalar), -1)
    raw_evaluation = np.einsum("ti,tj->tij", evaluation_axes[0], evaluation_axes[1]).reshape(len(evaluation_scalar), -1)
    against_train = np.column_stack([np.ones(len(train_scalar)), train_axes[0], train_axes[1]])
    residual_train, certificate = residualize(raw_train, against_train, relative_tolerance=1e-8)
    coefficient = np.linalg.lstsq(against_train, raw_train, rcond=1e-10)[0]
    against_evaluation = np.column_stack([np.ones(len(evaluation_scalar)), evaluation_axes[0], evaluation_axes[1]])
    residual_evaluation = raw_evaluation - against_evaluation @ coefficient
    return residual_train, residual_evaluation, {"pair": list(pair), "knots": knots_meta, "orthogonality": certificate, "projection": coefficient.tolist()}


def run_c_view(shared: Path, project: Path, output: Path, view: ViewSpec) -> dict[str, Any]:
    started = time.time()
    destination = output / "DEVELOPMENT" / "JOINT_BASIS" / view.head.head_id / view.proxy_policy
    destination.mkdir(parents=True, exist_ok=True)
    try:
        config = load_frozen_config(project)
        active = _load_active_channels(output, view)
        maximum = int(config["K_module"]["active_channel_gate"]["maximum_active_channels"])
        active = sorted(active, key=lambda item: (-float(item["activation"].get("mean_relative_improvement", 0.0)), item["channel"]))[:maximum]
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        folds = inner_folds(train, int(config["folds_and_selection"]["inner_folds"]))
        alpha_grid = [float(value) for value in config["C_module"]["joint_basis"]["ridge_alpha_grid"]]
        family_losses = {"ADDITIVE_COMPRESSED": [], "ADDITIVE_JOINT_BASIS": []}
        fold_cache = []
        pair_pool: set[tuple[int, int]] = set()
        for fit_index, evaluation_index in folds:
            fit = _cap(train.iloc[fit_index], int(config["row_caps"]["joint_physical_fit"]))
            evaluation = _cap(train.iloc[evaluation_index], int(config["row_caps"]["validation_selection_per_fold"]))
            features = fit_physical_features(shared, view, fit, evaluation, active, config, fit_split="train", evaluation_split="train")
            y_fit = fit["y_true"].to_numpy(dtype=np.float64)
            y_evaluation = evaluation["y_true"].to_numpy(dtype=np.float64)
            path = {}
            for family, key in (("ADDITIVE_COMPRESSED", "compressed"), ("ADDITIVE_JOINT_BASIS", "joint")):
                x_fit, x_evaluation = features[f"{key}_train"], features[f"{key}_evaluation"]
                if x_fit.shape[1] == 0:
                    path[family] = [float(np.mean(y_evaluation * y_evaluation))] * len(alpha_grid)
                else:
                    path[family] = [mse(y_evaluation, _ridge_fit(x_fit, y_fit, x_evaluation, alpha)[0]) for alpha in alpha_grid]
            fold_cache.append((fit, evaluation, features, y_fit, y_evaluation, path))
            if features["compressed_train"].shape[1] >= 2 and len(fit) >= int(config["C_module"]["pairwise_anova"]["minimum_fit_rows"]):
                baseline = _ridge_fit(features["compressed_train"], y_fit, features["compressed_train"], alpha_grid[0])[0]
                residual = y_fit - baseline
                standardized = (features["compressed_train"] - features["compressed_train"].mean(axis=0)) / np.where(features["compressed_train"].std(axis=0) > 0, features["compressed_train"].std(axis=0), 1.0)
                scores = []
                for pair in itertools.combinations(range(standardized.shape[1]), 2):
                    product = standardized[:, pair[0]] * standardized[:, pair[1]]
                    correlation = abs(float(np.corrcoef(product, residual)[0, 1])) if np.std(product) > 0 and np.std(residual) > 0 else 0.0
                    if correlation >= float(config["C_module"]["pairwise_anova"]["screening_absolute_train_only_residual_correlation_min"]):
                        scores.append((correlation, pair))
                pair_pool.update(pair for _, pair in sorted(scores, reverse=True)[: int(config["C_module"]["pairwise_anova"]["candidate_pair_pool_max"])] )
        candidate_losses = {}
        for family in family_losses:
            for alpha_index, alpha in enumerate(alpha_grid):
                candidate_losses[(family, alpha)] = [cache[-1][family][alpha_index] for cache in fold_cache]
        compressed_selection = one_se_select(
            {key: value for key, value in candidate_losses.items() if key[0] == "ADDITIVE_COMPRESSED"},
            lambda value: (-value[1],), minimum_usable_folds=3,
        )
        compressed_candidate = compressed_selection.selected
        joint_selection = one_se_select(
            {key: value for key, value in candidate_losses.items() if key[0] == "ADDITIVE_JOINT_BASIS"},
            lambda value: (-value[1],), minimum_usable_folds=3,
        )
        c_selection = one_se_select(candidate_losses, lambda value: (0 if value[0] == "ADDITIVE_COMPRESSED" else 1, -value[1]), neutral=compressed_candidate, minimum_usable_folds=3)
        selected_family, selected_alpha = c_selection.selected
        joint_best = min((key for key in candidate_losses if key[0] == "ADDITIVE_JOINT_BASIS"), key=lambda key: np.mean(candidate_losses[key]))
        joint_activation = practical_activation(
            candidate_losses[compressed_candidate], candidate_losses[joint_best],
            minimum_relative_improvement=float(config["C_module"]["joint_basis"]["activation_relative_mse_min"]),
            minimum_positive_fraction=float(config["C_module"]["joint_basis"]["positive_fold_fraction_min"]),
        )
        if selected_family == "ADDITIVE_JOINT_BASIS" and not joint_activation["pass"]:
            selected_family, selected_alpha = compressed_candidate
        selected_pairs: list[tuple[int, int]] = []
        pair_audit = []
        base_key = "joint" if selected_family == "ADDITIVE_JOINT_BASIS" else "compressed"
        current_losses = candidate_losses[(selected_family, selected_alpha)]
        remaining = sorted(pair_pool)
        for _ in range(int(config["C_module"]["pairwise_anova"]["selected_pair_max"])):
            pair_losses = {}
            for pair in remaining:
                losses = []
                for fit, evaluation, features, y_fit, y_evaluation, _ in fold_cache:
                    pair_fit, pair_evaluation, _ = _pair_columns(
                        features["compressed_train"], features["compressed_evaluation"], pair,
                        int(config["C_module"]["pairwise_anova"]["per_axis_spline_knot_count"]),
                    )
                    x_fit = np.concatenate([features[f"{base_key}_train"], pair_fit], axis=1)
                    x_evaluation = np.concatenate([features[f"{base_key}_evaluation"], pair_evaluation], axis=1)
                    losses.append(mse(y_evaluation, _ridge_fit(x_fit, y_fit, x_evaluation, selected_alpha)[0]))
                pair_losses[pair] = losses
            if not pair_losses:
                break
            neutral = "NO_PAIR"
            selection = one_se_select({neutral: current_losses, **pair_losses}, lambda value: (0,) if value == neutral else (1, value), neutral=neutral, minimum_usable_folds=3)
            if selection.selected == neutral:
                break
            pair = selection.selected
            activation = practical_activation(current_losses, pair_losses[pair], minimum_relative_improvement=0.01, minimum_positive_fraction=0.75)
            pair_audit.append({"pair": list(pair), "selection": selection.to_json(), "activation": activation})
            if not activation["pass"]:
                break
            selected_pairs.append(pair)
            current_losses = pair_losses[pair]
            remaining.remove(pair)
        if selected_pairs:
            selected_family = "SPARSE_PAIRWISE_ANOVA_MAX3"
        final_train = _cap(train, int(config["row_caps"]["joint_physical_fit"]))
        features = fit_physical_features(shared, view, final_train, validation, active, config, fit_split="train", evaluation_split="validation")
        component_predictions = {}
        component_contracts = {}
        for component_name, key, candidate in (
            ("K_COMPRESSED", "compressed", compressed_candidate),
            ("K_JOINT_BASIS", "joint", joint_selection.selected),
        ):
            component_train, component_validation = features[f"{key}_train"], features[f"{key}_evaluation"]
            if component_train.shape[1]:
                component_predictions[component_name], component_contracts[component_name] = _ridge_fit(
                    component_train, final_train["y_true"].to_numpy(dtype=np.float64), component_validation, float(candidate[1])
                )
            else:
                component_predictions[component_name] = np.zeros(len(validation), dtype=np.float64)
                component_contracts[component_name] = {"status": "EXACT_ZERO", "parameter_count": 0}
        base_key = "joint" if selected_family in {"ADDITIVE_JOINT_BASIS", "SPARSE_PAIRWISE_ANOVA_MAX3"} else "compressed"
        x_train, x_validation = features[f"{base_key}_train"], features[f"{base_key}_evaluation"]
        pair_contracts = []
        for pair in selected_pairs:
            pair_train, pair_validation, contract = _pair_columns(features["compressed_train"], features["compressed_evaluation"], pair, 4)
            x_train = np.concatenate([x_train, pair_train], axis=1)
            x_validation = np.concatenate([x_validation, pair_validation], axis=1)
            pair_contracts.append(contract)
        if x_train.shape[1]:
            prediction, fusion_contract = _ridge_fit(x_train, final_train["y_true"].to_numpy(dtype=np.float64), x_validation, selected_alpha)
        else:
            prediction = np.zeros(len(validation), dtype=np.float64)
            fusion_contract = {"mean": [], "scale": [], "coefficient": [], "intercept": 0.0, "alpha": selected_alpha, "numerical_certificate": {"status": "EXACT_ZERO"}}
        frame = validation[["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]].copy()
        frame["y_pred"] = prediction; frame["model"] = "PRISM_V2_C_PHYSICAL"; frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"; frame.to_parquet(prediction_path, index=False, compression="zstd")
        component_files = {}
        for component_name, component_prediction in component_predictions.items():
            component_frame = validation[["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]].copy()
            component_frame["y_pred"] = component_prediction; component_frame["model"] = f"PRISM_V2_{component_name}"; component_frame["dtype"] = "float64"
            component_path = destination / f"{component_name}.validation.parquet"; component_frame.to_parquet(component_path, index=False, compression="zstd")
            component_files[component_name] = {"path": str(component_path.relative_to(output)), "sha256": sha256_file(component_path),
                                               "contract": component_contracts[component_name]}
        result = {
            "status": "PASS", "stage": "V3_C_FUSION", "dataset": view.head.dataset, "task": view.head.task_id,
            "target_head": view.head.head_id, "proxy_policy": view.proxy_policy, "active_channels": features["channels"],
            "selected_family": selected_family, "selected_alpha": selected_alpha, "selected_pairs": [list(value) for value in selected_pairs],
            "channel_contracts": features["channel_contracts"], "global_joint_columns": features.get("global_joint_columns", []),
            "fusion_contract": fusion_contract, "pair_contracts": pair_contracts,
            "compressed_selection": compressed_selection.to_json(), "c_selection": c_selection.to_json(),
            "joint_selection": joint_selection.to_json(), "component_files": component_files,
            "candidate_fold_losses": {str(key): value for key, value in candidate_losses.items()},
            "joint_activation": joint_activation, "pair_audit": pair_audit,
            "prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": sha256_file(prediction_path),
            "test_accessed": False, "elapsed_seconds": time.time() - started,
            **regression_metrics(frame["y_true"].to_numpy(), prediction),
        }
    except Exception as error:
        result = {"status": "SOLVER_FAILED_RETAINED", "stage": "V3_C_FUSION", "target_head": view.head.head_id,
                  "proxy_policy": view.proxy_policy, "test_accessed": False, "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started}
    write_json(destination / "RESULT.json", result)
    return result


def run_v3_c(shared: Path, project: Path, output: Path, n_jobs: int) -> dict[str, Any]:
    results, pending = [], []
    for view in development_input_views(shared):
        path = output / "DEVELOPMENT" / "JOINT_BASIS" / view.head.head_id / view.proxy_policy / "RESULT.json"
        if path.is_file():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior.get("status") in {"PASS", "SOLVER_FAILED_RETAINED"}: results.append(prior); continue
        pending.append(view)
    results.extend(
        run_parallel(
            run_c_view,
            [(shared, project, output, view) for view in pending],
            n_jobs,
            per_worker_gib=5.0,
            label="V3_C_FUSION",
        )
    )
    summary = {"status": "PASS" if all(item["status"] == "PASS" for item in results) else "COMPLETED_WITH_RETAINED_FAILURES",
               "stage": "V3_C_FUSION", "views": len(results), "pass": sum(item["status"] == "PASS" for item in results),
               "joint_basis": sum(item.get("selected_family") == "ADDITIVE_JOINT_BASIS" for item in results),
               "pairwise": sum(item.get("selected_family") == "SPARSE_PAIRWISE_ANOVA_MAX3" for item in results), "test_accessed": False}
    write_json(output / "DEVELOPMENT" / "JOINT_BASIS" / "SUMMARY.json", summary)
    return summary
