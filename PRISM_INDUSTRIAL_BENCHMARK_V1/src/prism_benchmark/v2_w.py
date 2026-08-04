from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.interpolate import BSpline

from .cpu_data import ViewSpec, inner_folds, load_samples, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_basis import natural_cubic_columns
from .v2_c import _load_active_channels, _pair_columns, _ridge_fit, fit_physical_features
from .v2_config import load_frozen_config
from .v2_k import _cap
from .v2_numerics import difference_penalty, residualize, solve_certified
from .v2_selection import one_se_select, practical_activation
from .v2_runtime import run_parallel
from .v2_views import development_input_views


def _load_c(output: Path, view: ViewSpec) -> dict[str, Any]:
    path = output / "DEVELOPMENT" / "JOINT_BASIS" / view.head.head_id / view.proxy_policy / "RESULT.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError(f"V3 prerequisite is not PASS: {path}")
    return result


def fit_c_fold_prediction(
    shared: Path,
    view: ViewSpec,
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    active: list[dict[str, Any]],
    config: dict[str, Any],
    c_result: dict[str, Any],
    *,
    fit_split: str,
    evaluation_split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features = fit_physical_features(shared, view, fit, evaluation, active, config, fit_split=fit_split, evaluation_split=evaluation_split)
    family = c_result["selected_family"]
    key = "joint" if family in {"ADDITIVE_JOINT_BASIS", "SPARSE_PAIRWISE_ANOVA_MAX3"} else "compressed"
    train_x, evaluation_x = features[f"{key}_train"], features[f"{key}_evaluation"]
    pair_contracts = []
    for pair_value in c_result.get("selected_pairs", []):
        pair = tuple(int(value) for value in pair_value)
        pair_train, pair_evaluation, contract = _pair_columns(features["compressed_train"], features["compressed_evaluation"], pair, 4)
        train_x = np.concatenate([train_x, pair_train], axis=1)
        evaluation_x = np.concatenate([evaluation_x, pair_evaluation], axis=1)
        pair_contracts.append(contract)
    if train_x.shape[1] == 0:
        return np.zeros(len(fit)), np.zeros(len(evaluation)), {"status": "EXACT_ZERO"}
    evaluation_prediction, contract = _ridge_fit(train_x, fit["y_true"].to_numpy(dtype=np.float64), evaluation_x, float(c_result["selected_alpha"]))
    train_prediction = ((train_x - np.asarray(contract["mean"])) / np.asarray(contract["scale"])) @ np.asarray(contract["coefficient"]) + float(contract["intercept"])
    return train_prediction, evaluation_prediction, {"fusion": contract, "channels": features["channel_contracts"], "pairwise": pair_contracts}


def _ispline_raw(values: np.ndarray, internal_knots: np.ndarray, degree: int = 3) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    lower, upper = float(np.min(x)), float(np.max(x))
    if not upper > lower:
        return np.empty((len(x), 0), dtype=np.float64)
    knots = np.concatenate([np.full(degree + 1, lower), internal_knots, np.full(degree + 1, upper)])
    count = len(knots) - degree - 1
    columns = []
    for index in range(count):
        coefficient = np.zeros(count); coefficient[index] = 1.0
        basis = BSpline(knots, coefficient, degree, extrapolate=True)
        denominator = knots[index + degree + 1] - knots[index]
        if denominator <= 0:
            continue
        antiderivative = basis.antiderivative()
        multiplier = (degree + 1) / denominator
        base = float(antiderivative(lower))
        total = float(multiplier * (antiderivative(upper) - base))
        if total <= 0:
            continue
        columns.append(np.clip(multiplier * (antiderivative(x) - base) / total, 0.0, 1.0))
    return np.column_stack(columns) if columns else np.empty((len(x), 0), dtype=np.float64)


def _ispline_fixed(values:np.ndarray,internal_knots:np.ndarray,lower:float,upper:float,degree:int=3)->np.ndarray:
    x=np.clip(np.asarray(values,dtype=np.float64).reshape(-1),lower,upper)
    knots=np.concatenate([np.full(degree+1,lower),internal_knots,np.full(degree+1,upper)]);count=len(knots)-degree-1;columns=[]
    for index in range(count):
        coefficient=np.zeros(count);coefficient[index]=1.0;basis=BSpline(knots,coefficient,degree,extrapolate=True)
        denominator=knots[index+degree+1]-knots[index]
        if denominator<=0:continue
        antiderivative=basis.antiderivative();multiplier=(degree+1)/denominator;base=float(antiderivative(lower));total=float(multiplier*(antiderivative(upper)-base))
        if total>0:columns.append(np.clip(multiplier*(antiderivative(x)-base)/total,0.0,1.0))
    return np.column_stack(columns) if columns else np.empty((len(x),0),dtype=np.float64)


def _w_design(train_latent: np.ndarray, evaluation_latent: np.ndarray, family: str, knot_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = float(np.mean(train_latent)); scale = float(np.std(train_latent)) or 1.0
    train = (train_latent - mean) / scale; evaluation = (evaluation_latent - mean) / scale
    knots = np.unique(np.quantile(train, np.arange(1, knot_count + 1) / (knot_count + 1)))
    if family == "MONOTONE":
        train_raw = _ispline_raw(train, knots); evaluation_raw = _ispline_raw(evaluation, knots)
        # Re-evaluate with train support for exact basis identity.
        lower, upper = float(np.min(train)), float(np.max(train))
        clipped_eval = np.clip(evaluation, lower, upper)
        evaluation_raw = _ispline_fixed(clipped_eval, knots, lower, upper)
        return train_raw, evaluation_raw, {"mean": mean, "scale": scale, "knots": knots.tolist(), "train_min": lower, "train_max": upper}
    train_raw = natural_cubic_columns(train, knots)[:, 2:]
    evaluation_raw = natural_cubic_columns(evaluation, knots)[:, 2:]
    against = np.column_stack([np.ones(len(train)), train])
    train_design, orthogonality = residualize(train_raw, against, relative_tolerance=1e-8)
    projection = np.linalg.lstsq(against, train_raw, rcond=1e-10)[0]
    evaluation_design = evaluation_raw - np.column_stack([np.ones(len(evaluation)), evaluation]) @ projection
    return train_design, evaluation_design, {"mean": mean, "scale": scale, "knots": knots.tolist(), "projection": projection.tolist(), "orthogonality": orthogonality,
                                                 "train_min": float(np.min(train)), "train_max": float(np.max(train))}


def fit_w_candidate(train_latent: np.ndarray, target: np.ndarray, evaluation_latent: np.ndarray, family: str, knot_count: int, smoothness: float, direction: int = 1) -> tuple[np.ndarray, dict[str, Any]]:
    if family == "IDENTITY":
        return evaluation_latent.copy(), {"family": family, "parameter_count": 0, "effective_df": 1.0}
    train_design, evaluation_design, metadata = _w_design(train_latent, evaluation_latent, family, knot_count)
    residual_target = target - train_latent
    penalty = smoothness * difference_penalty(train_design.shape[1])
    if family == "MONOTONE":
        root = np.linalg.cholesky(penalty + np.eye(penalty.shape[0]) * 1e-15)
        augmented_x = np.vstack([direction * train_design, root])
        augmented_y = np.concatenate([residual_target, np.zeros(root.shape[0])])
        fit = optimize.lsq_linear(augmented_x, augmented_y, bounds=(0.0, np.inf), lsmr_tol=1e-12)
        coefficient = fit.x
        certificate = {"status": "PASS" if fit.success else "NUMERICALLY_INVALID", "solver": "NONNEGATIVE_LSQ", "optimality": float(fit.optimality)}
        prediction = evaluation_latent + direction * evaluation_design @ coefficient
        effective_df = float(np.count_nonzero(coefficient > 1e-12))
    else:
        coefficient, linear_certificate = solve_certified(train_design, residual_target, penalty)
        certificate = linear_certificate.to_json()
        prediction = evaluation_latent + evaluation_design @ coefficient
        gram = train_design.T @ train_design
        effective_df = float(np.trace(np.linalg.pinv(gram + penalty, rcond=1e-12) @ gram))
    return prediction, {"family": family, "knot_count": knot_count, "smoothness": smoothness, "direction": direction,
                        "coefficient": coefficient.tolist(), "basis": metadata, "effective_df": effective_df,
                        "parameter_count": int(np.count_nonzero(np.abs(coefficient) > 1e-12)), "numerical_certificate": certificate}


def predict_w_contract(latent:np.ndarray,contract:dict[str,Any])->np.ndarray:
    values=np.asarray(latent,dtype=np.float64)
    if contract["family"]=="IDENTITY":return values.copy()
    metadata=contract["basis"];standardized=(values-float(metadata["mean"]))/float(metadata["scale"]);knots=np.asarray(metadata["knots"],dtype=np.float64)
    if contract["family"]=="MONOTONE":
        design=_ispline_fixed(standardized,knots,float(metadata["train_min"]),float(metadata["train_max"]))
        return values+int(contract["direction"])*design@np.asarray(contract["coefficient"],dtype=np.float64)
    raw=natural_cubic_columns(standardized,knots)[:,2:];against=np.column_stack([np.ones(len(values)),standardized])
    design=raw-against@np.asarray(metadata["projection"],dtype=np.float64)
    return values+design@np.asarray(contract["coefficient"],dtype=np.float64)


def run_w_view(shared: Path, project: Path, output: Path, view: ViewSpec) -> dict[str, Any]:
    started = time.time(); destination = output / "DEVELOPMENT" / "WIENER" / view.head.head_id / view.proxy_policy
    destination.mkdir(parents=True, exist_ok=True)
    try:
        config = load_frozen_config(project); c_result = _load_c(output, view); active = _load_active_channels(output, view)
        train = load_samples(shared, view, "train"); validation = load_samples(shared, view, "validation")
        folds = inner_folds(train, 4); fold_data = []; signs = []
        for fold, (fit_index, evaluation_index) in enumerate(folds):
            fit = _cap(train.iloc[fit_index], int(config["row_caps"]["wiener_fit"]))
            evaluation = _cap(train.iloc[evaluation_index], int(config["row_caps"]["validation_selection_per_fold"]))
            fit_latent, evaluation_latent, _ = fit_c_fold_prediction(shared, view, fit, evaluation, active, config, c_result, fit_split="train", evaluation_split="train")
            target = fit["y_true"].to_numpy(dtype=np.float64); evaluation_target = evaluation["y_true"].to_numpy(dtype=np.float64)
            correlation = float(stats.spearmanr(fit_latent, target - fit_latent).statistic) if np.std(fit_latent) > 0 else 0.0
            signs.append(correlation); fold_data.append((fold, fit_latent, target, evaluation_latent, evaluation_target, evaluation))
        applicable_signs = [value for value in signs if abs(value) >= float(config["W_module"]["monotone_direction"]["absolute_correlation_min"])]
        direction = 1 if sum(value > 0 for value in applicable_signs) >= sum(value < 0 for value in applicable_signs) else -1
        monotone_applicable = bool(applicable_signs) and sum((value > 0) == (direction > 0) for value in applicable_signs) / len(signs) >= float(config["W_module"]["monotone_direction"]["same_sign_fold_fraction_min"])
        losses: dict[Any, list[float]] = {"IDENTITY": []}
        for _, _, _, evaluation_latent, evaluation_target, _ in fold_data:
            losses["IDENTITY"].append(mse(evaluation_target, evaluation_latent))
        for family in ("MONOTONE", "NATURAL_CUBIC"):
            if family == "MONOTONE" and not monotone_applicable: continue
            knot_counts = [4, 6] if family == "MONOTONE" else [4, 6, 8]
            for knots in knot_counts:
                for smoothness in config["W_module"]["smoothness_penalties"]:
                    candidate = (family, knots, float(smoothness)); losses[candidate] = []
                    for _, fit_latent, target, evaluation_latent, evaluation_target, _ in fold_data:
                        prediction, contract = fit_w_candidate(fit_latent, target, evaluation_latent, family, knots, float(smoothness), direction)
                        if contract["effective_df"] > float(config["W_module"]["effective_degrees_of_freedom_hard_max"]): losses[candidate].append(float("inf"))
                        else: losses[candidate].append(mse(evaluation_target, prediction))
        def complexity(value):
            if value == "IDENTITY": return (0,)
            family, knots, smoothness = value
            return (1, 0 if family == "MONOTONE" else 1, knots, -smoothness)
        selection = one_se_select(losses, complexity, neutral="IDENTITY", minimum_usable_folds=3)
        selected = selection.selected; activation = {"pass": False, "reason": "IDENTITY_SELECTED"}
        if selected != "IDENTITY":
            activation = practical_activation(losses["IDENTITY"], losses[selected], minimum_relative_improvement=0.01, minimum_positive_fraction=0.75)
            if not activation["pass"]: selected = "IDENTITY"
        oof_frames = []
        for fold, fit_latent, fit_target, evaluation_latent, _, evaluation in fold_data:
            frame = evaluation[["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]].copy()
            frame["physical_oof"] = evaluation_latent
            if selected == "IDENTITY":
                frame["physical_w_oof"] = evaluation_latent
            else:
                family, knots, smoothness = selected
                frame["physical_w_oof"] = fit_w_candidate(
                    fit_latent, fit_target, evaluation_latent, family, knots, smoothness, direction
                )[0]
            frame["oof_fold"] = fold; oof_frames.append(frame)
        oof = pd.concat(oof_frames, ignore_index=True)
        _, validation_latent, c_final_contract = fit_c_fold_prediction(shared, view, _cap(train, int(config["row_caps"]["joint_physical_fit"])), validation, active, config, c_result, fit_split="train", evaluation_split="validation")
        if selected == "IDENTITY": prediction = validation_latent; w_contract = {"family": "IDENTITY", "parameter_count": 0, "effective_df": 1.0}
        else:
            family, knots, smoothness = selected
            prediction, w_contract = fit_w_candidate(oof["physical_oof"].to_numpy(), oof["y_true"].to_numpy(), validation_latent, family, knots, smoothness, direction)
        oof_path = destination / "PHYSICAL_OOF.parquet"; oof.to_parquet(oof_path, index=False, compression="zstd")
        frame = validation[["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]].copy(); frame["physical_latent"] = validation_latent
        frame["y_pred"] = prediction; frame["model"] = "PRISM_V2_K_C_W"; frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"; frame.to_parquet(prediction_path, index=False, compression="zstd")
        hard_exceed = float(np.mean((validation_latent < np.min(oof["physical_oof"])) | (validation_latent > np.max(oof["physical_oof"]))))
        result = {"status": "PASS", "stage": "V4_WIENER", "target_head": view.head.head_id, "proxy_policy": view.proxy_policy,
                  "selected_candidate": str(selected), "w_contract": w_contract, "c_final_contract": c_final_contract,
                  "one_se": selection.to_json(), "activation": activation, "fold_losses": {str(key): value for key, value in losses.items()},
                  "monotone_correlations": signs, "monotone_applicable": monotone_applicable, "monotone_direction": direction,
                  "latent_hard_support_exceedance_rate": hard_exceed, "oof_path": str(oof_path.relative_to(output)), "oof_sha256": sha256_file(oof_path),
                  "prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": sha256_file(prediction_path),
                  "test_accessed": False, "elapsed_seconds": time.time()-started, **regression_metrics(frame["y_true"].to_numpy(), prediction)}
    except Exception as error:
        result = {"status": "SOLVER_FAILED_RETAINED", "stage": "V4_WIENER", "target_head": view.head.head_id, "proxy_policy": view.proxy_policy,
                  "test_accessed": False, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.time()-started}
    write_json(destination / "RESULT.json", result); return result


def run_v4_w(shared: Path, project: Path, output: Path, n_jobs: int) -> dict[str, Any]:
    results, pending = [], []
    for view in development_input_views(shared):
        path = output / "DEVELOPMENT" / "WIENER" / view.head.head_id / view.proxy_policy / "RESULT.json"
        if path.is_file():
            prior=json.loads(path.read_text());
            if prior.get("status") in {"PASS","SOLVER_FAILED_RETAINED"}: results.append(prior); continue
        pending.append(view)
    results.extend(
        run_parallel(
            run_w_view,
            [(shared, project, output, view) for view in pending],
            n_jobs,
            per_worker_gib=6.0,
            label="V4_WIENER",
        )
    )
    summary={"status":"PASS" if all(x["status"]=="PASS" for x in results) else "COMPLETED_WITH_RETAINED_FAILURES","stage":"V4_WIENER",
             "views":len(results),"pass":sum(x["status"]=="PASS" for x in results),"activated":sum(x.get("selected_candidate")!="IDENTITY" for x in results),"test_accessed":False}
    write_json(output/"DEVELOPMENT"/"WIENER"/"SUMMARY.json",summary); return summary
