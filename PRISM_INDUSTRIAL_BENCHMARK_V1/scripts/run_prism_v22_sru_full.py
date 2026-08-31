from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.prism_ct import CTBasisConfig, feature_audit
from prism_benchmark.prism_ct_assembly import (
    fit_simplex_assembly,
    predict_simplex_assembly,
)
from prism_benchmark.v2_c import _ridge_fit, _ridge_predict
from prism_benchmark.v2_selection import one_se_select
from prism_benchmark.v2_urysohn import fit_contract, predict_contract
from prism_benchmark.v21_a import (
    EXACT_ZERO as A_EXACT_ZERO,
    MATURE_RESIDUAL_AR,
    fit_mature_residual_ar,
    mature_residual_features,
    predict_mature_residual_ar,
)
from prism_benchmark.v21_selection import guarded_local_one_se_select
from prism_benchmark.v211_selection import numerical_contract_passes
from prism_benchmark.v211_w import (
    IDENTITY,
    MONOTONE,
    NATURAL_CUBIC,
    fit_w_correction,
    predict_w_correction,
)


K_ZERO = "EXACT_ZERO"
BRANCHES = ("D", "M", "S")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }


def _expanding_folds(indices: np.ndarray, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(indices, dtype=np.int64)
    blocks = [block for block in np.array_split(values, count + 1) if len(block)]
    if len(blocks) != count + 1:
        raise ValueError(f"cannot build {count} expanding folds from {len(values)} rows")
    result = []
    for fold in range(count):
        fit = np.concatenate(blocks[: fold + 1])
        evaluation = blocks[fold + 1]
        if not len(fit) or not len(evaluation) or int(fit[-1]) >= int(evaluation[0]):
            raise RuntimeError("noncausal expanding fold")
        result.append((fit, evaluation))
    return result


def _build_temporal_representations(x: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    temporal = config["temporal_representation"]
    lags = [int(value) for value in temporal["D_lags_steps"]]
    taus = np.asarray(temporal["CT_tau_steps"], dtype=np.float64)
    n, channels = x.shape
    maximum_lag = max(lags)

    delay = np.full((n, channels, len(lags)), np.nan, dtype=np.float64)
    for column, lag in enumerate(lags):
        delay[maximum_lag:, :, column] = x[maximum_lag - lag : n - lag]

    decay = np.exp(-1.0 / taus)
    states = np.empty((n, channels, len(taus)), dtype=np.float64)
    states[0] = x[0, :, None]
    for row in range(1, n):
        states[row] = decay[None, :] * states[row - 1] + (1.0 - decay)[None, :] * x[row, :, None]

    multires = np.empty_like(states)
    multires[:, :, 0] = x - states[:, :, 0]
    for scale in range(len(taus) - 1):
        multires[:, :, scale + 1] = states[:, :, scale] - states[:, :, scale + 1]

    return {"D": delay, "M": multires, "S": states}


def _k_candidates(config: dict[str, Any]) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = [(K_ZERO, 1), ("LINEAR_DISTRIBUTED_LAG", 1)]
    widths = [int(value) for value in config["K"]["candidate_m_x"]]
    for family in ("RANK_1", "RANK_2", "RANK_3", "FULL_FINITE_URYSOHN"):
        result.extend((family, width) for width in widths)
    return result


def _k_complexity(candidate: tuple[str, int]) -> tuple[Any, ...]:
    family, width = candidate
    order = {
        K_ZERO: 0,
        "LINEAR_DISTRIBUTED_LAG": 1,
        "RANK_1": 2,
        "RANK_2": 3,
        "RANK_3": 4,
        "FULL_FINITE_URYSOHN": 5,
    }
    return (order[family], int(width))


def _fit_k(
    values: np.ndarray,
    target: np.ndarray,
    candidate: tuple[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    family, width = candidate
    penalty = config["K"]["penalties"]
    contract = fit_contract(
        values,
        target,
        family,
        int(width),
        (
            float(penalty["lambda_0"]),
            float(penalty["lambda_tau"]),
            float(penalty["lambda_x"]),
        ),
        als_seeds=tuple(int(value) for value in config["K"]["als_seeds"]),
        als_max_iterations=int(config["K"]["als_max_iterations"]),
        als_tolerance=float(config["K"]["als_tolerance"]),
        als_max_increases=int(config["K"]["als_max_increases"]),
        als_divergence_factor=float(config["K"]["als_divergence_factor"]),
    )
    return contract


def _select_k_channel(
    representation: np.ndarray,
    target_delta: np.ndarray,
    development: np.ndarray,
    channel: int,
    config: dict[str, Any],
) -> tuple[tuple[str, int], dict[str, Any]]:
    candidates = _k_candidates(config)
    folds = _expanding_folds(development, int(config["selection"]["inner_expanding_folds"]))
    losses: dict[tuple[str, int], list[float]] = {candidate: [] for candidate in candidates}
    numeric: dict[str, list[dict[str, Any]]] = {str(candidate): [] for candidate in candidates}
    for fit_index, evaluation_index in folds:
        fit_values = representation[fit_index, channel, :]
        evaluation_values = representation[evaluation_index, channel, :]
        y_fit = target_delta[fit_index]
        y_evaluation = target_delta[evaluation_index]
        for candidate in candidates:
            try:
                contract = _fit_k(fit_values, y_fit, candidate, config)
                valid = numerical_contract_passes(contract)
                prediction = predict_contract(evaluation_values, contract) if valid else np.full(len(evaluation_index), np.nan)
                loss = float(np.mean(np.square(y_evaluation - prediction))) if valid else float("nan")
                numeric[str(candidate)].append({"pass": bool(valid), "certificate": contract.get("certificate", {})})
            except Exception as error:
                loss = float("nan")
                numeric[str(candidate)].append({"pass": False, "error": f"{type(error).__name__}: {error}"})
            losses[candidate].append(loss)

    selection = guarded_local_one_se_select(
        losses,
        _k_complexity,
        neutral=(K_ZERO, 1),
        minimum_relative_improvement=float(config["selection"]["minimum_relative_improvement"]),
        minimum_positive_fraction=float(config["selection"]["minimum_positive_fold_fraction"]),
        minimum_usable_folds=int(config["selection"]["minimum_usable_folds"]),
    )
    return selection.final_selected_candidate, {
        "selected": str(selection.final_selected_candidate),
        "selection": selection.to_json(),
        "candidate_numeric_audit": numeric,
    }


def _k_contribution_matrices(
    representation: np.ndarray,
    target_delta: np.ndarray,
    fit_index: np.ndarray,
    evaluation_index: np.ndarray,
    selected_by_channel: dict[int, tuple[str, int]],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_columns = []
    evaluation_columns = []
    contracts = []
    for channel in sorted(selected_by_channel):
        candidate = selected_by_channel[channel]
        if candidate[0] == K_ZERO:
            continue
        contract = _fit_k(
            representation[fit_index, channel, :],
            target_delta[fit_index],
            candidate,
            config,
        )
        if not numerical_contract_passes(contract):
            raise RuntimeError(f"selected K candidate became numerically invalid: channel={channel}: {candidate}")
        train_columns.append(predict_contract(representation[fit_index, channel, :], contract))
        evaluation_columns.append(predict_contract(representation[evaluation_index, channel, :], contract))
        contracts.append({"channel": int(channel), "candidate": str(candidate), "contract": contract})
    if not train_columns:
        return (
            np.empty((len(fit_index), 0), dtype=np.float64),
            np.empty((len(evaluation_index), 0), dtype=np.float64),
            contracts,
        )
    return np.column_stack(train_columns), np.column_stack(evaluation_columns), contracts


def _select_branch_structure(
    branch: str,
    representation: np.ndarray,
    target_delta: np.ndarray,
    development: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    audit: dict[str, Any] = {"branch": branch, "eligible": True}
    if branch == "S":
        temporal = config["temporal_representation"]
        ct_config = CTBasisConfig(
            dt_seconds=float(config["cadence_seconds"]),
            taus_seconds=tuple(float(value) for value in temporal["CT_tau_seconds"]),
            condition_hard_fail=float(temporal["S_condition_number_hard_fail"]),
        )
        condition = feature_audit(representation[development].reshape(len(development), -1), ct_config)
        audit["conditioning"] = condition.__dict__
        if not condition.passed_conditioning:
            audit["eligible"] = False
            audit["reason"] = "S_CONDITIONING_REJECTED_BEFORE_K"
            return {
                "branch": branch,
                "eligible": False,
                "selected_by_channel": {},
                "selected_alpha": None,
                "audit": audit,
            }

    selected_by_channel: dict[int, tuple[str, int]] = {}
    k_audit: dict[str, Any] = {}
    for channel in range(representation.shape[1]):
        selected, result = _select_k_channel(
            representation, target_delta, development, channel, config
        )
        selected_by_channel[channel] = selected
        k_audit[str(channel)] = result
    active = [channel for channel, candidate in selected_by_channel.items() if candidate[0] != K_ZERO]
    audit["active_channels"] = active
    audit["K"] = k_audit
    if not active:
        audit["eligible"] = False
        audit["reason"] = "ALL_K_CHANNELS_EXACT_ZERO"
        return {
            "branch": branch,
            "eligible": False,
            "selected_by_channel": selected_by_channel,
            "selected_alpha": None,
            "audit": audit,
        }

    alpha_grid = [float(value) for value in config["C"]["ridge_alpha_grid"]]
    losses = {alpha: [] for alpha in alpha_grid}
    folds = _expanding_folds(development, int(config["selection"]["inner_expanding_folds"]))
    for fit_index, evaluation_index in folds:
        x_fit, x_evaluation, _ = _k_contribution_matrices(
            representation,
            target_delta,
            fit_index,
            evaluation_index,
            selected_by_channel,
            config,
        )
        y_fit = target_delta[fit_index]
        y_evaluation = target_delta[evaluation_index]
        for alpha in alpha_grid:
            try:
                prediction, _ = _ridge_fit(x_fit, y_fit, x_evaluation, alpha)
                losses[alpha].append(float(np.mean(np.square(y_evaluation - prediction))))
            except Exception:
                losses[alpha].append(float("nan"))
    selection = one_se_select(
        losses,
        lambda alpha: (-float(alpha),),
        minimum_usable_folds=int(config["selection"]["minimum_usable_folds"]),
    )
    selected_alpha = float(selection.selected)
    x_fit, x_same, contracts = _k_contribution_matrices(
        representation,
        target_delta,
        development,
        development,
        selected_by_channel,
        config,
    )
    fitted, c_contract = _ridge_fit(x_fit, target_delta[development], x_same, selected_alpha)
    target_variance = float(np.var(target_delta[development], dtype=np.float64))
    prediction_variance = float(np.var(fitted, dtype=np.float64))
    coefficient = np.asarray(c_contract.get("coefficient", []), dtype=np.float64)
    path_audit = {
        "prediction_variance_ratio_to_target": prediction_variance / max(target_variance, np.finfo(np.float64).tiny),
        "maximum_nonintercept_coefficient_abs": float(np.max(np.abs(coefficient), initial=0.0)),
        "variance_pass": bool(prediction_variance / max(target_variance, np.finfo(np.float64).tiny) >= float(config["C"]["input_path_min_variance_ratio"])),
        "coefficient_pass": bool(float(np.max(np.abs(coefficient), initial=0.0)) >= float(config["C"]["input_path_min_nonintercept_coefficient_abs"])),
    }
    path_audit["pass"] = bool(path_audit["variance_pass"] and path_audit["coefficient_pass"])
    audit["C"] = {
        "family": "ADDITIVE_COMPRESSED",
        "selected_alpha": selected_alpha,
        "selection": selection.to_json(),
        "input_path_preservation": path_audit,
        "fit_k_contract_count": len(contracts),
    }
    if not path_audit["pass"]:
        audit["eligible"] = False
        audit["reason"] = "C_INPUT_PATH_NOT_PRESERVED"
    return {
        "branch": branch,
        "eligible": bool(audit["eligible"]),
        "selected_by_channel": selected_by_channel,
        "selected_alpha": selected_alpha,
        "audit": audit,
    }


def _predict_branch_fixed(
    representation: np.ndarray,
    target_delta: np.ndarray,
    fit_index: np.ndarray,
    evaluation_index: np.ndarray,
    structure: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not structure["eligible"]:
        return np.zeros(len(evaluation_index), dtype=np.float64), {
            "branch": structure["branch"],
            "status": "EXACT_ZERO_INELIGIBLE_BRANCH",
        }
    x_fit, x_evaluation, k_contracts = _k_contribution_matrices(
        representation,
        target_delta,
        fit_index,
        evaluation_index,
        structure["selected_by_channel"],
        config,
    )
    prediction, c_contract = _ridge_fit(
        x_fit,
        target_delta[fit_index],
        x_evaluation,
        float(structure["selected_alpha"]),
    )
    return prediction, {
        "branch": structure["branch"],
        "status": "PASS",
        "k_contracts": k_contracts,
        "c_contract": c_contract,
    }


def _branch_predictions(
    representations: dict[str, np.ndarray],
    target_delta: np.ndarray,
    fit_index: np.ndarray,
    evaluation_index: np.ndarray,
    structures: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    predictions: dict[str, np.ndarray] = {}
    contracts: dict[str, Any] = {}
    for branch in BRANCHES:
        structure = structures[branch]
        if not structure["eligible"]:
            continue
        prediction, contract = _predict_branch_fixed(
            representations[branch],
            target_delta,
            fit_index,
            evaluation_index,
            structure,
            config,
        )
        predictions[branch] = prediction
        contracts[branch] = contract
    if not predictions:
        raise RuntimeError("no D/M/S branch survived K/C selection")
    return predictions, contracts


def _w_candidates(config: dict[str, Any], direction: int, monotone_allowed: bool) -> list[Any]:
    result: list[Any] = [IDENTITY]
    if monotone_allowed:
        result.extend(
            (MONOTONE, int(knots), float(smoothness), float(mu), int(direction))
            for knots in config["W"]["monotone_knots"]
            for smoothness in config["W"]["smoothness"]
            for mu in config["W"]["soft_overlap_mu"]
        )
    result.extend(
        (NATURAL_CUBIC, int(knots), float(smoothness), float(mu), 1)
        for knots in config["W"]["natural_cubic_knots"]
        for smoothness in config["W"]["smoothness"]
        for mu in config["W"]["soft_overlap_mu"]
    )
    return result


def _w_complexity(candidate: Any) -> tuple[Any, ...]:
    if candidate == IDENTITY:
        return (0,)
    family, knots, smoothness, mu, _ = candidate
    return (1, 0 if family == MONOTONE else 1, int(knots), -float(smoothness), -float(mu))


def _fit_w_candidate(
    candidate: Any,
    fit_gamma: np.ndarray,
    fit_residual: np.ndarray,
    evaluation_gamma: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate == IDENTITY:
        return fit_w_correction(
            fit_gamma,
            fit_residual,
            evaluation_gamma,
            family=IDENTITY,
        )
    family, knots, smoothness, mu, direction = candidate
    return fit_w_correction(
        fit_gamma,
        fit_residual,
        evaluation_gamma,
        family=family,
        knot_count=int(knots),
        smoothness=float(smoothness),
        mu=float(mu),
        upstream_predictions=fit_gamma[:, None],
        direction=int(direction),
    )


def _select_and_materialize_w(
    oof: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Any, dict[str, Any], dict[str, Any]]:
    fold_ids = sorted(int(value) for value in oof["oof_fold"].unique())
    evaluation_folds = fold_ids[1:]
    correlations = []
    for fold in evaluation_folds:
        fit = oof[oof["oof_fold"] < fold]
        residual = fit["target_delta"].to_numpy(dtype=np.float64) - fit["gamma_oof"].to_numpy(dtype=np.float64)
        latent = fit["gamma_oof"].to_numpy(dtype=np.float64)
        correlation = stats.spearmanr(latent, residual).statistic if np.std(latent) > 0 and np.std(residual) > 0 else 0.0
        correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
    median = float(np.median(correlations)) if correlations else 0.0
    direction = 1 if median >= 0 else -1
    same_sign = float(np.mean(np.sign(correlations) == direction)) if correlations else 0.0
    monotone_allowed = bool(abs(median) >= 0.05 and same_sign >= 0.75)
    candidates = _w_candidates(config, direction, monotone_allowed)
    losses = {candidate: [] for candidate in candidates}
    numeric = {str(candidate): [] for candidate in candidates}
    for fold in evaluation_folds:
        fit = oof[oof["oof_fold"] < fold]
        evaluation = oof[oof["oof_fold"] == fold]
        fit_gamma = fit["gamma_oof"].to_numpy(dtype=np.float64)
        fit_residual = fit["target_delta"].to_numpy(dtype=np.float64) - fit_gamma
        eval_gamma = evaluation["gamma_oof"].to_numpy(dtype=np.float64)
        eval_target = evaluation["target_delta"].to_numpy(dtype=np.float64)
        for candidate in candidates:
            try:
                correction, contract = _fit_w_candidate(candidate, fit_gamma, fit_residual, eval_gamma)
                valid = bool(np.isfinite(correction).all() and numerical_contract_passes(contract))
                loss = float(np.mean(np.square(eval_target - eval_gamma - correction))) if valid else float("nan")
                numeric[str(candidate)].append({"pass": valid, "certificate": contract.get("numerical_certificate", {})})
            except Exception as error:
                loss = float("nan")
                numeric[str(candidate)].append({"pass": False, "error": f"{type(error).__name__}: {error}"})
            losses[candidate].append(loss)
    selection = guarded_local_one_se_select(
        losses,
        _w_complexity,
        neutral=IDENTITY,
        minimum_relative_improvement=float(config["selection"]["minimum_relative_improvement"]),
        minimum_positive_fraction=float(config["selection"]["minimum_positive_fold_fraction"]),
        minimum_usable_folds=int(config["selection"]["minimum_usable_folds"]),
    )
    selected = selection.final_selected_candidate
    frames = []
    for fold in evaluation_folds:
        fit = oof[oof["oof_fold"] < fold]
        evaluation = oof[oof["oof_fold"] == fold].copy()
        fit_gamma = fit["gamma_oof"].to_numpy(dtype=np.float64)
        correction, _ = _fit_w_candidate(
            selected,
            fit_gamma,
            fit["target_delta"].to_numpy(dtype=np.float64) - fit_gamma,
            evaluation["gamma_oof"].to_numpy(dtype=np.float64),
        )
        evaluation["delta_w_oof"] = correction
        evaluation["gamma_w_oof"] = evaluation["gamma_oof"].to_numpy(dtype=np.float64) + correction
        frames.append(evaluation)
    materialized = pd.concat(frames, ignore_index=True).sort_values("origin").reset_index(drop=True)
    full_gamma = oof["gamma_oof"].to_numpy(dtype=np.float64)
    _, final_contract = _fit_w_candidate(
        selected,
        full_gamma,
        oof["target_delta"].to_numpy(dtype=np.float64) - full_gamma,
        full_gamma,
    )
    audit = {
        "selected": str(selected),
        "selection": selection.to_json(),
        "monotone_direction": direction,
        "monotone_allowed": monotone_allowed,
        "median_fold_spearman": median,
        "same_sign_fold_fraction": same_sign,
        "candidate_numeric_audit": numeric,
        "final_contract": final_contract,
    }
    return materialized, selected, final_contract, audit


def _a_samples(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["origin"]].copy()
    result["entity_id"] = "sru_line4"
    result["latest_available_target_index"] = result["origin"].astype(np.int64) - 1
    return result


def _a_complexity(candidate: Any) -> tuple[Any, ...]:
    if candidate == A_EXACT_ZERO:
        return (0,)
    _, profile, alpha, mu = candidate
    return (1, int(profile[1]), -int(profile[0]), -float(alpha), -float(mu))


def _select_a(
    w_oof: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    work = w_oof.copy()
    work["residual"] = work["target_delta"] - work["gamma_w_oof"]
    work["entity_id"] = "sru_line4"
    work["latest_available_target_index"] = work["origin"].astype(np.int64) - 1
    profiles = [tuple(int(value) for value in profile) for profile in config["A"]["profiles_delta_history_steps"]]
    alphas = [float(value) for value in config["A"]["ridge_alpha_grid"]]
    mus = [float(value) for value in config["A"]["soft_overlap_mu"]]
    candidates: list[Any] = [A_EXACT_ZERO]
    candidates.extend((MATURE_RESIDUAL_AR, profile, alpha, mu) for profile in profiles for alpha in alphas for mu in mus)
    losses = {candidate: [] for candidate in candidates}
    coverage: dict[str, list[float]] = {str(profile): [] for profile in profiles}
    fold_ids = sorted(int(value) for value in work["oof_fold"].unique())
    evaluation_folds = fold_ids[1:]
    for fold in evaluation_folds:
        fit = work[work["oof_fold"] < fold].copy()
        evaluation = work[work["oof_fold"] == fold].copy()
        residual_mean = float(fit["residual"].mean())
        y_fit = fit["residual"].to_numpy(dtype=np.float64)
        y_eval = evaluation["residual"].to_numpy(dtype=np.float64)
        losses[A_EXACT_ZERO].append(float(np.mean(np.square(y_eval))))
        upstream = fit[["gamma_oof", "delta_w_oof"]].to_numpy(dtype=np.float64)
        for profile in profiles:
            delta, history = profile
            x_fit, observed_fit, _ = mature_residual_features(
                fit,
                work,
                h_steps=0,
                w_steps=1,
                delta=delta,
                history=history,
                maximum_lags=int(config["A"]["maximum_lags"]),
                residual_mean=residual_mean,
            )
            x_eval, observed_eval, _ = mature_residual_features(
                evaluation,
                work,
                h_steps=0,
                w_steps=1,
                delta=delta,
                history=history,
                maximum_lags=int(config["A"]["maximum_lags"]),
                residual_mean=residual_mean,
            )
            coverage[str(profile)].extend([observed_fit, observed_eval])
            for alpha in alphas:
                for mu in mus:
                    candidate = (MATURE_RESIDUAL_AR, profile, alpha, mu)
                    try:
                        prediction, contract = fit_mature_residual_ar(
                            x_fit,
                            y_fit,
                            x_eval,
                            alpha=alpha,
                            mu=mu,
                            upstream_predictions=upstream,
                        )
                        valid = bool(np.isfinite(prediction).all() and numerical_contract_passes(contract))
                        losses[candidate].append(float(np.mean(np.square(y_eval - prediction))) if valid else float("nan"))
                    except Exception:
                        losses[candidate].append(float("nan"))
    selection = guarded_local_one_se_select(
        losses,
        _a_complexity,
        neutral=A_EXACT_ZERO,
        minimum_relative_improvement=float(config["selection"]["minimum_relative_improvement"]),
        minimum_positive_fraction=float(config["selection"]["minimum_positive_fold_fraction"]),
        minimum_usable_folds=int(config["selection"]["minimum_usable_folds"]),
    )
    return selection.final_selected_candidate, {
        "selected": str(selection.final_selected_candidate),
        "selection": selection.to_json(),
        "profile_coverage": coverage,
    }


def _fit_final_a(
    selected: Any,
    train_residual_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if selected == A_EXACT_ZERO:
        return np.zeros(len(test_frame), dtype=np.float64), {
            "family": A_EXACT_ZERO,
            "parameter_count": 0,
            "numerical_certificate": {"status": "EXACT_ZERO"},
        }
    train = train_residual_frame.copy()
    train["entity_id"] = "sru_line4"
    train["latest_available_target_index"] = train["origin"].astype(np.int64) - 1
    test = test_frame.copy()
    test["entity_id"] = "sru_line4"
    test["latest_available_target_index"] = test["origin"].astype(np.int64) - 1
    source = pd.concat(
        [
            train[["entity_id", "origin", "residual"]],
            test[["entity_id", "origin", "residual"]],
        ],
        ignore_index=True,
    )
    residual_mean = float(train["residual"].mean())
    _, profile, alpha, mu = selected
    delta, history = profile
    x_train, observed_train, train_audit = mature_residual_features(
        train,
        source,
        h_steps=0,
        w_steps=1,
        delta=int(delta),
        history=int(history),
        maximum_lags=int(config["A"]["maximum_lags"]),
        residual_mean=residual_mean,
    )
    x_test, observed_test, test_audit = mature_residual_features(
        test,
        source,
        h_steps=0,
        w_steps=1,
        delta=int(delta),
        history=int(history),
        maximum_lags=int(config["A"]["maximum_lags"]),
        residual_mean=residual_mean,
    )
    _, contract = fit_mature_residual_ar(
        x_train,
        train["residual"].to_numpy(dtype=np.float64),
        x_train,
        alpha=float(alpha),
        mu=float(mu),
        upstream_predictions=train[["gamma", "delta_w"]].to_numpy(dtype=np.float64),
    )
    prediction = predict_mature_residual_ar(x_test, contract)
    contract.update(
        {
            "profile": [int(delta), int(history)],
            "residual_mean": residual_mean,
            "maturity_train_audit": train_audit,
            "maturity_test_audit": test_audit,
            "observed_train_fraction": observed_train,
            "observed_test_fraction": observed_test,
        }
    )
    return prediction, contract


def run(data_path: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    started = time.time()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(data_path, sep=r"\s+", header=None)
    data = frame.to_numpy(dtype=np.float64)
    if data.shape != (10081, 7):
        raise RuntimeError(f"corrected SRU shape mismatch: {data.shape}")
    x = data[:, :5]
    y = data[:, int(config["target_column"])]
    representations = _build_temporal_representations(x, config)
    maximum_lag = max(int(value) for value in config["temporal_representation"]["D_lags_steps"])
    train_size = int(config["author_train_size"])
    train_index = np.arange(maximum_lag, train_size, dtype=np.int64)
    test_index = np.arange(train_size, len(data), dtype=np.int64)
    if len(train_index) != int(config["author_train_target_count"]):
        raise RuntimeError("author train-target count mismatch")
    target_delta = np.zeros(len(y), dtype=np.float64)
    target_delta[1:] = y[1:] - y[:-1]
    persistence = np.roll(y, 1)
    persistence[0] = y[0]

    development_count = int(math.floor(float(config["selection"]["development_fraction_of_author_train"]) * len(train_index)))
    development = train_index[:development_count]
    tail = train_index[development_count:]
    midpoint = len(tail) // 2
    gamma_calibration = tail[:midpoint]
    residual_bridge = tail[midpoint:]
    if min(len(development), len(gamma_calibration), len(residual_bridge)) < 100:
        raise RuntimeError("author-train subpartition is too small")

    structures: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        structures[branch] = _select_branch_structure(
            branch,
            representations[branch],
            target_delta,
            development,
            config,
        )

    calibration_predictions, calibration_contracts = _branch_predictions(
        representations,
        target_delta,
        development,
        gamma_calibration,
        structures,
        config,
    )
    gamma_state = fit_simplex_assembly(
        calibration_predictions,
        target_delta[gamma_calibration],
        ridge=float(config["Gamma_CT"]["ridge"]),
    )

    oof_frames = []
    outer_folds = _expanding_folds(
        development, int(config["selection"]["gamma_oof_expanding_folds"])
    )
    for fold, (fit_index, evaluation_index) in enumerate(outer_folds):
        branch_predictions, _ = _branch_predictions(
            representations,
            target_delta,
            fit_index,
            evaluation_index,
            structures,
            config,
        )
        gamma = predict_simplex_assembly(gamma_state, branch_predictions)
        block = pd.DataFrame(
            {
                "origin": evaluation_index,
                "target_delta": target_delta[evaluation_index],
                "persistence": persistence[evaluation_index],
                "gamma_oof": gamma,
                "oof_fold": fold,
            }
        )
        for branch in gamma_state.branch_names:
            block[f"branch_{branch}_oof"] = branch_predictions[branch]
        oof_frames.append(block)
    gamma_oof = pd.concat(oof_frames, ignore_index=True).sort_values("origin").reset_index(drop=True)

    w_oof, selected_w, w_contract, w_audit = _select_and_materialize_w(gamma_oof, config)
    selected_a, a_audit = _select_a(w_oof, config)

    tail_predictions, tail_contracts = _branch_predictions(
        representations,
        target_delta,
        development,
        tail,
        structures,
        config,
    )
    tail_gamma = predict_simplex_assembly(gamma_state, tail_predictions)
    tail_w = predict_w_correction(tail_gamma, w_contract)
    tail_table = pd.DataFrame(
        {
            "origin": tail,
            "target_delta": target_delta[tail],
            "gamma": tail_gamma,
            "delta_w": tail_w,
        }
    )
    tail_table["residual"] = tail_table["target_delta"] - tail_table["gamma"] - tail_table["delta_w"]
    bridge_set = set(int(value) for value in residual_bridge)
    bridge = tail_table[tail_table["origin"].isin(bridge_set)].copy()

    a_train = w_oof[["origin", "target_delta", "gamma_oof", "delta_w_oof", "gamma_w_oof"]].copy()
    a_train = a_train.rename(columns={"gamma_oof": "gamma", "delta_w_oof": "delta_w"})
    a_train["residual"] = a_train["target_delta"] - a_train["gamma"] - a_train["delta_w"]
    a_train = pd.concat(
        [a_train[["origin", "target_delta", "gamma", "delta_w", "residual"]], bridge[["origin", "target_delta", "gamma", "delta_w", "residual"]]],
        ignore_index=True,
    ).sort_values("origin").drop_duplicates("origin", keep="last").reset_index(drop=True)

    final_branch_predictions, final_branch_contracts = _branch_predictions(
        representations,
        target_delta,
        train_index,
        test_index,
        structures,
        config,
    )
    gamma_test = predict_simplex_assembly(gamma_state, final_branch_predictions)
    w_test = predict_w_correction(gamma_test, w_contract)
    test_residual = target_delta[test_index] - gamma_test - w_test
    test_for_a = pd.DataFrame(
        {
            "origin": test_index,
            "target_delta": target_delta[test_index],
            "gamma": gamma_test,
            "delta_w": w_test,
            "residual": test_residual,
        }
    )
    a_test, a_contract = _fit_final_a(selected_a, a_train, test_for_a, config)

    prediction_k = persistence[test_index] + gamma_test
    prediction_kw = persistence[test_index] + gamma_test + w_test
    prediction_kwa = persistence[test_index] + gamma_test + w_test + a_test
    target = y[test_index]

    gamma_weights = {
        branch: float(weight)
        for branch, weight in zip(gamma_state.branch_names, gamma_state.weights, strict=True)
    }
    gamma_weights["PERSISTENCE"] = float(gamma_state.persistence_weight)
    result = {
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "dataset_shape": list(data.shape),
        "target": config["target"],
        "information_set": config["information_set"],
        "target_transform": config["target_transform"],
        "n_train": int(len(train_index)),
        "n_development": int(len(development)),
        "n_gamma_calibration": int(len(gamma_calibration)),
        "n_residual_bridge": int(len(residual_bridge)),
        "n_test": int(len(test_index)),
        "author_boundary": int(train_size),
        "temporal_representation": config["temporal_representation"],
        "branch_structures": {branch: structures[branch]["audit"] for branch in BRANCHES},
        "gamma_ct": {
            "branch_names": list(gamma_state.branch_names),
            "weights": gamma_weights,
            "ridge": float(gamma_state.ridge),
            "iterations": int(gamma_state.iterations),
            "fit_partition": "author_train_tail_gamma_calibration_only",
            "calibration_branch_contracts": calibration_contracts,
        },
        "W": {
            "selected": str(selected_w),
            "active": selected_w != IDENTITY,
            "audit": w_audit,
        },
        "A": {
            "selected": str(selected_a),
            "active": selected_a != A_EXACT_ZERO,
            "audit": a_audit,
            "final_contract": a_contract,
        },
        "nested_routes": {
            "K_GAMMA": _metrics(target, prediction_k),
            "K_GAMMA_W": _metrics(target, prediction_kw),
            "K_GAMMA_W_A": _metrics(target, prediction_kwa),
            "PERSISTENCE": _metrics(target, persistence[test_index]),
        },
        "primary_model": "PRISM_V2_2_BETA_FULL_KWA_DYNAMIC",
        "primary_metrics": _metrics(target, prediction_kwa),
        "test_target_used_for_selection": False,
        "future_inputs_used": False,
        "historical_target_usage": "y[t-1] persistence plus mature residuals only",
        "maturity_rule": config["A"]["maturity_rule"],
        "fairness_note": config["fairness_note"],
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "elapsed_seconds": time.time() - started,
        "contracts": {
            "tail_branch_contracts": tail_contracts,
            "final_branch_contracts": final_branch_contracts,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full PRISM v2.2(beta) K/C -> Gamma_CT -> W -> A on the RTA-TCN SRU author split")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "prism_v22_sru_rta_tcn_full.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data.resolve(), args.config.resolve(), args.output.resolve())
    summary = {
        "status": result["status"],
        "primary_model": result["primary_model"],
        "primary_metrics": result["primary_metrics"],
        "nested_routes": result["nested_routes"],
        "gamma_weights": result["gamma_ct"]["weights"],
        "W_selected": result["W"]["selected"],
        "A_selected": result["A"]["selected"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
