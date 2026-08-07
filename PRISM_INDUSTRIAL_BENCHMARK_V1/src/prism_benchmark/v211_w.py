from __future__ import annotations

import json
import hashlib
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .cpu_data import ViewSpec, inner_folds, load_samples, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_basis import natural_cubic_columns
from .v2_c import _ridge_fit, fit_physical_features
from .v2_k import _cap
from .v2_numerics import difference_penalty, solve_certified
from .v2_w import _ispline_fixed, fit_w_candidate as fit_hard_projection_replay
from .v21_selection import guarded_local_one_se_select
from .v21_w import _psd_root, _standardized_columns, soft_overlap_penalty
from .v211_config import load_v211_configs
from .v2_runtime import ordered_fork_map, run_parallel
from .v211_k import load_active_channels
from .v211_selection import numerical_contract_passes


IDENTITY = "IDENTITY_CORRECTION"
MONOTONE = "MONOTONE_I_SPLINE_CORRECTION"
NATURAL_CUBIC = "NATURAL_CUBIC_CORRECTION"
W_FAMILIES = (IDENTITY, MONOTONE, NATURAL_CUBIC)
BEST_ACTIVE_K = "BEST_ACTIVE_K_CHANNEL"
W_INNER_WORKERS_ENV = "PRISM_V211_W_INNER_WORKERS"
_W_CANDIDATE_CONTEXT: tuple[list[Any], list[tuple[Any, ...]], list[dict[str, Any]]] | None = None
_W_FOLD_CONTEXT: tuple[
    Path,
    ViewSpec,
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[tuple[np.ndarray, np.ndarray]],
] | None = None


def _w_inner_workers() -> int:
    raw = os.environ.get(W_INNER_WORKERS_ENV, "1")
    try:
        workers = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{W_INNER_WORKERS_ENV} must be an integer") from error
    cpu_count = os.cpu_count() or 1
    if workers < 1 or workers > cpu_count:
        raise RuntimeError(
            f"{W_INNER_WORKERS_ENV} must be within [1, {cpu_count}]"
        )
    return workers


def _row_id_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in frame["base_origin_id"].astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _candidate_descriptor(candidate: Any) -> dict[str, Any]:
    if candidate == IDENTITY:
        return {"family": IDENTITY}
    family, knots, smoothness, mu, direction = candidate
    return {
        "family": str(family),
        "knot_count": int(knots),
        "smoothness": float(smoothness),
        "soft_overlap_mu": float(mu),
        "direction": int(direction),
    }


@dataclass(frozen=True)
class WDesign:
    train: np.ndarray
    evaluation: np.ndarray
    metadata: dict[str, Any]


def _relative_scale_floor(values: np.ndarray, multiplier: float = 64.0) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    maximum = float(np.max(np.abs(array), initial=0.0))
    return float(multiplier) * np.finfo(np.float64).eps * max(1.0, maximum)


def latent_fold_usable(
    latent: np.ndarray,
    *,
    minimum_distinct_values: int = 20,
    minimum_rank: int = 2,
    scale_floor_multiplier: float = 64.0,
) -> dict[str, Any]:
    """Apply the registered scale-independent W fold eligibility test."""
    values = np.asarray(latent, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    finite_values = values[finite]
    distinct = int(len(np.unique(finite_values)))
    standard_deviation = (
        float(np.std(finite_values, dtype=np.float64))
        if len(finite_values)
        else float("nan")
    )
    scale_floor = (
        _relative_scale_floor(finite_values, scale_floor_multiplier)
        if len(finite_values)
        else float("inf")
    )
    rank = 0
    if len(finite_values):
        rank = int(
            np.linalg.matrix_rank(
                np.column_stack(
                    [np.ones(len(finite_values), dtype=np.float64), finite_values]
                )
            )
        )
    checks = {
        "all_finite": bool(finite.all()),
        "distinct_values": distinct >= int(minimum_distinct_values),
        "intercept_latent_rank": rank >= int(minimum_rank),
        "relative_scale": bool(
            np.isfinite(standard_deviation) and standard_deviation > scale_floor
        ),
    }
    passed = all(checks.values())
    return {
        "status": "USABLE" if passed else "NOT_APPLICABLE",
        "pass": bool(passed),
        "checks": checks,
        "rows": len(values),
        "finite_values": int(finite.sum()),
        "distinct_finite_values": distinct,
        "intercept_latent_rank": rank,
        "standard_deviation": standard_deviation,
        "relative_scale_floor": scale_floor,
        "absolute_variance_gate_used": False,
    }


def _internal_knots(values: np.ndarray, knot_count: int) -> np.ndarray:
    if knot_count < 1:
        raise ValueError("knot_count must be positive")
    return np.unique(
        np.quantile(values, np.arange(1, knot_count + 1) / (knot_count + 1))
    )


def build_w_design(
    train_latent: np.ndarray,
    evaluation_latent: np.ndarray,
    family: str,
    knot_count: int,
) -> WDesign:
    """Build W on a relative-scale-standardized latent, including tiny signals."""
    if family not in {MONOTONE, NATURAL_CUBIC}:
        raise ValueError(f"W basis is unavailable for {family!r}")
    train_latent = np.asarray(train_latent, dtype=np.float64).reshape(-1)
    evaluation_latent = np.asarray(evaluation_latent, dtype=np.float64).reshape(-1)
    if not np.isfinite(train_latent).all() or not np.isfinite(evaluation_latent).all():
        raise ValueError("W latent contains non-finite values")
    mean = float(np.mean(train_latent, dtype=np.float64))
    scale = float(np.std(train_latent, dtype=np.float64))
    floor = _relative_scale_floor(train_latent)
    if not scale > floor:
        raise ValueError("W latent has no numerically resolvable train-fold variation")
    train = (train_latent - mean) / scale
    evaluation = (evaluation_latent - mean) / scale
    lower = float(np.min(train))
    upper = float(np.max(train))
    if not upper > lower:
        raise ValueError("W latent has no train-fold variation")
    knots = _internal_knots(train, knot_count)
    if family == MONOTONE:
        train_design = _ispline_fixed(train, knots, lower, upper)
        evaluation_design = _ispline_fixed(evaluation, knots, lower, upper)
    else:
        train_design = natural_cubic_columns(train, knots)[:, 2:]
        evaluation_design = natural_cubic_columns(evaluation, knots)[:, 2:]
    if train_design.shape[1] == 0:
        raise ValueError("W basis is empty")
    if not np.isfinite(train_design).all() or not np.isfinite(evaluation_design).all():
        raise ValueError("W basis contains non-finite values")
    metadata = {
        "family": family,
        "mean": mean,
        "scale": scale,
        "relative_scale_floor": floor,
        "knots": knots.tolist(),
        "train_min": lower,
        "train_max": upper,
        "hard_feature_residualization": False,
        "basis_contract": "UNPROJECTED_STAGEWISE_RESIDUAL_CORRECTION",
    }
    return WDesign(train_design, evaluation_design, metadata)


def fit_w_correction(
    train_latent: np.ndarray,
    residual_target: np.ndarray,
    evaluation_latent: np.ndarray,
    *,
    family: str,
    knot_count: int = 0,
    smoothness: float = 0.0,
    mu: float = 0.0,
    upstream_predictions: np.ndarray | None = None,
    direction: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_latent = np.asarray(train_latent, dtype=np.float64).reshape(-1)
    residual_target = np.asarray(residual_target, dtype=np.float64).reshape(-1)
    evaluation_latent = np.asarray(evaluation_latent, dtype=np.float64).reshape(-1)
    if residual_target.shape != train_latent.shape:
        raise ValueError("W target/latent row mismatch")
    if family == IDENTITY:
        return np.zeros(len(evaluation_latent), dtype=np.float64), {
            "family": IDENTITY,
            "coefficient": [],
            "parameter_count": 0,
            "effective_df": 0.0,
            "hard_feature_residualization": False,
            "soft_overlap_mu": 0.0,
            "numerical_certificate": {"status": "EXACT_ZERO"},
        }
    if family not in {MONOTONE, NATURAL_CUBIC}:
        raise ValueError(f"unknown W family: {family}")
    if smoothness < 0 or mu < 0:
        raise ValueError("W penalties must be nonnegative")
    if direction not in {-1, 1}:
        raise ValueError("monotone direction must be -1 or +1")
    basis = build_w_design(train_latent, evaluation_latent, family, knot_count)
    sign = direction if family == MONOTONE else 1
    signed_train = basis.train * sign
    signed_evaluation = basis.evaluation * sign
    penalty = float(smoothness) * difference_penalty(signed_train.shape[1])
    penalty += soft_overlap_penalty(
        signed_train, upstream_predictions, float(mu)
    )
    if family == MONOTONE:
        root = _psd_root(penalty)
        augmented_x = np.vstack([signed_train, root])
        augmented_y = np.concatenate(
            [residual_target, np.zeros(len(root), dtype=np.float64)]
        )
        fit = optimize.lsq_linear(
            augmented_x,
            augmented_y,
            bounds=(0.0, np.inf),
            lsmr_tol=1e-12,
        )
        coefficient = np.asarray(fit.x, dtype=np.float64)
        certificate = {
            "status": "PASS" if fit.success else "NUMERICALLY_INVALID",
            "solver": "NONNEGATIVE_LSQ",
            "optimality": float(fit.optimality),
        }
        effective_df = float(np.count_nonzero(coefficient > 1e-12))
    else:
        coefficient, numeric = solve_certified(
            signed_train, residual_target, penalty
        )
        certificate = numeric.to_json()
        gram = signed_train.T @ signed_train
        effective_df = float(
            np.trace(np.linalg.pinv(gram + penalty, rcond=1e-12) @ gram)
        )
    correction = signed_evaluation @ coefficient
    train_correction = signed_train @ coefficient
    overlap = 0.0
    upstream = (
        _standardized_columns(upstream_predictions)
        if upstream_predictions is not None
        else np.empty((len(train_latent), 0), dtype=np.float64)
    )
    if upstream.shape[1]:
        standardized_correction = _standardized_columns(train_correction)
        if standardized_correction.shape[1]:
            overlap = float(
                np.linalg.norm(
                    upstream.T @ standardized_correction / len(train_latent)
                )
            )
    contract = {
        "family": family,
        "knot_count": int(knot_count),
        "smoothness": float(smoothness),
        "soft_overlap_mu": float(mu),
        "direction": int(direction),
        "coefficient": coefficient.tolist(),
        "basis": basis.metadata,
        "effective_df": effective_df,
        "parameter_count": int(np.count_nonzero(np.abs(coefficient) > 1e-12)),
        "hard_feature_residualization": False,
        "soft_overlap_norm": overlap,
        "numerical_certificate": certificate,
    }
    return correction, contract


def predict_w_correction(latent: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    values = np.asarray(latent, dtype=np.float64).reshape(-1)
    if contract["family"] == IDENTITY:
        return np.zeros(len(values), dtype=np.float64)
    metadata = contract["basis"]
    standardized = (values - float(metadata["mean"])) / float(metadata["scale"])
    knots = np.asarray(metadata["knots"], dtype=np.float64)
    if contract["family"] == MONOTONE:
        design = _ispline_fixed(
            standardized,
            knots,
            float(metadata["train_min"]),
            float(metadata["train_max"]),
        ) * int(contract["direction"])
    elif contract["family"] == NATURAL_CUBIC:
        design = natural_cubic_columns(standardized, knots)[:, 2:]
    else:
        raise ValueError(f"unknown W contract family: {contract['family']}")
    return design @ np.asarray(contract["coefficient"], dtype=np.float64)


def w_support_derivative_audit(
    latent: np.ndarray,
    contract: dict[str, Any],
    *,
    grid_points: int = 2048,
) -> dict[str, Any]:
    values = np.asarray(latent, dtype=np.float64).reshape(-1)
    if contract["family"] == IDENTITY:
        return {
            "status": "PASS",
            "family": IDENTITY,
            "rows": len(values),
            "below_support_fraction": 0.0,
            "above_support_fraction": 0.0,
            "maximum_absolute_derivative": 0.0,
            "finite_derivative": True,
            "monotonicity_violation_fraction": 0.0,
            "grid_points": 0,
        }
    metadata = contract["basis"]
    standardized = (
        values - float(metadata["mean"])
    ) / float(metadata["scale"])
    lower = float(metadata["train_min"])
    upper = float(metadata["train_max"])
    grid_standardized = np.linspace(lower, upper, int(grid_points))
    grid = grid_standardized * float(metadata["scale"]) + float(metadata["mean"])
    correction = predict_w_correction(grid, contract)
    derivative = np.diff(correction) / np.diff(grid)
    finite = bool(np.isfinite(derivative).all())
    direction = int(contract.get("direction", 1))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(derivative), initial=0.0))
    )
    violation = (
        float(np.mean(direction * derivative < -tolerance))
        if contract["family"] == MONOTONE and len(derivative)
        else 0.0
    )
    return {
        "status": "PASS" if finite and violation == 0.0 else "NUMERICAL_WARNING",
        "family": contract["family"],
        "rows": len(values),
        "below_support_fraction": float(np.mean(standardized < lower)),
        "above_support_fraction": float(np.mean(standardized > upper)),
        "maximum_absolute_derivative": float(
            np.max(np.abs(derivative), initial=0.0)
        ),
        "finite_derivative": finite,
        "monotonicity_violation_fraction": violation,
        "grid_points": int(grid_points),
    }


def _load_c_result(output: Path, view: ViewSpec) -> dict[str, Any]:
    path = (
        output
        / "DEVELOPMENT"
        / "C"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") not in {"PASS", "C_INPUT_PATH_COLLAPSE_BUG"}:
        raise RuntimeError(f"E2R C prerequisite is incomplete: {path}")
    return value


def _w_candidates(
    v211: dict[str, Any],
    v21: dict[str, Any],
    v2: dict[str, Any],
    direction: int,
    monotone: bool,
) -> list[Any]:
    candidates: list[Any] = [IDENTITY]
    smoothness = [float(value) for value in v2["W_module"]["smoothness_penalties"]]
    mus = [float(value) for value in v211["W"]["soft_overlap_mu"]]
    include_monotone = bool(
        monotone
        or MONOTONE in set(v211.get("W", {}).get("candidates", ()))
    )
    if include_monotone:
        candidates.extend(
            (MONOTONE, int(knots), penalty, mu, direction)
            for knots in v21["W"]["monotone_knots"]
            for penalty in smoothness
            for mu in mus
        )
    candidates.extend(
        (NATURAL_CUBIC, int(knots), penalty, mu, 1)
        for knots in v21["W"]["natural_cubic_knots"]
        for penalty in smoothness
        for mu in mus
    )
    return candidates


def w_candidate_scope(
    v211: dict[str, Any],
    v21: dict[str, Any],
    v2: dict[str, Any],
    direction: int,
    monotone: bool,
    *,
    input_path_preserved: bool,
    k_exact_zero: bool,
    fold_train_latents: list[np.ndarray],
) -> tuple[list[Any], dict[str, Any]]:
    rule = v211["W"]
    audits = [
        latent_fold_usable(
            values,
            minimum_distinct_values=int(
                rule["minimum_distinct_latent_values_per_fold"]
            ),
            minimum_rank=int(rule["minimum_rank_of_intercept_and_latent"]),
            scale_floor_multiplier=float(
                rule["relative_scale_floor_multiplier_float64_eps"]
            ),
        )
        for values in fold_train_latents
    ]
    usable = sum(bool(audit["pass"]) for audit in audits)
    minimum = int(rule["minimum_usable_folds"])
    if k_exact_zero:
        reason = "K_EXACT_ZERO"
        nonlinear_allowed = False
    elif usable < minimum:
        reason = "INSUFFICIENT_USABLE_LATENT_FOLDS"
        nonlinear_allowed = False
    elif not input_path_preserved:
        reason = "C_INPUT_PATH_NOT_PRESERVED_DIAGNOSTIC_ONLY"
        nonlinear_allowed = True
    else:
        reason = "NONLINEAR_W_ELIGIBLE"
        nonlinear_allowed = True
    candidates = (
        _w_candidates(v211, v21, v2, direction, monotone)
        if nonlinear_allowed
        else [IDENTITY]
    )
    return candidates, {
        "input_path_preserved": bool(input_path_preserved),
        "k_exact_zero": bool(k_exact_zero),
        "fold_usability": audits,
        "usable_fold_count": int(usable),
        "minimum_usable_folds": minimum,
        "nonlinear_candidates_allowed": bool(nonlinear_allowed),
        "formal_nonlinear_interpretation_allowed": bool(
            nonlinear_allowed and input_path_preserved
        ),
        "identity_forced": not nonlinear_allowed,
        "raw_absolute_variance_gate_used": False,
        "reason": reason,
    }


def _fit_c_routed(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    features = fit_physical_features(
        shared,
        view,
        fit,
        evaluation,
        active,
        config,
        fit_split=fit_split,
        evaluation_split=evaluation_split,
    )
    target = fit["y_true"].to_numpy(dtype=np.float64)
    family = str(c_result["selected_family"])
    if family == "K_EXACT_ZERO" or not active:
        intercept = float(np.mean(target, dtype=np.float64))
        train_prediction = np.full(len(fit), intercept, dtype=np.float64)
        evaluation_prediction = np.full(len(evaluation), intercept, dtype=np.float64)
        contract = {
            "family": "K_EXACT_ZERO",
            "coefficient": [],
            "intercept": intercept,
            "parameter_count": 1,
            "numerical_certificate": {"status": "EXACT_ZERO"},
        }
    elif family == BEST_ACTIVE_K:
        channel = str(c_result["best_active_k_channel"])
        index = features["channels"].index(channel)
        train_prediction = features["compressed_train"][:, index].copy()
        evaluation_prediction = features["compressed_evaluation"][:, index].copy()
        contract = {
            "family": BEST_ACTIVE_K,
            "channel": channel,
            "coefficient": [1.0],
            "intercept": 0.0,
            "parameter_count": 1,
            "numerical_certificate": {"status": "PASS"},
        }
    else:
        key = "joint" if family == "ADDITIVE_JOINT_BASIS" else "compressed"
        train_x = features[f"{key}_train"]
        evaluation_x = features[f"{key}_evaluation"]
        if train_x.shape[1] == 0:
            raise RuntimeError("active K route produced an empty C design")
        evaluation_prediction, contract = _ridge_fit(
            train_x,
            target,
            evaluation_x,
            float(c_result["selected_alpha"]),
        )
        train_prediction = (
            (train_x - np.asarray(contract["mean"], dtype=np.float64))
            / np.asarray(contract["scale"], dtype=np.float64)
        ) @ np.asarray(contract["coefficient"], dtype=np.float64) + float(
            contract["intercept"]
        )
        contract = {"family": family, **contract}
    return (
        train_prediction,
        evaluation_prediction,
        features["compressed_train"],
        features["compressed_evaluation"],
        contract,
    )


def _fit_registered_w(
    candidate: Any,
    train_latent: np.ndarray,
    train_target: np.ndarray,
    evaluation_latent: np.ndarray,
    upstream_predictions: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate == IDENTITY:
        return fit_w_correction(
            train_latent,
            train_target - train_latent,
            evaluation_latent,
            family=IDENTITY,
        )
    family, knots, smoothness, mu, direction = candidate
    return fit_w_correction(
        train_latent,
        train_target - train_latent,
        evaluation_latent,
        family=family,
        knot_count=int(knots),
        smoothness=float(smoothness),
        mu=float(mu),
        upstream_predictions=upstream_predictions,
        direction=int(direction),
    )


def _evaluate_w_candidate(
    candidate: Any,
    fit_latent: np.ndarray,
    fit_target: np.ndarray,
    evaluation_latent: np.ndarray,
    fit_upstream: np.ndarray,
    evaluation_target: np.ndarray,
    fold_index: int,
    fold_usable: bool,
) -> tuple[float, dict[str, Any]]:
    if candidate != IDENTITY and not fold_usable:
        return float("nan"), {"status": "NOT_APPLICABLE", "fold": fold_index}
    try:
        correction, contract = _fit_registered_w(
            candidate,
            fit_latent,
            fit_target,
            evaluation_latent,
            fit_upstream,
        )
        valid = bool(
            np.isfinite(correction).all() and numerical_contract_passes(contract)
        )
        return (
            mse(evaluation_target, evaluation_latent + correction)
            if valid
            else float("nan"),
            {
                "status": "PASS" if valid else "NUMERICALLY_INVALID",
                "fold": fold_index,
                "certificate": contract.get("numerical_certificate", {}),
            },
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        return float("nan"), {
            "status": "NOT_APPLICABLE",
            "fold": fold_index,
            "error": str(error),
        }


def _evaluate_w_indexed(fold_index: int, candidate_index: int) -> tuple[float, dict[str, Any]]:
    if _W_CANDIDATE_CONTEXT is None:
        raise RuntimeError("W candidate context was not initialized before fork")
    candidates, fold_inputs, fold_audits = _W_CANDIDATE_CONTEXT
    item = fold_inputs[fold_index]
    return _evaluate_w_candidate(
        candidates[candidate_index],
        item[2],
        item[6],
        item[3],
        item[4],
        item[7],
        fold_index,
        bool(fold_audits[fold_index]["pass"]),
    )


def _prepare_w_fold_indexed(
    fold_index: int,
) -> tuple[tuple[Any, ...], float, dict[str, Any]]:
    if _W_FOLD_CONTEXT is None:
        raise RuntimeError("W fold context was not initialized before fork")
    shared, view, train, active, v2, c_result, rule, folds = _W_FOLD_CONTEXT
    fit_index, evaluation_index = folds[fold_index]
    fit = _cap(train.iloc[fit_index], int(v2["row_caps"]["wiener_fit"]))
    evaluation = _cap(
        train.iloc[evaluation_index],
        int(v2["row_caps"]["validation_selection_per_fold"]),
    )
    (
        fit_latent,
        evaluation_latent,
        fit_upstream,
        evaluation_upstream,
        _,
    ) = _fit_c_routed(
        shared,
        view,
        fit,
        evaluation,
        active,
        v2,
        c_result,
        fit_split="train",
        evaluation_split="train",
    )
    fit_target = fit["y_true"].to_numpy(dtype=np.float64)
    latent_audit = latent_fold_usable(
        fit_latent,
        minimum_distinct_values=int(
            rule["minimum_distinct_latent_values_per_fold"]
        ),
        minimum_rank=int(rule["minimum_rank_of_intercept_and_latent"]),
        scale_floor_multiplier=float(
            rule["relative_scale_floor_multiplier_float64_eps"]
        ),
    )
    correlation = (
        float(stats.spearmanr(fit_latent, fit_target - fit_latent).statistic)
        if latent_audit["pass"]
        else float("nan")
    )
    fold_input = (
        fold_index,
        evaluation,
        fit_latent,
        evaluation_latent,
        fit_upstream,
        evaluation_upstream,
        fit_target,
        evaluation["y_true"].to_numpy(dtype=np.float64),
    )
    return fold_input, correlation, latent_audit


def run_w_view(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    protocol: str = "sru",
) -> dict[str, Any]:
    global _W_FOLD_CONTEXT, _W_CANDIDATE_CONTEXT
    started = time.time()
    destination = output / "DEVELOPMENT" / "W" / view.head.head_id / view.proxy_policy
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v211, v21, v2 = load_v211_configs(project, protocol=protocol)
        inner_workers = _w_inner_workers()
        c_result = _load_c_result(output, view)
        active = load_active_channels(output, view)
        frozen_channels = set(c_result.get("active_channels", ()))
        active = [item for item in active if item.get("channel") in frozen_channels]
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        fold_inputs: list[tuple[Any, ...]] = []
        correlations: list[float] = []
        fold_latent_audits: list[dict[str, Any]] = []
        rule = v211["W"]
        fold_indices = list(
            inner_folds(train, int(v21["selection"]["inner_folds"]))
        )
        _W_FOLD_CONTEXT = (
            shared,
            view,
            train,
            active,
            v2,
            c_result,
            rule,
            fold_indices,
        )
        try:
            prepared_folds = ordered_fork_map(
                _prepare_w_fold_indexed,
                [(fold_index,) for fold_index in range(len(fold_indices))],
                min(inner_workers, len(fold_indices)),
                label="PRISM_V211_METRO_M3_W_FOLDS",
            )
        finally:
            _W_FOLD_CONTEXT = None
        for fold_input, correlation, latent_audit in prepared_folds:
            fold_inputs.append(fold_input)
            correlations.append(correlation)
            fold_latent_audits.append(latent_audit)
        finite = np.asarray(
            [value for value in correlations if np.isfinite(value)], dtype=np.float64
        )
        direction = 1 if len(finite) == 0 or float(np.median(finite)) >= 0 else -1
        direction_rule = v2["W_module"]["monotone_direction"]
        monotone = bool(
            len(finite) >= int(rule["minimum_usable_folds"])
            and abs(float(np.median(finite)))
            >= float(direction_rule["absolute_correlation_min"])
            and float(np.mean(np.sign(finite) == direction))
            >= float(direction_rule["same_sign_fold_fraction_min"])
        )
        input_path_preserved = bool(
            c_result.get("input_path_preservation", {}).get("pass", False)
        )
        k_exact_zero = not active or c_result.get("selected_family") == "K_EXACT_ZERO"
        candidates, candidate_scope = w_candidate_scope(
            v211,
            v21,
            v2,
            direction,
            monotone,
            input_path_preserved=input_path_preserved,
            k_exact_zero=k_exact_zero,
            fold_train_latents=[item[2] for item in fold_inputs],
        )
        losses = {candidate: [] for candidate in candidates}
        candidate_numeric_audit: dict[str, list[dict[str, Any]]] = {
            str(candidate): [] for candidate in candidates
        }
        _W_CANDIDATE_CONTEXT = (candidates, fold_inputs, fold_latent_audits)
        try:
            indexed_jobs = [
                (fold_index, candidate_index)
                for fold_index in range(len(fold_inputs))
                for candidate_index in range(len(candidates))
            ]
            indexed_results = ordered_fork_map(
                _evaluate_w_indexed,
                indexed_jobs,
                inner_workers,
                label="PRISM_V211_METRO_M3_W_INNER",
            )
        finally:
            _W_CANDIDATE_CONTEXT = None
        for (fold_index, candidate_index), (loss, audit) in zip(
            indexed_jobs, indexed_results, strict=True
        ):
            candidate = candidates[candidate_index]
            losses[candidate].append(loss)
            candidate_numeric_audit[str(candidate)].append(audit)

        def complexity(candidate: Any) -> tuple[Any, ...]:
            if candidate == IDENTITY:
                return (0,)
            family, knots, smoothness, mu, _ = candidate
            return (
                1,
                0 if family == MONOTONE else 1,
                int(knots),
                -float(smoothness),
                -float(mu),
            )

        minimum_usable = int(rule["minimum_usable_folds"])
        applicable_nonlinear = [
            candidate
            for candidate in candidates
            if candidate != IDENTITY
            and int(np.count_nonzero(np.isfinite(losses[candidate])))
            >= minimum_usable
        ]
        joint_basis_candidate = (
            min(
                applicable_nonlinear,
                key=lambda candidate: (
                    float(
                        np.mean(
                            np.asarray(losses[candidate], dtype=np.float64)[
                                np.isfinite(losses[candidate])
                            ],
                            dtype=np.float64,
                        )
                    ),
                    complexity(candidate),
                ),
            )
            if applicable_nonlinear
            else None
        )

        selection = guarded_local_one_se_select(
            losses,
            complexity,
            neutral=IDENTITY,
            minimum_relative_improvement=float(
                rule["minimum_relative_mse_improvement"]
            ),
            minimum_positive_fraction=float(rule["minimum_positive_fold_fraction"]),
            minimum_usable_folds=int(rule["minimum_usable_folds"]),
        )
        diagnostic_selected = selection.final_selected_candidate
        formal_allowed = bool(
            candidate_scope["formal_nonlinear_interpretation_allowed"]
        )
        selected = diagnostic_selected if formal_allowed else IDENTITY
        # The formal PF activation decision remains exactly the guarded v2.1.1
        # selection above.  The Metro transfer audit also pre-registers a W-on
        # ablation when PF selects identity.  Freeze that ablation from the
        # development-loss-best applicable non-identity construction; it is
        # never selection eligible and never changes ``selected``.
        pf_ablation_candidate = (
            selected if selected != IDENTITY else joint_basis_candidate
        )
        if pf_ablation_candidate is None:
            raise RuntimeError(
                "no applicable non-identity W construction for the registered ablation"
            )
        selection_status = (
            "W_RESCUE_DIAGNOSTIC_ONLY"
            if not input_path_preserved and not k_exact_zero
            else "W_FORMAL_SELECTION"
        )
        oof_frames = []
        for item in fold_inputs:
            (
                fold,
                evaluation,
                fit_latent,
                evaluation_latent,
                fit_upstream,
                evaluation_upstream,
                fit_target,
                _,
            ) = item
            correction, _ = _fit_registered_w(
                selected,
                fit_latent,
                fit_target,
                evaluation_latent,
                fit_upstream,
            )
            if pf_ablation_candidate == selected:
                ablation_correction = correction.copy()
            else:
                ablation_correction, _ = _fit_registered_w(
                    pf_ablation_candidate,
                    fit_latent,
                    fit_target,
                    evaluation_latent,
                    fit_upstream,
                )
            frame = evaluation[
                [
                    "base_origin_id",
                    "view_sample_id",
                    "entity_id",
                    "origin",
                    "latest_available_target_index",
                    "y_true",
                ]
            ].copy()
            frame["physical_oof"] = evaluation_latent
            frame["delta_w_oof"] = correction
            frame["physical_w_oof"] = evaluation_latent + correction
            frame["delta_w_ablation_oof"] = ablation_correction
            frame["physical_w_ablation_oof"] = (
                evaluation_latent + ablation_correction
            )
            frame["oof_fold"] = fold
            for index in range(evaluation_upstream.shape[1]):
                frame[f"k_channel_contribution_{index:03d}"] = evaluation_upstream[
                    :, index
                ]
            oof_frames.append(frame)
        oof = pd.concat(oof_frames, ignore_index=True)
        final_train = _cap(train, int(v2["row_caps"]["wiener_fit"]))
        (
            fit_latent,
            validation_latent,
            fit_upstream,
            _,
            c_contract,
        ) = _fit_c_routed(
            shared,
            view,
            final_train,
            validation,
            active,
            v2,
            c_result,
            fit_split="train",
            evaluation_split="validation",
        )
        correction, contract = _fit_registered_w(
            selected,
            fit_latent,
            final_train["y_true"].to_numpy(dtype=np.float64),
            validation_latent,
            fit_upstream,
        )
        if not numerical_contract_passes(contract):
            raise RuntimeError("selected W final refit failed its numerical certificate")
        if pf_ablation_candidate == selected:
            ablation_correction = correction.copy()
            ablation_contract = dict(contract)
        else:
            ablation_correction, ablation_contract = _fit_registered_w(
                pf_ablation_candidate,
                fit_latent,
                final_train["y_true"].to_numpy(dtype=np.float64),
                validation_latent,
                fit_upstream,
            )
        if not numerical_contract_passes(ablation_contract):
            raise RuntimeError("registered W ablation final refit failed its numerical certificate")
        selected_support_derivative_audit = w_support_derivative_audit(
            validation_latent, contract
        )
        ablation_support_derivative_audit = w_support_derivative_audit(
            validation_latent, ablation_contract
        )
        identity_correction, identity_contract = _fit_registered_w(
            IDENTITY,
            fit_latent,
            final_train["y_true"].to_numpy(dtype=np.float64),
            validation_latent,
            fit_upstream,
        )
        identity_prediction_error = float(
            np.max(np.abs(validation_latent + identity_correction - validation_latent), initial=0.0)
        )
        validation_target = validation["y_true"].to_numpy(dtype=np.float64)
        identity_residual_error = float(
            np.max(
                np.abs(
                    (validation_target - validation_latent - identity_correction)
                    - (validation_target - validation_latent)
                ),
                initial=0.0,
            )
        )
        joint_basis_contract = None
        if joint_basis_candidate is not None:
            _, joint_basis_fit = _fit_registered_w(
                joint_basis_candidate,
                fit_latent,
                final_train["y_true"].to_numpy(dtype=np.float64),
                validation_latent,
                fit_upstream,
            )
            if numerical_contract_passes(joint_basis_fit):
                joint_basis_contract = {
                    key: joint_basis_fit[key]
                    for key in (
                        "family",
                        "knot_count",
                        "smoothness",
                        "direction",
                        "basis",
                        "numerical_certificate",
                    )
                }
                joint_basis_contract.update(
                    {
                        "source": "BEST_MEAN_APPLICABLE_NONIDENTITY_W_BASIS",
                        "pf_activation_eligible": False,
                        "candidate": _candidate_descriptor(joint_basis_candidate),
                        "support_derivative_audit": w_support_derivative_audit(
                            validation_latent, joint_basis_fit
                        ),
                    }
                )
        if selected == IDENTITY:
            contract = {
                **contract,
                "identity_reason": candidate_scope["reason"],
                "input_path_preserved": input_path_preserved,
            }
        prediction = validation_latent + correction
        if selected == IDENTITY:
            replay_prediction = validation_latent.copy()
            replay_contract = {"family": "IDENTITY", "parameter_count": 0}
        else:
            family, knots, smoothness, _, selected_direction = selected
            legacy_family = "MONOTONE" if family == MONOTONE else "NATURAL_CUBIC"
            replay_prediction, replay_contract = fit_hard_projection_replay(
                fit_latent,
                final_train["y_true"].to_numpy(dtype=np.float64),
                validation_latent,
                legacy_family,
                int(knots),
                float(smoothness),
                int(selected_direction),
            )
        frame = validation[
            [
                "base_origin_id",
                "view_sample_id",
                "entity_id",
                "origin",
                "latest_available_target_index",
                "y_true",
            ]
        ].copy()
        frame["physical_latent"] = validation_latent
        frame["delta_w"] = correction
        frame["delta_w_ablation"] = ablation_correction
        frame["physical_w_ablation"] = validation_latent + ablation_correction
        frame["y_pred"] = prediction
        frame["model"] = "PRISM_V2_1_1_K_C_W"
        frame["dtype"] = "float64"
        oof_path = destination / "PHYSICAL_OOF.parquet"
        prediction_path = destination / "validation.parquet"
        oof.to_parquet(oof_path, index=False, compression="zstd")
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        final_loss = mse(frame["y_true"].to_numpy(dtype=np.float64), prediction)
        status = (
            "W_RESCUE_DIAGNOSTIC_ONLY"
            if selection_status == "W_RESCUE_DIAGNOSTIC_ONLY"
            else "PASS"
        )
        result = {
            "status": status,
            "stage": "E3R_W",
            "inner_candidate_workers": inner_workers,
            "inner_parallelism_scope": "ORDERED_INDEPENDENT_CANDIDATES_ONLY",
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "selected_candidate": str(selected),
            "diagnostic_selected_candidate": str(diagnostic_selected),
            "selection_status": selection_status,
            "w_contract": contract,
            "pf_ablation_w_candidate": _candidate_descriptor(
                pf_ablation_candidate
            ),
            "pf_ablation_w_contract": ablation_contract,
            "pf_ablation_selection_eligible": False,
            "selected_support_derivative_audit": selected_support_derivative_audit,
            "pf_ablation_support_derivative_audit": ablation_support_derivative_audit,
            "w_candidate_scope": candidate_scope,
            "usable_fold_count": candidate_scope["usable_fold_count"],
            "input_path_preservation": c_result.get("input_path_preservation", {}),
            "input_path_nonzero": input_path_preserved,
            "c_final_contract": c_contract,
            "selection": selection.to_json(),
            "candidate_fold_losses": {
                str(key): value for key, value in losses.items()
            },
            "candidate_numerical_audit": candidate_numeric_audit,
            "candidate_families_compared": sorted(
                {
                    _candidate_descriptor(candidate)["family"]
                    for candidate in candidates
                    if candidate == IDENTITY
                    or int(np.count_nonzero(np.isfinite(losses[candidate])))
                    >= minimum_usable
                }
            ),
            "candidate_descriptors": [
                _candidate_descriptor(candidate) for candidate in candidates
            ],
            "fold_row_id_hashes": [
                {
                    "fold": int(item[0]),
                    "row_id_sha256": _row_id_hash(item[1]),
                }
                for item in fold_inputs
            ],
            "nested_candidates_share_fold_mask_and_row_ids": True,
            "identity_equivalence": {
                "prediction_max_abs_error": identity_prediction_error,
                "residual_max_abs_error": identity_residual_error,
                "identity_contract": identity_contract,
                "pass": identity_prediction_error == 0.0
                and identity_residual_error == 0.0,
            },
            "joint_w_basis_contract": joint_basis_contract,
            "monotone_correlations": correlations,
            "monotone_applicable": monotone,
            "monotone_direction": direction,
            "hard_feature_residualization": False,
            "hard_projection_replay": {
                "selection_eligible": False,
                "split": "validation",
                "contract": replay_contract,
                "mse": mse(
                    validation["y_true"].to_numpy(dtype=np.float64),
                    replay_prediction,
                ),
            },
            "oof_path": str(oof_path.relative_to(output)),
            "oof_sha256": sha256_file(oof_path),
            "final_selected_candidate": str(selected),
            "final_selected_fold_losses": list(losses[selected]),
            "final_selected_prediction_path": str(prediction_path.relative_to(output)),
            "final_selected_contract": contract,
            "final_prediction_loss": final_loss,
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "row_cap_audit": {
                "cap_name": "wiener_fit",
                "cap": int(v2["row_caps"]["wiener_fit"]),
                "fit_rows": len(final_train),
                "validation_rows": len(validation),
                "fit_source": "train_only",
            },
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
        }
    except Exception as error:
        result = {
            "status": "SOLVER_FAILED_RETAINED",
            "stage": "E3R_W",
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "test_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    write_json(destination / "RESULT.json", result)
    return result


def _count_activated_w(results: list[dict[str, Any]]) -> int:
    return sum(
        item.get("status") == "PASS"
        and item.get("w_contract", {}).get("family") not in {None, IDENTITY}
        for item in results
    )


def run_e3r_w(shared: Path, project: Path, output: Path) -> dict[str, Any]:
    from .v21_views import sru_input_views

    views = sru_input_views(shared)
    results = run_parallel(
        run_w_view,
        [(shared, project, output, view) for view in views],
        int(os.environ.get("PRISM_V211_WORKERS", "8")),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "4")),
        label="PRISM_V211_E3R_W",
    )
    summary = {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in results)
        else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": "E3R_W",
        "views": len(results),
        "pass": sum(item["status"] == "PASS" for item in results),
        "activated": _count_activated_w(results),
        "rescue_diagnostic_only": sum(
            item["status"] == "W_RESCUE_DIAGNOSTIC_ONLY" for item in results
        ),
        "test_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / "W" / "SUMMARY.json", summary)
    return summary
