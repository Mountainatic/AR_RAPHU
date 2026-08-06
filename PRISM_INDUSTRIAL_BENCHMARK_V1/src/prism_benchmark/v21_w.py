from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats

from .cpu_data import ViewSpec, inner_folds, load_samples, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_basis import natural_cubic_columns
from .v2_c import _ridge_fit, fit_physical_features
from .v2_k import _cap
from .v2_numerics import difference_penalty, solve_certified
from .v2_w import _ispline_fixed, fit_w_candidate as fit_hard_projection_replay
from .v21_config import load_v21_and_v2_config
from .v21_k import load_active_channels
from .v21_selection import guarded_local_one_se_select


IDENTITY = "IDENTITY_CORRECTION"
MONOTONE = "MONOTONE_I_SPLINE_CORRECTION"
NATURAL_CUBIC = "NATURAL_CUBIC_CORRECTION"
W_FAMILIES = (IDENTITY, MONOTONE, NATURAL_CUBIC)


@dataclass(frozen=True)
class WDesign:
    train: np.ndarray
    evaluation: np.ndarray
    metadata: dict[str, Any]


def _internal_knots(values: np.ndarray, knot_count: int) -> np.ndarray:
    if knot_count < 1:
        raise ValueError("knot_count must be positive")
    return np.unique(np.quantile(values, np.arange(1, knot_count + 1) / (knot_count + 1)))


def build_w_design(
    train_latent: np.ndarray,
    evaluation_latent: np.ndarray,
    family: str,
    knot_count: int,
) -> WDesign:
    """Build the v2.1 W basis without hard feature residualization."""
    if family not in {MONOTONE, NATURAL_CUBIC}:
        raise ValueError(f"W basis is unavailable for {family!r}")
    train_latent = np.asarray(train_latent, dtype=np.float64).reshape(-1)
    evaluation_latent = np.asarray(evaluation_latent, dtype=np.float64).reshape(-1)
    mean = float(np.mean(train_latent, dtype=np.float64))
    scale = float(np.std(train_latent, dtype=np.float64))
    if scale * scale < 1e-12:
        scale = 1.0
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
    metadata = {
        "family": family,
        "mean": mean,
        "scale": scale,
        "knots": knots.tolist(),
        "train_min": lower,
        "train_max": upper,
        "hard_feature_residualization": False,
        "basis_contract": "UNPROJECTED_STAGEWISE_RESIDUAL_CORRECTION",
    }
    return WDesign(train_design, evaluation_design, metadata)


def _standardized_columns(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] == 0:
        return matrix.copy()
    centered = matrix - matrix.mean(axis=0, dtype=np.float64)
    scale = matrix.std(axis=0, dtype=np.float64)
    keep = scale * scale >= 1e-12
    if not np.any(keep):
        return np.empty((len(matrix), 0), dtype=np.float64)
    return centered[:, keep] / scale[keep]


def soft_overlap_penalty(
    design: np.ndarray,
    upstream_predictions: np.ndarray | None,
    mu: float,
) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    if mu < 0:
        raise ValueError("soft-overlap mu must be nonnegative")
    if mu == 0.0 or upstream_predictions is None:
        return np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    upstream = _standardized_columns(upstream_predictions)
    if upstream.shape[1] == 0:
        return np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    cross = upstream.T @ design / max(len(design), 1)
    return float(mu) * (cross.T @ cross)


def _psd_root(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(symmetric)
    keep = values > max(values.max(initial=0.0) * 1e-14, 1e-18)
    if not np.any(keep):
        return np.empty((0, len(matrix)), dtype=np.float64)
    return np.sqrt(values[keep])[:, None] * vectors[:, keep].T


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
    signed_train = basis.train * (direction if family == MONOTONE else 1)
    signed_evaluation = basis.evaluation * (direction if family == MONOTONE else 1)
    penalty = float(smoothness) * difference_penalty(signed_train.shape[1])
    penalty += soft_overlap_penalty(signed_train, upstream_predictions, float(mu))
    if family == MONOTONE:
        root = _psd_root(penalty)
        augmented_x = np.vstack([signed_train, root])
        augmented_y = np.concatenate([residual_target, np.zeros(len(root), dtype=np.float64)])
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
        coefficient, numeric = solve_certified(signed_train, residual_target, penalty)
        certificate = numeric.to_json()
        gram = signed_train.T @ signed_train
        effective_df = float(np.trace(np.linalg.pinv(gram + penalty, rcond=1e-12) @ gram))
    correction = signed_evaluation @ coefficient
    train_correction = signed_train @ coefficient
    overlap = 0.0
    upstream = _standardized_columns(upstream_predictions) if upstream_predictions is not None else np.empty((len(train_latent), 0))
    if upstream.shape[1]:
        standardized_correction = _standardized_columns(train_correction)
        if standardized_correction.shape[1]:
            overlap = float(np.linalg.norm(upstream.T @ standardized_correction / len(train_latent)))
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


def run_e3_w(shared: Path, project: Path, output: Path) -> dict[str, Any]:
    """Run the registered SRU W stage after E2 K/C has completed."""
    from .v21_views import sru_input_views

    results = [run_w_view(shared, project, output, view) for view in sru_input_views(shared)]
    summary = {
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": "E3_W",
        "views": len(results),
        "pass": sum(item["status"] == "PASS" for item in results),
        "activated": sum(item.get("w_contract", {}).get("family") != IDENTITY for item in results),
        "test_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / "W" / "SUMMARY.json", summary)
    return summary


def _load_c_result(output: Path, view: ViewSpec) -> dict[str, Any]:
    path = output / "DEVELOPMENT" / "C" / view.head.head_id / view.proxy_policy / "RESULT.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError(f"E2 C prerequisite is not PASS: {path}")
    return value


def _w_candidates(v21: dict[str, Any], v2: dict[str, Any], direction: int, monotone: bool) -> list[Any]:
    candidates: list[Any] = [IDENTITY]
    smoothness = [float(value) for value in v2["W_module"]["smoothness_penalties"]]
    mus = [float(value) for value in v21["W"]["soft_overlap_mu"]]
    if monotone:
        candidates.extend((MONOTONE, int(k), s, mu, direction) for k in v21["W"]["monotone_knots"] for s in smoothness for mu in mus)
    candidates.extend((NATURAL_CUBIC, int(k), s, mu, 1) for k in v21["W"]["natural_cubic_knots"] for s in smoothness for mu in mus)
    return candidates


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
    key = "joint" if c_result["selected_family"] == "ADDITIVE_JOINT_BASIS" else "compressed"
    train_x = features[f"{key}_train"]
    evaluation_x = features[f"{key}_evaluation"]
    target = fit["y_true"].to_numpy(dtype=np.float64)
    if train_x.shape[1] == 0:
        intercept = float(np.mean(target, dtype=np.float64))
        train_prediction = np.full(len(fit), intercept, dtype=np.float64)
        evaluation_prediction = np.full(len(evaluation), intercept, dtype=np.float64)
        contract = {
            "status": "K_EXACT_ZERO",
            "parameter_count": 1,
            "intercept": intercept,
        }
    else:
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
        return np.zeros(len(evaluation_latent), dtype=np.float64), {"family": IDENTITY, "parameter_count": 0, "effective_df": 0.0, "soft_overlap_mu": 0.0, "hard_feature_residualization": False}
    family, knots, smoothness, mu, direction = candidate
    return fit_w_correction(train_latent, train_target - train_latent, evaluation_latent, family=family, knot_count=knots, smoothness=smoothness, mu=mu, upstream_predictions=upstream_predictions, direction=direction)


def run_w_view(shared: Path, project: Path, output: Path, view: ViewSpec) -> dict[str, Any]:
    started = time.time()
    destination = output / "DEVELOPMENT" / "W" / view.head.head_id / view.proxy_policy
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v21, v2 = load_v21_and_v2_config(project)
        c_result = _load_c_result(output, view)
        active = load_active_channels(output, view)
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        fold_inputs = []
        correlations = []
        for fold, (fit_index, evaluation_index) in enumerate(inner_folds(train, int(v21["selection"]["inner_folds"]))):
            fit = _cap(train.iloc[fit_index], int(v2["row_caps"]["wiener_fit"]))
            evaluation = _cap(train.iloc[evaluation_index], int(v2["row_caps"]["validation_selection_per_fold"]))
            fit_latent, evaluation_latent, fit_upstream, evaluation_upstream, _ = _fit_c_routed(shared, view, fit, evaluation, active, v2, c_result, fit_split="train", evaluation_split="train")
            fit_target = fit["y_true"].to_numpy(dtype=np.float64)
            correlations.append(float(stats.spearmanr(fit_latent, fit_target - fit_latent).statistic))
            fold_inputs.append((fold, evaluation, fit_latent, evaluation_latent, fit_upstream, evaluation_upstream, fit_target, evaluation["y_true"].to_numpy(dtype=np.float64)))
        finite = np.asarray([value for value in correlations if np.isfinite(value)], dtype=np.float64)
        direction = 1 if len(finite) == 0 or float(np.median(finite)) >= 0 else -1
        rule = v2["W_module"]["monotone_direction"]
        monotone = bool(len(finite) >= int(v21["selection"]["minimum_usable_folds"]) and abs(float(np.median(finite))) >= float(rule["absolute_correlation_min"]) and float(np.mean(np.sign(finite) == direction)) >= float(rule["same_sign_fold_fraction_min"]))
        candidates = _w_candidates(v21, v2, direction, monotone)
        losses = {candidate: [] for candidate in candidates}
        for _, _, fit_latent, evaluation_latent, fit_upstream, _, fit_target, evaluation_target in fold_inputs:
            for candidate in candidates:
                correction, _ = _fit_registered_w(candidate, fit_latent, fit_target, evaluation_latent, fit_upstream)
                losses[candidate].append(mse(evaluation_target, evaluation_latent + correction))
        def complexity(candidate: Any) -> tuple[Any, ...]:
            if candidate == IDENTITY:
                return (0,)
            family, knots, smoothness, mu, _ = candidate
            return (1, 0 if family == MONOTONE else 1, int(knots), -float(smoothness), -float(mu))
        selection = guarded_local_one_se_select(losses, complexity, neutral=IDENTITY, minimum_relative_improvement=float(v21["selection"]["minimum_relative_improvement"]["W"]), minimum_positive_fraction=float(v21["selection"]["minimum_positive_fold_fraction"]), minimum_usable_folds=int(v21["selection"]["minimum_usable_folds"]))
        selected = selection.final_selected_candidate
        oof_frames = []
        for fold, evaluation, fit_latent, evaluation_latent, fit_upstream, evaluation_upstream, fit_target, _ in fold_inputs:
            correction, _ = _fit_registered_w(selected, fit_latent, fit_target, evaluation_latent, fit_upstream)
            frame = evaluation[["base_origin_id", "view_sample_id", "entity_id", "origin", "latest_available_target_index", "y_true"]].copy()
            frame["physical_oof"], frame["delta_w_oof"], frame["physical_w_oof"], frame["oof_fold"] = evaluation_latent, correction, evaluation_latent + correction, fold
            for index in range(evaluation_upstream.shape[1]):
                frame[f"k_channel_contribution_{index:03d}"] = evaluation_upstream[:, index]
            oof_frames.append(frame)
        oof = pd.concat(oof_frames, ignore_index=True)
        final_train = _cap(train, int(v2["row_caps"]["wiener_fit"]))
        fit_latent, validation_latent, fit_upstream, _, c_contract = _fit_c_routed(shared, view, final_train, validation, active, v2, c_result, fit_split="train", evaluation_split="validation")
        correction, contract = _fit_registered_w(selected, fit_latent, final_train["y_true"].to_numpy(dtype=np.float64), validation_latent, fit_upstream)
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
        frame = validation[["base_origin_id", "view_sample_id", "entity_id", "origin", "latest_available_target_index", "y_true"]].copy()
        frame["physical_latent"], frame["delta_w"], frame["y_pred"] = validation_latent, correction, prediction
        frame["model"], frame["dtype"] = "PRISM_V2_1_K_C_W", "float64"
        oof_path, prediction_path = destination / "PHYSICAL_OOF.parquet", destination / "validation.parquet"
        oof.to_parquet(oof_path, index=False, compression="zstd")
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        final_loss = mse(frame["y_true"].to_numpy(dtype=np.float64), prediction)
        result = {"status": "PASS", "stage": "E3_W", "dataset": view.head.dataset, "target_head": view.head.head_id, "proxy_policy": view.proxy_policy, "selected_candidate": str(selected), "w_contract": contract, "c_final_contract": c_contract, "selection": selection.to_json(), "candidate_fold_losses": {str(key): value for key, value in losses.items()}, "monotone_correlations": correlations, "monotone_applicable": monotone, "monotone_direction": direction, "hard_feature_residualization": False, "hard_projection_replay": {"selection_eligible": False, "split": "validation", "contract": replay_contract, "mse": mse(validation["y_true"].to_numpy(dtype=np.float64), replay_prediction)}, "oof_path": str(oof_path.relative_to(output)), "oof_sha256": sha256_file(oof_path), "final_selected_candidate": str(selected), "final_selected_fold_losses": list(selection.final_selected_fold_losses), "final_selected_prediction_path": str(prediction_path.relative_to(output)), "final_selected_contract": contract, "final_prediction_loss": final_loss, "prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": sha256_file(prediction_path), "test_accessed": False, "elapsed_seconds": time.time() - started, **regression_metrics(frame["y_true"].to_numpy(dtype=np.float64), prediction)}
    except Exception as error:
        result = {"status": "SOLVER_FAILED_RETAINED", "stage": "E3_W", "target_head": view.head.head_id, "proxy_policy": view.proxy_policy, "test_accessed": False, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started}
    write_json(destination / "RESULT.json", result)
    return result
