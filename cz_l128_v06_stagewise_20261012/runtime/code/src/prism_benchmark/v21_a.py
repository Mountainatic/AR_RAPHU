from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .v2_numerics import solve_certified
from .v21_w import soft_overlap_penalty


EXACT_ZERO = "EXACT_ZERO"
MATURE_RESIDUAL_AR = "MATURE_RESIDUAL_AR"


def _residual_lookup(
    residuals: pd.DataFrame,
    residual_mean: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    required = {"entity_id", "origin", "residual"}
    missing = required - set(residuals.columns)
    if missing:
        raise KeyError(f"residual frame is missing columns: {sorted(missing)}")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for entity, group in residuals.groupby("entity_id", sort=False):
        ordered = group.sort_values("origin")
        origins = ordered["origin"].to_numpy(dtype=np.int64)
        if len(np.unique(origins)) != len(origins):
            raise ValueError(f"duplicate residual origins for entity {entity}")
        values = ordered["residual"].to_numpy(dtype=np.float64) - float(residual_mean)
        result[str(entity)] = (origins, values)
    return result


def mature_residual_features(
    samples: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    h_steps: int,
    w_steps: int,
    delta: int,
    history: int,
    maximum_lags: int,
    residual_mean: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Create A features from residuals whose target windows are already mature.

    The sample-level ``latest_available_target_index`` embeds analyzer/label
    delay.  For a residual anchored at origin ``s``, the final target index is
    ``s + h + W - 1``; hence the latest legal residual origin is
    ``latest_available_target_index - h - W + 1``.
    """
    required = {"entity_id", "origin", "latest_available_target_index"}
    missing = required - set(samples.columns)
    if missing:
        raise KeyError(f"sample frame is missing maturity columns: {sorted(missing)}")
    if h_steps < 0 or w_steps < 1 or delta < 1 or history < 1 or maximum_lags < 1:
        raise ValueError("invalid maturity/profile parameters")
    latest_target = samples["latest_available_target_index"].to_numpy(dtype=np.int64)
    origins = samples["origin"].to_numpy(dtype=np.int64)
    if np.any(latest_target >= origins):
        raise ValueError("latest available target must be strictly before the prediction origin")
    count = min(maximum_lags, max(1, history // delta))
    offsets = np.unique(
        np.rint(np.linspace(0, max(0, history - delta), count)).astype(np.int64)
    )
    latest_residual_origin = latest_target - h_steps - w_steps + 1
    lookup = _residual_lookup(residuals, residual_mean)
    result = np.zeros((len(samples), len(offsets)), dtype=np.float64)
    observed = np.zeros_like(result, dtype=bool)
    entities = samples["entity_id"].astype(str).to_numpy()
    for entity in np.unique(entities):
        mask = np.flatnonzero(entities == entity)
        if entity not in lookup:
            continue
        source_origins, source_values = lookup[entity]
        queries = latest_residual_origin[mask, None] - offsets[None, :]
        positions = np.searchsorted(source_origins, queries)
        valid = positions < len(source_origins)
        safe = np.minimum(positions, max(len(source_origins) - 1, 0))
        valid &= source_origins[safe] == queries
        values = np.zeros(queries.shape, dtype=np.float64)
        values[valid] = source_values[safe[valid]]
        result[mask] = values
        observed[mask] = valid
    coverage = float(observed.mean()) if observed.size else 0.0
    audit = {
        "maturity_rule": "s_plus_h_plus_W_plus_D_le_t",
        "uses_latest_available_target_index": True,
        "latest_residual_origin_min": int(latest_residual_origin.min(initial=0)),
        "latest_residual_origin_max": int(latest_residual_origin.max(initial=0)),
        "offsets": offsets.tolist(),
        "observed_fraction": coverage,
        "cross_entity_reads": False,
    }
    return result, coverage, audit


def fit_mature_residual_ar(
    train_features: np.ndarray,
    residual_target: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    alpha: float,
    mu: float,
    upstream_predictions: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    train = np.asarray(train_features, dtype=np.float64)
    evaluation = np.asarray(evaluation_features, dtype=np.float64)
    target = np.asarray(residual_target, dtype=np.float64).reshape(-1)
    if train.ndim != 2 or evaluation.ndim != 2 or train.shape[1] != evaluation.shape[1]:
        raise ValueError("A feature matrices are incompatible")
    if target.shape != (len(train),):
        raise ValueError("A target/feature row mismatch")
    if alpha < 0 or mu < 0:
        raise ValueError("A penalties must be nonnegative")
    mean = train.mean(axis=0, dtype=np.float64)
    scale = train.std(axis=0, dtype=np.float64)
    scale[scale * scale < 1e-12] = 1.0
    x = (train - mean) / scale
    z = (evaluation - mean) / scale
    target_mean = float(np.mean(target, dtype=np.float64))
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    penalty += soft_overlap_penalty(x, upstream_predictions, float(mu))
    coefficient, certificate = solve_certified(x, target - target_mean, penalty)
    prediction = z @ coefficient + target_mean
    train_prediction = x @ coefficient + target_mean
    upstream = np.asarray(upstream_predictions, dtype=np.float64) if upstream_predictions is not None else np.empty((len(x), 0))
    if upstream.ndim == 1:
        upstream = upstream[:, None]
    overlap = 0.0
    if upstream.shape[1] and np.std(train_prediction) > 0:
        up_centered = upstream - upstream.mean(axis=0, dtype=np.float64)
        up_scale = upstream.std(axis=0, dtype=np.float64)
        keep = up_scale * up_scale >= 1e-12
        if np.any(keep):
            up_standardized = up_centered[:, keep] / up_scale[keep]
            pred_standardized = (train_prediction - np.mean(train_prediction)) / np.std(train_prediction)
            overlap = float(np.linalg.norm(up_standardized.T @ pred_standardized / len(x)))
    contract = {
        "family": MATURE_RESIDUAL_AR,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": target_mean,
        "alpha": float(alpha),
        "soft_overlap_mu": float(mu),
        "soft_overlap_norm": overlap,
        "hard_feature_residualization": False,
        "uses_input_features": False,
        "numerical_certificate": certificate.to_json(),
        "parameter_count": len(coefficient) + 1,
    }
    return prediction, contract


def predict_mature_residual_ar(features: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if contract["family"] == EXACT_ZERO:
        return np.zeros(len(matrix), dtype=np.float64)
    if contract["family"] != MATURE_RESIDUAL_AR:
        raise ValueError(f"unsupported A family: {contract['family']}")
    standardized = (
        matrix - np.asarray(contract["mean"], dtype=np.float64)
    ) / np.asarray(contract["scale"], dtype=np.float64)
    return standardized @ np.asarray(contract["coefficient"], dtype=np.float64) + float(
        contract["intercept"]
    )


def run_a_view(shared: "Path", project: "Path", output: "Path", view: Any) -> dict[str, Any]:
    import json
    import time
    import traceback
    from pathlib import Path
    from .cpu_data import load_samples, realized_state_profiles, sha256_file
    from .cpu_selection import mse, regression_metrics
    from .stage0 import write_json
    from .v21_config import load_v21_and_v2_config
    from .v21_selection import guarded_local_one_se_select

    started = time.time()
    destination = output / "DEVELOPMENT" / "A" / view.head.head_id / view.availability_scenario / view.proxy_policy
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v21, v2 = load_v21_and_v2_config(project)
        w_root = output / "DEVELOPMENT" / "W" / view.head.head_id / view.proxy_policy
        w_result = json.loads((w_root / "RESULT.json").read_text(encoding="utf-8"))
        if w_result.get("status") != "PASS":
            raise RuntimeError("E3 W prerequisite is not PASS")
        oof = pd.read_parquet(output / w_result["oof_path"])
        w_validation = pd.read_parquet(output / w_result["prediction_path"])
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        contribution_columns = sorted(
            column
            for column in oof.columns
            if column.startswith("k_channel_contribution_")
        )
        oof = train.merge(
            oof[
                [
                    "base_origin_id",
                    "physical_oof",
                    "delta_w_oof",
                    "physical_w_oof",
                    "oof_fold",
                    *contribution_columns,
                ]
            ],
            on="base_origin_id",
            how="inner",
            validate="one_to_one",
        )
        oof["residual"] = oof["y_true"] - oof["physical_w_oof"]
        validation_frame = validation.merge(w_validation[["base_origin_id", "physical_latent", "delta_w", "y_pred"]].rename(columns={"y_pred": "physical_w"}), on="base_origin_id", how="inner", validate="one_to_one")
        if len(validation_frame) != len(validation):
            raise RuntimeError("dynamic/input-only base_origin_id mismatch")
        validation_frame["residual"] = validation_frame["y_true"] - validation_frame["physical_w"]
        profiles = realized_state_profiles(view.head)
        alphas = [float(value) for value in v2["A_module"]["ridge_alpha_grid"]]
        mus = [float(value) for value in v21["A"]["soft_overlap_mu"]]
        candidates: list[Any] = [EXACT_ZERO]
        candidates.extend((MATURE_RESIDUAL_AR, profile, alpha, mu) for profile in profiles for alpha in alphas for mu in mus)
        losses = {candidate: [] for candidate in candidates}
        fold_means: dict[str, float] = {}
        coverage: dict[str, list[float]] = {str(profile): [] for profile in profiles}
        usable_folds = sorted(int(value) for value in oof["oof_fold"].unique())[1:]
        for fold in usable_folds:
            fit = oof[oof["oof_fold"] < fold]
            evaluation = oof[oof["oof_fold"] == fold]
            residual_mean = float(fit["residual"].mean())
            fold_means[str(fold)] = residual_mean
            y_fit = fit["residual"].to_numpy(dtype=np.float64)
            y_eval = evaluation["residual"].to_numpy(dtype=np.float64)
            losses[EXACT_ZERO].append(float(np.mean(y_eval * y_eval, dtype=np.float64)))
            upstream_columns = [*contribution_columns, "delta_w_oof"]
            if not contribution_columns:
                upstream_columns.insert(0, "physical_oof")
            upstream = fit[upstream_columns].to_numpy(dtype=np.float64)
            for profile in profiles:
                delta, history = profile
                x_fit, observed_fit, _ = mature_residual_features(fit, oof, h_steps=view.head.h_steps, w_steps=view.head.w_steps, delta=delta, history=history, maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]), residual_mean=residual_mean)
                x_eval, observed_eval, _ = mature_residual_features(evaluation, oof, h_steps=view.head.h_steps, w_steps=view.head.w_steps, delta=delta, history=history, maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]), residual_mean=residual_mean)
                coverage[str(profile)].extend([observed_fit, observed_eval])
                for alpha in alphas:
                    for mu in mus:
                        prediction, _ = fit_mature_residual_ar(x_fit, y_fit, x_eval, alpha=alpha, mu=mu, upstream_predictions=upstream)
                        losses[(MATURE_RESIDUAL_AR, profile, alpha, mu)].append(mse(y_eval, prediction))
        def complexity(candidate: Any) -> tuple[Any, ...]:
            if candidate == EXACT_ZERO:
                return (0,)
            _, profile, alpha, mu = candidate
            return (1, int(profile[1]), -int(profile[0]), -float(alpha), -float(mu))
        selection = guarded_local_one_se_select(losses, complexity, neutral=EXACT_ZERO, minimum_relative_improvement=float(v21["selection"]["minimum_relative_improvement"]["A"]), minimum_positive_fraction=float(v21["selection"]["minimum_positive_fold_fraction"]), minimum_usable_folds=int(v21["selection"]["minimum_usable_folds"]))
        selected = selection.final_selected_candidate
        residual_mean = float(oof["residual"].mean())
        combined = pd.concat([oof[["entity_id", "origin", "residual"]], validation_frame[["entity_id", "origin", "residual"]]], ignore_index=True)
        if selected == EXACT_ZERO:
            residual_prediction = np.zeros(len(validation_frame), dtype=np.float64)
            contract = {"family": EXACT_ZERO, "parameter_count": 0, "soft_overlap_mu": 0.0, "hard_feature_residualization": False}
            selected_coverage = 1.0
        else:
            _, profile, alpha, mu = selected
            delta, history = profile
            x_train, observed_train, train_audit = mature_residual_features(oof, oof, h_steps=view.head.h_steps, w_steps=view.head.w_steps, delta=delta, history=history, maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]), residual_mean=residual_mean)
            x_validation, observed_validation, validation_audit = mature_residual_features(validation_frame, combined, h_steps=view.head.h_steps, w_steps=view.head.w_steps, delta=delta, history=history, maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]), residual_mean=residual_mean)
            upstream_columns = [*contribution_columns, "delta_w_oof"]
            if not contribution_columns:
                upstream_columns.insert(0, "physical_oof")
            residual_prediction, contract = fit_mature_residual_ar(x_train, oof["residual"].to_numpy(dtype=np.float64), x_validation, alpha=alpha, mu=mu, upstream_predictions=oof[upstream_columns].to_numpy(dtype=np.float64))
            contract.update({"profile": list(profile), "maturity_train_audit": train_audit, "maturity_validation_audit": validation_audit})
            selected_coverage = min(observed_train, observed_validation)
        prediction = validation_frame["physical_w"].to_numpy(dtype=np.float64) + residual_prediction
        frame = validation_frame[["base_origin_id", "view_sample_id", "entity_id", "origin", "latest_available_target_index", "y_true"]].copy()
        frame["physical_w"], frame["residual_pred"], frame["y_pred"] = validation_frame["physical_w"], residual_prediction, prediction
        frame["model"], frame["dtype"] = "PRISM_V2_1_PHYSICS_FIRST", "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        final_loss = mse(frame["y_true"].to_numpy(dtype=np.float64), prediction)
        result = {"status": "PASS", "stage": "E4_A", "dataset": view.head.dataset, "target_head": view.head.head_id, "availability_scenario": view.availability_scenario, "proxy_policy": view.proxy_policy, "selected_candidate": str(selected), "a_contract": contract, "selection": selection.to_json(), "candidate_fold_losses": {str(key): value for key, value in losses.items()}, "fold_local_residual_means": fold_means, "fold_local_residual_centering": True, "maturity_rule": "s_plus_h_plus_W_plus_D_le_t", "observed_mature_feature_fraction": selected_coverage, "hard_feature_residualization": False, "final_selected_candidate": str(selected), "final_selected_fold_losses": list(selection.final_selected_fold_losses), "final_selected_prediction_path": str(prediction_path.relative_to(output)), "final_selected_contract": contract, "final_prediction_loss": final_loss, "prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": sha256_file(prediction_path), "test_accessed": False, "elapsed_seconds": time.time() - started, **regression_metrics(frame["y_true"].to_numpy(dtype=np.float64), prediction)}
    except Exception as error:
        result = {"status": "SOLVER_FAILED_RETAINED", "stage": "E4_A", "target_head": view.head.head_id, "availability_scenario": view.availability_scenario, "proxy_policy": view.proxy_policy, "test_accessed": False, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started}
    write_json(destination / "RESULT.json", result)
    return result


def run_e4_a(shared: "Path", project: "Path", output: "Path") -> dict[str, Any]:
    import json
    from .stage0 import write_json
    from .v21_assembly import build_physics_first_card
    from .v21_views import sru_dynamic_views

    results = []
    for view in sru_dynamic_views(shared):
        result = run_a_view(shared, project, output, view)
        results.append(result)
        if result.get("status") == "PASS":
            c_path = output / "DEVELOPMENT" / "C" / view.head.head_id / view.proxy_policy / "RESULT.json"
            w_path = output / "DEVELOPMENT" / "W" / view.head.head_id / view.proxy_policy / "RESULT.json"
            c_result = json.loads(c_path.read_text(encoding="utf-8"))
            w_result = json.loads(w_path.read_text(encoding="utf-8"))
            k_result = {"status": "PASS", "final_selected_candidate": c_result.get("active_channels", []), "input_path_nonzero": c_result.get("input_path_nonzero", False)}
            card = build_physics_first_card(k_result, c_result, w_result, result)
            card.update({"target_head": view.head.head_id, "availability_scenario": view.availability_scenario, "proxy_policy": view.proxy_policy, "validation_prediction_path": result["prediction_path"]})
            write_json(output / "ASSEMBLY_CARDS" / view.head.head_id / view.availability_scenario / view.proxy_policy / "PF_ASSEMBLY_CARD.json", card)
    summary = {"status": "PASS" if all(item["status"] == "PASS" for item in results) else "COMPLETED_WITH_RETAINED_FAILURES", "stage": "E4_A", "views": len(results), "pass": sum(item["status"] == "PASS" for item in results), "activated": sum(item.get("a_contract", {}).get("family") != EXACT_ZERO for item in results), "test_accessed": False}
    write_json(output / "DEVELOPMENT" / "A" / "SUMMARY.json", summary)
    return summary
