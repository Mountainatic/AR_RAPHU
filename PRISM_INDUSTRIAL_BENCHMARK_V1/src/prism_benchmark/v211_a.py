from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cpu_data import realized_state_profiles, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_runtime import ordered_fork_map, run_parallel
from .v21_a import (
    EXACT_ZERO,
    MATURE_RESIDUAL_AR,
    fit_mature_residual_ar,
    mature_residual_features,
    predict_mature_residual_ar,
)
from .v21_selection import guarded_local_one_se_select
from .v211_config import load_v211_configs
from .v211_support import load_native_samples, support_id_hash
A_INNER_WORKERS_ENV = "PRISM_V211_A_INNER_WORKERS"
_A_CANDIDATE_CONTEXT: tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float]],
] | None = None


def _a_inner_workers() -> int:
    raw = os.environ.get(A_INNER_WORKERS_ENV, "1")
    try:
        workers = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{A_INNER_WORKERS_ENV} must be an integer") from error
    cpu_count = os.cpu_count() or 1
    if workers < 1 or workers > cpu_count:
        raise RuntimeError(
            f"{A_INNER_WORKERS_ENV} must be within [1, {cpu_count}]"
        )
    return workers


def _evaluate_a_candidate(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    alpha: float,
    mu: float,
    upstream: np.ndarray,
) -> float:
    prediction, _ = fit_mature_residual_ar(
        x_fit,
        y_fit,
        x_evaluation,
        alpha=alpha,
        mu=mu,
        upstream_predictions=upstream,
    )
    return mse(y_evaluation, prediction)


def _evaluate_a_indexed(candidate_index: int) -> float:
    if _A_CANDIDATE_CONTEXT is None:
        raise RuntimeError("A candidate context was not initialized before fork")
    x_fit, y_fit, x_evaluation, y_evaluation, upstream, candidate_pairs = (
        _A_CANDIDATE_CONTEXT
    )
    alpha, mu = candidate_pairs[candidate_index]
    return _evaluate_a_candidate(
        x_fit,
        y_fit,
        x_evaluation,
        y_evaluation,
        alpha,
        mu,
        upstream,
    )


def _merge_w_oof_for_a(
    train: pd.DataFrame,
    w_oof: pd.DataFrame,
    contribution_columns: list[str],
) -> pd.DataFrame:
    columns = [
        "base_origin_id",
        "physical_oof",
        "delta_w_oof",
        "physical_w_oof",
        "delta_w_ablation_oof",
        "physical_w_ablation_oof",
        "oof_fold",
        *contribution_columns,
    ]
    missing = [column for column in columns if column not in w_oof.columns]
    if missing:
        raise RuntimeError(f"W OOF is missing registered A-route columns: {missing}")
    return train.merge(
        w_oof[columns],
        on="base_origin_id",
        how="inner",
        validate="one_to_one",
    )


def _merge_w_validation_for_a(
    validation: pd.DataFrame,
    w_validation: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict a dynamic view to its legal intersection with W assembly support."""
    columns = [
        "base_origin_id",
        "physical_latent",
        "delta_w",
        "delta_w_ablation",
        "physical_w_ablation",
        "y_pred",
    ]
    missing = [column for column in columns if column not in w_validation.columns]
    if missing:
        raise RuntimeError(
            f"W validation is missing registered A-route columns: {missing}"
        )
    frame = validation.merge(
        w_validation[columns].rename(columns={"y_pred": "physical_w"}),
        on="base_origin_id",
        how="inner",
        validate="one_to_one",
    )
    validation_ids = set(validation["base_origin_id"].astype(str))
    w_support_ids = set(w_validation["base_origin_id"].astype(str))
    expected_ids = validation_ids.intersection(w_support_ids)
    observed_ids = set(frame["base_origin_id"].astype(str))
    if observed_ids != expected_ids or len(frame) != len(expected_ids):
        raise RuntimeError(
            "A did not inherit the complete dynamic/W assembly support intersection"
        )
    if frame.empty:
        raise RuntimeError("A has no rows on the dynamic/W assembly support intersection")
    return frame


def _fit_frozen_a_route(
    oof: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    oof_physical_column: str,
    validation_physical_column: str,
    oof_w_column: str | None,
    selected: Any,
    contribution_columns: list[str],
    view: Any,
    v2: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Refit a frozen A construction on one pre-registered PF route."""
    route_oof = oof.copy()
    route_validation = validation.copy()
    route_oof["residual"] = (
        route_oof["y_true"].to_numpy(dtype=np.float64)
        - route_oof[oof_physical_column].to_numpy(dtype=np.float64)
    )
    route_validation["residual"] = (
        route_validation["y_true"].to_numpy(dtype=np.float64)
        - route_validation[validation_physical_column].to_numpy(dtype=np.float64)
    )
    if selected == EXACT_ZERO:
        return (
            np.zeros(len(route_validation), dtype=np.float64),
            {
                "family": EXACT_ZERO,
                "parameter_count": 0,
                "soft_overlap_mu": 0.0,
                "hard_feature_residualization": False,
                "numerical_certificate": {"status": "EXACT_ZERO"},
            },
            1.0,
        )
    _, profile, alpha, mu = selected
    delta, history = profile
    residual_mean = float(route_oof["residual"].mean())
    source = pd.concat(
        [
            route_oof[["entity_id", "origin", "residual"]],
            route_validation[["entity_id", "origin", "residual"]],
        ],
        ignore_index=True,
    )
    x_train, observed_train, train_audit = mature_residual_features(
        route_oof,
        route_oof,
        h_steps=view.head.h_steps,
        w_steps=view.head.w_steps,
        delta=delta,
        history=history,
        maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]),
        residual_mean=residual_mean,
    )
    x_validation, observed_validation, validation_audit = mature_residual_features(
        route_validation,
        source,
        h_steps=view.head.h_steps,
        w_steps=view.head.w_steps,
        delta=delta,
        history=history,
        maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]),
        residual_mean=residual_mean,
    )
    upstream_columns = list(contribution_columns)
    if oof_w_column is not None:
        upstream_columns.append(oof_w_column)
    if not upstream_columns:
        upstream_columns = [oof_physical_column]
    prediction, contract = fit_mature_residual_ar(
        x_train,
        route_oof["residual"].to_numpy(dtype=np.float64),
        x_validation,
        alpha=float(alpha),
        mu=float(mu),
        upstream_predictions=route_oof[upstream_columns].to_numpy(dtype=np.float64),
    )
    contract.update(
        {
            "profile": list(profile),
            "residual_mean": residual_mean,
            "maturity_train_audit": train_audit,
            "maturity_validation_audit": validation_audit,
        }
    )
    return prediction, contract, min(observed_train, observed_validation)


def run_a_view(
    shared: Path,
    project: Path,
    output: Path,
    view: Any,
    protocol: str = "sru",
) -> dict[str, Any]:
    global _A_CANDIDATE_CONTEXT
    started = time.time()
    destination = (
        output
        / "DEVELOPMENT"
        / "A"
        / view.head.head_id
        / view.availability_scenario
        / view.proxy_policy
    )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v211, v21, v2 = load_v211_configs(project, protocol=protocol)
        inner_workers = _a_inner_workers()
        w_root = output / "DEVELOPMENT" / "W" / view.head.head_id / view.proxy_policy
        w_result = json.loads((w_root / "RESULT.json").read_text(encoding="utf-8"))
        if w_result.get("status") != "PASS":
            raise RuntimeError("E3R W prerequisite is not PASS")
        oof = pd.read_parquet(output / w_result["oof_path"])
        w_validation = pd.read_parquet(output / w_result["prediction_path"])
        train = load_native_samples(shared, view, "train")
        validation = load_native_samples(shared, view, "validation")
        contribution_columns = sorted(
            column
            for column in oof.columns
            if column.startswith("k_channel_contribution_")
        )
        oof = _merge_w_oof_for_a(train, oof, contribution_columns)
        if len(oof) > int(v2["row_caps"]["state_fit"]):
            raise RuntimeError("A OOF fit rows exceed the frozen state_fit cap")
        oof["residual"] = oof["y_true"] - oof["physical_w_oof"]
        validation_frame = _merge_w_validation_for_a(validation, w_validation)
        validation_frame["residual"] = (
            validation_frame["y_true"] - validation_frame["physical_w"]
        )
        profiles = realized_state_profiles(view.head)
        alphas = [float(value) for value in v2["A_module"]["ridge_alpha_grid"]]
        mus = [float(value) for value in v21["A"]["soft_overlap_mu"]]
        candidates: list[Any] = [EXACT_ZERO]
        candidates.extend(
            (MATURE_RESIDUAL_AR, profile, alpha, mu)
            for profile in profiles
            for alpha in alphas
            for mu in mus
        )
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
            losses[EXACT_ZERO].append(
                float(np.mean(y_eval * y_eval, dtype=np.float64))
            )
            upstream_columns = [*contribution_columns, "delta_w_oof"]
            if not contribution_columns:
                upstream_columns.insert(0, "physical_oof")
            upstream = fit[upstream_columns].to_numpy(dtype=np.float64)
            for profile in profiles:
                delta, history = profile
                x_fit, observed_fit, _ = mature_residual_features(
                    fit,
                    oof,
                    h_steps=view.head.h_steps,
                    w_steps=view.head.w_steps,
                    delta=delta,
                    history=history,
                    maximum_lags=int(
                        v2["A_module"]["state_profile"]["maximum_lags"]
                    ),
                    residual_mean=residual_mean,
                )
                x_eval, observed_eval, _ = mature_residual_features(
                    evaluation,
                    oof,
                    h_steps=view.head.h_steps,
                    w_steps=view.head.w_steps,
                    delta=delta,
                    history=history,
                    maximum_lags=int(
                        v2["A_module"]["state_profile"]["maximum_lags"]
                    ),
                    residual_mean=residual_mean,
                )
                coverage[str(profile)].extend([observed_fit, observed_eval])
                candidate_pairs = [
                    (alpha, mu) for alpha in alphas for mu in mus
                ]
                _A_CANDIDATE_CONTEXT = (
                    x_fit,
                    y_fit,
                    x_eval,
                    y_eval,
                    upstream,
                    candidate_pairs,
                )
                try:
                    candidate_losses = ordered_fork_map(
                        _evaluate_a_indexed,
                        [(index,) for index in range(len(candidate_pairs))],
                        inner_workers,
                        label="PRISM_V211_METRO_M4_A_INNER",
                    )
                finally:
                    _A_CANDIDATE_CONTEXT = None
                for (alpha, mu), candidate_loss in zip(
                    candidate_pairs, candidate_losses, strict=True
                ):
                    losses[(MATURE_RESIDUAL_AR, profile, alpha, mu)].append(
                        candidate_loss
                    )

        def complexity(candidate: Any) -> tuple[Any, ...]:
            if candidate == EXACT_ZERO:
                return (0,)
            _, profile, alpha, mu = candidate
            return (
                1,
                int(profile[1]),
                -int(profile[0]),
                -float(alpha),
                -float(mu),
            )

        selection = guarded_local_one_se_select(
            losses,
            complexity,
            neutral=EXACT_ZERO,
            minimum_relative_improvement=float(
                v21["selection"]["minimum_relative_improvement"]["A"]
            ),
            minimum_positive_fraction=float(
                v21["selection"]["minimum_positive_fold_fraction"]
            ),
            minimum_usable_folds=int(v21["selection"]["minimum_usable_folds"]),
        )
        selected = selection.final_selected_candidate
        residual_mean = float(oof["residual"].mean())
        combined = pd.concat(
            [
                oof[["entity_id", "origin", "residual"]],
                validation_frame[["entity_id", "origin", "residual"]],
            ],
            ignore_index=True,
        )
        if selected == EXACT_ZERO:
            residual_prediction = np.zeros(len(validation_frame), dtype=np.float64)
            contract = {
                "family": EXACT_ZERO,
                "parameter_count": 0,
                "soft_overlap_mu": 0.0,
                "hard_feature_residualization": False,
                "numerical_certificate": {"status": "EXACT_ZERO"},
            }
            selected_coverage = 1.0
        else:
            _, profile, alpha, mu = selected
            delta, history = profile
            x_train, observed_train, train_audit = mature_residual_features(
                oof,
                oof,
                h_steps=view.head.h_steps,
                w_steps=view.head.w_steps,
                delta=delta,
                history=history,
                maximum_lags=int(
                    v2["A_module"]["state_profile"]["maximum_lags"]
                ),
                residual_mean=residual_mean,
            )
            x_validation, observed_validation, validation_audit = (
                mature_residual_features(
                    validation_frame,
                    combined,
                    h_steps=view.head.h_steps,
                    w_steps=view.head.w_steps,
                    delta=delta,
                    history=history,
                    maximum_lags=int(
                        v2["A_module"]["state_profile"]["maximum_lags"]
                    ),
                    residual_mean=residual_mean,
                )
            )
            upstream_columns = [*contribution_columns, "delta_w_oof"]
            if not contribution_columns:
                upstream_columns.insert(0, "physical_oof")
            residual_prediction, contract = fit_mature_residual_ar(
                x_train,
                oof["residual"].to_numpy(dtype=np.float64),
                x_validation,
                alpha=alpha,
                mu=mu,
                upstream_predictions=oof[upstream_columns].to_numpy(
                    dtype=np.float64
                ),
            )
            contract.update(
                {
                    "profile": list(profile),
                    "maturity_train_audit": train_audit,
                    "maturity_validation_audit": validation_audit,
                }
            )
            selected_coverage = min(observed_train, observed_validation)
        prediction = (
            validation_frame["physical_w"].to_numpy(dtype=np.float64)
            + residual_prediction
        )
        validation_residual = validation_frame["residual"].to_numpy(
            dtype=np.float64
        )
        residual_variance = float(np.var(validation_residual, dtype=np.float64))
        prediction_variance = float(
            np.var(residual_prediction, dtype=np.float64)
        )
        variance_floor = np.finfo(np.float64).tiny
        effective_prediction_variance_ratio = prediction_variance / max(
            residual_variance, variance_floor
        )
        coefficient = np.asarray(contract.get("coefficient", []), dtype=np.float64)
        maximum_nonintercept_coefficient_abs = float(
            np.max(np.abs(coefficient), initial=0.0)
        )
        neutral_validation_loss = float(
            np.mean(validation_residual * validation_residual, dtype=np.float64)
        )
        active_validation_loss = mse(validation_residual, residual_prediction)
        validation_relative_gain = (
            neutral_validation_loss - active_validation_loss
        ) / max(neutral_validation_loss, variance_floor)
        variance_threshold = float(
            v211["C"]["input_path_preservation"][
                "minimum_variance_ratio_to_target"
            ]
        )
        coefficient_threshold = float(
            v211["C"]["input_path_preservation"][
                "minimum_nonintercept_coefficient_abs"
            ]
        )
        gain_threshold = float(
            v21["selection"]["minimum_relative_improvement"]["A"]
        )
        active_near_zero = bool(
            selected != EXACT_ZERO
            and effective_prediction_variance_ratio < variance_threshold
            and maximum_nonintercept_coefficient_abs < coefficient_threshold
            and validation_relative_gain < gain_threshold
        )
        active_near_zero_audit = {
            "required": bool(
                v211.get("A", {}).get(
                    "active_near_zero_must_materialize_as_exact_zero", False
                )
            ),
            "effective_prediction_variance_ratio": effective_prediction_variance_ratio,
            "variance_ratio_threshold": variance_threshold,
            "maximum_nonintercept_coefficient_abs": maximum_nonintercept_coefficient_abs,
            "coefficient_abs_threshold": coefficient_threshold,
            "validation_relative_gain": validation_relative_gain,
            "relative_gain_threshold": gain_threshold,
            "all_three_below_threshold": active_near_zero,
            "threshold_source": "FROZEN_V211_NUMERICAL_ZERO_AND_A_ACTIVATION_GATES",
            "materialized_as_exact_zero": False,
        }
        if active_near_zero and active_near_zero_audit["required"]:
            selected = EXACT_ZERO
            residual_prediction = np.zeros(len(validation_frame), dtype=np.float64)
            contract = {
                "family": EXACT_ZERO,
                "parameter_count": 0,
                "soft_overlap_mu": 0.0,
                "hard_feature_residualization": False,
                "numerical_certificate": {"status": "EXACT_ZERO"},
                "reason": "ACTIVE_NEAR_ZERO_REMATERIALIZED",
            }
            selected_coverage = 1.0
            prediction = validation_frame["physical_w"].to_numpy(dtype=np.float64)
            active_near_zero_audit["materialized_as_exact_zero"] = True
        kca_residual, kca_contract, kca_coverage = _fit_frozen_a_route(
            oof,
            validation_frame,
            oof_physical_column="physical_oof",
            validation_physical_column="physical_latent",
            oof_w_column=None,
            selected=selected,
            contribution_columns=contribution_columns,
            view=view,
            v2=v2,
        )
        kcwa_residual, kcwa_contract, kcwa_coverage = _fit_frozen_a_route(
            oof,
            validation_frame,
            oof_physical_column="physical_w_ablation_oof",
            validation_physical_column="physical_w_ablation",
            oof_w_column="delta_w_ablation_oof",
            selected=selected,
            contribution_columns=contribution_columns,
            view=view,
            v2=v2,
        )
        route_predictions = {
            "KC": validation_frame["physical_latent"].to_numpy(dtype=np.float64),
            "KCW": validation_frame["physical_w_ablation"].to_numpy(dtype=np.float64),
        }
        route_predictions["KCA"] = route_predictions["KC"] + kca_residual
        route_predictions["KCWA"] = route_predictions["KCW"] + kcwa_residual
        w_active = w_result.get("w_contract", {}).get("family") != "IDENTITY_CORRECTION"
        a_active = selected != EXACT_ZERO
        pf_selected_route = (
            "KCWA"
            if w_active and a_active
            else "KCW"
            if w_active
            else "KCA"
            if a_active
            else "KC"
        )
        route_predictions["PF_SELECTED"] = route_predictions[pf_selected_route].copy()
        selected_route_error = float(
            np.max(
                np.abs(route_predictions["PF_SELECTED"] - prediction),
                initial=0.0,
            )
        )
        if selected_route_error > 1e-10:
            raise RuntimeError(
                "PF selected route does not reproduce the selected A validation prediction"
            )
        nested_frames = []
        for candidate, route_prediction in route_predictions.items():
            nested = validation_frame[
                [
                    "base_origin_id",
                    "view_sample_id",
                    "entity_id",
                    "origin",
                    "latest_available_target_index",
                    "y_true",
                ]
            ].copy().rename(columns={"view_sample_id": "sample_id"})
            nested["split"] = "validation"
            nested["candidate"] = candidate
            nested["y_pred"] = route_prediction
            nested_frames.append(nested)
        nested_path = destination / "validation_nested_pf.parquet"
        pd.concat(nested_frames, ignore_index=True).to_parquet(
            nested_path, index=False, compression="zstd"
        )
        frame = validation_frame[
            [
                "base_origin_id",
                "view_sample_id",
                "entity_id",
                "origin",
                "latest_available_target_index",
                "y_true",
            ]
        ].copy()
        frame["physical_w"] = validation_frame["physical_w"]
        frame["residual_pred"] = residual_prediction
        frame["y_pred"] = prediction
        frame["model"] = "PRISM_V2_1_1_PHYSICS_FIRST"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        final_loss = mse(frame["y_true"].to_numpy(dtype=np.float64), prediction)
        result = {
            "status": "PASS",
            "stage": "E4R_A",
            "inner_candidate_workers": inner_workers,
            "inner_parallelism_scope": "ORDERED_INDEPENDENT_CANDIDATES_ONLY",
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "selected_candidate": str(selected),
            "a_contract": contract,
            "selection": selection.to_json(),
            "candidate_fold_losses": {
                str(key): value for key, value in losses.items()
            },
            "fold_local_residual_means": fold_means,
            "fold_local_residual_centering": True,
            "maturity_rule": "s_plus_h_plus_W_plus_D_le_t",
            "uses_latest_available_target_index": True,
            "assembly_support_contract": w_result.get(
                "assembly_support_contract"
            ),
            "a_raw_input_support_hash": support_id_hash(validation_frame),
            "w_input_support_hash": w_result.get("w_input_support_hash"),
            "a_support_is_subset_of_w_support": set(
                validation_frame["base_origin_id"].astype(str)
            ).issubset(set(w_validation["base_origin_id"].astype(str))),
            "a_validation_anchor_rows": len(validation),
            "w_assembly_validation_rows": len(w_validation),
            "a_validation_intersection_rows": len(validation_frame),
            "observed_mature_feature_fraction": selected_coverage,
            "active_near_zero_audit": active_near_zero_audit,
            "hard_feature_residualization": False,
            "final_selected_candidate": str(selected),
            "final_selected_fold_losses": list(losses[selected]),
            "final_selected_prediction_path": str(prediction_path.relative_to(output)),
            "final_selected_contract": contract,
            "final_prediction_loss": final_loss,
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "nested_validation_prediction_path": str(
                nested_path.relative_to(output)
            ),
            "nested_validation_prediction_sha256": sha256_file(nested_path),
            "pf_selected_route": pf_selected_route,
            "pf_selected_route_max_abs_error": selected_route_error,
            "nested_a_contracts": {
                "KCA": kca_contract,
                "KCWA": kcwa_contract,
            },
            "nested_a_coverage": {
                "KCA": kca_coverage,
                "KCWA": kcwa_coverage,
            },
            "row_cap_audit": {
                "cap_name": "state_fit",
                "cap": int(v2["row_caps"]["state_fit"]),
                "fit_rows": len(oof),
                "validation_rows": len(validation_frame),
                "fit_source": "train_inner_oof_only",
                "within_cap": len(oof) <= int(v2["row_caps"]["state_fit"]),
            },
            "input_prediction_path": w_result["prediction_path"],
            "input_prediction_sha256": w_result["prediction_sha256"],
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
        }
    except Exception as error:
        result = {
            "status": "SOLVER_FAILED_RETAINED",
            "stage": "E4R_A",
            "target_head": view.head.head_id,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "test_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    write_json(destination / "RESULT.json", result)
    return result


def run_e4r_a(shared: Path, project: Path, output: Path) -> dict[str, Any]:
    from .v211_assembly import build_physics_first_card
    from .v21_views import sru_dynamic_views

    views = sru_dynamic_views(shared)
    results = run_parallel(
        run_a_view,
        [(shared, project, output, view) for view in views],
        int(os.environ.get("PRISM_V211_WORKERS", "8")),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "4")),
        label="PRISM_V211_E4R_A",
    )
    by_key = {
        (
            result.get("target_head"),
            result.get("availability_scenario"),
            result.get("proxy_policy"),
        ): result
        for result in results
    }
    for view in views:
        result = by_key[
            (view.head.head_id, view.availability_scenario, view.proxy_policy)
        ]
        if result.get("status") != "PASS":
            continue
        c_path = (
            output
            / "DEVELOPMENT"
            / "C"
            / view.head.head_id
            / view.proxy_policy
            / "RESULT.json"
        )
        w_path = (
            output
            / "DEVELOPMENT"
            / "W"
            / view.head.head_id
            / view.proxy_policy
            / "RESULT.json"
        )
        c_result = json.loads(c_path.read_text(encoding="utf-8"))
        w_result = json.loads(w_path.read_text(encoding="utf-8"))
        k_result = {
            "status": "PASS",
            "final_selected_candidate": c_result.get("active_channels", []),
            "input_path_preservation": c_result.get(
                "input_path_preservation", {}
            ),
        }
        card = build_physics_first_card(k_result, c_result, w_result, result)
        card.update(
            {
                "target_head": view.head.head_id,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "validation_prediction_path": result["prediction_path"],
            }
        )
        write_json(
            output
            / "ASSEMBLY_CARDS"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "PF_ASSEMBLY_CARD.json",
            card,
        )
    summary = {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in results)
        else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": "E4R_A",
        "views": len(results),
        "pass": sum(item["status"] == "PASS" for item in results),
        "activated": sum(
            item.get("a_contract", {}).get("family") != EXACT_ZERO
            for item in results
        ),
        "test_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / "A" / "SUMMARY.json", summary)
    return summary
