from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import load_samples, realized_state_profiles, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_runtime import run_parallel
from .v21_a import (
    EXACT_ZERO,
    MATURE_RESIDUAL_AR,
    fit_mature_residual_ar,
    mature_residual_features,
    predict_mature_residual_ar,
)
from .v21_selection import guarded_local_one_se_select
from .v211_config import load_v211_configs


def run_a_view(
    shared: Path,
    project: Path,
    output: Path,
    view: Any,
) -> dict[str, Any]:
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
        _, v21, v2 = load_v211_configs(project)
        w_root = output / "DEVELOPMENT" / "W" / view.head.head_id / view.proxy_policy
        w_result = json.loads((w_root / "RESULT.json").read_text(encoding="utf-8"))
        if w_result.get("status") != "PASS":
            raise RuntimeError("E3R W prerequisite is not PASS")
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
        validation_frame = validation.merge(
            w_validation[
                ["base_origin_id", "physical_latent", "delta_w", "y_pred"]
            ].rename(columns={"y_pred": "physical_w"}),
            on="base_origin_id",
            how="inner",
            validate="one_to_one",
        )
        if len(validation_frame) != len(validation):
            raise RuntimeError("dynamic/input-only base_origin_id mismatch")
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
                for alpha in alphas:
                    for mu in mus:
                        prediction, _ = fit_mature_residual_ar(
                            x_fit,
                            y_fit,
                            x_eval,
                            alpha=alpha,
                            mu=mu,
                            upstream_predictions=upstream,
                        )
                        losses[(MATURE_RESIDUAL_AR, profile, alpha, mu)].append(
                            mse(y_eval, prediction)
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
            "observed_mature_feature_fraction": selected_coverage,
            "hard_feature_residualization": False,
            "final_selected_candidate": str(selected),
            "final_selected_fold_losses": list(selection.final_selected_fold_losses),
            "final_selected_prediction_path": str(prediction_path.relative_to(output)),
            "final_selected_contract": contract,
            "final_prediction_loss": final_loss,
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
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
