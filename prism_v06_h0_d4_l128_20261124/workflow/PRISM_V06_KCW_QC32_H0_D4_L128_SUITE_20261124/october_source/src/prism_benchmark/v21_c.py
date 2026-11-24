from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .cpu_data import ViewSpec, inner_folds, load_samples, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_c import _ridge_fit, fit_physical_features
from .v2_k import _cap
from .v2_selection import one_se_select
from .v21_config import load_v21_and_v2_config
from .v21_k import load_active_channels
from .v21_selection import guarded_local_one_se_select
from .v21_views import sru_input_views


COMPRESSED = "ADDITIVE_COMPRESSED"
JOINT_BASIS = "ADDITIVE_JOINT_BASIS"


def _target_mean(values: Any) -> float:
    """Return an FP64 mean without passing NumPy kwargs to pandas."""
    return float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))


def run_c_view(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
) -> dict[str, Any]:
    started = time.time()
    destination = output / "DEVELOPMENT" / "C" / view.head.head_id / view.proxy_policy
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v21, v2 = load_v21_and_v2_config(project)
        active = load_active_channels(output, view)
        maximum = int(v2["K_module"]["active_channel_gate"]["maximum_active_channels"])
        active = sorted(
            active,
            key=lambda item: (
                -float(
                    item.get("activation_selection", {})
                    .get("activation_audit", {})
                    .get("LINEAR_DISTRIBUTED_LAG", {})
                    .get("mean_relative_improvement", 0.0)
                ),
                item["channel"],
            ),
        )[:maximum]
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        folds = inner_folds(train, int(v21["selection"]["inner_folds"]))
        alpha_grid = [float(value) for value in v2["C_module"]["joint_basis"]["ridge_alpha_grid"]]
        candidate_losses: dict[tuple[str, float], list[float]] = {
            (family, alpha): []
            for family in (COMPRESSED, JOINT_BASIS)
            for alpha in alpha_grid
        }
        for fit_index, evaluation_index in folds:
            fit = _cap(train.iloc[fit_index], int(v2["row_caps"]["joint_physical_fit"]))
            evaluation = _cap(
                train.iloc[evaluation_index],
                int(v2["row_caps"]["validation_selection_per_fold"]),
            )
            features = fit_physical_features(
                shared,
                view,
                fit,
                evaluation,
                active,
                v2,
                fit_split="train",
                evaluation_split="train",
            )
            target = fit["y_true"].to_numpy(dtype=np.float64)
            evaluation_target = evaluation["y_true"].to_numpy(dtype=np.float64)
            for family, key in ((COMPRESSED, "compressed"), (JOINT_BASIS, "joint")):
                train_x = features[f"{key}_train"]
                evaluation_x = features[f"{key}_evaluation"]
                for alpha in alpha_grid:
                    if train_x.shape[1] == 0:
                        prediction = np.full(
                            len(evaluation),
                            _target_mean(target),
                            dtype=np.float64,
                        )
                    else:
                        prediction = _ridge_fit(train_x, target, evaluation_x, alpha)[0]
                    candidate_losses[(family, alpha)].append(
                        mse(evaluation_target, prediction)
                    )
        minimum_folds = int(v21["selection"]["minimum_usable_folds"])
        compressed_selection = one_se_select(
            {key: value for key, value in candidate_losses.items() if key[0] == COMPRESSED},
            lambda value: (-value[1],),
            minimum_usable_folds=minimum_folds,
        )
        joint_selection = one_se_select(
            {key: value for key, value in candidate_losses.items() if key[0] == JOINT_BASIS},
            lambda value: (-value[1],),
            minimum_usable_folds=minimum_folds,
        )
        neutral = compressed_selection.selected
        joint = joint_selection.selected
        family_selection = guarded_local_one_se_select(
            {neutral: candidate_losses[neutral], joint: candidate_losses[joint]},
            lambda value: (0 if value[0] == COMPRESSED else 1, -value[1]),
            neutral=neutral,
            minimum_relative_improvement=float(v21["selection"]["minimum_relative_improvement"]["C"]),
            minimum_positive_fraction=float(v21["selection"]["minimum_positive_fold_fraction"]),
            minimum_usable_folds=minimum_folds,
        )
        selected_family, selected_alpha = family_selection.final_selected_candidate
        final_train = _cap(train, int(v2["row_caps"]["joint_physical_fit"]))
        features = fit_physical_features(
            shared,
            view,
            final_train,
            validation,
            active,
            v2,
            fit_split="train",
            evaluation_split="validation",
        )
        key = "joint" if selected_family == JOINT_BASIS else "compressed"
        train_x = features[f"{key}_train"]
        validation_x = features[f"{key}_evaluation"]
        if train_x.shape[1]:
            prediction, contract = _ridge_fit(
                train_x,
                final_train["y_true"].to_numpy(dtype=np.float64),
                validation_x,
                float(selected_alpha),
            )
        else:
            intercept = _target_mean(final_train["y_true"])
            prediction = np.full(len(validation), intercept, dtype=np.float64)
            contract = {
                "status": "K_EXACT_ZERO",
                "mean": [],
                "scale": [],
                "coefficient": [],
                "intercept": intercept,
                "alpha": float(selected_alpha),
                "parameter_count": 1,
            }
        frame = validation[
            ["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]
        ].copy()
        frame["y_pred"] = prediction
        frame["model"] = "PRISM_V2_1_K_C"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        input_path_nonzero = bool(
            train_x.shape[1]
            and np.var(prediction, dtype=np.float64) > 1e-12
            and any(item.get("active") for item in active)
        )
        result = {
            "status": "PASS",
            "stage": "E2_C",
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "active_channels": features["channels"],
            "selected_family": selected_family,
            "selected_alpha": float(selected_alpha),
            "selected_pairs": [],
            "pairwise_interactions_enabled": False,
            "channel_contracts": features["channel_contracts"],
            "global_joint_columns": features.get("global_joint_columns", []),
            "fusion_contract": contract,
            "compressed_selection": compressed_selection.to_json(),
            "joint_selection": joint_selection.to_json(),
            "family_selection": family_selection.to_json(),
            "candidate_fold_losses": {
                str(key): values for key, values in candidate_losses.items()
            },
            "input_path_nonzero": input_path_nonzero,
            "final_selected_candidate": selected_family,
            "final_selected_fold_losses": list(family_selection.final_selected_fold_losses),
            "final_selected_prediction_path": str(prediction_path.relative_to(output)),
            "final_selected_contract": contract,
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(frame["y_true"].to_numpy(dtype=np.float64), prediction),
        }
    except Exception as error:
        result = {
            "status": "SOLVER_FAILED_RETAINED",
            "stage": "E2_C",
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


def run_e2_c(shared: Path, project: Path, output: Path) -> dict[str, Any]:
    results = [run_c_view(shared, project, output, view) for view in sru_input_views(shared)]
    summary = {
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": "E2_C",
        "views": len(results),
        "pass": sum(item["status"] == "PASS" for item in results),
        "input_paths_nonzero": sum(bool(item.get("input_path_nonzero")) for item in results),
        "test_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / "C" / "SUMMARY.json", summary)
    return summary
