from __future__ import annotations

import os
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
from .v2_runtime import run_parallel
from .v21_views import sru_input_views
from .v211_config import load_v211_configs
from .v211_k import load_active_channels
from .v211_selection import (
    input_path_preservation_gate,
    numerical_contract_passes,
)


COMPRESSED = "ADDITIVE_COMPRESSED"
JOINT_BASIS = "ADDITIVE_JOINT_BASIS"
BEST_ACTIVE_K = "BEST_ACTIVE_K_CHANNEL"


def select_c_family_with_fallback(
    requested_family: str,
    family_gates: dict[str, dict[str, Any]],
) -> str:
    """Keep the registered representation order while preserving active K."""
    if requested_family not in {COMPRESSED, JOINT_BASIS}:
        raise ValueError(f"unregistered C representation: {requested_family}")
    order = [requested_family]
    if requested_family == JOINT_BASIS:
        order.append(COMPRESSED)
    order.append(BEST_ACTIVE_K)
    for family in order:
        if family == BEST_ACTIVE_K or bool(family_gates[family].get("pass", False)):
            return family
    raise RuntimeError("BEST_ACTIVE_K fallback was unexpectedly unreachable")


def _activation_strength(result: dict[str, Any]) -> float:
    audits = result.get("linear_activation_selection", {}).get(
        "activation_audit", {}
    )
    return max(
        (
            float(value.get("mean_relative_improvement", 0.0))
            for value in audits.values()
        ),
        default=0.0,
    )


def _coefficient(contract: dict[str, Any]) -> np.ndarray:
    return np.asarray(contract.get("coefficient", []), dtype=np.float64)


def _smallest_stable_alpha(
    alpha_grid: list[float],
    contracts_by_alpha: dict[float, list[dict[str, Any]]],
    minimum_folds: int,
) -> tuple[float, dict[str, Any]]:
    audit = {}
    for alpha in sorted(alpha_grid):
        contracts = contracts_by_alpha[alpha]
        passed = [numerical_contract_passes(contract) for contract in contracts]
        audit[str(alpha)] = {
            "passing_folds": int(sum(passed)),
            "total_folds": len(passed),
            "certificates": [
                contract.get("numerical_certificate", {}) for contract in contracts
            ],
        }
        if sum(passed) >= minimum_folds:
            return float(alpha), {
                "selection": "SMALLEST_STABLE_RIDGE",
                "selected_alpha": float(alpha),
                "candidates": audit,
            }
    raise RuntimeError("no C ridge alpha passed the minimum numerical folds")


def _gate_config(v211: dict[str, Any]) -> dict[str, float]:
    value = v211["C"]["input_path_preservation"]
    return {
        "minimum_variance_ratio_to_target": float(
            value["minimum_variance_ratio_to_target"]
        ),
        "minimum_fraction_of_best_active_k_variance_ratio": float(
            value["minimum_fraction_of_best_active_k_variance_ratio"]
        ),
        "maximum_mse_ratio_vs_best_active_k": float(
            value["maximum_mse_ratio_vs_best_active_k"]
        ),
        "minimum_nonintercept_coefficient_abs": float(
            value["minimum_nonintercept_coefficient_abs"]
        ),
    }


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
        v211, v21, v2 = load_v211_configs(project)
        active = load_active_channels(output, view)
        maximum = int(v2["K_module"]["active_channel_gate"]["maximum_active_channels"])
        active = sorted(
            active,
            key=lambda item: (-_activation_strength(item), item["channel"]),
        )[:maximum]
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        folds = inner_folds(train, int(v21["selection"]["inner_folds"]))
        minimum_folds = int(v21["selection"]["minimum_usable_folds"])
        alpha_grid = sorted(
            float(value)
            for value in v2["C_module"]["joint_basis"]["ridge_alpha_grid"]
        )
        families = (COMPRESSED, JOINT_BASIS)
        candidate_losses = {
            (family, alpha): [] for family in families for alpha in alpha_grid
        }
        candidate_predictions: dict[tuple[str, float], list[np.ndarray]] = {
            key: [] for key in candidate_losses
        }
        candidate_contracts: dict[tuple[str, float], list[dict[str, Any]]] = {
            key: [] for key in candidate_losses
        }
        fold_targets: list[np.ndarray] = []
        fold_best_channel_predictions: dict[str, list[np.ndarray]] = {
            item["channel"]: [] for item in active
        }
        fold_best_channel_losses: dict[str, list[float]] = {
            item["channel"]: [] for item in active
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
            fold_targets.append(evaluation_target)
            for index, channel in enumerate(features["channels"]):
                prediction = features["compressed_evaluation"][:, index]
                fold_best_channel_predictions[channel].append(prediction)
                fold_best_channel_losses[channel].append(
                    mse(evaluation_target, prediction)
                )
            for family, key in ((COMPRESSED, "compressed"), (JOINT_BASIS, "joint")):
                train_x = features[f"{key}_train"]
                evaluation_x = features[f"{key}_evaluation"]
                for alpha in alpha_grid:
                    if train_x.shape[1] == 0:
                        prediction = np.full(
                            len(evaluation),
                            float(np.mean(target, dtype=np.float64)),
                            dtype=np.float64,
                        )
                        contract = {
                            "family": family,
                            "coefficient": [],
                            "intercept": float(np.mean(target, dtype=np.float64)),
                            "alpha": alpha,
                            "numerical_certificate": {"status": "EXACT_ZERO"},
                        }
                    else:
                        prediction, contract = _ridge_fit(
                            train_x, target, evaluation_x, alpha
                        )
                        contract = {"family": family, **contract}
                    candidate_predictions[(family, alpha)].append(prediction)
                    candidate_contracts[(family, alpha)].append(contract)
                    candidate_losses[(family, alpha)].append(
                        mse(evaluation_target, prediction)
                    )

        if not active:
            intercept = float(
                np.mean(train["y_true"].to_numpy(dtype=np.float64), dtype=np.float64)
            )
            prediction = np.full(len(validation), intercept, dtype=np.float64)
            contract = {
                "family": "K_EXACT_ZERO",
                "coefficient": [],
                "intercept": intercept,
                "parameter_count": 1,
                "numerical_certificate": {"status": "EXACT_ZERO"},
            }
            selected_family = "K_EXACT_ZERO"
            selected_alpha = 0.0
            selected_fold_losses = [
                float(np.mean(np.square(values), dtype=np.float64))
                for values in fold_targets
            ]
            input_gate = {
                "status": "INPUT_PATH_COLLAPSED",
                "pass": False,
                "reason": "NO_ACTIVE_K_CHANNEL",
            }
            family_selection_json = None
            ridge_audits: dict[str, Any] = {}
            best_channel = None
            best_k_validation = np.full(len(validation), intercept, dtype=np.float64)
            final_features = {
                "channels": [],
                "channel_contracts": [],
                "global_joint_columns": [],
            }
            selection_status = "K_EXACT_ZERO"
        else:
            best_channel = min(
                fold_best_channel_losses,
                key=lambda channel: (
                    float(np.mean(fold_best_channel_losses[channel], dtype=np.float64)),
                    channel,
                ),
            )
            selected_alphas: dict[str, float] = {}
            ridge_audits = {}
            for family in families:
                selected_alphas[family], ridge_audits[family] = _smallest_stable_alpha(
                    alpha_grid,
                    {
                        alpha: candidate_contracts[(family, alpha)]
                        for alpha in alpha_grid
                    },
                    minimum_folds,
                )
            family_losses = {
                family: candidate_losses[(family, selected_alphas[family])]
                for family in families
            }
            family_selection = one_se_select(
                family_losses,
                lambda family: (0 if family == COMPRESSED else 1,),
                minimum_usable_folds=minimum_folds,
            )
            family_selection_json = family_selection.to_json()
            y_oof = np.concatenate(fold_targets)
            best_k_oof = np.concatenate(
                fold_best_channel_predictions[best_channel]
            )
            family_gates: dict[str, dict[str, Any]] = {}
            gate_parameters = _gate_config(v211)
            for family in families:
                alpha = selected_alphas[family]
                contracts = candidate_contracts[(family, alpha)]
                coefficients = np.concatenate(
                    [_coefficient(contract) for contract in contracts]
                )
                family_gates[family] = input_path_preservation_gate(
                    y_oof,
                    np.concatenate(candidate_predictions[(family, alpha)]),
                    best_k_oof,
                    nonintercept_coefficients=coefficients,
                    numerical_certificate_passed=all(
                        numerical_contract_passes(contract) for contract in contracts
                    ),
                    **gate_parameters,
                )
            requested_family = str(family_selection.selected)
            oof_selected_family = select_c_family_with_fallback(
                requested_family, family_gates
            )
            final_train = _cap(
                train, int(v2["row_caps"]["joint_physical_fit"])
            )
            final_features = fit_physical_features(
                shared,
                view,
                final_train,
                validation,
                active,
                v2,
                fit_split="train",
                evaluation_split="validation",
            )
            best_index = final_features["channels"].index(best_channel)
            best_k_validation = final_features["compressed_evaluation"][:, best_index]
            final_order = [oof_selected_family]
            if oof_selected_family == JOINT_BASIS:
                final_order.append(COMPRESSED)
            if BEST_ACTIVE_K not in final_order:
                final_order.append(BEST_ACTIVE_K)
            final_attempts: dict[str, dict[str, Any]] = {}
            selected_payload: dict[str, Any] | None = None
            for family in final_order:
                if family == BEST_ACTIVE_K:
                    family_prediction = best_k_validation.copy()
                    family_contract = {
                        "family": BEST_ACTIVE_K,
                        "channel": best_channel,
                        "coefficient": [1.0],
                        "intercept": 0.0,
                        "alpha": 0.0,
                        "parameter_count": 1,
                        "numerical_certificate": {"status": "PASS"},
                    }
                    family_alpha = 0.0
                    family_fold_losses = fold_best_channel_losses[best_channel]
                    family_oof_gate = input_path_preservation_gate(
                        y_oof,
                        best_k_oof,
                        best_k_oof,
                        nonintercept_coefficients=np.asarray([1.0]),
                        numerical_certificate_passed=True,
                        **gate_parameters,
                    )
                else:
                    key = "joint" if family == JOINT_BASIS else "compressed"
                    train_x = final_features[f"{key}_train"]
                    validation_x = final_features[f"{key}_evaluation"]
                    family_alpha = selected_alphas[family]
                    family_prediction, family_contract = _ridge_fit(
                        train_x,
                        final_train["y_true"].to_numpy(dtype=np.float64),
                        validation_x,
                        family_alpha,
                    )
                    family_contract = {
                        "family": family,
                        "parameter_count": len(family_contract["coefficient"]) + 1,
                        **family_contract,
                    }
                    family_fold_losses = family_losses[family]
                    family_oof_gate = family_gates[family]
                validation_gate = input_path_preservation_gate(
                    validation["y_true"].to_numpy(dtype=np.float64),
                    family_prediction,
                    best_k_validation,
                    nonintercept_coefficients=_coefficient(family_contract),
                    numerical_certificate_passed=numerical_contract_passes(
                        family_contract
                    ),
                    **gate_parameters,
                )
                combined_pass = bool(
                    family_oof_gate.get("pass", False)
                    and validation_gate.get("pass", False)
                )
                combined_gate = {
                    **family_oof_gate,
                    "status": "INPUT_PATH_PRESERVED"
                    if combined_pass
                    else "INPUT_PATH_COLLAPSED",
                    "pass": combined_pass,
                    "oof_confirmation_passed": bool(
                        family_oof_gate.get("pass", False)
                    ),
                    "validation_confirmation": validation_gate,
                }
                payload = {
                    "family": family,
                    "alpha": family_alpha,
                    "prediction": family_prediction,
                    "contract": family_contract,
                    "fold_losses": family_fold_losses,
                    "gate": combined_gate,
                }
                final_attempts[family] = {
                    "alpha": family_alpha,
                    "input_path_preservation": combined_gate,
                }
                if combined_pass:
                    selected_payload = payload
                    break
                if family == BEST_ACTIVE_K:
                    selected_payload = payload
            if selected_payload is None:
                raise RuntimeError("C fallback order produced no final candidate")
            selected_family = str(selected_payload["family"])
            selected_alpha = float(selected_payload["alpha"])
            prediction = np.asarray(selected_payload["prediction"], dtype=np.float64)
            contract = dict(selected_payload["contract"])
            selected_fold_losses = list(selected_payload["fold_losses"])
            input_gate = {
                **selected_payload["gate"],
                "requested_family": requested_family,
                "oof_selected_family": oof_selected_family,
                "family_gates": family_gates,
                "final_attempts": final_attempts,
            }
            if not input_gate["pass"]:
                selection_status = "C_INPUT_PATH_COLLAPSE_BUG"
            elif selected_family == BEST_ACTIVE_K:
                selection_status = "C_FALLBACK_TO_BEST_ACTIVE_K"
            elif selected_family != oof_selected_family:
                selection_status = "C_FALLBACK_TO_COMPRESSED"
            else:
                selection_status = "C_REPRESENTATION_SELECTED"

        frame = validation[
            ["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]
        ].copy()
        frame["y_pred"] = prediction
        frame["model"] = "PRISM_V2_1_1_K_C"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        status = "PASS" if selection_status != "C_INPUT_PATH_COLLAPSE_BUG" else selection_status
        result = {
            "status": status,
            "stage": "E2R_C",
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "active_channels": final_features["channels"],
            "best_active_k_channel": best_channel,
            "selected_family": selected_family,
            "selected_alpha": float(selected_alpha),
            "selection_status": selection_status,
            "ridge_semantics": "NUMERICAL_STABILITY_ONLY",
            "channel_contracts": final_features.get("channel_contracts", []),
            "global_joint_columns": final_features.get("global_joint_columns", []),
            "fusion_contract": contract,
            "family_selection": family_selection_json,
            "ridge_audits": ridge_audits,
            "candidate_fold_losses": {
                str(key): values for key, values in candidate_losses.items()
            },
            "best_active_k_fold_losses": []
            if best_channel is None
            else fold_best_channel_losses[best_channel],
            "input_path_preservation": input_gate,
            "input_path_nonzero": bool(input_gate.get("pass", False)),
            "final_selected_candidate": selected_family,
            "final_selected_fold_losses": list(selected_fold_losses),
            "final_selected_prediction_path": str(
                prediction_path.relative_to(output)
            ),
            "final_selected_contract": contract,
            "final_prediction_loss": mse(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
        }
    except Exception as error:
        result = {
            "status": "SOLVER_FAILED_RETAINED",
            "stage": "E2R_C",
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


def run_e2r_c(shared: Path, project: Path, output: Path) -> dict[str, Any]:
    views = sru_input_views(shared)
    results = run_parallel(
        run_c_view,
        [(shared, project, output, view) for view in views],
        int(os.environ.get("PRISM_V211_WORKERS", "8")),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "4")),
        label="PRISM_V211_E2R_C",
    )
    summary = {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in results)
        else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": "E2R_C",
        "views": len(results),
        "pass": sum(item["status"] == "PASS" for item in results),
        "input_paths_nonzero": sum(
            bool(item.get("input_path_nonzero")) for item in results
        ),
        "fallback_to_best_active_k": sum(
            item.get("selection_status") == "C_FALLBACK_TO_BEST_ACTIVE_K"
            for item in results
        ),
        "collapse_bugs": sum(
            item.get("selection_status") == "C_INPUT_PATH_COLLAPSE_BUG"
            for item in results
        ),
        "test_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / "C" / "SUMMARY.json", summary)
    return summary
