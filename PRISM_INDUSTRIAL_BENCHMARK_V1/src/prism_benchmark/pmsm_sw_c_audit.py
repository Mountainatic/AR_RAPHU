from __future__ import annotations

"""Train-only corrected-C audit for the preregistered PMSM_SW D branch.

The D channel structures are treated as immutable upstream selections.  This
module replays those K structures on v2.1.1 assembly-common support, selects the
smallest numerically stable inherited C ridge on train-only OOF folds, applies
the inherited input-path-preservation gate, freezes the C family, and only then
materializes the validation target for a non-selecting holdout diagnostic.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .c4_prism import _fit_candidate, _numerical_status, profile_values
from .cpu_data import BaseAccessor, deterministic_subsample
from .cpu_selection import regression_metrics
from .pmsm_sw_c_contract import (
    EXPECTED_GATE,
    active_d_descriptors,
    assert_c_freeze_consistency,
)
from .pmsm_sw_d_audit import assert_prelockbox, primary_view
from .stage0 import write_json
from .v2_c import _ridge_fit
from .v211_selection import input_path_preservation_gate, numerical_contract_passes
from .v211_support import (
    apply_assembly_support,
    fold_evaluation_causal_floor,
    load_native_samples,
    support_audit,
    support_id_hash,
)
from .cpu_data import inner_folds


COMPRESSED = "ADDITIVE_COMPRESSED"
BEST_ACTIVE_K = "BEST_ACTIVE_K_CHANNEL"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _c4(project: Path) -> dict[str, Any]:
    return _json(project / "configs/cpu_model_freeze_v1.json")["c4"]


def _cap_after_support(samples, cap: int):
    index = deterministic_subsample(samples, int(cap))
    return samples.iloc[index].reset_index(drop=True)


def _k_contributions(
    accessor: BaseAccessor,
    fit,
    evaluation,
    active: list[dict[str, Any]],
    c4: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    fit_columns: list[np.ndarray] = []
    evaluation_columns: list[np.ndarray] = []
    contracts: list[dict[str, Any]] = []
    for item in active:
        channel = str(item["channel"])
        profile = tuple(int(value) for value in item["selected_profile"])
        m_tau = int(item["selected_m_tau"])
        kind = str(item["selected_kind"])
        m_x = int(item["selected_m_x"])
        lambdas = tuple(float(value) for value in item["selected_lambdas"])
        if kind == "exact_zero":
            raise RuntimeError("inactive exact-zero K entered active C replay")
        fit_values, intervals = profile_values(accessor, fit, channel, profile, m_tau)
        evaluation_values, _ = profile_values(
            accessor, evaluation, channel, profile, m_tau
        )
        combined = np.concatenate([fit_values, evaluation_values], axis=0)
        prediction, contract, parameter_count = _fit_candidate(
            fit_values,
            fit["y_true"].to_numpy(dtype=np.float64),
            combined,
            kind,
            m_x,
            lambdas,
            c4["solver"],
        )
        numerical_status, relative_kkt, condition_number = _numerical_status(
            contract, c4["numerical_thresholds"]
        )
        if numerical_status != "PASS":
            raise RuntimeError(
                f"frozen K replay became numerically invalid: {channel} {kind}"
            )
        fit_columns.append(prediction[: len(fit)])
        evaluation_columns.append(prediction[len(fit) :])
        contracts.append(
            {
                "channel": channel,
                "selected_profile": list(profile),
                "selected_kind": kind,
                "selected_m_tau": m_tau,
                "selected_m_x": m_x,
                "selected_lambdas": list(lambdas),
                "selected_intervals": [list(value) for value in intervals],
                "parameter_count": int(parameter_count),
                "relative_kkt": float(relative_kkt),
                "condition_number": float(condition_number),
                "model_contract": contract,
            }
        )
    if not fit_columns:
        return (
            np.empty((len(fit), 0), dtype=np.float64),
            np.empty((len(evaluation), 0), dtype=np.float64),
            contracts,
        )
    return np.column_stack(fit_columns), np.column_stack(evaluation_columns), contracts


def _coefficients(contract: dict[str, Any]) -> np.ndarray:
    return np.asarray(contract.get("coefficient", []), dtype=np.float64)


def run_primary_corrected_c(
    shared: Path,
    project: Path,
    d_output: Path,
    output: Path,
) -> dict[str, Any]:
    freeze = assert_c_freeze_consistency(project)
    assert_prelockbox(shared, project)
    view = primary_view(shared)
    active = active_d_descriptors(d_output)
    c4 = _c4(project)
    destination = output / "C_ONLY" / view.head.head_id / view.proxy_policy
    destination.mkdir(parents=True, exist_ok=True)

    # Selection boundary: only train targets/accessors exist until structure_frozen.
    train = load_native_samples(shared, view, "train")
    train_accessor = BaseAccessor(
        shared, view.head.dataset, "train", [item["channel"] for item in active]
    ) if active else None

    if not active:
        structure_frozen = {
            "selection_partition": "train_only",
            "validation_used_for_selection": False,
            "test_accessed": False,
            "active_channels": [],
            "selected_family": "EXACT_ZERO_INELIGIBLE_D",
            "selected_alpha": None,
            "best_active_k_channel": None,
            "fallback_used": False,
            "reason": "ALL_D_K_CHANNELS_EXACT_ZERO",
        }
    else:
        folds = inner_folds(train, int(freeze["selection_partition"]["inner_fold_count"]))
        minimum_folds = int(freeze["selection_partition"]["minimum_usable_folds"])
        fit_cap = int(freeze["selection_partition"]["row_caps"]["D_C4_fit_row_cap"])
        evaluation_cap = int(
            freeze["selection_partition"]["row_caps"]["D_C4_selection_validation_row_cap"]
        )
        alpha_grid = sorted(float(value) for value in freeze["C_contract"]["ridge_alpha_grid"])

        fold_targets: list[np.ndarray] = []
        k_predictions = {item["channel"]: [] for item in active}
        k_losses = {item["channel"]: [] for item in active}
        c_predictions = {alpha: [] for alpha in alpha_grid}
        c_contracts = {alpha: [] for alpha in alpha_grid}
        fold_support: list[dict[str, Any]] = []

        for fold_index, (fit_index, evaluation_index) in enumerate(folds):
            fit_raw = train.iloc[fit_index]
            evaluation_raw = train.iloc[evaluation_index]
            evaluation_floor = fold_evaluation_causal_floor(fit_raw, evaluation_raw)
            fit_common = apply_assembly_support(fit_raw, active)
            evaluation_common = apply_assembly_support(
                evaluation_raw, active, evaluation_floor
            )
            fit = _cap_after_support(fit_common, fit_cap)
            evaluation = _cap_after_support(evaluation_common, evaluation_cap)
            if not len(fit) or not len(evaluation):
                raise RuntimeError(f"empty C assembly fold {fold_index}")
            x_fit, x_evaluation, k_contracts = _k_contributions(
                train_accessor, fit, evaluation, active, c4
            )
            y_fit = fit["y_true"].to_numpy(dtype=np.float64)
            y_evaluation = evaluation["y_true"].to_numpy(dtype=np.float64)
            fold_targets.append(y_evaluation)
            for column, item in enumerate(active):
                channel = item["channel"]
                prediction = x_evaluation[:, column]
                k_predictions[channel].append(prediction)
                k_losses[channel].append(
                    float(np.mean(np.square(y_evaluation - prediction), dtype=np.float64))
                )
            for alpha in alpha_grid:
                try:
                    prediction, contract = _ridge_fit(
                        x_fit, y_fit, x_evaluation, alpha
                    )
                    valid = bool(numerical_contract_passes(contract))
                except Exception as error:
                    prediction = np.full(len(evaluation), np.nan, dtype=np.float64)
                    contract = {
                        "coefficient": [],
                        "numerical_certificate": {"status": "NUMERICALLY_INVALID"},
                        "error": f"{type(error).__name__}: {error}",
                    }
                    valid = False
                c_predictions[alpha].append(
                    prediction if valid else np.full(len(evaluation), np.nan)
                )
                c_contracts[alpha].append(contract)
            fold_support.append(
                {
                    "fold_index": int(fold_index),
                    "fit_common_before_cap": support_audit(fit_common),
                    "fit_after_cap": support_audit(fit),
                    "evaluation_common_before_cap": support_audit(evaluation_common),
                    "evaluation_after_cap": support_audit(evaluation),
                    "support_mask_applied_before_row_cap": True,
                    "active_k_contract_count": len(k_contracts),
                }
            )

        ridge_audit: dict[str, Any] = {}
        selected_alpha: float | None = None
        for alpha in alpha_grid:
            passed = [
                bool(numerical_contract_passes(contract))
                for contract in c_contracts[alpha]
            ]
            ridge_audit[str(alpha)] = {
                "passing_folds": int(sum(passed)),
                "total_folds": int(len(passed)),
                "certificates": [
                    contract.get("numerical_certificate", {})
                    for contract in c_contracts[alpha]
                ],
            }
            if selected_alpha is None and sum(passed) >= minimum_folds:
                selected_alpha = float(alpha)
        if selected_alpha is None:
            raise RuntimeError("no registered C ridge alpha passed minimum folds")

        best_channel = min(
            (item["channel"] for item in active),
            key=lambda channel: (
                float(np.mean(k_losses[channel], dtype=np.float64)),
                str(channel),
            ),
        )
        y_oof = np.concatenate(fold_targets)
        best_k_oof = np.concatenate(k_predictions[best_channel])
        c_oof = np.concatenate(c_predictions[selected_alpha])
        selected_contracts = c_contracts[selected_alpha]
        coefficients = np.concatenate(
            [_coefficients(contract) for contract in selected_contracts]
        )
        gate = input_path_preservation_gate(
            y_oof,
            c_oof,
            best_k_oof,
            nonintercept_coefficients=coefficients,
            numerical_certificate_passed=all(
                numerical_contract_passes(contract)
                for contract in selected_contracts
            ),
            **EXPECTED_GATE,
        )
        selected_family = COMPRESSED if bool(gate.get("pass")) else BEST_ACTIVE_K
        structure_frozen = {
            "selection_partition": "train_only",
            "validation_used_for_selection": False,
            "test_accessed": False,
            "active_channels": [item["channel"] for item in active],
            "active_k_selections": active,
            "selected_family": selected_family,
            "selected_alpha": float(selected_alpha),
            "best_active_k_channel": str(best_channel),
            "fallback_used": selected_family == BEST_ACTIVE_K,
            "ridge_semantics": "NUMERICAL_STABILIZATION_ONLY",
            "ridge_selection": "SMALLEST_STABLE_RIDGE",
            "ridge_audit": ridge_audit,
            "best_active_k_fold_losses": [float(value) for value in k_losses[best_channel]],
            "input_path_preservation_oof": gate,
            "fold_support": fold_support,
            "reason": "PASS" if selected_family == COMPRESSED else "C_FALLBACK_TO_BEST_ACTIVE_K",
        }

    # Holdout boundary: validation is materialized only after structure_frozen.
    validation = load_native_samples(shared, view, "validation")
    if not active:
        selected_validation = validation.copy()
        prediction = np.zeros(len(selected_validation), dtype=np.float64)
        validation_contract: dict[str, Any] = {"family": "EXACT_ZERO_INELIGIBLE_D"}
        train_support = train.copy()
    else:
        train_support = apply_assembly_support(train, active)
        selected_train = _cap_after_support(
            train_support,
            int(freeze["selection_partition"]["row_caps"]["D_C4_fit_row_cap"]),
        )
        selected_validation = apply_assembly_support(validation, active).reset_index(drop=True)
        validation_accessor = BaseAccessor(
            shared,
            view.head.dataset,
            "validation",
            [item["channel"] for item in active],
        )
        x_train, x_validation, k_contracts = _k_contributions(
            validation_accessor, selected_train, selected_validation, active, c4
        )
        if structure_frozen["selected_family"] == BEST_ACTIVE_K:
            best_channel = str(structure_frozen["best_active_k_channel"])
            column = [item["channel"] for item in active].index(best_channel)
            prediction = x_validation[:, column]
            validation_contract = {
                "family": BEST_ACTIVE_K,
                "channel": best_channel,
                "k_contracts": k_contracts,
            }
        else:
            prediction, c_contract = _ridge_fit(
                x_train,
                selected_train["y_true"].to_numpy(dtype=np.float64),
                x_validation,
                float(structure_frozen["selected_alpha"]),
            )
            if not numerical_contract_passes(c_contract):
                raise RuntimeError("frozen corrected-C refit failed numerical certificate")
            validation_contract = {
                "family": COMPRESSED,
                "c_contract": c_contract,
                "k_contracts": k_contracts,
            }

    y_validation = selected_validation["y_true"].to_numpy(dtype=np.float64)
    metrics = regression_metrics(y_validation, prediction)
    persistence = regression_metrics(y_validation, np.zeros(len(y_validation)))
    result = {
        "status": "PASS",
        "stage": "PMSM_SW_CORRECTED_C_IMPLEMENTATION_AUDIT",
        "evidence_role": "DEVELOPMENT_IMPLEMENTATION_AUDIT_NOT_FINAL_CONFIRMATION",
        "target_head": view.head.head_id,
        "test_accessed": False,
        "validation_used_for_selection": False,
        "structure_frozen_before_validation_materialization": True,
        "structure": structure_frozen,
        "validation_metrics": metrics,
        "persistence_validation_metrics": persistence,
        "validation_rows": int(len(selected_validation)),
        "train_assembly_support_audit": support_audit(train_support),
        "validation_assembly_support_audit": support_audit(selected_validation),
        "validation_prediction_support_hash": support_id_hash(selected_validation),
        "validation_contract": validation_contract,
    }
    write_json(destination / "RESULT.json", result)
    return result
