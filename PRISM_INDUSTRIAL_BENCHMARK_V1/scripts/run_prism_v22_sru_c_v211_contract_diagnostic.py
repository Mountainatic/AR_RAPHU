from __future__ import annotations

"""Diagnostic-only SRU v2.2 wrapper with v2.1.1-compatible C semantics.

Purpose
-------
The current SRU v2.2 adapter treated the C ridge alpha as a predictive
hyperparameter and applied one-SE directly across alpha.  Frozen v2.1.1 C
semantics instead define ridge as NUMERICAL_STABILITY_ONLY, select the smallest
stable ridge, then apply an OOF input-path-preservation gate, with
BEST_ACTIVE_K_CHANNEL as the terminal fallback if the C fusion collapses.

This wrapper preserves all strict-v2.2 K numerical admission, D/M/S temporal
representations, Gamma_CT, W, A, author split, target transform and no-test-
selection rules.  It changes only the C contract for diagnosis.
"""

from typing import Any

import numpy as np

# Installs strict K numerical admission and persistence-only Gamma fallback.
import run_prism_v22_sru_full_strict  # noqa: F401
import run_prism_v22_sru_full as base
from prism_benchmark.v211_selection import input_path_preservation_gate


COMPRESSED = "ADDITIVE_COMPRESSED"
BEST_ACTIVE_K = "BEST_ACTIVE_K_CHANNEL"
_ORIGINAL_PREDICT_BRANCH_FIXED = base._predict_branch_fixed


def _coefficients(contract: dict[str, Any]) -> np.ndarray:
    return np.asarray(contract.get("coefficient", []), dtype=np.float64)


def _gate_kwargs(config: dict[str, Any]) -> dict[str, float]:
    # These are the frozen v2.1.1 SRU C input-path-preservation values.
    return {
        "minimum_variance_ratio_to_target": float(
            config["C"]["input_path_min_variance_ratio"]
        ),
        "minimum_fraction_of_best_active_k_variance_ratio": 0.10,
        "maximum_mse_ratio_vs_best_active_k": 1.02,
        "minimum_nonintercept_coefficient_abs": float(
            config["C"]["input_path_min_nonintercept_coefficient_abs"]
        ),
    }


def _v211_c_select_branch_structure(
    branch: str,
    representation: np.ndarray,
    target_delta: np.ndarray,
    development: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    audit: dict[str, Any] = {"branch": branch, "eligible": True}

    if branch == "S":
        temporal = config["temporal_representation"]
        ct_config = base.CTBasisConfig(
            dt_seconds=float(config["cadence_seconds"]),
            taus_seconds=tuple(float(value) for value in temporal["CT_tau_seconds"]),
            condition_hard_fail=float(temporal["S_condition_number_hard_fail"]),
        )
        condition = base.feature_audit(
            representation[development].reshape(len(development), -1), ct_config
        )
        audit["conditioning"] = condition.__dict__
        if not condition.passed_conditioning:
            audit["eligible"] = False
            audit["reason"] = "S_CONDITIONING_REJECTED_BEFORE_K"
            return {
                "branch": branch,
                "eligible": False,
                "selected_by_channel": {},
                "selected_alpha": None,
                "selected_family": None,
                "best_active_k_channel": None,
                "audit": audit,
            }

    selected_by_channel: dict[int, tuple[str, int]] = {}
    k_audit: dict[str, Any] = {}
    for channel in range(representation.shape[1]):
        selected, result = base._select_k_channel(
            representation, target_delta, development, channel, config
        )
        selected_by_channel[channel] = selected
        k_audit[str(channel)] = result

    active = [
        channel
        for channel, candidate in selected_by_channel.items()
        if candidate[0] != base.K_ZERO
    ]
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
            "selected_family": None,
            "best_active_k_channel": None,
            "audit": audit,
        }

    active_order = sorted(active)
    alpha_grid = sorted(float(value) for value in config["C"]["ridge_alpha_grid"])
    minimum_folds = int(config["selection"]["minimum_usable_folds"])
    folds = base._expanding_folds(
        development, int(config["selection"]["inner_expanding_folds"])
    )

    fold_targets: list[np.ndarray] = []
    c_losses = {alpha: [] for alpha in alpha_grid}
    c_predictions = {alpha: [] for alpha in alpha_grid}
    c_contracts = {alpha: [] for alpha in alpha_grid}
    k_predictions = {channel: [] for channel in active_order}
    k_losses = {channel: [] for channel in active_order}

    for fit_index, evaluation_index in folds:
        x_fit, x_evaluation, _ = base._k_contribution_matrices(
            representation,
            target_delta,
            fit_index,
            evaluation_index,
            selected_by_channel,
            config,
        )
        if x_fit.shape[1] != len(active_order):
            raise RuntimeError("active K column/order mismatch in C diagnostic")
        y_fit = target_delta[fit_index]
        y_evaluation = target_delta[evaluation_index]
        fold_targets.append(y_evaluation)

        for column, channel in enumerate(active_order):
            prediction = x_evaluation[:, column]
            k_predictions[channel].append(prediction)
            k_losses[channel].append(
                float(np.mean(np.square(y_evaluation - prediction)))
            )

        for alpha in alpha_grid:
            try:
                prediction, contract = base._ridge_fit(
                    x_fit, y_fit, x_evaluation, alpha
                )
                valid = bool(base.numerical_contract_passes(contract))
                c_predictions[alpha].append(
                    prediction if valid else np.full(len(evaluation_index), np.nan)
                )
                c_contracts[alpha].append(contract)
                c_losses[alpha].append(
                    float(np.mean(np.square(y_evaluation - prediction)))
                    if valid
                    else float("nan")
                )
            except Exception as error:
                c_predictions[alpha].append(
                    np.full(len(evaluation_index), np.nan, dtype=np.float64)
                )
                c_contracts[alpha].append(
                    {
                        "numerical_certificate": {"status": "NUMERICALLY_INVALID"},
                        "coefficient": [],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                c_losses[alpha].append(float("nan"))

    # Frozen v2.1.1 ridge semantics: smallest registered alpha with the minimum
    # number of numerically certified folds. Ridge is not a predictive one-SE
    # hyperparameter at C.
    ridge_audit: dict[str, Any] = {}
    selected_alpha: float | None = None
    for alpha in alpha_grid:
        passed = [
            bool(base.numerical_contract_passes(contract))
            for contract in c_contracts[alpha]
        ]
        ridge_audit[str(alpha)] = {
            "passing_folds": int(sum(passed)),
            "total_folds": len(passed),
            "certificates": [
                contract.get("numerical_certificate", {})
                for contract in c_contracts[alpha]
            ],
        }
        if selected_alpha is None and sum(passed) >= minimum_folds:
            selected_alpha = float(alpha)
    if selected_alpha is None:
        raise RuntimeError("no C ridge alpha passed the minimum numerical folds")

    best_channel = min(
        active_order,
        key=lambda channel: (
            float(np.mean(k_losses[channel], dtype=np.float64)),
            int(channel),
        ),
    )
    y_oof = np.concatenate(fold_targets)
    best_k_oof = np.concatenate(k_predictions[best_channel])
    c_oof = np.concatenate(c_predictions[selected_alpha])
    contracts = c_contracts[selected_alpha]
    coefficients = np.concatenate(
        [_coefficients(contract) for contract in contracts]
    )
    oof_gate = input_path_preservation_gate(
        y_oof,
        c_oof,
        best_k_oof,
        nonintercept_coefficients=coefficients,
        numerical_certificate_passed=all(
            base.numerical_contract_passes(contract) for contract in contracts
        ),
        **_gate_kwargs(config),
    )

    selected_family = COMPRESSED if bool(oof_gate.get("pass", False)) else BEST_ACTIVE_K

    # Non-selecting full-development materialization diagnostic.
    x_fit, x_same, k_contracts = base._k_contribution_matrices(
        representation,
        target_delta,
        development,
        development,
        selected_by_channel,
        config,
    )
    best_column = active_order.index(best_channel)
    best_k_full = x_same[:, best_column]
    if selected_family == COMPRESSED:
        fitted, final_contract = base._ridge_fit(
            x_fit,
            target_delta[development],
            x_same,
            selected_alpha,
        )
        final_numeric = bool(base.numerical_contract_passes(final_contract))
        full_gate = input_path_preservation_gate(
            target_delta[development],
            fitted,
            best_k_full,
            nonintercept_coefficients=_coefficients(final_contract),
            numerical_certificate_passed=final_numeric,
            **_gate_kwargs(config),
        )
    else:
        fitted = best_k_full.copy()
        final_contract = {
            "family": BEST_ACTIVE_K,
            "channel": int(best_channel),
            "coefficient": [1.0],
            "intercept": 0.0,
            "numerical_certificate": {"status": "PASS"},
        }
        full_gate = input_path_preservation_gate(
            target_delta[development],
            fitted,
            best_k_full,
            nonintercept_coefficients=np.asarray([1.0]),
            numerical_certificate_passed=True,
            **_gate_kwargs(config),
        )

    audit["C"] = {
        "family": selected_family,
        "selected_alpha": selected_alpha,
        "ridge_semantics": "NUMERICAL_STABILITY_ONLY",
        "ridge_selection": "SMALLEST_REGISTERED_ALPHA_PASSING_MINIMUM_NUMERICAL_FOLDS",
        "ridge_audit": ridge_audit,
        "best_active_k_channel": int(best_channel),
        "best_active_k_fold_losses": k_losses[best_channel],
        "selected_c_fold_losses": c_losses[selected_alpha],
        "input_path_preservation": oof_gate,
        "full_development_confirmation": full_gate,
        "fallback_used": selected_family == BEST_ACTIVE_K,
        "fallback_order": [COMPRESSED, BEST_ACTIVE_K],
        "fit_k_contract_count": len(k_contracts),
    }
    audit["reason"] = (
        "PASS" if selected_family == COMPRESSED else "C_FALLBACK_TO_BEST_ACTIVE_K"
    )
    return {
        "branch": branch,
        "eligible": True,
        "selected_by_channel": selected_by_channel,
        "selected_alpha": selected_alpha,
        "selected_family": selected_family,
        "best_active_k_channel": int(best_channel),
        "audit": audit,
    }


def _v211_c_predict_branch_fixed(
    representation: np.ndarray,
    target_delta: np.ndarray,
    fit_index: np.ndarray,
    evaluation_index: np.ndarray,
    structure: dict[str, Any],
    config: dict[str, Any],
):
    if not structure["eligible"]:
        return np.zeros(len(evaluation_index), dtype=np.float64), {
            "branch": structure["branch"],
            "status": "EXACT_ZERO_INELIGIBLE_BRANCH",
        }

    if structure.get("selected_family") != BEST_ACTIVE_K:
        return _ORIGINAL_PREDICT_BRANCH_FIXED(
            representation,
            target_delta,
            fit_index,
            evaluation_index,
            structure,
            config,
        )

    channel = int(structure["best_active_k_channel"])
    candidate = structure["selected_by_channel"][channel]
    contract = base._fit_k(
        representation[fit_index, channel, :],
        target_delta[fit_index],
        candidate,
        config,
    )
    if not base.numerical_contract_passes(contract):
        raise RuntimeError("BEST_ACTIVE_K fallback failed numerical certificate")
    prediction = base.predict_contract(
        representation[evaluation_index, channel, :], contract
    )
    return prediction, {
        "branch": structure["branch"],
        "status": "PASS_BEST_ACTIVE_K_FALLBACK",
        "family": BEST_ACTIVE_K,
        "channel": channel,
        "k_contract": contract,
    }


base._select_branch_structure = _v211_c_select_branch_structure
base._predict_branch_fixed = _v211_c_predict_branch_fixed


if __name__ == "__main__":
    raise SystemExit(base.main())
