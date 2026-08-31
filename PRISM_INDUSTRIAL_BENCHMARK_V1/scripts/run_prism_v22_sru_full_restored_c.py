from __future__ import annotations

"""Restore the frozen PRISM v2.1.1 SRU C semantics inside the v2.2 adapter.

This is a contract-restoration wrapper, not a score-tuning patch.  It imports
`run_prism_v22_sru_full_strict`, retaining its strict K numerical admission,
and restores the already-frozen v2.1.1 SRU rules that C:

1. treats ridge only as numerical stabilization and selects the smallest stable
   alpha rather than using larger alpha as structural simplicity;
2. must preserve an already-active K input path;
3. falls back to BEST_ACTIVE_K_CHANNEL instead of silently shrinking the input
   prediction to zero.

The historical frozen SRU thresholds are retained: minimum C variance is the
larger of 1e-8 of target variance and 10% of best-active-K variance; C MSE may
be at most 1.02 times best-active-K MSE; and at least one non-intercept
coefficient must exceed 1e-10 in magnitude.
"""

from typing import Any

import numpy as np

import run_prism_v22_sru_full_strict as strict

base = strict.base

BEST_ACTIVE_K = "BEST_ACTIVE_K_CHANNEL"
MINIMUM_FRACTION_OF_BEST_K_VARIANCE = 0.10
MAXIMUM_MSE_RATIO_VS_BEST_K = 1.02


def _mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(y) - np.asarray(prediction)), dtype=np.float64))


def _smallest_stable_c_alpha(
    representation: np.ndarray,
    target_delta: np.ndarray,
    development: np.ndarray,
    selected_by_channel: dict[int, tuple[str, int]],
    config: dict[str, Any],
) -> tuple[float, dict[str, Any], dict[int, list[float]], int]:
    folds = base._expanding_folds(
        development, int(config["selection"]["inner_expanding_folds"])
    )
    minimum_folds = int(config["selection"]["minimum_usable_folds"])
    alpha_grid = sorted(float(value) for value in config["C"]["ridge_alpha_grid"])
    active_channels = sorted(
        channel
        for channel, candidate in selected_by_channel.items()
        if candidate[0] != base.K_ZERO
    )
    channel_losses: dict[int, list[float]] = {channel: [] for channel in active_channels}
    alpha_audit: dict[str, Any] = {
        str(alpha): {"folds": [], "full_refit_pass": False}
        for alpha in alpha_grid
    }

    for fold_number, (fit_index, evaluation_index) in enumerate(folds):
        x_fit, x_evaluation, _ = base._k_contribution_matrices(
            representation,
            target_delta,
            fit_index,
            evaluation_index,
            selected_by_channel,
            config,
        )
        y_fit = target_delta[fit_index]
        y_evaluation = target_delta[evaluation_index]
        if x_evaluation.shape[1] != len(active_channels):
            raise RuntimeError("C active-channel/contribution mismatch")
        for column, channel in enumerate(active_channels):
            channel_losses[channel].append(_mse(y_evaluation, x_evaluation[:, column]))
        for alpha in alpha_grid:
            try:
                prediction, contract = base._ridge_fit(x_fit, y_fit, x_evaluation, alpha)
                valid = bool(
                    np.isfinite(prediction).all()
                    and base.numerical_contract_passes(contract)
                )
                alpha_audit[str(alpha)]["folds"].append(
                    {
                        "fold": int(fold_number),
                        "pass": valid,
                        "mse": _mse(y_evaluation, prediction) if valid else None,
                        "certificate": contract.get("numerical_certificate", {}),
                    }
                )
            except Exception as error:
                alpha_audit[str(alpha)]["folds"].append(
                    {
                        "fold": int(fold_number),
                        "pass": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    x_development, x_same, _ = base._k_contribution_matrices(
        representation,
        target_delta,
        development,
        development,
        selected_by_channel,
        config,
    )
    selected_alpha = None
    for alpha in alpha_grid:
        try:
            _, contract = base._ridge_fit(
                x_development, target_delta[development], x_same, alpha
            )
            full_pass = bool(base.numerical_contract_passes(contract))
            alpha_audit[str(alpha)]["full_refit_pass"] = full_pass
            alpha_audit[str(alpha)]["full_refit_certificate"] = contract.get(
                "numerical_certificate", {}
            )
        except Exception as error:
            full_pass = False
            alpha_audit[str(alpha)]["full_refit_error"] = (
                f"{type(error).__name__}: {error}"
            )
        fold_passes = sum(
            bool(item.get("pass", False))
            for item in alpha_audit[str(alpha)]["folds"]
        )
        alpha_audit[str(alpha)]["passing_folds"] = int(fold_passes)
        if selected_alpha is None and full_pass and fold_passes >= minimum_folds:
            selected_alpha = float(alpha)

    if selected_alpha is None:
        raise RuntimeError("no C ridge alpha passed smallest-stable numerical admission")

    best_channel = min(
        active_channels,
        key=lambda channel: (
            float(np.mean(channel_losses[channel], dtype=np.float64)), channel
        ),
    )
    return selected_alpha, alpha_audit, channel_losses, int(best_channel)


def _c_input_path_gate(
    representation: np.ndarray,
    target_delta: np.ndarray,
    development: np.ndarray,
    selected_by_channel: dict[int, tuple[str, int]],
    selected_alpha: float,
    best_channel: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    folds = base._expanding_folds(
        development, int(config["selection"]["inner_expanding_folds"])
    )
    active_channels = sorted(
        channel
        for channel, candidate in selected_by_channel.items()
        if candidate[0] != base.K_ZERO
    )
    best_column = active_channels.index(best_channel)
    targets = []
    c_predictions = []
    best_k_predictions = []
    coefficients = []
    numerical_passes = []

    for fit_index, evaluation_index in folds:
        x_fit, x_evaluation, _ = base._k_contribution_matrices(
            representation,
            target_delta,
            fit_index,
            evaluation_index,
            selected_by_channel,
            config,
        )
        prediction, contract = base._ridge_fit(
            x_fit,
            target_delta[fit_index],
            x_evaluation,
            float(selected_alpha),
        )
        targets.append(target_delta[evaluation_index])
        c_predictions.append(prediction)
        best_k_predictions.append(x_evaluation[:, best_column])
        coefficients.extend(
            np.asarray(contract.get("coefficient", []), dtype=np.float64).tolist()
        )
        numerical_passes.append(bool(base.numerical_contract_passes(contract)))

    y = np.concatenate(targets)
    c_prediction = np.concatenate(c_predictions)
    best_k = np.concatenate(best_k_predictions)
    target_variance = float(np.var(y, dtype=np.float64))
    denominator = max(target_variance, np.finfo(np.float64).tiny)
    c_variance_ratio = float(np.var(c_prediction, dtype=np.float64) / denominator)
    best_k_variance_ratio = float(np.var(best_k, dtype=np.float64) / denominator)
    required_variance_ratio = max(
        float(config["C"]["input_path_min_variance_ratio"]),
        MINIMUM_FRACTION_OF_BEST_K_VARIANCE * best_k_variance_ratio,
    )
    c_mse = _mse(y, c_prediction)
    best_k_mse = _mse(y, best_k)
    coefficient_max = float(
        np.max(np.abs(np.asarray(coefficients, dtype=np.float64)), initial=0.0)
    )
    variance_pass = bool(c_variance_ratio >= required_variance_ratio)
    mse_pass = bool(c_mse <= MAXIMUM_MSE_RATIO_VS_BEST_K * best_k_mse)
    coefficient_pass = bool(
        coefficient_max
        >= float(config["C"]["input_path_min_nonintercept_coefficient_abs"])
    )
    numerical_pass = bool(numerical_passes and all(numerical_passes))
    passed = bool(variance_pass and mse_pass and coefficient_pass and numerical_pass)
    return {
        "status": "INPUT_PATH_PRESERVED" if passed else "INPUT_PATH_COLLAPSED",
        "pass": passed,
        "prediction_variance_ratio_to_target": c_variance_ratio,
        "best_active_k_variance_ratio_to_target": best_k_variance_ratio,
        "required_variance_ratio": required_variance_ratio,
        "mse": c_mse,
        "best_active_k_mse": best_k_mse,
        "mse_ratio_vs_best_active_k": c_mse / max(best_k_mse, np.finfo(np.float64).tiny),
        "maximum_mse_ratio_vs_best_active_k": MAXIMUM_MSE_RATIO_VS_BEST_K,
        "maximum_nonintercept_coefficient_abs": coefficient_max,
        "variance_pass": variance_pass,
        "mse_pass": mse_pass,
        "coefficient_pass": coefficient_pass,
        "numerical_pass": numerical_pass,
        "frozen_source": "PRISM_V2_1_1_SRU_IMPLEMENTATION_CORRECTION_V1",
    }


def restored_select_branch_structure(
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
                "c_family": None,
                "best_active_channel": None,
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
            "c_family": None,
            "best_active_channel": None,
            "audit": audit,
        }

    selected_alpha, ridge_audit, channel_losses, best_channel = _smallest_stable_c_alpha(
        representation,
        target_delta,
        development,
        selected_by_channel,
        config,
    )
    gate = _c_input_path_gate(
        representation,
        target_delta,
        development,
        selected_by_channel,
        selected_alpha,
        best_channel,
        config,
    )
    if gate["pass"]:
        c_family = "ADDITIVE_COMPRESSED"
        fallback = False
    else:
        c_family = BEST_ACTIVE_K
        fallback = True

    audit["C"] = {
        "family": c_family,
        "selected_alpha": float(selected_alpha) if not fallback else 0.0,
        "ridge_semantics": "NUMERICAL_STABILITY_ONLY",
        "ridge_selection": "SMALLEST_STABLE_RIDGE",
        "ridge_audit": ridge_audit,
        "input_path_preservation": gate,
        "best_active_channel": int(best_channel),
        "best_active_channel_fold_losses": channel_losses[int(best_channel)],
        "fallback_to_best_active_k": fallback,
        "silent_k_erasure_forbidden": True,
        "frozen_source": "PRISM_V2_1_1_SRU_IMPLEMENTATION_CORRECTION_V1",
    }
    audit["reason"] = (
        "PASS" if not fallback else "C_FALLBACK_TO_BEST_ACTIVE_K_CHANNEL"
    )
    return {
        "branch": branch,
        "eligible": True,
        "selected_by_channel": selected_by_channel,
        "selected_alpha": float(selected_alpha) if not fallback else 0.0,
        "c_family": c_family,
        "best_active_channel": int(best_channel),
        "audit": audit,
    }


def restored_predict_branch_fixed(
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
    if structure.get("c_family") == BEST_ACTIVE_K:
        channel = int(structure["best_active_channel"])
        candidate = structure["selected_by_channel"][channel]
        contract = base._fit_k(
            representation[fit_index, channel, :],
            target_delta[fit_index],
            candidate,
            config,
        )
        if not base.numerical_contract_passes(contract):
            raise RuntimeError(
                f"BEST_ACTIVE_K refit became numerically invalid: branch={structure['branch']} channel={channel}"
            )
        prediction = base.predict_contract(
            representation[evaluation_index, channel, :], contract
        )
        return prediction, {
            "branch": structure["branch"],
            "status": "PASS_BEST_ACTIVE_K_FALLBACK",
            "c_family": BEST_ACTIVE_K,
            "channel": channel,
            "candidate": str(candidate),
            "k_contract": contract,
        }

    x_fit, x_evaluation, k_contracts = base._k_contribution_matrices(
        representation,
        target_delta,
        fit_index,
        evaluation_index,
        structure["selected_by_channel"],
        config,
    )
    prediction, c_contract = base._ridge_fit(
        x_fit,
        target_delta[fit_index],
        x_evaluation,
        float(structure["selected_alpha"]),
    )
    if not base.numerical_contract_passes(c_contract):
        raise RuntimeError(
            f"restored C refit became numerically invalid: branch={structure['branch']}"
        )
    return prediction, {
        "branch": structure["branch"],
        "status": "PASS",
        "c_family": "ADDITIVE_COMPRESSED",
        "k_contracts": k_contracts,
        "c_contract": c_contract,
    }


base._select_branch_structure = restored_select_branch_structure
base._predict_branch_fixed = restored_predict_branch_fixed


if __name__ == "__main__":
    raise SystemExit(base.main())
