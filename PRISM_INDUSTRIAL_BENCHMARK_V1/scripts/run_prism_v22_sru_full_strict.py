from __future__ import annotations

"""Strict numerical-refit wrapper for the SRU v2.2 full KWA benchmark.

This wrapper intentionally leaves the full runner unchanged and replaces only
its K channel-selection hook. An active K structural candidate is eligible for
guarded one-SE selection only when that same structure is numerically valid on:

1. every registered K/C inner expanding fold;
2. every registered Gamma_CT OOF expanding fold used later to construct W
   residuals; and
3. the full development refit.

No numerical threshold is relaxed and no candidate is changed after selection.
This makes the matched adapter inherit the v2.1.1 principle that a selected K
structure must certify everywhere it will be materialized downstream.
"""

import numpy as np

import run_prism_v22_sru_full as base


def _numeric_fold_audit(
    representation: np.ndarray,
    target_delta: np.ndarray,
    channel: int,
    candidates: list[tuple[str, int]],
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict,
) -> dict[str, list[dict]]:
    audit = {str(candidate): [] for candidate in candidates}
    for fold_number, (fit_index, _) in enumerate(folds):
        fit_values = representation[fit_index, channel, :]
        y_fit = target_delta[fit_index]
        for candidate in candidates:
            try:
                contract = base._fit_k(fit_values, y_fit, candidate, config)
                valid = bool(base.numerical_contract_passes(contract))
                audit[str(candidate)].append(
                    {
                        "fold": int(fold_number),
                        "pass": valid,
                        "certificate": contract.get("certificate", {}),
                    }
                )
            except Exception as error:
                audit[str(candidate)].append(
                    {
                        "fold": int(fold_number),
                        "pass": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return audit


def strict_select_k_channel(
    representation: np.ndarray,
    target_delta: np.ndarray,
    development: np.ndarray,
    channel: int,
    config: dict,
):
    candidates = base._k_candidates(config)
    selection_folds = base._expanding_folds(
        development, int(config["selection"]["inner_expanding_folds"])
    )
    gamma_oof_folds = base._expanding_folds(
        development, int(config["selection"]["gamma_oof_expanding_folds"])
    )

    losses = {candidate: [] for candidate in candidates}
    selection_numeric = {str(candidate): [] for candidate in candidates}

    # Predictive scoring remains exactly on the registered K/C selection folds.
    for fold_number, (fit_index, evaluation_index) in enumerate(selection_folds):
        fit_values = representation[fit_index, channel, :]
        evaluation_values = representation[evaluation_index, channel, :]
        y_fit = target_delta[fit_index]
        y_evaluation = target_delta[evaluation_index]
        for candidate in candidates:
            try:
                contract = base._fit_k(fit_values, y_fit, candidate, config)
                valid = bool(base.numerical_contract_passes(contract))
                if valid:
                    prediction = base.predict_contract(evaluation_values, contract)
                    loss = float(np.mean(np.square(y_evaluation - prediction)))
                else:
                    loss = float("nan")
                selection_numeric[str(candidate)].append(
                    {
                        "fold": int(fold_number),
                        "pass": valid,
                        "certificate": contract.get("certificate", {}),
                    }
                )
            except Exception as error:
                loss = float("nan")
                selection_numeric[str(candidate)].append(
                    {
                        "fold": int(fold_number),
                        "pass": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            losses[candidate].append(loss)

    # W is trained from Gamma_CT OOF residuals.  These folds may have different
    # boundaries from the K/C selection folds, so certify them explicitly too.
    gamma_oof_numeric = _numeric_fold_audit(
        representation,
        target_delta,
        channel,
        candidates,
        gamma_oof_folds,
        config,
    )

    full_refit = {}
    stable_candidates = []
    for candidate in candidates:
        try:
            contract = base._fit_k(
                representation[development, channel, :],
                target_delta[development],
                candidate,
                config,
            )
            full_valid = bool(base.numerical_contract_passes(contract))
            full_refit[str(candidate)] = {
                "pass": full_valid,
                "certificate": contract.get("certificate", {}),
            }
        except Exception as error:
            full_valid = False
            full_refit[str(candidate)] = {
                "pass": False,
                "error": f"{type(error).__name__}: {error}",
            }

        selection_all_valid = bool(selection_numeric[str(candidate)]) and all(
            bool(item.get("pass", False))
            for item in selection_numeric[str(candidate)]
        )
        gamma_all_valid = bool(gamma_oof_numeric[str(candidate)]) and all(
            bool(item.get("pass", False))
            for item in gamma_oof_numeric[str(candidate)]
        )
        stable = bool(full_valid and selection_all_valid and gamma_all_valid)
        full_refit[str(candidate)].update(
            {
                "all_selection_folds_pass": selection_all_valid,
                "all_gamma_oof_folds_pass": gamma_all_valid,
                "eligible_for_selection": stable,
            }
        )
        if stable:
            stable_candidates.append(candidate)
        else:
            # Unstable structures are removed before one-SE/activation.  This
            # is a numerical admission rule only; no evaluation target is read.
            losses[candidate] = [float("nan")] * len(selection_folds)

    neutral = (base.K_ZERO, 1)
    if neutral not in stable_candidates:
        raise RuntimeError(
            f"K exact-zero neutral failed strict numerical stability: channel={channel}"
        )

    selection = base.guarded_local_one_se_select(
        losses,
        base._k_complexity,
        neutral=neutral,
        minimum_relative_improvement=float(
            config["selection"]["minimum_relative_improvement"]
        ),
        minimum_positive_fraction=float(
            config["selection"]["minimum_positive_fold_fraction"]
        ),
        minimum_usable_folds=int(config["selection"]["minimum_usable_folds"]),
    )
    selected = selection.final_selected_candidate
    if selected not in stable_candidates:
        raise RuntimeError(
            f"guarded K selection returned a non-stable candidate: channel={channel}: {selected}"
        )

    return selected, {
        "selected": str(selected),
        "selection": selection.to_json(),
        "candidate_numeric_audit": selection_numeric,
        "gamma_oof_numeric_audit": gamma_oof_numeric,
        "full_refit_numeric_audit": full_refit,
        "strict_stability_rule": (
            "FULL_DEVELOPMENT_REFIT_AND_ALL_SELECTION_AND_GAMMA_OOF_FOLDS"
        ),
        "stable_candidates": [str(candidate) for candidate in stable_candidates],
    }


base._select_k_channel = strict_select_k_channel


if __name__ == "__main__":
    raise SystemExit(base.main())
