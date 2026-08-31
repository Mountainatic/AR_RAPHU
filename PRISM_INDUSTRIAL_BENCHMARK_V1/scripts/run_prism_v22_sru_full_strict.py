from __future__ import annotations

"""Strict numerical-refit wrapper for the SRU v2.2 full KWA benchmark.

This wrapper intentionally leaves the full runner unchanged and replaces only
its K channel-selection hook.  An active K candidate is eligible for guarded
one-SE selection only when the exact same structural candidate is numerically
valid on the full development refit and on every registered inner expanding
fold.  This mirrors the v2.1.1 rule that materialized K structures may not be
selected from a merely partial set of numerically valid folds.
"""

import numpy as np

import run_prism_v22_sru_full as base


def strict_select_k_channel(
    representation: np.ndarray,
    target_delta: np.ndarray,
    development: np.ndarray,
    channel: int,
    config: dict,
):
    candidates = base._k_candidates(config)
    folds = base._expanding_folds(
        development, int(config["selection"]["inner_expanding_folds"])
    )
    losses = {candidate: [] for candidate in candidates}
    numeric = {str(candidate): [] for candidate in candidates}

    for fit_index, evaluation_index in folds:
        fit_values = representation[fit_index, channel, :]
        evaluation_values = representation[evaluation_index, channel, :]
        y_fit = target_delta[fit_index]
        y_evaluation = target_delta[evaluation_index]
        for candidate in candidates:
            try:
                contract = base._fit_k(fit_values, y_fit, candidate, config)
                valid = base.numerical_contract_passes(contract)
                if valid:
                    prediction = base.predict_contract(evaluation_values, contract)
                    loss = float(np.mean(np.square(y_evaluation - prediction)))
                else:
                    loss = float("nan")
                numeric[str(candidate)].append(
                    {
                        "pass": bool(valid),
                        "certificate": contract.get("certificate", {}),
                    }
                )
            except Exception as error:
                loss = float("nan")
                numeric[str(candidate)].append(
                    {
                        "pass": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            losses[candidate].append(loss)

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

        all_fold_valid = bool(numeric[str(candidate)]) and all(
            bool(item.get("pass", False)) for item in numeric[str(candidate)]
        )
        stable = bool(full_valid and all_fold_valid)
        full_refit[str(candidate)]["all_inner_folds_pass"] = all_fold_valid
        full_refit[str(candidate)]["eligible_for_selection"] = stable
        if stable:
            stable_candidates.append(candidate)
        else:
            # A numerically unstable structure must not enter one-SE or
            # practical-activation selection.  The exact-zero neutral remains
            # available and is expected to certify on every fold.
            losses[candidate] = [float("nan")] * len(folds)

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
        "candidate_numeric_audit": numeric,
        "full_refit_numeric_audit": full_refit,
        "strict_stability_rule": "FULL_REFIT_AND_ALL_INNER_FOLDS",
        "stable_candidates": [str(candidate) for candidate in stable_candidates],
    }


base._select_k_channel = strict_select_k_channel


if __name__ == "__main__":
    raise SystemExit(base.main())
