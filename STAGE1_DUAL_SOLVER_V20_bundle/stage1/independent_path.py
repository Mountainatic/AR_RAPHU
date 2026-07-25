"""Shared-warmup, independent pruning forks for non-convex KAN models.

Nested homotopy can propagate a bad support from one regularization level to
all stronger levels.  V20 keeps the computationally valuable shared dense
warmup, but each regularization point starts from the same warmup checkpoint.
The final scale is selected across seeds with a proper validation-only one-SE
rule in :mod:`stage1.experiment_utils`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict

from .protocol import refit_fixed_support, run_pruning_to_stable_support


@dataclass
class IndependentForkResult:
    seed: int
    requested_scale: float
    lambda_group: float
    median_warmup_branch_norm: float
    terminal_support: list[int]
    support_at_best_validation: list[int]
    stable: bool
    support_stable_epoch: int | None
    prune_best_val_rmse: float
    refit_val_rmse: float
    prune_history: list[dict]
    refit_history: list[dict]
    terminal_state: dict
    refit_state: dict

    def summary(self) -> dict:
        row = asdict(self)
        for key in ("prune_history", "refit_history", "terminal_state", "refit_state"):
            row.pop(key)
        row["config_id"] = f"s0={self.requested_scale:.8g}"
        row["active_count"] = len(self.terminal_support)
        row["refit_best_val_rmse"] = self.refit_val_rmse
        return row


def calibrate_lambda(requested_scale: float, median_branch_norm: float, learning_rate: float) -> float:
    if learning_rate <= 0 or median_branch_norm < 0 or requested_scale < 0:
        raise ValueError("invalid lambda calibration inputs")
    return float(requested_scale) * float(median_branch_norm) / float(learning_rate)


def run_independent_fork(
    model,
    warmup_state: dict,
    *,
    seed: int,
    requested_scale: float,
    median_warmup_branch_norm: float,
    x_train,
    y_train,
    x_val,
    y_val,
    lr: float,
    prune_epochs: int,
    ramp_epochs: int,
    full_penalty_min_epochs: int,
    stable_epochs: int,
    refit_epochs: int,
    refit_patience: int,
    validation_interval: int = 5,
    diagnostic_interval: int = 20,
    history_interval: int = 5,
) -> IndependentForkResult:
    model.load_state_dict(copy.deepcopy(warmup_state))
    lambda_group = calibrate_lambda(requested_scale, median_warmup_branch_norm, lr)
    prune = run_pruning_to_stable_support(
        model, x_train, y_train, x_val, y_val,
        ramp_epochs=ramp_epochs,
        full_penalty_min_epochs=full_penalty_min_epochs,
        stable_epochs=stable_epochs,
        max_epochs=prune_epochs,
        lr=lr,
        lambda_group=lambda_group,
        validation_interval=validation_interval,
        diagnostic_interval=diagnostic_interval,
        history_interval=history_interval,
    )
    refit_model = copy.deepcopy(model)
    refit_model.load_state_dict(copy.deepcopy(prune.terminal_state))
    refit = refit_fixed_support(
        refit_model, prune.terminal_support,
        x_train, y_train, x_val, y_val,
        epochs=refit_epochs,
        lr=lr,
        patience=refit_patience,
        validation_interval=validation_interval,
        diagnostic_interval=diagnostic_interval,
        history_interval=history_interval,
    )
    if refit.best_state is None:
        raise RuntimeError("refit returned no best state")
    return IndependentForkResult(
        seed=seed,
        requested_scale=float(requested_scale),
        lambda_group=lambda_group,
        median_warmup_branch_norm=float(median_warmup_branch_norm),
        terminal_support=list(prune.terminal_support),
        support_at_best_validation=list(prune.support_at_best_validation),
        stable=bool(prune.stable),
        support_stable_epoch=prune.support_stable_epoch,
        prune_best_val_rmse=float(prune.best_val_rmse),
        refit_val_rmse=float(refit.best_val_rmse),
        prune_history=prune.history,
        refit_history=refit.history,
        terminal_state=copy.deepcopy(prune.terminal_state),
        refit_state=copy.deepcopy(refit.best_state),
    )
