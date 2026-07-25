"""Warm-started group-sparsity continuation path.

Instead of re-running warmup for every regularization level, this module trains
one dense model and follows a nested support path.  Each path point may be
refitted without penalty for validation-only model selection.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

from .protocol import (StageResult, refit_fixed_support,
                       run_pruning_to_stable_support, train_warmup)


@dataclass
class HomotopyPoint:
    lambda_group: float
    requested_scale: float
    terminal_support: list[int]
    stable: bool
    stable_epoch: int | None
    prune_best_val_rmse: float
    refit_best_val_rmse: float
    prune_state: dict
    refit_state: dict
    prune_history: list[dict]
    refit_history: list[dict]

    def summary(self) -> dict:
        row = asdict(self)
        row.pop("prune_state")
        row.pop("refit_state")
        row.pop("prune_history")
        row.pop("refit_history")
        row["active_count"] = len(self.terminal_support)
        return row


@dataclass
class HomotopyResult:
    warmup: StageResult
    points: list[HomotopyPoint]
    selected_index: int

    @property
    def selected(self) -> HomotopyPoint:
        return self.points[self.selected_index]


def one_standard_error_path_select(points: list[HomotopyPoint]) -> int:
    """Validation-only sparse selection over a continuation path."""
    if not points:
        raise ValueError("empty homotopy path")
    values = np.asarray([p.refit_best_val_rmse for p in points], dtype=float)
    best = int(values.argmin())
    # A single path has no seed standard error. Use the robust standard error
    # of available validation values as a conservative tolerance. Multi-seed
    # orchestration can replace this with a cross-seed SE before final choice.
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    admissible = [
        i for i, point in enumerate(points)
        if point.refit_best_val_rmse <= values[best] + se
    ]
    return min(
        admissible,
        key=lambda i: (
            len(points[i].terminal_support),
            not points[i].stable,
            -points[i].lambda_group,
        ),
    )


def run_group_homotopy_path(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    requested_scales: Iterable[float],
    lr: float,
    warmup_epochs: int,
    warmup_patience: int,
    prune_epochs: int,
    ramp_epochs: int,
    full_penalty_min_epochs: int,
    stable_epochs: int,
    refit_epochs: int,
    refit_patience: int,
    validation_interval: int = 5,
    diagnostic_interval: int = 20,
) -> HomotopyResult:
    """Train one warmup and follow increasing regularization with warm starts.

    ``requested_scale`` is the dimensionless shrink fraction used by the
    project: lambda = scale * median_branch_norm / lr.
    """
    scales = sorted(set(float(v) for v in requested_scales))
    if not scales or scales[0] < 0:
        raise ValueError("requested_scales must be nonnegative")
    warmup = train_warmup(
        model, x_train, y_train, x_val, y_val,
        epochs=warmup_epochs, lr=lr, patience=warmup_patience,
        validation_interval=validation_interval,
        diagnostic_interval=diagnostic_interval,
        history_interval=validation_interval,
    )
    median_norm = float(model.response_branches.compute_branch_norms().median().cpu())
    points: list[HomotopyPoint] = []
    continuation_model = model
    for scale in scales:
        lambda_group = scale * median_norm / lr
        prune = run_pruning_to_stable_support(
            continuation_model, x_train, y_train, x_val, y_val,
            ramp_epochs=ramp_epochs,
            full_penalty_min_epochs=full_penalty_min_epochs,
            stable_epochs=stable_epochs,
            max_epochs=prune_epochs,
            lr=lr,
            lambda_group=lambda_group,
            validation_interval=validation_interval,
            diagnostic_interval=diagnostic_interval,
            history_interval=validation_interval,
        )
        # Refit a clone so the continuation model remains at the authoritative
        # terminal pruning state for the next, stronger lambda.
        refit_model = copy.deepcopy(continuation_model)
        refit = refit_fixed_support(
            refit_model, prune.terminal_support,
            x_train, y_train, x_val, y_val,
            epochs=refit_epochs, lr=lr, patience=refit_patience,
            validation_interval=validation_interval,
            diagnostic_interval=diagnostic_interval,
            history_interval=validation_interval,
        )
        points.append(HomotopyPoint(
            lambda_group=lambda_group,
            requested_scale=scale,
            terminal_support=list(prune.terminal_support),
            stable=prune.stable,
            stable_epoch=prune.support_stable_epoch,
            prune_best_val_rmse=prune.best_val_rmse,
            refit_best_val_rmse=refit.best_val_rmse,
            prune_state=copy.deepcopy(prune.terminal_state),
            refit_state=copy.deepcopy(refit.best_state),
            prune_history=prune.history,
            refit_history=refit.history,
        ))
        if not prune.terminal_support:
            break
    return HomotopyResult(
        warmup=warmup,
        points=points,
        selected_index=one_standard_error_path_select(points),
    )
