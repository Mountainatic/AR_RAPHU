"""Shared-history spectral smoke fit for CZ protocol stage R2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from ar_raphu.spectral.design import build_ar_nuisance_design, build_spectral_design
from ar_raphu.spectral.penalties import tensor_penalty
from ar_raphu.spectral.solver import solve_full_kernel

from .linear import regression_metrics, target_indices
from .protocol import DevelopmentFold


@dataclass(frozen=True, slots=True)
class SpectralSmokeResult:
    metrics: dict[str, float]
    relative_kkt_residual: float
    coefficients: int
    train_target_count: int
    validation_target_count: int


def fit_shared_history_smoke(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fold: DevelopmentFold,
    horizon: int,
    L_shared: int = 32,
    lag_basis_count: int = 16,
    amplitude_basis_count: int = 16,
    smoothness_weight: float = 1.0e-3,
    ridge_weight: float = 1.0e-8,
) -> SpectralSmokeResult:
    """Fit one full spectral XAR model without selecting any hyperparameter."""

    train_targets = target_indices(
        start=0,
        stop=fold.effective_train_stop,
        horizon=horizon,
        max_history=L_shared,
    )
    validation_targets = target_indices(
        start=fold.validation_start,
        stop=fold.validation_stop,
        horizon=horizon,
        max_history=L_shared,
    )
    train_external = build_spectral_design(
        x,
        target_indices=train_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_x=L_shared,
        lag_basis_count=lag_basis_count,
        amplitude_basis_count=amplitude_basis_count,
    )
    validation_external = build_spectral_design(
        x,
        target_indices=validation_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_x=L_shared,
        lag_basis_count=lag_basis_count,
        amplitude_basis_count=amplitude_basis_count,
    )
    train_ar = build_ar_nuisance_design(
        y,
        target_indices=train_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_y=L_shared,
        lag_basis_count=lag_basis_count,
        amplitude_basis_count=amplitude_basis_count,
    )
    validation_ar = build_ar_nuisance_design(
        y,
        target_indices=validation_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_y=L_shared,
        lag_basis_count=lag_basis_count,
        amplitude_basis_count=amplitude_basis_count,
    )
    external_penalty = tensor_penalty(
        train_external.lag_gram,
        train_external.amplitude_grams,
        lag_smoothness=smoothness_weight,
        amplitude_smoothness=smoothness_weight,
        ridge_weight=ridge_weight,
    )
    ar_penalty = tensor_penalty(
        train_external.lag_gram,
        [np.eye(amplitude_basis_count, dtype=np.float64)],
        lag_smoothness=smoothness_weight,
        amplitude_smoothness=smoothness_weight,
        ridge_weight=ridge_weight,
    )
    # The AR amplitude Gram is not separately exposed by the legacy helper.
    # A normalized identity ridge preserves a positive pilot system without
    # affecting any formal R3 penalty choice.
    ar_penalty += ridge_weight * np.eye(ar_penalty.shape[0], dtype=np.float64)
    penalty = scipy.linalg.block_diag(external_penalty, ar_penalty)
    train_matrix = np.column_stack((train_external.matrix, train_ar))
    validation_matrix = np.column_stack(
        (validation_external.matrix, validation_ar)
    )
    fit = solve_full_kernel(
        train_matrix,
        y[train_targets],
        penalty,
        fit_intercept=True,
        compute_condition_number=False,
    )
    prediction = validation_matrix @ fit.coefficients + fit.intercept
    return SpectralSmokeResult(
        metrics=regression_metrics(y[validation_targets], prediction),
        relative_kkt_residual=fit.relative_kkt_residual,
        coefficients=int(len(fit.coefficients)),
        train_target_count=int(len(train_targets)),
        validation_target_count=int(len(validation_targets)),
    )
