"""Forward blocked cross-fitting and double residualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nuisance import fit_nuisance


@dataclass(frozen=True, slots=True)
class CrossFitResult:
    y_residual: np.ndarray
    phi_residual: np.ndarray
    evaluation_indices: np.ndarray
    fold_summaries: tuple[dict[str, float | int], ...]
    condition_number: float
    max_abs_nuisance_correlation: float


def forward_crossfit(
    y: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    *,
    folds: int,
    initial_nuisance_prefix_targets: int,
    purge_gap: int,
    ridge_candidates: tuple[float, ...],
    nuisance_selection_tail_fraction: float,
) -> CrossFitResult:
    y = np.asarray(y, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if y.shape != (len(phi),) or psi.shape[0] != len(y):
        raise ValueError("Cross-fit arrays must share their first dimension.")
    if initial_nuisance_prefix_targets + folds > len(y):
        raise ValueError("Insufficient samples for requested forward folds.")
    evaluation_blocks = np.array_split(
        np.arange(initial_nuisance_prefix_targets, len(y)), folds
    )
    y_parts: list[np.ndarray] = []
    phi_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    summaries: list[dict[str, float | int]] = []
    for fold, evaluation in enumerate(evaluation_blocks):
        train_stop = int(evaluation[0]) - purge_gap
        if train_stop <= 2:
            raise ValueError("Purge gap leaves no nuisance training prefix.")
        nuisance = fit_nuisance(
            psi[:train_stop],
            y[:train_stop],
            phi[:train_stop],
            ridge_candidates=ridge_candidates,
            validation_fraction=nuisance_selection_tail_fraction,
        )
        y_hat, phi_hat = nuisance.predict(psi[evaluation])
        y_part = y[evaluation] - y_hat
        phi_part = phi[evaluation] - phi_hat
        y_parts.append(y_part)
        phi_parts.append(phi_part)
        index_parts.append(evaluation)
        summaries.append(
            {
                "fold": fold,
                "train_start": 0,
                "train_stop": train_stop,
                "evaluation_start": int(evaluation[0]),
                "evaluation_stop": int(evaluation[-1]) + 1,
                "purge_gap": purge_gap,
                "ridge_y": nuisance.ridge_y,
                "ridge_phi": nuisance.ridge_phi,
                "y_rmse": float(np.sqrt(np.mean(y_part**2))),
            }
        )
    y_residual = np.concatenate(y_parts)
    phi_residual = np.concatenate(phi_parts)
    indices = np.concatenate(index_parts)
    gram = phi_residual.T @ phi_residual / len(phi_residual)
    condition_number = float(np.linalg.cond(gram))
    combined = np.concatenate([phi_residual, psi[indices]], axis=1)
    correlations = np.corrcoef(combined, rowvar=False)
    p = phi_residual.shape[1]
    cross = correlations[:p, p:]
    finite = np.abs(cross[np.isfinite(cross)])
    max_correlation = float(finite.max()) if finite.size else 0.0
    return CrossFitResult(
        y_residual=y_residual,
        phi_residual=phi_residual,
        evaluation_indices=indices,
        fold_summaries=tuple(summaries),
        condition_number=condition_number,
        max_abs_nuisance_correlation=max_correlation,
    )
