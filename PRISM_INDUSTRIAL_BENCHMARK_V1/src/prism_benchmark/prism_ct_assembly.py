from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AssemblyState:
    branch_names: tuple[str, ...]
    weights: np.ndarray
    persistence_weight: float
    ridge: float
    iterations: int


def project_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w) = 1}."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("simplex vector cannot be empty")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    candidates = np.flatnonzero(
        ordered * (np.arange(len(ordered)) + 1) > (cumulative - 1.0)
    )
    if not len(candidates):
        return np.full_like(values, 1.0 / len(values))
    rho = int(candidates[-1])
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(values - theta, 0.0)


def fit_simplex_assembly(
    branch_predictions: Mapping[str, np.ndarray],
    target_delta: np.ndarray,
    ridge: float = 1e-3,
    max_iterations: int = 20000,
    tolerance: float = 1e-12,
) -> AssemblyState:
    """Fit an interpretable PRISM A-level convex assembly.

    Each dynamic branch contributes a predicted future increment. An additional zero-delta
    column represents persistence. The fitted weights solve

        min_w ||P_aug w - delta_y||^2 / n + ridge * ||w_dynamic||^2
        s.t. w >= 0, sum(w) = 1.

    The persistence anchor means the total dynamic mass can shrink toward zero without an
    unconstrained intercept or a negative cancellation between temporal branches.
    """
    if not branch_predictions:
        raise ValueError("at least one dynamic branch is required")
    names = tuple(sorted(branch_predictions))
    columns = [
        np.asarray(branch_predictions[name], dtype=np.float64).reshape(-1)
        for name in names
    ]
    n_rows = len(columns[0])
    if any(len(column) != n_rows for column in columns):
        raise ValueError("branch prediction lengths differ")
    target = np.asarray(target_delta, dtype=np.float64).reshape(-1)
    if len(target) != n_rows:
        raise ValueError("target length differs from branch predictions")

    design = np.column_stack(
        [*columns, np.zeros(n_rows, dtype=np.float64)]
    )
    dimension = design.shape[1]
    penalty = np.diag(
        np.r_[np.full(dimension - 1, float(ridge)), 0.0]
    )
    hessian = (design.T @ design) / max(n_rows, 1) + penalty
    linear = (design.T @ target) / max(n_rows, 1)
    lipschitz = float(np.linalg.eigvalsh(hessian).max())
    step = 1.0 / max(2.0 * lipschitz, 1e-12)

    weights = np.full(dimension, 1.0 / dimension, dtype=np.float64)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        gradient = 2.0 * (hessian @ weights - linear)
        updated = project_probability_simplex(weights - step * gradient)
        if np.linalg.norm(updated - weights) <= tolerance:
            weights = updated
            break
        weights = updated

    return AssemblyState(
        branch_names=names,
        weights=weights[:-1],
        persistence_weight=float(weights[-1]),
        ridge=float(ridge),
        iterations=iterations,
    )


def predict_simplex_assembly(
    state: AssemblyState,
    branch_predictions: Mapping[str, np.ndarray],
) -> np.ndarray:
    columns = [
        np.asarray(branch_predictions[name], dtype=np.float64).reshape(-1)
        for name in state.branch_names
    ]
    if not columns:
        raise ValueError("assembly state has no dynamic branches")
    matrix = np.column_stack(columns)
    return matrix @ state.weights
