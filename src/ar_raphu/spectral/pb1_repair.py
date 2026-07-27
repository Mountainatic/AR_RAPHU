"""Numerical and selection primitives frozen by PB1 Development Repair V2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .penalty_interval import LogPenaltyInterval, numerical_jitter


@dataclass(frozen=True, slots=True)
class PB1SystemSolution:
    coefficients: np.ndarray
    relative_kkt_residual: float
    solver_stage: str
    converged: bool
    numerical_jitter: float
    diag_equilibration_used: bool
    iterative_refinement_steps: int
    condition_estimate_before: float
    condition_estimate_after: float
    effective_rank: int
    smallest_retained_singular_value: float
    svd_rcond: float
    solution_is_minimum_norm: bool


def zero_inclusive_penalty_grid(
    interval: LogPenaltyInterval, positive_points: int
) -> np.ndarray:
    """Preserve the registered positive grid and prepend an exact zero."""

    positive = interval.grid(positive_points)
    grid = np.concatenate((np.zeros(1, dtype=np.float64), positive))
    if grid[0] != 0.0 or not np.all(grid[1:] > 0.0):
        raise AssertionError("PB1 nonnegative penalty grid lost its exact endpoint.")
    return grid


def positive_lower_expansion_required(
    *,
    selected_index: int,
    axis_zero_best_loss: float,
    global_minimum_loss: float,
    global_minimum_se: float,
) -> bool:
    """Expand only when the minimum positive value wins and zero is outside 1-SE."""

    return bool(
        selected_index == 1
        and axis_zero_best_loss > global_minimum_loss + global_minimum_se
    )


def automatic_time_block_length(values: np.ndarray) -> int:
    """Select a deterministic ACF-based block length for one time record."""

    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(series) < 16:
        raise ValueError("Automatic block selection needs at least 16 values.")
    centered = series - float(np.mean(series))
    variance = float(centered @ centered)
    if variance <= np.finfo(np.float64).eps:
        return max(2, min(len(series) // 2, int(np.ceil(len(series) ** (1 / 3)))))
    maximum_lag = max(2, min(len(series) // 4, int(np.ceil(np.sqrt(len(series))))))
    correlations = np.array(
        [
            float(centered[:-lag] @ centered[lag:] / variance)
            for lag in range(1, maximum_lag + 1)
        ]
    )
    threshold = 1.96 / np.sqrt(len(series))
    below = np.flatnonzero(np.abs(correlations) <= threshold)
    decorrelation_lag = int(below[0] + 1) if len(below) else maximum_lag
    return max(2, min(maximum_lag, 2 * decorrelation_lag))


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("PB1 solver requires a square system matrix.")
    return 0.5 * (array + array.T)


def _relative_kkt(matrix: np.ndarray, coefficients: np.ndarray, rhs: np.ndarray) -> float:
    residual = matrix @ coefficients - rhs
    return float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
    )


def _condition_estimate(matrix: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")


def _equilibrated_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    jitter: float,
) -> tuple[np.ndarray, float]:
    epsilon_d = max(np.finfo(np.float64).eps, jitter)
    scale = 1.0 / np.sqrt(np.maximum(np.abs(np.diag(matrix)), epsilon_d))
    balanced = (scale[:, None] * matrix) * scale[None, :]
    factor = scipy.linalg.cho_factor(
        balanced, lower=True, check_finite=True
    )
    z = scipy.linalg.cho_solve(factor, scale * rhs, check_finite=False)
    return scale * z, _condition_estimate(balanced)


def iterative_refine_pb1(
    matrix: np.ndarray,
    rhs: np.ndarray,
    initial: np.ndarray,
    *,
    maximum_steps: int = 5,
    minimum_improvement: float = 1.0e-3,
    target_kkt: float = 1.0e-8,
) -> tuple[np.ndarray, float, float, int]:
    """Refine in original coordinates and report before/after KKT."""

    system = _symmetric(matrix)
    vector = np.asarray(rhs, dtype=np.float64)
    coefficients = np.asarray(initial, dtype=np.float64).copy()
    before = _relative_kkt(system, coefficients, vector)
    previous = before
    steps = 0
    for step in range(1, maximum_steps + 1):
        residual = vector - system @ coefficients
        correction, _ = _equilibrated_solve(system, residual, jitter=0.0)
        coefficients += correction
        steps = step
        current = _relative_kkt(system, coefficients, vector)
        if current <= target_kkt:
            break
        improvement = (previous - current) / max(previous, np.finfo(float).eps)
        if improvement < minimum_improvement:
            break
        previous = current
    return coefficients, before, _relative_kkt(system, coefficients, vector), steps


def solve_pb1_system(
    system_without_jitter: np.ndarray,
    rhs: np.ndarray,
    *,
    kkt_threshold: float = 1.0e-8,
    maximum_refinement_steps: int = 5,
    minimum_refinement_improvement: float = 1.0e-3,
    svd_rcond: float | None = None,
) -> PB1SystemSolution:
    """Solve one frozen PB1 normal equation without changing its objective."""

    matrix = _symmetric(system_without_jitter)
    vector = np.asarray(rhs, dtype=np.float64)
    if vector.shape != (len(matrix),):
        raise ValueError("PB1 solver RHS has incompatible shape.")
    if kkt_threshold != 1.0e-8:
        raise ValueError("PB1 Repair V2 freezes the KKT threshold at 1e-8.")
    if maximum_refinement_steps != 5:
        raise ValueError("PB1 Repair V2 freezes five refinement steps.")
    jitter = numerical_jitter(matrix)
    condition_before = _condition_estimate(matrix)
    condition_after = condition_before
    coefficients: np.ndarray | None = None
    stage = "SYMMETRIC_FP64_PRIMARY"
    equilibration_used = False
    refinement_steps = 0

    try:
        factor = scipy.linalg.cho_factor(
            matrix, lower=True, check_finite=True
        )
        coefficients = scipy.linalg.cho_solve(
            factor, vector, check_finite=False
        )
    except (np.linalg.LinAlgError, ValueError):
        coefficients = None
    if coefficients is not None:
        kkt = _relative_kkt(matrix, coefficients, vector)
        if kkt <= kkt_threshold:
            return PB1SystemSolution(
                coefficients=coefficients,
                relative_kkt_residual=kkt,
                solver_stage=stage,
                converged=True,
                numerical_jitter=jitter,
                diag_equilibration_used=False,
                iterative_refinement_steps=0,
                condition_estimate_before=condition_before,
                condition_estimate_after=condition_after,
                effective_rank=len(matrix),
                smallest_retained_singular_value=float("nan"),
                svd_rcond=float("nan"),
                solution_is_minimum_norm=False,
            )

    try:
        coefficients, condition_after = _equilibrated_solve(
            matrix, vector, jitter=jitter
        )
        stage = "DIAGONAL_EQUILIBRATION"
        equilibration_used = True
    except (np.linalg.LinAlgError, ValueError):
        coefficients = None
    if coefficients is not None:
        kkt = _relative_kkt(matrix, coefficients, vector)
        if kkt <= kkt_threshold:
            return PB1SystemSolution(
                coefficients=coefficients,
                relative_kkt_residual=kkt,
                solver_stage=stage,
                converged=True,
                numerical_jitter=jitter,
                diag_equilibration_used=True,
                iterative_refinement_steps=0,
                condition_estimate_before=condition_before,
                condition_estimate_after=condition_after,
                effective_rank=len(matrix),
                smallest_retained_singular_value=float("nan"),
                svd_rcond=float("nan"),
                solution_is_minimum_norm=False,
            )
        try:
            coefficients, _, kkt, refinement_steps = iterative_refine_pb1(
                matrix,
                vector,
                coefficients,
                maximum_steps=maximum_refinement_steps,
                minimum_improvement=minimum_refinement_improvement,
                target_kkt=kkt_threshold,
            )
        except (np.linalg.LinAlgError, ValueError):
            kkt = _relative_kkt(matrix, coefficients, vector)
        stage = "ITERATIVE_REFINEMENT"
        if kkt <= kkt_threshold:
            return PB1SystemSolution(
                coefficients=coefficients,
                relative_kkt_residual=kkt,
                solver_stage=stage,
                converged=True,
                numerical_jitter=jitter,
                diag_equilibration_used=True,
                iterative_refinement_steps=refinement_steps,
                condition_estimate_before=condition_before,
                condition_estimate_after=condition_after,
                effective_rank=len(matrix),
                smallest_retained_singular_value=float("nan"),
                svd_rcond=float("nan"),
                solution_is_minimum_norm=False,
            )

    values = scipy.linalg.svdvals(matrix, check_finite=True)
    largest = max(float(values[0]), np.finfo(np.float64).eps)
    resolved_rcond = (
        float(svd_rcond)
        if svd_rcond is not None
        else np.finfo(np.float64).eps * max(matrix.shape)
    )
    cutoff = resolved_rcond * largest
    retained = values[values > cutoff]
    coefficients, _, effective_rank, _ = scipy.linalg.lstsq(
        matrix,
        vector,
        cond=resolved_rcond,
        lapack_driver="gelsd",
        check_finite=True,
    )
    kkt = _relative_kkt(matrix, coefficients, vector)
    return PB1SystemSolution(
        coefficients=coefficients,
        relative_kkt_residual=kkt,
        solver_stage="SVD_MINIMUM_NORM",
        converged=kkt <= kkt_threshold,
        numerical_jitter=jitter,
        diag_equilibration_used=equilibration_used,
        iterative_refinement_steps=refinement_steps,
        condition_estimate_before=condition_before,
        condition_estimate_after=condition_after,
        effective_rank=int(effective_rank),
        smallest_retained_singular_value=(
            float(retained[-1]) if len(retained) else 0.0
        ),
        svd_rcond=resolved_rcond,
        solution_is_minimum_norm=True,
    )
