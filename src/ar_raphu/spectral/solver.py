"""FP64 strongly convex full-kernel solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse.linalg


@dataclass(frozen=True, slots=True)
class SpectralFit:
    coefficients: np.ndarray
    predictions: np.ndarray
    relative_kkt_residual: float
    condition_number: float
    intercept: float = 0.0
    converged: bool = True
    iterations: int = 0


def solve_full_kernel(
    matrix: np.ndarray,
    target: np.ndarray,
    penalty: np.ndarray,
    *,
    numerical_jitter_relative: float = 1.0e-10,
    fit_intercept: bool = False,
    compute_condition_number: bool = True,
) -> SpectralFit:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(penalty, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or p.shape != (x.shape[1],) * 2:
        raise ValueError("Incompatible design, target, or penalty shape.")
    if fit_intercept:
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        x_work = x - x_mean
        y_work = y - y_mean
    else:
        x_mean = np.zeros(x.shape[1], dtype=np.float64)
        y_mean = 0.0
        x_work = x
        y_work = y
    system = x_work.T @ x_work / len(x_work) + p
    rhs = x_work.T @ y_work / len(x_work)
    scale = max(float(np.trace(system) / len(system)), 1.0)
    jitter = numerical_jitter_relative * scale
    solved_system = system + jitter * np.eye(len(system))
    factor = scipy.linalg.cho_factor(solved_system, lower=True, check_finite=True)
    coefficients = scipy.linalg.cho_solve(factor, rhs, check_finite=True)
    residual = solved_system @ coefficients - rhs
    denominator = max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
    intercept = y_mean - float(x_mean @ coefficients)
    predictions = x @ coefficients + intercept
    return SpectralFit(
        coefficients=coefficients,
        predictions=predictions,
        relative_kkt_residual=float(np.linalg.norm(residual) / denominator),
        condition_number=(
            float(np.linalg.cond(solved_system))
            if compute_condition_number
            else float("nan")
        ),
        intercept=intercept,
    )


def solve_full_kernel_pcg(
    matrix: np.ndarray,
    target: np.ndarray,
    penalty: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-8,
    max_iterations: int = 2000,
    block_slices: tuple[slice, ...] | None = None,
    numerical_jitter_relative: float = 1.0e-10,
    fit_intercept: bool = False,
    warm_start: np.ndarray | None = None,
) -> SpectralFit:
    """Matrix-free FP64 PCG with block-Jacobi Cholesky preconditioning."""

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(penalty, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or p.shape != (x.shape[1],) * 2:
        raise ValueError("Incompatible design, target, or penalty shape.")
    if fit_intercept:
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        x_work = x - x_mean
        y_work = y - y_mean
    else:
        x_mean = np.zeros(x.shape[1], dtype=np.float64)
        y_mean = 0.0
        x_work = x
        y_work = y
    rhs = x_work.T @ y_work / len(x_work)
    diagonal_scale = float(
        np.mean(np.sum(x_work * x_work, axis=0) / len(x_work) + np.diag(p))
    )
    jitter = numerical_jitter_relative * max(diagonal_scale, 1.0)

    def matvec(vector: np.ndarray) -> np.ndarray:
        return (
            x_work.T @ (x_work @ vector) / len(x_work)
            + p @ vector
            + jitter * vector
        )

    operator = scipy.sparse.linalg.LinearOperator(
        (x.shape[1], x.shape[1]), matvec=matvec, dtype=np.float64
    )
    slices = block_slices or (slice(0, x.shape[1]),)
    factors: list[tuple[np.ndarray, bool]] = []
    for block in slices:
        local = (
            x_work[:, block].T @ x_work[:, block] / len(x_work)
            + p[block, block]
            + jitter * np.eye(block.stop - block.start)
        )
        factors.append(
            scipy.linalg.cho_factor(local, lower=True, check_finite=True)
        )

    def precondition(vector: np.ndarray) -> np.ndarray:
        result = np.empty_like(vector)
        for block, factor in zip(slices, factors, strict=True):
            result[block] = scipy.linalg.cho_solve(
                factor, vector[block], check_finite=False
            )
        return result

    preconditioner = scipy.sparse.linalg.LinearOperator(
        operator.shape, matvec=precondition, dtype=np.float64
    )
    iteration_count = 0

    def callback(_: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    coefficients, info = scipy.sparse.linalg.cg(
        operator,
        rhs,
        x0=(
            None
            if warm_start is None
            else np.asarray(warm_start, dtype=np.float64)
        ),
        rtol=relative_tolerance,
        atol=0.0,
        maxiter=max_iterations,
        M=preconditioner,
        callback=callback,
    )
    residual = matvec(coefficients) - rhs
    relative_residual = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
    )
    converged = info == 0 and relative_residual <= relative_tolerance
    intercept = y_mean - float(x_mean @ coefficients)
    return SpectralFit(
        coefficients=coefficients,
        predictions=x @ coefficients + intercept,
        relative_kkt_residual=relative_residual,
        condition_number=float("nan"),
        intercept=intercept,
        converged=converged,
        iterations=iteration_count,
    )
