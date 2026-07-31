from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class GCVFit:
    coefficient: np.ndarray
    intercept: float
    prediction: np.ndarray
    selected_lambda: float
    effective_df: float
    gcv: float
    kkt_residual: float
    condition_number: float
    curve: list[dict[str, float]]
    generalized_eigenvalues: np.ndarray


def fit_gcv(
    matrix: np.ndarray,
    target: np.ndarray,
    penalty: np.ndarray,
    *,
    log10_min: float,
    log10_max: float,
    bracket_points: int,
    brent_xatol: float,
) -> GCVFit:
    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    if matrix.dtype != np.float64 or target.dtype != np.float64:
        raise RuntimeError("FP64_REQUIRED")
    x_mean = np.mean(matrix, axis=0)
    y_mean = float(np.mean(target))
    centered_x = matrix - x_mean
    centered_y = target - y_mean
    n_rows = len(target)
    gram = centered_x.T @ centered_x / n_rows
    rhs = centered_x.T @ centered_y / n_rows
    penalty = (penalty + penalty.T) / 2.0
    eigenvalues, vectors = eigh(gram, penalty, check_finite=False, driver="gvd")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = vectors.T @ rhs
    y_energy = float(centered_y @ centered_y / n_rows)

    def statistics(log10_lambda: float) -> tuple[float, float, float]:
        regularization = 10.0 ** float(log10_lambda)
        denominator = eigenvalues + regularization
        linear = float(np.sum(projected_rhs**2 / denominator))
        quadratic = float(np.sum(eigenvalues * projected_rhs**2 / denominator**2))
        rss = max(y_energy - 2.0 * linear + quadratic, 0.0)
        df = 1.0 + float(np.sum(eigenvalues / denominator))
        gcv = rss / max((1.0 - df / n_rows) ** 2, 1e-30)
        return gcv, df, rss

    grid = np.linspace(log10_min, log10_max, int(bracket_points), dtype=np.float64)
    curve = []
    for value in grid:
        gcv, df, rss = statistics(float(value))
        curve.append({"log10_lambda": float(value), "lambda": float(10.0**value), "gcv": gcv, "effective_df": df, "rss_mean": rss})
    best = int(np.argmin([row["gcv"] for row in curve]))
    lower = float(grid[max(best - 1, 0)])
    upper = float(grid[min(best + 1, len(grid) - 1)])
    if lower == upper:
        lower, upper = float(log10_min), float(log10_max)
    result = minimize_scalar(lambda value: statistics(float(value))[0], bounds=(lower, upper), method="bounded", options={"xatol": float(brent_xatol), "maxiter": 200})
    selected_log = float(result.x)
    selected_lambda = float(10.0**selected_log)
    gcv, df, _ = statistics(selected_log)
    coefficient = vectors @ (projected_rhs / (eigenvalues + selected_lambda))
    intercept = y_mean - float(x_mean @ coefficient)
    prediction = intercept + matrix @ coefficient
    hessian = gram + selected_lambda * penalty
    residual = hessian @ coefficient - rhs
    kkt = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30))
    condition = float(np.linalg.cond(hessian))
    curve.append({"log10_lambda": selected_log, "lambda": selected_lambda, "gcv": gcv, "effective_df": df, "rss_mean": float(np.mean((target - prediction) ** 2)), "selected": 1.0})
    return GCVFit(
        coefficient=coefficient,
        intercept=intercept,
        prediction=prediction,
        selected_lambda=selected_lambda,
        effective_df=df,
        gcv=gcv,
        kkt_residual=kkt,
        condition_number=condition,
        curve=curve,
        generalized_eigenvalues=eigenvalues,
    )


def ridge_fit(matrix: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    augmented = np.column_stack((np.ones(len(matrix)), matrix))
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    gram = augmented.T @ augmented / len(target) + penalty
    rhs = augmented.T @ target / len(target)
    coefficient = np.linalg.solve(gram, rhs)
    return coefficient[1:], float(coefficient[0])


def one_se_select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    minimum = min(rows, key=lambda row: float(row["mean_mse"]))
    threshold = float(minimum["mean_mse"]) + float(minimum["se_mse"])
    eligible = [row for row in rows if float(row["mean_mse"]) <= threshold + 1e-15]
    def complexity(row: dict[str, Any]) -> tuple[float, float, float]:
        candidate = str(row["candidate"])
        if candidate == "A0":
            return (0.0, 0.0, 0.0)
        if candidate.startswith("AR:"):
            return (1.0, float(candidate.split(":")[1]), -float(row.get("ridge", 0.0)))
        return (2.0, float(candidate.split(":")[1]), -float(row.get("ridge", 0.0)))
    selected = min(eligible, key=complexity)
    return {**selected, "one_se_threshold": threshold}
