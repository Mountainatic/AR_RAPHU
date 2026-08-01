from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NaturalCubicBasis:
    mean: float
    scale: float
    knots: np.ndarray
    feature_mean: np.ndarray

    @property
    def dimension(self) -> int:
        return max(1, len(self.knots) - 1)

    @staticmethod
    def _raw(standardized: np.ndarray, knots: np.ndarray) -> np.ndarray:
        x = np.asarray(standardized, dtype=np.float64).reshape(-1)
        if len(knots) < 3:
            return x[:, None]
        last = knots[-1]
        penultimate = knots[-2]

        def d(knot: float) -> np.ndarray:
            denominator = last - knot
            if denominator <= 0:
                raise ValueError("natural spline knots must be strictly increasing")
            return (np.maximum(x - knot, 0.0) ** 3 - np.maximum(x - last, 0.0) ** 3) / denominator

        reference = d(float(penultimate))
        nonlinear = [d(float(knot)) - reference for knot in knots[:-2]]
        return np.column_stack([x, *nonlinear])

    @classmethod
    def fit(cls, values: np.ndarray, requested_dimension: int) -> "NaturalCubicBasis":
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if requested_dimension < 1:
            raise ValueError(requested_dimension)
        mean = float(np.mean(values, dtype=np.float64))
        scale = float(np.std(values, ddof=0)) or 1.0
        standardized = (values - mean) / scale
        quantiles = np.linspace(0.0, 1.0, requested_dimension + 1)
        knots = np.unique(np.quantile(standardized, quantiles))
        raw = cls._raw(standardized, knots)
        feature_mean = np.mean(raw, axis=0, dtype=np.float64)
        return cls(mean=mean, scale=scale, knots=knots, feature_mean=feature_mean)

    def transform(self, values: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(values, dtype=np.float64).reshape(-1) - self.mean) / self.scale
        return self._raw(standardized, self.knots) - self.feature_mean

    def metadata(self) -> dict[str, Any]:
        return {
            "family": "centered_natural_cubic_b_spline",
            "mean": self.mean,
            "scale": self.scale,
            "knots": self.knots.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "dimension": self.dimension,
        }

    @classmethod
    def from_metadata(cls, value: dict[str, Any]) -> "NaturalCubicBasis":
        return cls(
            mean=float(value["mean"]),
            scale=float(value["scale"]),
            knots=np.asarray(value["knots"], dtype=np.float64),
            feature_mean=np.asarray(value["feature_mean"], dtype=np.float64),
        )


def tensor_design(values: np.ndarray, basis: NaturalCubicBasis) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(values.shape)
    rows = [basis.transform(values[:, block]) for block in range(values.shape[1])]
    return np.stack(rows, axis=1)


def roughness_matrix(size: int, order: int = 2) -> np.ndarray:
    if size <= order:
        return np.zeros((0, size), dtype=np.float64)
    return np.diff(np.eye(size, dtype=np.float64), n=order, axis=0)


def full_penalty(
    m_tau: int,
    m_x: int,
    lambda_0: float,
    lambda_tau: float,
    lambda_x: float,
) -> np.ndarray:
    identity_tau = np.eye(m_tau, dtype=np.float64)
    identity_x = np.eye(m_x, dtype=np.float64)
    d_tau = roughness_matrix(m_tau)
    d_x = roughness_matrix(m_x)
    return (
        lambda_0 * np.eye(m_tau * m_x, dtype=np.float64)
        + lambda_tau * np.kron(d_tau.T @ d_tau, identity_x)
        + lambda_x * np.kron(identity_tau, d_x.T @ d_x)
    )


def solve_penalized(
    design: np.ndarray,
    target: np.ndarray,
    evaluation: np.ndarray,
    penalty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    design = np.asarray(design, dtype=np.float64)
    evaluation = np.asarray(evaluation, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if penalty.shape != (design.shape[1], design.shape[1]):
        raise ValueError((penalty.shape, design.shape))
    center = np.mean(design, axis=0, dtype=np.float64)
    centered = design - center
    evaluation_centered = evaluation - center
    target_mean = float(np.mean(target, dtype=np.float64))
    centered_target = target - target_mean
    gram = centered.T @ centered
    rhs = centered.T @ centered_target
    system = gram + penalty
    try:
        coefficient = np.linalg.solve(system, rhs)
        solver = "cholesky_or_lu"
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(system, rhs, rcond=1e-12)[0]
        solver = "svd_rescue"
    residual = system @ coefficient - rhs
    relative_kkt = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0))
    condition = float(np.linalg.cond(system))
    prediction = evaluation_centered @ coefficient + target_mean
    intercept = target_mean - float(center @ coefficient)
    certificate = {
        "solver": solver,
        "relative_kkt": relative_kkt,
        "condition_number": condition,
        "effective_rank": int(np.linalg.matrix_rank(system)),
        "intercept": intercept,
    }
    return np.asarray(prediction, dtype=np.float64), coefficient, certificate


def fit_full(
    phi_train: np.ndarray,
    target: np.ndarray,
    phi_eval: np.ndarray,
    lambda_0: float,
    lambda_tau: float,
    lambda_x: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    m_tau, m_x = phi_train.shape[1:]
    design = phi_train.reshape(len(phi_train), m_tau * m_x)
    evaluation = phi_eval.reshape(len(phi_eval), m_tau * m_x)
    prediction, coefficient, certificate = solve_penalized(
        design,
        target,
        evaluation,
        full_penalty(m_tau, m_x, lambda_0, lambda_tau, lambda_x),
    )
    return prediction, coefficient.reshape(m_tau, m_x), certificate


def _factor_penalty(size: int, rank: int, lambda_0: float, smoothness: float) -> np.ndarray:
    d = roughness_matrix(size)
    block = lambda_0 * np.eye(size, dtype=np.float64) + smoothness * (d.T @ d)
    return np.kron(np.eye(rank, dtype=np.float64), block)


def fit_rank_als(
    phi_train: np.ndarray,
    target: np.ndarray,
    phi_eval: np.ndarray,
    rank: int,
    lambda_0: float,
    lambda_tau: float,
    lambda_x: float,
    max_iterations: int,
    tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if rank < 1:
        raise ValueError(rank)
    m_tau, m_x = phi_train.shape[1:]
    _, full_theta, _ = fit_full(phi_train, target, phi_train, lambda_0, lambda_tau, lambda_x)
    left, singular, right_t = np.linalg.svd(full_theta, full_matrices=False)
    effective_rank = min(rank, len(singular))
    u = left[:, :effective_rank] * np.sqrt(singular[:effective_rank])[None, :]
    v = right_t[:effective_rank].T * np.sqrt(singular[:effective_rank])[None, :]
    rng = np.random.default_rng(seed)
    if seed != 0:
        u = u + 1e-3 * rng.normal(size=u.shape)
        v = v + 1e-3 * rng.normal(size=v.shape)
    if effective_rank < rank:
        u = np.column_stack([u, 1e-3 * rng.normal(size=(m_tau, rank - effective_rank))])
        v = np.column_stack([v, 1e-3 * rng.normal(size=(m_x, rank - effective_rank))])
    previous = float("inf")
    iterations = 0
    last_certificate: dict[str, Any] = {}
    for iteration in range(max_iterations):
        design_u = np.einsum("tbx,xr->tbr", phi_train, v).transpose(0, 2, 1).reshape(len(phi_train), rank * m_tau)
        eval_u = design_u
        _, u_vector, certificate_u = solve_penalized(
            design_u,
            target,
            eval_u,
            _factor_penalty(m_tau, rank, lambda_0, lambda_tau),
        )
        u = u_vector.reshape(rank, m_tau).T
        design_v = np.einsum("tbx,br->txr", phi_train, u).transpose(0, 2, 1).reshape(len(phi_train), rank * m_x)
        _, v_vector, certificate_v = solve_penalized(
            design_v,
            target,
            design_v,
            _factor_penalty(m_x, rank, lambda_0, lambda_x),
        )
        v = v_vector.reshape(rank, m_x).T
        theta = u @ v.T
        fitted = np.einsum("tbx,bx->t", phi_train, theta)
        intercept = float(np.mean(target - fitted, dtype=np.float64))
        objective = float(np.mean(np.square(target - fitted - intercept), dtype=np.float64))
        iterations = iteration + 1
        last_certificate = {
            "u": certificate_u,
            "v": certificate_v,
            "objective": objective,
        }
        if np.isfinite(previous) and abs(previous - objective) <= tolerance * max(abs(previous), 1.0):
            break
        previous = objective
    theta = u @ v.T
    train_component = np.einsum("tbx,bx->t", phi_train, theta)
    intercept = float(np.mean(target - train_component, dtype=np.float64))
    prediction = np.einsum("tbx,bx->t", phi_eval, theta) + intercept
    singular_values = np.linalg.svd(theta, compute_uv=False)
    certificate = {
        "iterations": iterations,
        "requested_rank": rank,
        "realized_rank": int(np.linalg.matrix_rank(theta)),
        "singular_values": singular_values.tolist(),
        "intercept": intercept,
        "train_mse": float(np.mean(np.square(target - train_component - intercept), dtype=np.float64)),
        "subproblem_certificates": last_certificate,
    }
    return np.asarray(prediction, dtype=np.float64), theta, certificate
