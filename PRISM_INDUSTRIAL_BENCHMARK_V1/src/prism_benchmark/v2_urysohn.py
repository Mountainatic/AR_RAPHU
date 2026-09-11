from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from threading import RLock
from typing import Any

import numpy as np

from .v2_basis import AmplitudeBasis, tensor_design
from .v2_numerics import (
    centered_sufficient_statistics,
    difference_penalty,
    solve_centered_certified_gram,
)


STREAM_CHUNK_ROWS = 16384


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


@lru_cache(maxsize=None)
def _difference_penalty_cached(size: int) -> np.ndarray:
    return _readonly(difference_penalty(int(size)))


@lru_cache(maxsize=None)
def _surface_penalty_cached(
    m_tau: int,
    m_x: int,
    lambda_0: float,
    lambda_tau: float,
    lambda_x: float,
) -> np.ndarray:
    return _readonly(
        float(lambda_0) * np.eye(int(m_tau) * int(m_x), dtype=np.float64)
        + float(lambda_tau)
        * np.kron(
            _difference_penalty_cached(int(m_tau)),
            np.eye(int(m_x), dtype=np.float64),
        )
        + float(lambda_x)
        * np.kron(
            np.eye(int(m_tau), dtype=np.float64),
            _difference_penalty_cached(int(m_x)),
        )
    )


def surface_penalty(m_tau: int, m_x: int, lambda_0: float, lambda_tau: float, lambda_x: float) -> np.ndarray:
    return _surface_penalty_cached(
        int(m_tau),
        int(m_x),
        float(lambda_0),
        float(lambda_tau),
        float(lambda_x),
    )


def _centered_solve(design: np.ndarray, target: np.ndarray, penalty: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
    statistics = centered_sufficient_statistics(design, target, chunk_rows=STREAM_CHUNK_ROWS)
    coefficient, intercept, certificate = solve_centered_certified_gram(*statistics, penalty)
    return coefficient, intercept, certificate.to_json()


def _transformed_centered_solve(
    values: np.ndarray,
    target: np.ndarray,
    basis: AmplitudeBasis,
    penalty: np.ndarray,
    *,
    linear_only: bool,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    rows, m_tau = values.shape
    width = m_tau if linear_only else m_tau * basis.dimension
    gram = np.zeros((width, width), dtype=np.float64)
    rhs = np.zeros(width, dtype=np.float64)
    feature_sum = np.zeros(width, dtype=np.float64)
    target_sum = 0.0
    for start in range(0, rows, STREAM_CHUNK_ROWS):
        stop = min(start + STREAM_CHUNK_ROWS, rows)
        transformed = basis.transform(values[start:stop])
        design = transformed[:, :, 0] if linear_only else transformed.reshape(stop - start, -1)
        block_y = target[start:stop]
        gram += design.T @ design
        rhs += design.T @ block_y
        feature_sum += np.sum(design, axis=0, dtype=np.float64)
        target_sum += float(np.sum(block_y, dtype=np.float64))
    coefficient, intercept, certificate = solve_centered_certified_gram(
        gram, rhs, feature_sum, target_sum, rows, penalty
    )
    return coefficient, intercept, certificate.to_json()


@lru_cache(maxsize=None)
def _factor_penalty_cached(
    size: int, rank: int, lambda_0: float, smoothness: float
) -> np.ndarray:
    block = float(lambda_0) * np.eye(int(size), dtype=np.float64) + float(
        smoothness
    ) * _difference_penalty_cached(int(size))
    return _readonly(np.kron(np.eye(int(rank), dtype=np.float64), block))


def _factor_penalty(size: int, rank: int, lambda_0: float, smoothness: float) -> np.ndarray:
    return _factor_penalty_cached(
        int(size), int(rank), float(lambda_0), float(smoothness)
    )


@dataclass(frozen=True)
class DesignStatistics:
    """Uncentered FP64 sufficient statistics for one immutable design."""

    gram: np.ndarray
    rhs: np.ndarray
    feature_sum: np.ndarray
    target_sum: float
    target_square_sum: float
    rows: int

    @classmethod
    def from_design(
        cls, design: np.ndarray, target: np.ndarray
    ) -> "DesignStatistics":
        gram, rhs, feature_sum, target_sum, rows = centered_sufficient_statistics(
            design, target, chunk_rows=STREAM_CHUNK_ROWS
        )
        return cls(
            _readonly(gram),
            _readonly(rhs),
            _readonly(feature_sum),
            float(target_sum),
            float(np.dot(target, target)),
            int(rows),
        )

    def project(self, transform: np.ndarray) -> "DesignStatistics":
        projection = np.asarray(transform, dtype=np.float64)
        if projection.ndim != 2 or projection.shape[0] != len(self.rhs):
            raise ValueError("sufficient-statistic projection shape mismatch")
        gram = projection.T @ self.gram @ projection
        rhs = projection.T @ self.rhs
        feature_sum = projection.T @ self.feature_sum
        return DesignStatistics(
            gram,
            rhs,
            feature_sum,
            self.target_sum,
            self.target_square_sum,
            self.rows,
        )


def _statistics_solve(
    statistics: DesignStatistics, penalty: np.ndarray
) -> tuple[np.ndarray, float, dict[str, Any]]:
    coefficient, intercept, certificate = solve_centered_certified_gram(
        statistics.gram,
        statistics.rhs,
        statistics.feature_sum,
        statistics.target_sum,
        statistics.rows,
        penalty,
    )
    return coefficient, intercept, certificate.to_json()


def _statistics_mse(
    statistics: DesignStatistics,
    coefficient: np.ndarray,
    intercept: float,
) -> float:
    beta = np.asarray(coefficient, dtype=np.float64)
    offset = float(intercept)
    squared_error = (
        statistics.target_square_sum
        - 2.0 * float(beta @ statistics.rhs)
        - 2.0 * offset * statistics.target_sum
        + float(beta @ statistics.gram @ beta)
        + 2.0 * offset * float(beta @ statistics.feature_sum)
        + statistics.rows * offset * offset
    )
    scale = max(statistics.target_square_sum, 1.0)
    if squared_error < 0.0 and abs(squared_error) <= 1024.0 * np.finfo(np.float64).eps * scale:
        squared_error = 0.0
    return float(squared_error / statistics.rows)


def _u_transform(v: np.ndarray, m_tau: int) -> np.ndarray:
    m_x, rank = v.shape
    transform = np.zeros((m_tau * m_x, rank * m_tau), dtype=np.float64)
    for component in range(rank):
        for lag in range(m_tau):
            transform[
                lag * m_x : (lag + 1) * m_x,
                component * m_tau + lag,
            ] = v[:, component]
    return transform


def _v_transform(u: np.ndarray, m_x: int) -> np.ndarray:
    m_tau, rank = u.shape
    transform = np.zeros((m_tau * m_x, rank * m_x), dtype=np.float64)
    for component in range(rank):
        for amplitude in range(m_x):
            transform[
                amplitude::m_x,
                component * m_x + amplitude,
            ] = u[:, component]
    return transform


def _component_transform(
    left: np.ndarray, right: np.ndarray, count: int
) -> np.ndarray:
    return np.column_stack(
        [np.outer(left[:, index], right[index]).reshape(-1) for index in range(count)]
    )


@dataclass
class PreparedContractFit:
    """Reusable, fit-only transforms for one fixed fold/support and amplitude width."""

    values: np.ndarray
    target: np.ndarray
    requested_m_x: int
    basis: AmplitudeBasis
    phi: np.ndarray
    full_statistics: DesignStatistics
    linear_statistics: DesignStatistics
    _full_solution_cache: dict[
        tuple[float, float, float], tuple[np.ndarray, float, dict[str, Any]]
    ] = field(default_factory=dict, init=False, repr=False)
    _full_svd_cache: dict[
        tuple[float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = field(default_factory=dict, init=False, repr=False)
    _cache_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def full_solution(
        self, lambdas: tuple[float, float, float]
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Share one full-surface solve across FULL and every ALS rank/seed."""

        key = tuple(float(value) for value in lambdas)
        with self._cache_lock:
            cached = self._full_solution_cache.get(key)
            if cached is not None:
                return cached
            m_tau, m_x = self.phi.shape[1:]
            coefficient, intercept, certificate = _statistics_solve(
                self.full_statistics,
                surface_penalty(m_tau, m_x, *key),
            )
            value = (coefficient.reshape(m_tau, m_x), intercept, certificate)
            self._full_solution_cache[key] = value
            return value

    def full_svd(
        self, lambdas: tuple[float, float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the deterministic full-surface initialization shared by ALS seeds/ranks."""

        key = tuple(float(value) for value in lambdas)
        with self._cache_lock:
            cached = self._full_svd_cache.get(key)
            if cached is not None:
                return cached
            full_theta, _, _ = self.full_solution(key)
            value = np.linalg.svd(
                full_theta, full_matrices=False
            )
            self._full_svd_cache[key] = value
            return value


def prepare_contract_fit(
    train_values: np.ndarray,
    target: np.ndarray,
    requested_m_x: int,
) -> PreparedContractFit:
    """Materialize deterministic basis features once for repeated hyperparameter fits."""

    values = np.asarray(train_values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if values.ndim != 2 or y.shape != (len(values),):
        raise ValueError("invalid prepared Urysohn fit inputs")
    basis = AmplitudeBasis.fit(values, int(requested_m_x))
    phi = tensor_design(values, basis)
    return PreparedContractFit(
        values,
        y,
        int(requested_m_x),
        basis,
        phi,
        DesignStatistics.from_design(phi.reshape(len(phi), -1), y),
        DesignStatistics.from_design(phi[:, :, 0], y),
    )


def _rank_als(
    phi: np.ndarray,
    target: np.ndarray,
    rank: int,
    lambdas: tuple[float, float, float],
    *,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
    maximum_increases: int,
    divergence_factor: float,
    initial_svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    sufficient_statistics: DesignStatistics | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    lambda_0, lambda_tau, lambda_x = lambdas
    m_tau, m_x = phi.shape[1:]
    if initial_svd is None:
        full_design = phi.reshape(len(phi), -1)
        full_theta, _, _ = _centered_solve(
            full_design, target, surface_penalty(m_tau, m_x, *lambdas)
        )
        left, singular, right = np.linalg.svd(
            full_theta.reshape(m_tau, m_x), full_matrices=False
        )
    else:
        left, singular, right = initial_svd
    rng = np.random.default_rng(seed)
    used = min(rank, len(singular))
    u = np.zeros((m_tau, rank), dtype=np.float64)
    v = np.zeros((m_x, rank), dtype=np.float64)
    u[:, :used] = left[:, :used] * np.sqrt(singular[:used])[None, :]
    v[:, :used] = right[:used].T * np.sqrt(singular[:used])[None, :]
    if used < rank:
        u[:, used:] = 1e-3 * rng.normal(size=(m_tau, rank - used))
        v[:, used:] = 1e-3 * rng.normal(size=(m_x, rank - used))
    u += 1e-6 * rng.normal(size=u.shape)
    v += 1e-6 * rng.normal(size=v.shape)
    previous = float("inf")
    initial = None
    increases = 0
    history: list[float] = []
    subproblems: dict[str, Any] = {}
    status = "PASS"
    for iteration in range(maximum_iterations):
        if sufficient_statistics is None:
            design_u = np.einsum("tbx,xr->trb", phi, v).reshape(len(phi), rank * m_tau)
            u_vector, intercept_u, cert_u = _centered_solve(
                design_u, target, _factor_penalty(m_tau, rank, lambda_0, lambda_tau)
            )
        else:
            u_vector, intercept_u, cert_u = _statistics_solve(
                sufficient_statistics.project(_u_transform(v, m_tau)),
                _factor_penalty(m_tau, rank, lambda_0, lambda_tau),
            )
        u = u_vector.reshape(rank, m_tau).T
        if sufficient_statistics is None:
            design_v = np.einsum("tbx,br->trx", phi, u).reshape(len(phi), rank * m_x)
            v_vector, intercept_v, cert_v = _centered_solve(
                design_v, target, _factor_penalty(m_x, rank, lambda_0, lambda_x)
            )
        else:
            v_vector, intercept_v, cert_v = _statistics_solve(
                sufficient_statistics.project(_v_transform(u, m_x)),
                _factor_penalty(m_x, rank, lambda_0, lambda_x),
            )
        v = v_vector.reshape(rank, m_x).T
        theta = u @ v.T
        if sufficient_statistics is None:
            component = np.einsum("tbx,bx->t", phi, theta)
            intercept = float(np.mean(target - component, dtype=np.float64))
            objective = float(np.mean(np.square(target - component - intercept), dtype=np.float64))
        else:
            coefficient = theta.reshape(-1)
            intercept = float(
                (sufficient_statistics.target_sum - coefficient @ sufficient_statistics.feature_sum)
                / sufficient_statistics.rows
            )
            objective = _statistics_mse(
                sufficient_statistics, coefficient, intercept
            )
        history.append(objective)
        initial = objective if initial is None else initial
        if objective > previous:
            increases += 1
        else:
            increases = 0
        if objective > divergence_factor * max(initial, np.finfo(np.float64).tiny) or increases > maximum_increases:
            status = "NUMERICALLY_INVALID"
            break
        if np.isfinite(previous) and abs(previous - objective) <= tolerance * max(abs(previous), 1.0):
            break
        previous = objective
        subproblems = {"u": cert_u, "v": cert_v, "intercepts": [intercept_u, intercept_v]}
    theta = u @ v.T
    # Fixed-support refit: preserve singular directions, refit component weights and intercept.
    left, singular, right = np.linalg.svd(theta, full_matrices=False)
    component_count = min(rank, len(singular))
    if sufficient_statistics is None:
        component_design = np.column_stack(
            [np.einsum("tbx,b,x->t", phi, left[:, index], right[index]) for index in range(component_count)]
        )
        weights, intercept, refit_certificate = _centered_solve(
            component_design,
            target,
            lambda_0 * np.eye(component_design.shape[1], dtype=np.float64),
        )
    else:
        component_statistics = sufficient_statistics.project(
            _component_transform(left, right, component_count)
        )
        weights, intercept, refit_certificate = _statistics_solve(
            component_statistics,
            lambda_0 * np.eye(component_count, dtype=np.float64),
        )
    theta = sum(
        (weights[index] * np.outer(left[:, index], right[index]) for index in range(len(weights))),
        start=np.zeros((m_tau, m_x), dtype=np.float64),
    )
    train_mse = (
        float(np.mean(np.square(target - (np.einsum("tbx,bx->t", phi, theta) + intercept)), dtype=np.float64))
        if sufficient_statistics is None
        else _statistics_mse(sufficient_statistics, theta.reshape(-1), intercept)
    )
    values = np.linalg.svd(theta, compute_uv=False)
    return theta, intercept, {
        "status": status,
        "iterations": len(history),
        "objective_history": history,
        "train_mse": train_mse,
        "requested_rank": rank,
        "realized_rank": int(np.linalg.matrix_rank(theta, tol=values.max(initial=0.0) * 1e-10)),
        "singular_values": values.tolist(),
        "fixed_support_refit": True,
        "fixed_support_refit_certificate": refit_certificate,
        "subproblem_certificates": subproblems,
    }


def fit_prepared_contract(
    prepared: PreparedContractFit,
    family: str,
    m_x: int,
    lambdas: tuple[float, float, float],
    *,
    als_seeds: tuple[int, ...] = (20260804, 20260805, 20260806),
    als_max_iterations: int = 100,
    als_tolerance: float = 1e-8,
    als_max_increases: int = 5,
    als_divergence_factor: float = 1e6,
) -> dict[str, Any]:
    """Fit one contract while reusing immutable fold-local transforms/statistics."""

    if family == "EXACT_ZERO":
        return {
            "family": family,
            "intercept": 0.0,
            "theta": [],
            "basis": None,
            "certificate": {"status": "EXACT_ZERO"},
            "parameter_count": 0,
        }
    requested = 1 if family == "LINEAR_DISTRIBUTED_LAG" else int(m_x)
    if requested != prepared.requested_m_x:
        raise ValueError(
            "prepared amplitude width does not match the requested contract"
        )
    basis = prepared.basis
    if basis.dimension == 0:
        return {
            "family": "EXACT_ZERO",
            "intercept": 0.0,
            "theta": [],
            "basis": basis.metadata(),
            "certificate": {
                "status": "NOT_APPLICABLE",
                "reason": "CONSTANT_CHANNEL",
            },
            "parameter_count": 0,
        }
    y = prepared.target
    phi = prepared.phi
    m_tau, realized_m_x = phi.shape[1:]
    if family == "LINEAR_DISTRIBUTED_LAG":
        coefficient, intercept, certificate = _statistics_solve(
            prepared.linear_statistics,
            lambdas[0] * np.eye(m_tau)
            + lambdas[1] * _difference_penalty_cached(m_tau),
        )
        theta = coefficient[:, None]
    elif family == "FULL_FINITE_URYSOHN":
        theta, intercept, certificate = prepared.full_solution(lambdas)
    elif family.startswith("RANK_"):
        rank = int(family.split("_")[1])
        initial_svd = prepared.full_svd(lambdas)
        candidates = [
            _rank_als(
                phi,
                y,
                rank,
                lambdas,
                seed=seed,
                maximum_iterations=als_max_iterations,
                tolerance=als_tolerance,
                maximum_increases=als_max_increases,
                divergence_factor=als_divergence_factor,
                initial_svd=initial_svd,
                sufficient_statistics=prepared.full_statistics,
            )
            for seed in als_seeds
        ]
        theta, intercept, certificate = min(
            candidates, key=lambda item: item[2]["train_mse"]
        )
        certificate = {
            **certificate,
            "initializations": len(candidates),
            "initialization_train_mse": [
                item[2]["train_mse"] for item in candidates
            ],
        }
    else:
        raise KeyError(family)
    rank = int(family.split("_")[1]) if family.startswith("RANK_") else None
    free_parameters = (
        rank * (m_tau + realized_m_x) if rank is not None else theta.size
    ) + 1
    return {
        "family": family,
        "intercept": float(intercept),
        "theta": theta.tolist(),
        "basis": basis.metadata(),
        "certificate": certificate,
        "parameter_count": int(free_parameters),
    }


def fit_contract(
    train_values: np.ndarray,
    target: np.ndarray,
    family: str,
    m_x: int,
    lambdas: tuple[float, float, float],
    *,
    als_seeds: tuple[int, ...] = (20260804, 20260805, 20260806),
    als_max_iterations: int = 100,
    als_tolerance: float = 1e-8,
    als_max_increases: int = 5,
    als_divergence_factor: float = 1e6,
) -> dict[str, Any]:
    values = np.asarray(train_values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if family == "EXACT_ZERO":
        return {"family": family, "intercept": 0.0, "theta": [], "basis": None, "certificate": {"status": "EXACT_ZERO"}, "parameter_count": 0}
    requested = 1 if family == "LINEAR_DISTRIBUTED_LAG" else m_x
    basis = AmplitudeBasis.fit(values, requested)
    if basis.dimension == 0:
        return {"family": "EXACT_ZERO", "intercept": 0.0, "theta": [], "basis": basis.metadata(), "certificate": {"status": "NOT_APPLICABLE", "reason": "CONSTANT_CHANNEL"}, "parameter_count": 0}
    prepared = prepare_contract_fit(values, y, requested)
    return fit_prepared_contract(
        prepared,
        family,
        m_x,
        lambdas,
        als_seeds=als_seeds,
        als_max_iterations=als_max_iterations,
        als_tolerance=als_tolerance,
        als_max_increases=als_max_increases,
        als_divergence_factor=als_divergence_factor,
    )


def basis_from_metadata(value: dict[str, Any]) -> AmplitudeBasis:
    return AmplitudeBasis(
        kind=str(value["kind"]), mean=float(value["mean"]), scale=float(value["scale"]),
        knots=tuple(float(x) for x in value["knots"]), levels=tuple(float(x) for x in value["levels"]),
        feature_mean=tuple(float(x) for x in value["feature_mean"]),
    )


def predict_contract_from_design(
    phi: np.ndarray, contract: dict[str, Any]
) -> np.ndarray:
    """Predict from a transform created by the contract's fit-local basis."""

    design = np.asarray(phi, dtype=np.float64)
    if contract["family"] == "EXACT_ZERO":
        return np.zeros(len(design), dtype=np.float64)
    theta = np.asarray(contract["theta"], dtype=np.float64)
    if (
        design.ndim != 3
        or design.shape[1] != theta.shape[0]
        or design.shape[2] < theta.shape[1]
    ):
        raise ValueError("prepared prediction design does not match contract")
    design = design[:, :, : theta.shape[1]]
    result = np.empty(len(design), dtype=np.float64)
    for start in range(0, len(design), STREAM_CHUNK_ROWS):
        stop = min(start + STREAM_CHUNK_ROWS, len(design))
        result[start:stop] = (
            np.einsum("tbx,bx->t", design[start:stop], theta)
            + float(contract["intercept"])
        )
    return result


def predict_contract(values: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    if contract["family"] == "EXACT_ZERO":
        return np.zeros(len(values), dtype=np.float64)
    basis = basis_from_metadata(contract["basis"])
    theta = np.asarray(contract["theta"], dtype=np.float64)
    result = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), STREAM_CHUNK_ROWS):
        stop = min(start + STREAM_CHUNK_ROWS, len(values))
        phi = tensor_design(values[start:stop], basis)
        result[start:stop] = np.einsum("tbx,bx->t", phi, theta) + float(contract["intercept"])
    return result
