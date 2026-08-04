from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .v2_basis import AmplitudeBasis, tensor_design
from .v2_numerics import difference_penalty, solve_certified


def surface_penalty(m_tau: int, m_x: int, lambda_0: float, lambda_tau: float, lambda_x: float) -> np.ndarray:
    return (
        lambda_0 * np.eye(m_tau * m_x, dtype=np.float64)
        + lambda_tau * np.kron(difference_penalty(m_tau), np.eye(m_x, dtype=np.float64))
        + lambda_x * np.kron(np.eye(m_tau, dtype=np.float64), difference_penalty(m_x))
    )


def _centered_solve(design: np.ndarray, target: np.ndarray, penalty: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
    center = design.mean(axis=0, dtype=np.float64)
    y_mean = float(np.mean(target, dtype=np.float64))
    coefficient, certificate = solve_certified(design - center, target - y_mean, penalty)
    intercept = y_mean - float(center @ coefficient)
    return coefficient, intercept, certificate.to_json()


def _factor_penalty(size: int, rank: int, lambda_0: float, smoothness: float) -> np.ndarray:
    block = lambda_0 * np.eye(size, dtype=np.float64) + smoothness * difference_penalty(size)
    return np.kron(np.eye(rank, dtype=np.float64), block)


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
) -> tuple[np.ndarray, float, dict[str, Any]]:
    lambda_0, lambda_tau, lambda_x = lambdas
    m_tau, m_x = phi.shape[1:]
    full_design = phi.reshape(len(phi), -1)
    full_theta, _, _ = _centered_solve(full_design, target, surface_penalty(m_tau, m_x, *lambdas))
    left, singular, right = np.linalg.svd(full_theta.reshape(m_tau, m_x), full_matrices=False)
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
        design_u = np.einsum("tbx,xr->trb", phi, v).reshape(len(phi), rank * m_tau)
        u_vector, intercept_u, cert_u = _centered_solve(
            design_u, target, _factor_penalty(m_tau, rank, lambda_0, lambda_tau)
        )
        u = u_vector.reshape(rank, m_tau).T
        design_v = np.einsum("tbx,br->trx", phi, u).reshape(len(phi), rank * m_x)
        v_vector, intercept_v, cert_v = _centered_solve(
            design_v, target, _factor_penalty(m_x, rank, lambda_0, lambda_x)
        )
        v = v_vector.reshape(rank, m_x).T
        theta = u @ v.T
        component = np.einsum("tbx,bx->t", phi, theta)
        intercept = float(np.mean(target - component, dtype=np.float64))
        objective = float(np.mean(np.square(target - component - intercept), dtype=np.float64))
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
    component_design = np.column_stack(
        [np.einsum("tbx,b,x->t", phi, left[:, index], right[index]) for index in range(min(rank, len(singular)))]
    )
    weights, intercept, refit_certificate = _centered_solve(
        component_design, target, lambda_0 * np.eye(component_design.shape[1], dtype=np.float64)
    )
    theta = sum(
        (weights[index] * np.outer(left[:, index], right[index]) for index in range(len(weights))),
        start=np.zeros((m_tau, m_x), dtype=np.float64),
    )
    fitted = np.einsum("tbx,bx->t", phi, theta) + intercept
    values = np.linalg.svd(theta, compute_uv=False)
    return theta, intercept, {
        "status": status,
        "iterations": len(history),
        "objective_history": history,
        "train_mse": float(np.mean(np.square(target - fitted), dtype=np.float64)),
        "requested_rank": rank,
        "realized_rank": int(np.linalg.matrix_rank(theta, tol=values.max(initial=0.0) * 1e-10)),
        "singular_values": values.tolist(),
        "fixed_support_refit": True,
        "fixed_support_refit_certificate": refit_certificate,
        "subproblem_certificates": subproblems,
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
    phi = tensor_design(values, basis)
    m_tau, realized_m_x = phi.shape[1:]
    if family == "LINEAR_DISTRIBUTED_LAG":
        design = phi[:, :, 0]
        coefficient, intercept, certificate = _centered_solve(
            design, y, lambdas[0] * np.eye(m_tau) + lambdas[1] * difference_penalty(m_tau)
        )
        theta = coefficient[:, None]
    elif family == "FULL_FINITE_URYSOHN":
        coefficient, intercept, certificate = _centered_solve(
            phi.reshape(len(phi), -1), y, surface_penalty(m_tau, realized_m_x, *lambdas)
        )
        theta = coefficient.reshape(m_tau, realized_m_x)
    elif family.startswith("RANK_"):
        rank = int(family.split("_")[1])
        candidates = [
            _rank_als(
                phi, y, rank, lambdas, seed=seed, maximum_iterations=als_max_iterations,
                tolerance=als_tolerance, maximum_increases=als_max_increases,
                divergence_factor=als_divergence_factor,
            )
            for seed in als_seeds
        ]
        theta, intercept, certificate = min(candidates, key=lambda item: item[2]["train_mse"])
        certificate = {**certificate, "initializations": len(candidates), "initialization_train_mse": [item[2]["train_mse"] for item in candidates]}
    else:
        raise KeyError(family)
    rank = int(family.split("_")[1]) if family.startswith("RANK_") else None
    free_parameters = (rank * (m_tau + realized_m_x) if rank is not None else theta.size) + 1
    return {
        "family": family,
        "intercept": float(intercept),
        "theta": theta.tolist(),
        "basis": basis.metadata(),
        "certificate": certificate,
        "parameter_count": int(free_parameters),
    }


def basis_from_metadata(value: dict[str, Any]) -> AmplitudeBasis:
    return AmplitudeBasis(
        kind=str(value["kind"]), mean=float(value["mean"]), scale=float(value["scale"]),
        knots=tuple(float(x) for x in value["knots"]), levels=tuple(float(x) for x in value["levels"]),
        feature_mean=tuple(float(x) for x in value["feature_mean"]),
    )


def predict_contract(values: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    if contract["family"] == "EXACT_ZERO":
        return np.zeros(len(values), dtype=np.float64)
    basis = basis_from_metadata(contract["basis"])
    phi = tensor_design(values, basis)
    theta = np.asarray(contract["theta"], dtype=np.float64)
    return np.einsum("tbx,bx->t", phi, theta) + float(contract["intercept"])
