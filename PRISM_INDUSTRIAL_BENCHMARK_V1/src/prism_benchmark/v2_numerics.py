from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class LinearCertificate:
    status: str
    solver: str
    jitter_relative: float
    relative_kkt: float
    condition_number: float
    numerical_rank: int
    coefficient_l2: float

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


def deterministic_hash_order(values: Iterable[object]) -> np.ndarray:
    """Full SHA256 lexicographic order required by the V2 freeze."""
    hashes = np.asarray([hashlib.sha256(str(value).encode("utf-8")).digest() for value in values], dtype="|S32")
    return np.argsort(hashes, kind="stable")


def deterministic_subsample(values: Iterable[object], cap: int) -> np.ndarray:
    order = deterministic_hash_order(values)
    return order if len(order) <= cap else order[:cap]


def difference_penalty(size: int, order: int = 2) -> np.ndarray:
    if size <= order:
        return np.zeros((size, size), dtype=np.float64)
    difference = np.diff(np.eye(size, dtype=np.float64), n=order, axis=0)
    return difference.T @ difference


def solve_certified(
    design: np.ndarray,
    target: np.ndarray,
    penalty: np.ndarray | float = 0.0,
    *,
    jitter_grid: tuple[float, ...] = (0.0, 1e-12, 1e-10, 1e-8),
    qr_tolerance: float = 1e-10,
    svd_rcond: float = 1e-12,
    kkt_warning: float = 1e-10,
    kkt_hard: float = 1e-8,
    condition_warning: float = 1e12,
    condition_hard: float = 1e14,
    coefficient_hard: float = 1e6,
) -> tuple[np.ndarray, LinearCertificate]:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite or invalid linear system")
    p = x.shape[1]
    gram = x.T @ x
    rhs = x.T @ y
    if np.isscalar(penalty):
        penalty_matrix = np.eye(p, dtype=np.float64) * float(penalty)
    else:
        penalty_matrix = np.asarray(penalty, dtype=np.float64)
        if penalty_matrix.shape != (p, p):
            raise ValueError("penalty shape mismatch")
    system0 = gram + penalty_matrix
    scale = max(float(np.trace(gram)) / max(p, 1), np.finfo(np.float64).tiny)
    coefficient: np.ndarray | None = None
    solver = ""
    used_jitter = 0.0
    system = system0
    for relative in jitter_grid:
        system = system0 + np.eye(p, dtype=np.float64) * (relative * scale)
        try:
            factor = linalg.cho_factor(system, lower=True, check_finite=False)
            coefficient = linalg.cho_solve(factor, rhs, check_finite=False)
            solver = "CHOLESKY"
            used_jitter = relative
            break
        except linalg.LinAlgError:
            continue
    if coefficient is None:
        try:
            q, r, piv = linalg.qr(system0, mode="economic", pivoting=True, check_finite=False)
            diagonal = np.abs(np.diag(r))
            rank = int(np.sum(diagonal > (diagonal.max(initial=0.0) * qr_tolerance)))
            if rank == p:
                pivot_solution = linalg.solve_triangular(r, q.T @ rhs, check_finite=False)
                coefficient = np.empty(p, dtype=np.float64)
                coefficient[piv] = pivot_solution
                solver = "PIVOTED_QR"
        except linalg.LinAlgError:
            coefficient = None
    if coefficient is None:
        coefficient = np.linalg.lstsq(system0, rhs, rcond=svd_rcond)[0]
        solver = "SVD_RESCUE"
        system = system0
    residual = system @ coefficient - rhs
    denominator = max(float(np.linalg.norm(rhs)), float(np.linalg.norm(system) * np.linalg.norm(coefficient)), 1.0)
    relative_kkt = float(np.linalg.norm(residual) / denominator)
    eigenvalues = np.linalg.eigvalsh((system + system.T) * 0.5)
    absolute = np.abs(eigenvalues)
    nonzero = absolute[absolute > absolute.max(initial=0.0) * 1e-10]
    condition = float(absolute.max(initial=0.0) / nonzero.min()) if len(nonzero) else float("inf")
    rank = int(np.linalg.matrix_rank(system, tol=absolute.max(initial=0.0) * 1e-10))
    coefficient_l2 = float(np.linalg.norm(coefficient))
    if (not np.isfinite(coefficient).all() or relative_kkt > kkt_hard or condition > condition_hard or coefficient_l2 > coefficient_hard):
        status = "NUMERICALLY_INVALID"
    elif relative_kkt > kkt_warning or condition > condition_warning:
        status = "PASS_WITH_WARNING"
    else:
        status = "PASS"
    return coefficient, LinearCertificate(status, solver, used_jitter, relative_kkt, condition, rank, coefficient_l2)


def residualize(
    candidate: np.ndarray,
    against: np.ndarray,
    *,
    relative_tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    z = np.asarray(candidate, dtype=np.float64)
    base = np.asarray(against, dtype=np.float64)
    if base.ndim == 1:
        base = base[:, None]
    if z.ndim == 1:
        z = z[:, None]
    if len(z) != len(base):
        raise ValueError("orthogonalization row mismatch")
    q, r, _ = linalg.qr(base, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    rank = int(np.sum(diagonal > diagonal.max(initial=0.0) * 1e-10))
    q = q[:, :rank]
    residual = z - q @ (q.T @ z) if rank else z.copy()
    denominator = max(float(np.linalg.norm(base) * np.linalg.norm(residual)), np.finfo(np.float64).tiny)
    relative = float(np.linalg.norm(base.T @ residual) / denominator)
    return residual, {
        "method": "pivoted_qr",
        "projection_rank": rank,
        "relative_orthogonality": relative,
        "pass": bool(relative <= relative_tolerance),
    }

