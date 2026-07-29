"""Explicit dense FP64 reference operations for ORSS equivalence gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .operator import UrysohnLinearOperator
from .penalties import PenaltyWeights, SeparablePenalty


@dataclass(frozen=True, slots=True)
class DenseReferenceFit:
    coefficients: torch.Tensor
    relative_kkt_residual: float


def dense_penalty_matrix(
    penalty: SeparablePenalty, weights: PenaltyWeights
) -> torch.Tensor:
    lag = penalty.lag_factor.T @ penalty.lag_factor
    amplitude = penalty.amplitude_factor.T @ penalty.amplitude_factor
    block = (
        weights.lag
        * torch.kron(
            lag,
            torch.eye(
                penalty.m_x, device=penalty.device, dtype=penalty.dtype
            ),
        )
        + weights.amplitude
        * torch.kron(
            torch.eye(
                penalty.m_tau,
                device=penalty.device,
                dtype=penalty.dtype,
            ),
            amplitude,
        )
        + weights.ridge
        * torch.eye(
            penalty.m_tau * penalty.m_x,
            device=penalty.device,
            dtype=penalty.dtype,
        )
    )
    return torch.block_diag(*([block] * penalty.channels))


def solve_dense_reference(
    operator: UrysohnLinearOperator,
    centered_target: torch.Tensor,
    penalty: SeparablePenalty,
    weights: PenaltyWeights,
) -> DenseReferenceFit:
    design = operator.dense_design().to(torch.float64)
    target = centered_target.to(design)
    gram = design.T @ design / len(design)
    rhs = design.T @ target / len(design)
    scientific_penalty = dense_penalty_matrix(penalty, weights).to(gram)
    system = 0.5 * (gram + gram.T) + scientific_penalty
    if weights.exact_zero:
        coefficients = torch.linalg.lstsq(design, target).solution
    else:
        coefficients = torch.linalg.solve(system, rhs)
    residual = rhs - system @ coefficients
    relative = float(
        torch.linalg.vector_norm(residual).item()
        / torch.linalg.vector_norm(rhs).clamp_min(
            torch.finfo(torch.float64).eps
        ).item()
    )
    return DenseReferenceFit(coefficients, relative)


def dense_fp64_refine(
    operator: UrysohnLinearOperator,
    centered_target: torch.Tensor,
    penalty: SeparablePenalty,
    weights: PenaltyWeights,
    initial: torch.Tensor,
) -> DenseReferenceFit:
    """One dense FP64 rescue step, used only for final certification."""

    design = operator.dense_design().to(torch.float64)
    target = centered_target.to(design)
    gram = design.T @ design / len(design)
    rhs = design.T @ target / len(design)
    scientific_penalty = dense_penalty_matrix(penalty, weights).to(gram)
    system = 0.5 * (gram + gram.T) + scientific_penalty
    current = initial.to(system)
    residual = rhs - system @ current
    correction = torch.linalg.solve(system, residual)
    coefficients = current + correction
    final_residual = rhs - system @ coefficients
    relative = float(
        torch.linalg.vector_norm(final_residual).item()
        / torch.linalg.vector_norm(rhs).clamp_min(
            torch.finfo(torch.float64).eps
        ).item()
    )
    return DenseReferenceFit(coefficients, relative)


def dense_fp64_spectral_equilibrated_refine(
    operator: UrysohnLinearOperator,
    centered_target: torch.Tensor,
    penalty: SeparablePenalty,
    weights: PenaltyWeights,
    *,
    relative_tolerance: float,
    maximum_refinement_steps: int = 64,
) -> DenseReferenceFit:
    """Certify an ill-conditioned selected fit without changing its objective.

    The scientific penalty is diagonalized in the exact lag-by-amplitude
    eigenbasis, the complete dense FP64 normal system is diagonally
    equilibrated there, and a pivoted LU solve is refined against the original
    matrix-free operator residual.  This path is intentionally restricted to
    final selected-model certification.
    """

    if weights.exact_zero:
        raise ValueError(
            "Spectral-equilibrated rescue requires a positive penalty."
        )
    if maximum_refinement_steps < 0:
        raise ValueError("Maximum refinement steps cannot be negative.")
    design = operator.dense_design().to(torch.float64)
    target = centered_target.to(design)
    gram = design.T @ design / len(design)
    rhs = design.T @ target / len(design)
    system = (
        0.5 * (gram + gram.T)
        + dense_penalty_matrix(penalty, weights).to(gram)
    )

    _, lag_vectors = torch.linalg.eigh(
        0.5
        * (
            penalty.lag_normal_matrix
            + penalty.lag_normal_matrix.T
        )
    )
    _, amplitude_vectors = torch.linalg.eigh(
        0.5
        * (
            penalty.amplitude_normal_matrix
            + penalty.amplitude_normal_matrix.T
        )
    )
    block_vectors = torch.kron(lag_vectors, amplitude_vectors)
    vectors = torch.block_diag(*([block_vectors] * penalty.channels))
    transformed_system = vectors.T @ system @ vectors
    transformed_system = 0.5 * (
        transformed_system + transformed_system.T
    )
    transformed_rhs = vectors.T @ rhs
    scale = torch.sqrt(
        torch.diagonal(transformed_system).clamp_min(
            torch.finfo(transformed_system.dtype).tiny
        )
    )
    equilibrated = (
        transformed_system / scale[:, None] / scale[None, :]
    )
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    factor, pivots, info = torch.linalg.lu_factor_ex(equilibrated)
    if int(info.item()) != 0:
        raise RuntimeError(
            f"EQUILIBRATED_LU_FAILED_AT_PIVOT_{int(info.item())}"
        )
    scaled = torch.linalg.lu_solve(
        factor, pivots, (transformed_rhs / scale)[:, None]
    ).squeeze(1)
    coefficients = vectors @ (scaled / scale)

    operator_rhs = operator.rhs(centered_target)
    rhs_norm = torch.linalg.vector_norm(operator_rhs).clamp_min(
        torch.finfo(operator_rhs.dtype).tiny
    )
    best_coefficients = coefficients.clone()
    best_relative = float("inf")
    for _ in range(maximum_refinement_steps + 1):
        residual = (
            operator_rhs
            - operator.normal(coefficients)
            - penalty.normal(coefficients, weights)
        )
        relative = float(
            (torch.linalg.vector_norm(residual) / rhs_norm).item()
        )
        if relative < best_relative:
            best_relative = relative
            best_coefficients = coefficients.clone()
        if best_relative <= relative_tolerance:
            break
        transformed_residual = vectors.T @ residual
        correction = torch.linalg.lu_solve(
            factor,
            pivots,
            (transformed_residual / scale)[:, None],
        ).squeeze(1)
        coefficients = coefficients + vectors @ (correction / scale)
    return DenseReferenceFit(best_coefficients, best_relative)
