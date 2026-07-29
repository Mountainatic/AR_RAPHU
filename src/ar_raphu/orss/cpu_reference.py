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
