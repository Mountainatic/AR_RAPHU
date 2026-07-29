"""FP64 residual certification and iterative refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True, slots=True)
class RefinementResult:
    coefficients: torch.Tensor
    relative_kkt_residual: float
    steps: int
    converged: bool
    fallback_required: bool


def iterative_refinement(
    coefficients: torch.Tensor,
    *,
    apply_fp64: Callable[[torch.Tensor], torch.Tensor],
    rhs_fp64: torch.Tensor,
    correction_solver: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    maximum_steps: int = 5,
    threshold: float = 1.0e-8,
) -> RefinementResult:
    current = coefficients.to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(rhs_fp64).clamp_min(
        torch.finfo(torch.float64).eps
    )
    relative = float("inf")
    for step in range(maximum_steps + 1):
        residual = rhs_fp64 - apply_fp64(current)
        relative = float(
            torch.linalg.vector_norm(residual).item() / denominator.item()
        )
        if relative <= threshold:
            return RefinementResult(
                coefficients=current,
                relative_kkt_residual=relative,
                steps=step,
                converged=True,
                fallback_required=False,
            )
        if step == maximum_steps:
            break
        correction = correction_solver(residual, current)
        current = current + correction.to(current)
    return RefinementResult(
        coefficients=current,
        relative_kkt_residual=relative,
        steps=maximum_steps,
        converged=False,
        fallback_required=True,
    )

