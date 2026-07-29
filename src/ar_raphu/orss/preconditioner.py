"""Matrix-free diagonal and channel-block preconditioners."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .operator import UrysohnLinearOperator
from .penalties import PenaltyWeights, SeparablePenalty


def data_normal_diagonal(operator: UrysohnLinearOperator) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    for branch in operator.branches:
        diagonal = torch.zeros(
            (operator.m_tau, operator.m_x),
            device=operator.device,
            dtype=operator.dtype,
        )
        for start in range(0, operator.observations, operator.chunk_time):
            stop = min(start + operator.chunk_time, operator.observations)
            features = torch.einsum(
                "la,nlb->nab",
                branch.lag_basis,
                branch.amplitude[start:stop],
            )
            diagonal += features.square().sum(dim=0)
        raw_mean = operator.feature_mean[len(blocks)]
        diagonal = diagonal / operator.observations - raw_mean.square()
        blocks.append(diagonal.clamp_min(torch.finfo(operator.dtype).eps))
    return torch.stack(blocks).reshape(-1)


@dataclass(slots=True)
class DiagonalPreconditioner:
    diagonal: torch.Tensor

    def solve(self, vector: torch.Tensor) -> torch.Tensor:
        return vector / self.diagonal


def build_diagonal_preconditioner(
    operator: UrysohnLinearOperator,
    penalty: SeparablePenalty,
    weights: PenaltyWeights,
    *,
    data_diagonal: torch.Tensor | None = None,
) -> DiagonalPreconditioner:
    if data_diagonal is None:
        data_diagonal = data_normal_diagonal(operator)
    diagonal = data_diagonal + penalty.diagonal(weights)
    floor = diagonal.abs().max().clamp_min(1.0) * 1.0e-12
    return DiagonalPreconditioner(diagonal.clamp_min(floor))
