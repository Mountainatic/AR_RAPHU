"""Augmented least-squares operator without explicit normal equations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch

from .operator import UrysohnLinearOperator
from .penalties import PenaltyWeights, SeparablePenalty


@dataclass(frozen=True, slots=True)
class AugmentedSlices:
    data: slice
    lag: slice
    amplitude: slice
    ridge: slice
    total: int


class AugmentedRegularizedOperator:
    def __init__(
        self,
        data_operator: UrysohnLinearOperator,
        penalty: SeparablePenalty,
        weights: PenaltyWeights,
    ) -> None:
        self.data_operator = data_operator
        self.penalty = penalty
        self.weights = weights
        n = data_operator.observations
        lag_size = (
            penalty.channels * penalty.lag_factor.shape[0] * penalty.m_x
            if weights.lag
            else 0
        )
        amplitude_size = (
            penalty.channels
            * penalty.m_tau
            * penalty.amplitude_factor.shape[0]
            if weights.amplitude
            else 0
        )
        ridge_size = penalty.dimension if weights.ridge else 0
        lag_start = n
        amplitude_start = lag_start + lag_size
        ridge_start = amplitude_start + amplitude_size
        self.slices = AugmentedSlices(
            data=slice(0, n),
            lag=slice(lag_start, amplitude_start),
            amplitude=slice(amplitude_start, ridge_start),
            ridge=slice(ridge_start, ridge_start + ridge_size),
            total=ridge_start + ridge_size,
        )
        self.dimension = penalty.dimension
        self.device = data_operator.device
        self.dtype = data_operator.dtype

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        parts = [
            self.data_operator.forward(theta)
            / sqrt(self.data_operator.observations)
        ]
        if self.weights.lag:
            parts.append(
                sqrt(self.weights.lag) * self.penalty.lag_forward(theta)
            )
        if self.weights.amplitude:
            parts.append(
                sqrt(self.weights.amplitude)
                * self.penalty.amplitude_forward(theta)
            )
        if self.weights.ridge:
            parts.append(sqrt(self.weights.ridge) * theta)
        return torch.cat(parts)

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        result = self.data_operator.adjoint(
            residual[self.slices.data]
        ) / sqrt(self.data_operator.observations)
        if self.weights.lag:
            result = result + sqrt(self.weights.lag) * self.penalty.lag_adjoint(
                residual[self.slices.lag]
            )
        if self.weights.amplitude:
            result = (
                result
                + sqrt(self.weights.amplitude)
                * self.penalty.amplitude_adjoint(
                    residual[self.slices.amplitude]
                )
            )
        if self.weights.ridge:
            result = result + sqrt(self.weights.ridge) * residual[
                self.slices.ridge
            ]
        return result

    def normal(self, theta: torch.Tensor) -> torch.Tensor:
        return self.data_operator.normal(theta) + self.penalty.normal(
            theta, self.weights
        )

    def augmented_rhs(self, centered_target: torch.Tensor) -> torch.Tensor:
        rhs = torch.zeros(
            self.slices.total, device=self.device, dtype=self.dtype
        )
        rhs[self.slices.data] = centered_target / sqrt(
            self.data_operator.observations
        )
        return rhs

    def normal_rhs(self, centered_target: torch.Tensor) -> torch.Tensor:
        return self.data_operator.rhs(centered_target)

