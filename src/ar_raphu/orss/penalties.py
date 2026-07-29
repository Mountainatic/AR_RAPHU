"""Separable roughness factors and affine penalty operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def normalized_second_difference_factor(
    size: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if size < 3:
        return torch.zeros((0, size), device=device, dtype=dtype)
    identity = torch.eye(size, device=device, dtype=dtype)
    difference = identity[:-2] - 2.0 * identity[1:-1] + identity[2:]
    trace = torch.sum(difference.square())
    return difference * torch.sqrt(
        torch.as_tensor(size, device=device, dtype=dtype) / trace
    )


@dataclass(frozen=True, slots=True)
class PenaltyWeights:
    lag: float
    amplitude: float
    ridge: float

    @property
    def exact_zero(self) -> bool:
        return self.lag == 0.0 and self.amplitude == 0.0 and self.ridge == 0.0


class SeparablePenalty:
    def __init__(
        self,
        *,
        channels: int,
        m_tau: int,
        m_x: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.channels = channels
        self.m_tau = m_tau
        self.m_x = m_x
        self.dimension = channels * m_tau * m_x
        self.device = device
        self.dtype = dtype
        self.lag_factor = normalized_second_difference_factor(
            m_tau, device=device, dtype=dtype
        )
        self.amplitude_factor = normalized_second_difference_factor(
            m_x, device=device, dtype=dtype
        )
        self.lag_normal_matrix = self.lag_factor.T @ self.lag_factor
        self.amplitude_normal_matrix = (
            self.amplitude_factor.T @ self.amplitude_factor
        )

    def reshape(self, theta: torch.Tensor) -> torch.Tensor:
        return theta.reshape(self.channels, self.m_tau, self.m_x)

    def lag_forward(self, theta: torch.Tensor) -> torch.Tensor:
        coefficients = self.reshape(theta)
        return torch.einsum("ra,cab->crb", self.lag_factor, coefficients).reshape(
            -1
        )

    def lag_adjoint(self, values: torch.Tensor) -> torch.Tensor:
        shaped = values.reshape(
            self.channels, self.lag_factor.shape[0], self.m_x
        )
        return torch.einsum(
            "ra,crb->cab", self.lag_factor, shaped
        ).reshape(-1)

    def amplitude_forward(self, theta: torch.Tensor) -> torch.Tensor:
        coefficients = self.reshape(theta)
        return torch.einsum(
            "sb,cab->cas", self.amplitude_factor, coefficients
        ).reshape(-1)

    def amplitude_adjoint(self, values: torch.Tensor) -> torch.Tensor:
        shaped = values.reshape(
            self.channels, self.m_tau, self.amplitude_factor.shape[0]
        )
        return torch.einsum(
            "sb,cas->cab", self.amplitude_factor, shaped
        ).reshape(-1)

    def normal(
        self, theta: torch.Tensor, weights: PenaltyWeights
    ) -> torch.Tensor:
        result = weights.ridge * theta
        if weights.lag:
            result = result + weights.lag * self.lag_adjoint(
                self.lag_forward(theta)
            )
        if weights.amplitude:
            result = result + weights.amplitude * self.amplitude_adjoint(
                self.amplitude_forward(theta)
            )
        return result

    def normal_batch(
        self,
        theta: torch.Tensor,
        weights: list[PenaltyWeights] | tuple[PenaltyWeights, ...],
    ) -> torch.Tensor:
        if theta.ndim != 2 or theta.shape[1] != self.dimension:
            raise ValueError(
                "Batched coefficients must have shape (batch, dimension)."
            )
        if len(weights) != theta.shape[0]:
            raise ValueError("One penalty weight is required per batch row.")
        coefficients = theta.reshape(
            -1, self.channels, self.m_tau, self.m_x
        )
        lag_weights = torch.as_tensor(
            [row.lag for row in weights],
            device=theta.device,
            dtype=theta.dtype,
        )
        amplitude_weights = torch.as_tensor(
            [row.amplitude for row in weights],
            device=theta.device,
            dtype=theta.dtype,
        )
        ridge_weights = torch.as_tensor(
            [row.ridge for row in weights],
            device=theta.device,
            dtype=theta.dtype,
        )
        result = ridge_weights[:, None, None, None] * coefficients
        if bool(torch.any(lag_weights != 0)):
            lagged = torch.einsum(
                "ij,kcjb->kcib", self.lag_normal_matrix, coefficients
            )
            result = result + lag_weights[:, None, None, None] * lagged
        if bool(torch.any(amplitude_weights != 0)):
            amplitude = torch.einsum(
                "ij,kcaj->kcai",
                self.amplitude_normal_matrix,
                coefficients,
            )
            result = (
                result
                + amplitude_weights[:, None, None, None] * amplitude
            )
        return result.reshape(theta.shape)

    def diagonal(self, weights: PenaltyWeights) -> torch.Tensor:
        lag_diagonal = torch.diagonal(
            self.lag_factor.T @ self.lag_factor
        )[:, None]
        amplitude_diagonal = torch.diagonal(
            self.amplitude_factor.T @ self.amplitude_factor
        )[None, :]
        block = (
            weights.lag * lag_diagonal
            + weights.amplitude * amplitude_diagonal
            + weights.ridge
        )
        return block.expand(self.channels, -1, -1).reshape(-1)

    def diagonal_batch(
        self,
        weights: list[PenaltyWeights] | tuple[PenaltyWeights, ...],
    ) -> torch.Tensor:
        lag_diagonal = torch.diagonal(self.lag_normal_matrix)[None, :, None]
        amplitude_diagonal = torch.diagonal(
            self.amplitude_normal_matrix
        )[None, None, :]
        lag_weights = torch.as_tensor(
            [row.lag for row in weights],
            device=self.device,
            dtype=self.dtype,
        )
        amplitude_weights = torch.as_tensor(
            [row.amplitude for row in weights],
            device=self.device,
            dtype=self.dtype,
        )
        ridge_weights = torch.as_tensor(
            [row.ridge for row in weights],
            device=self.device,
            dtype=self.dtype,
        )
        block = (
            lag_weights[:, None, None] * lag_diagonal
            + amplitude_weights[:, None, None] * amplitude_diagonal
            + ridge_weights[:, None, None]
        )
        return (
            block[:, None, ...]
            .expand(-1, self.channels, -1, -1)
            .reshape(len(weights), -1)
        )
