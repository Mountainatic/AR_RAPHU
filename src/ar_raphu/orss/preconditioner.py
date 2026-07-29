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


@dataclass(slots=True)
class SpectralDiagonalPreconditioner:
    channels: int
    m_tau: int
    m_x: int
    lag_vectors: torch.Tensor
    amplitude_vectors: torch.Tensor
    diagonal: torch.Tensor

    def solve(self, vector: torch.Tensor) -> torch.Tensor:
        coefficients = vector.reshape(
            self.channels, self.m_tau, self.m_x
        )
        spectral = torch.einsum(
            "ia,cij,jb->cab",
            self.lag_vectors,
            coefficients,
            self.amplitude_vectors,
        )
        resolved = spectral / self.diagonal
        return torch.einsum(
            "ia,cab,jb->cij",
            self.lag_vectors,
            resolved,
            self.amplitude_vectors,
        ).reshape(-1)


@dataclass(slots=True)
class BatchedSpectralDiagonalPreconditioner:
    channels: int
    m_tau: int
    m_x: int
    lag_vectors: torch.Tensor
    amplitude_vectors: torch.Tensor
    diagonal: torch.Tensor

    def _to_spectral(self, vector: torch.Tensor) -> torch.Tensor:
        coefficients = vector.reshape(
            -1, self.channels, self.m_tau, self.m_x
        )
        return torch.einsum(
            "ia,kcij,jb->kcab",
            self.lag_vectors,
            coefficients,
            self.amplitude_vectors,
        )

    def _from_spectral(self, spectral: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "ia,kcab,jb->kcij",
            self.lag_vectors,
            spectral,
            self.amplitude_vectors,
        ).reshape(spectral.shape[0], -1)

    def solve(self, vector: torch.Tensor) -> torch.Tensor:
        spectral = self._to_spectral(vector)
        resolved = spectral / self.diagonal
        return self._from_spectral(resolved)

    def coefficients_to_scaled(self, vector: torch.Tensor) -> torch.Tensor:
        return (
            self._to_spectral(vector) * torch.sqrt(self.diagonal)
        ).reshape(vector.shape)

    def scaled_to_coefficients(self, vector: torch.Tensor) -> torch.Tensor:
        spectral = vector.reshape(self.diagonal.shape)
        return self._from_spectral(
            spectral / torch.sqrt(self.diagonal)
        )

    def gradient_to_scaled(self, vector: torch.Tensor) -> torch.Tensor:
        return (
            self._to_spectral(vector) / torch.sqrt(self.diagonal)
        ).reshape(vector.shape)


@dataclass(slots=True)
class ChannelBlockPreconditioner:
    channels: int
    block_dimension: int
    factors: torch.Tensor
    numerical_jitter: tuple[float, ...]

    def solve(self, vector: torch.Tensor) -> torch.Tensor:
        shaped = vector.to(self.factors.dtype).reshape(
            self.channels, self.block_dimension, 1
        )
        return (
            torch.cholesky_solve(shaped, self.factors)
            .reshape(-1)
            .to(vector.dtype)
        )


def spectral_penalty_coordinates(
    penalty: SeparablePenalty,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lag_values, lag_vectors = torch.linalg.eigh(
        0.5 * (penalty.lag_normal_matrix + penalty.lag_normal_matrix.T)
    )
    amplitude_values, amplitude_vectors = torch.linalg.eigh(
        0.5
        * (
            penalty.amplitude_normal_matrix
            + penalty.amplitude_normal_matrix.T
        )
    )
    return (
        lag_values.clamp_min(0.0),
        lag_vectors,
        amplitude_values.clamp_min(0.0),
        amplitude_vectors,
    )


def spectral_data_normal_diagonal(
    data_diagonal: torch.Tensor,
    penalty: SeparablePenalty,
    lag_vectors: torch.Tensor,
    amplitude_vectors: torch.Tensor,
) -> torch.Tensor:
    shaped = data_diagonal.reshape(
        penalty.channels, penalty.m_tau, penalty.m_x
    )
    return torch.einsum(
        "ia,cij,jb->cab",
        lag_vectors.square(),
        shaped,
        amplitude_vectors.square(),
    )


def build_spectral_diagonal_preconditioner(
    operator: UrysohnLinearOperator,
    penalty: SeparablePenalty,
    weights: PenaltyWeights,
    *,
    data_diagonal: torch.Tensor | None = None,
) -> SpectralDiagonalPreconditioner:
    if data_diagonal is None:
        data_diagonal = data_normal_diagonal(operator)
    lag_values, lag_vectors, amplitude_values, amplitude_vectors = (
        spectral_penalty_coordinates(penalty)
    )
    spectral_data = spectral_data_normal_diagonal(
        data_diagonal,
        penalty,
        lag_vectors,
        amplitude_vectors,
    )
    diagonal = (
        spectral_data
        + weights.lag * lag_values[None, :, None]
        + weights.amplitude * amplitude_values[None, None, :]
        + weights.ridge
    )
    floor = spectral_data.clamp_min(
        torch.finfo(spectral_data.dtype).tiny
    ).amin() * 1.0e-6
    return SpectralDiagonalPreconditioner(
        channels=penalty.channels,
        m_tau=penalty.m_tau,
        m_x=penalty.m_x,
        lag_vectors=lag_vectors,
        amplitude_vectors=amplitude_vectors,
        diagonal=diagonal.clamp_min(floor),
    )


def build_batched_spectral_diagonal_preconditioner(
    operator: UrysohnLinearOperator,
    penalty: SeparablePenalty,
    weights: list[PenaltyWeights],
    *,
    data_diagonal: torch.Tensor | None = None,
) -> BatchedSpectralDiagonalPreconditioner:
    if data_diagonal is None:
        data_diagonal = data_normal_diagonal(operator)
    lag_values, lag_vectors, amplitude_values, amplitude_vectors = (
        spectral_penalty_coordinates(penalty)
    )
    spectral_data = spectral_data_normal_diagonal(
        data_diagonal,
        penalty,
        lag_vectors,
        amplitude_vectors,
    )
    lag_weights = torch.as_tensor(
        [row.lag for row in weights],
        device=operator.device,
        dtype=operator.dtype,
    )
    amplitude_weights = torch.as_tensor(
        [row.amplitude for row in weights],
        device=operator.device,
        dtype=operator.dtype,
    )
    ridge_weights = torch.as_tensor(
        [row.ridge for row in weights],
        device=operator.device,
        dtype=operator.dtype,
    )
    diagonal = (
        spectral_data[None, ...]
        + lag_weights[:, None, None, None]
        * lag_values[None, None, :, None]
        + amplitude_weights[:, None, None, None]
        * amplitude_values[None, None, None, :]
        + ridge_weights[:, None, None, None]
    )
    floor = (
        spectral_data.clamp_min(
            torch.finfo(spectral_data.dtype).tiny
        ).amin()
        * 1.0e-6
    )
    return BatchedSpectralDiagonalPreconditioner(
        channels=penalty.channels,
        m_tau=penalty.m_tau,
        m_x=penalty.m_x,
        lag_vectors=lag_vectors,
        amplitude_vectors=amplitude_vectors,
        diagonal=diagonal.clamp_min(floor),
    )


def data_normal_channel_blocks(
    operator: UrysohnLinearOperator,
) -> torch.Tensor:
    block_dimension = operator.m_tau * operator.m_x
    blocks = torch.zeros(
        (operator.channels, block_dimension, block_dimension),
        device=operator.device,
        dtype=operator.dtype,
    )
    for channel, branch in enumerate(operator.branches):
        mean = operator.feature_mean[channel].reshape(1, block_dimension)
        for start in range(0, operator.observations, operator.chunk_time):
            stop = min(start + operator.chunk_time, operator.observations)
            features = torch.einsum(
                "la,nlb->nab",
                branch.lag_basis,
                branch.amplitude[start:stop],
            ).reshape(stop - start, block_dimension)
            features = features - mean
            blocks[channel].addmm_(
                features.T,
                features,
                beta=1.0,
                alpha=1.0 / operator.observations,
            )
    return 0.5 * (blocks + blocks.transpose(1, 2))


def build_channel_block_preconditioner(
    operator: UrysohnLinearOperator,
    penalty: SeparablePenalty,
    weights: PenaltyWeights,
    *,
    data_blocks: torch.Tensor | None = None,
) -> ChannelBlockPreconditioner:
    if data_blocks is None:
        data_blocks = data_normal_channel_blocks(operator)
    factor_dtype = torch.float64
    data_blocks = data_blocks.to(factor_dtype)
    identity_tau = torch.eye(
        penalty.m_tau, device=operator.device, dtype=factor_dtype
    )
    identity_x = torch.eye(
        penalty.m_x, device=operator.device, dtype=factor_dtype
    )
    lag_normal = penalty.lag_normal_matrix.to(factor_dtype)
    amplitude_normal = penalty.amplitude_normal_matrix.to(factor_dtype)
    penalty_block = (
        weights.lag * torch.kron(lag_normal, identity_x)
        + weights.amplitude
        * torch.kron(identity_tau, amplitude_normal)
        + weights.ridge
        * torch.eye(
            penalty.m_tau * penalty.m_x,
            device=operator.device,
            dtype=factor_dtype,
        )
    )
    systems = data_blocks + penalty_block[None, ...]
    factors, info = torch.linalg.cholesky_ex(systems)
    jitters = torch.zeros(
        operator.channels, device=operator.device, dtype=factor_dtype
    )
    if bool(torch.any(info != 0)):
        identity = torch.eye(
            systems.shape[-1], device=operator.device, dtype=factor_dtype
        )
        scale = torch.diagonal(systems, dim1=1, dim2=2).abs().amax(dim=1)
        unresolved = info != 0
        for multiplier in (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6):
            trial_jitter = scale * multiplier
            trial, trial_info = torch.linalg.cholesky_ex(
                systems + trial_jitter[:, None, None] * identity
            )
            accepted = unresolved & (trial_info == 0)
            factors[accepted] = trial[accepted]
            jitters[accepted] = trial_jitter[accepted]
            unresolved = unresolved & (~accepted)
            if not bool(torch.any(unresolved)):
                break
        if bool(torch.any(unresolved)):
            raise RuntimeError("CHANNEL_BLOCK_PRECONDITIONER_CHOLESKY_FAILED")
    return ChannelBlockPreconditioner(
        channels=operator.channels,
        block_dimension=penalty.m_tau * penalty.m_x,
        factors=factors,
        numerical_jitter=tuple(
            float(value) for value in jitters.detach().cpu().tolist()
        ),
    )


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
