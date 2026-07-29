"""Demmler-Reinsch coordinates for separable spline penalties."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DRBasis:
    vectors: torch.Tensor
    eigenvalues: torch.Tensor
    mass: torch.Tensor

    def to_spectral(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.vectors.T @ self.mass @ coefficients

    def from_spectral(self, spectral: torch.Tensor) -> torch.Tensor:
        return self.vectors @ spectral


def generalized_dr_basis(
    penalty: torch.Tensor,
    mass: torch.Tensor,
    *,
    relative_floor: float = 1.0e-12,
) -> DRBasis:
    p = 0.5 * (penalty + penalty.T)
    m = 0.5 * (mass + mass.T)
    values, vectors = torch.linalg.eigh(m)
    scale = values.abs().max().clamp_min(1.0)
    if bool(torch.any(values <= relative_floor * scale)):
        m = m + relative_floor * scale * torch.eye(
            len(m), device=m.device, dtype=m.dtype
        )
    factor = torch.linalg.cholesky(m)
    inverse_factor = torch.linalg.solve_triangular(
        factor,
        torch.eye(len(m), device=m.device, dtype=m.dtype),
        upper=False,
    )
    whitened = inverse_factor @ p @ inverse_factor.T
    eigenvalues, eigenvectors = torch.linalg.eigh(
        0.5 * (whitened + whitened.T)
    )
    dr_vectors = inverse_factor.T @ eigenvectors
    return DRBasis(
        vectors=dr_vectors,
        eigenvalues=eigenvalues.clamp_min(0.0),
        mass=m,
    )


def tensor_to_dr(
    coefficients: torch.Tensor, lag: DRBasis, amplitude: DRBasis
) -> torch.Tensor:
    return lag.vectors.T @ lag.mass @ coefficients @ amplitude.mass @ amplitude.vectors


def tensor_from_dr(
    spectral: torch.Tensor, lag: DRBasis, amplitude: DRBasis
) -> torch.Tensor:
    return lag.vectors @ spectral @ amplitude.vectors.T

