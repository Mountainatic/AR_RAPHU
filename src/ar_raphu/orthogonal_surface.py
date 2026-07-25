"""Convex A-anchored orthogonal residual surfaces for v2 M8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from STAGE1_DUAL_SOLVER_V20_bundle.stage1.variational_spline import (
    bspline_basis_1d,
    make_spline_grid,
)

from .rank_audit import orthogonalize_lag_basis


def second_difference_roughness(
    size: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if size < 3:
        raise ValueError("A second-difference penalty requires size >= 3.")
    difference = torch.zeros(size - 2, size, device=device, dtype=dtype)
    indices = torch.arange(size - 2, device=device)
    difference[indices, indices] = 1.0
    difference[indices, indices + 1] = -2.0
    difference[indices, indices + 2] = 1.0
    return difference.T @ difference


def orthogonal_lag_bases(
    anchor_kernels: torch.Tensor,
    grid_size: int,
    *,
    spline_order: int = 3,
) -> torch.Tensor:
    """Return q-orthogonal lag bases shaped ``[N,L,M_tau]``."""

    if anchor_kernels.ndim != 2:
        raise ValueError("anchor_kernels must be [variables, lags].")
    kernels = anchor_kernels.detach().to(torch.float64)
    lag_count = kernels.shape[1]
    grid = make_spline_grid(
        0.0,
        float(lag_count - 1),
        int(grid_size),
        int(spline_order),
        device=kernels.device,
        dtype=kernels.dtype,
    )
    raw = bspline_basis_1d(
        torch.arange(lag_count, device=kernels.device, dtype=kernels.dtype),
        grid,
        spline_order,
    )
    parts = [
        torch.as_tensor(
            orthogonalize_lag_basis(
                raw.cpu().numpy(), kernels[index].cpu().numpy()
            ),
            device=kernels.device,
            dtype=kernels.dtype,
        )
        for index in range(kernels.shape[0])
    ]
    return torch.stack(parts)


def surface_design(
    amplitude_basis: torch.Tensor, lag_basis: torch.Tensor
) -> torch.Tensor:
    """Contract window bases into ``[observations, variables*K_tau*M_x]``."""

    if amplitude_basis.ndim != 4 or lag_basis.ndim != 3:
        raise ValueError("Expected amplitude [B,N,L,Mx] and lag [N,L,Mt].")
    if amplitude_basis.shape[1:3] != lag_basis.shape[:2]:
        raise ValueError("Variable/lag dimensions do not align.")
    features = torch.einsum("bnlm,nlk->bnkm", amplitude_basis, lag_basis)
    return features.reshape(features.shape[0], -1)


def surface_penalty(
    variable_count: int,
    lag_basis_count: int,
    amplitude_basis_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    lag = second_difference_roughness(
        lag_basis_count, device=device, dtype=dtype
    )
    amplitude = second_difference_roughness(
        amplitude_basis_count, device=device, dtype=dtype
    )
    block = torch.kron(
        lag, torch.eye(amplitude_basis_count, device=device, dtype=dtype)
    ) + torch.kron(
        torch.eye(lag_basis_count, device=device, dtype=dtype), amplitude
    )
    return torch.block_diag(*([block] * int(variable_count)))


@dataclass(frozen=True, slots=True)
class SurfaceRidgeResult:
    coefficients: torch.Tensor
    kkt_residual: float
    numerical_jitter: float
    objective: float


def solve_surface_ridge(
    design: torch.Tensor,
    target_residual: torch.Tensor,
    penalty: torch.Tensor,
    *,
    smoothness: float,
    relative_jitter: float = 1.0e-10,
) -> SurfaceRidgeResult:
    """Solve the fixed M8 residual surface in FP64 by normal equations."""

    x = design.to(torch.float64)
    y = target_residual.reshape(-1).to(x)
    roughness = penalty.to(x)
    if x.shape[0] != len(y) or roughness.shape != (x.shape[1], x.shape[1]):
        raise ValueError("M8 design, target, and penalty dimensions mismatch.")
    gram = x.T @ x / x.shape[0]
    right = x.T @ y / x.shape[0]
    system = gram + float(smoothness) * roughness
    scale = float(system.diagonal().abs().mean().detach().cpu())
    jitter = float(relative_jitter) * max(scale, 1.0)
    regularized = system + jitter * torch.eye(
        system.shape[0], device=x.device, dtype=x.dtype
    )
    coefficients = torch.linalg.solve(regularized, right)
    residual = x @ coefficients - y
    kkt = system @ coefficients - right
    objective = (
        0.5 * residual.square().mean()
        + 0.5
        * float(smoothness)
        * coefficients.dot(roughness @ coefficients)
    )
    return SurfaceRidgeResult(
        coefficients=coefficients,
        kkt_residual=float(kkt.norm().detach().cpu()),
        numerical_jitter=jitter,
        objective=float(objective.detach().cpu()),
    )
