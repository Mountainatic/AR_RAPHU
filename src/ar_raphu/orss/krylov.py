"""GPU-native LSQR and certified residual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .augmented import AugmentedRegularizedOperator


@dataclass(frozen=True, slots=True)
class KrylovResult:
    coefficients: torch.Tensor
    iterations: int
    converged: bool
    augmented_relative_residual: float
    relative_kkt_residual: float
    operator_forward_calls: int
    operator_adjoint_calls: int


@dataclass(frozen=True, slots=True)
class PCGResult:
    coefficients: torch.Tensor
    iterations: int
    converged: bool
    relative_residual: float


def _safe_norm(vector: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(vector).clamp_min(
        torch.finfo(vector.dtype).tiny
    )


def lsqr(
    operator: AugmentedRegularizedOperator,
    rhs: torch.Tensor,
    *,
    relative_tolerance: float,
    maximum_iterations: int,
    initial: torch.Tensor | None = None,
) -> KrylovResult:
    """Golub-Kahan LSQR implemented with CUDA tensor operations."""

    if initial is None:
        x = torch.zeros(
            operator.dimension, device=operator.device, dtype=operator.dtype
        )
        residual_rhs = rhs
    else:
        x = initial.clone()
        residual_rhs = rhs - operator.forward(x)
    rhs_norm = _safe_norm(rhs)
    u = residual_rhs.clone()
    beta = _safe_norm(u)
    u = u / beta
    v = operator.adjoint(u)
    alpha = _safe_norm(v)
    v = v / alpha
    w = v.clone()
    phi_bar = beta
    rho_bar = alpha
    normal_rhs = operator.adjoint(rhs)
    normal_rhs_norm = _safe_norm(normal_rhs)
    adjoint_calls = 2
    forward_calls = 1 if initial is not None else 0
    converged = False
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        u = operator.forward(v) - alpha * u
        forward_calls += 1
        beta = torch.linalg.vector_norm(u)
        if float(beta.item()) > 0.0:
            u = u / beta
        v = operator.adjoint(u) - beta * v
        adjoint_calls += 1
        alpha = torch.linalg.vector_norm(v)
        if float(alpha.item()) > 0.0:
            v = v / alpha
        rho = torch.hypot(rho_bar, beta)
        cosine = rho_bar / rho
        sine = beta / rho
        theta = sine * alpha
        rho_bar = -cosine * alpha
        phi = cosine * phi_bar
        phi_bar = sine * phi_bar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
        if (
            iteration % 10 == 0
            or float(torch.abs(phi_bar).item() / rhs_norm.item())
            <= relative_tolerance
        ):
            trial_residual = operator.forward(x) - rhs
            forward_calls += 1
            trial_normal = operator.adjoint(trial_residual)
            adjoint_calls += 1
            if (
                float(
                    torch.linalg.vector_norm(trial_normal).item()
                    / normal_rhs_norm.item()
                )
                <= relative_tolerance
            ):
                converged = True
                break

    augmented_residual = operator.forward(x) - rhs
    forward_calls += 1
    normal_residual = operator.adjoint(augmented_residual)
    adjoint_calls += 1
    relative_kkt = float(
        torch.linalg.vector_norm(normal_residual).item()
        / normal_rhs_norm.item()
    )
    return KrylovResult(
        coefficients=x,
        iterations=iteration,
        converged=converged,
        augmented_relative_residual=float(
            torch.linalg.vector_norm(augmented_residual).item()
            / rhs_norm.item()
        ),
        relative_kkt_residual=relative_kkt,
        operator_forward_calls=forward_calls,
        operator_adjoint_calls=adjoint_calls,
    )


def pcg_normal(
    apply,
    rhs: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    preconditioner=None,
    relative_tolerance: float = 1.0e-10,
    maximum_iterations: int = 500,
) -> PCGResult:
    """Preconditioned CG refinement on an SPD normal operator."""

    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    residual = rhs - apply(x)
    denominator = _safe_norm(rhs)
    relative = float(torch.linalg.vector_norm(residual).item() / denominator.item())
    if relative <= relative_tolerance:
        return PCGResult(x, 0, True, relative)
    z = (
        residual.clone()
        if preconditioner is None
        else preconditioner.solve(residual)
    )
    direction = z.clone()
    rz = torch.dot(residual, z)
    converged = False
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        image = apply(direction)
        curvature = torch.dot(direction, image)
        if float(curvature.item()) <= 0.0:
            break
        alpha = rz / curvature
        x = x + alpha * direction
        residual = residual - alpha * image
        relative = float(
            torch.linalg.vector_norm(residual).item() / denominator.item()
        )
        if relative <= relative_tolerance:
            converged = True
            break
        z = (
            residual.clone()
            if preconditioner is None
            else preconditioner.solve(residual)
        )
        next_rz = torch.dot(residual, z)
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    return PCGResult(x, iteration, converged, relative)
