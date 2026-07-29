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


@dataclass(frozen=True, slots=True)
class BatchedPCGResult:
    coefficients: torch.Tensor
    iterations: tuple[int, ...]
    converged: tuple[bool, ...]
    relative_residuals: tuple[float, ...]
    operator_calls: int


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
    residual_recompute_interval: int = 25,
    restart_interval: int = 100,
) -> PCGResult:
    """Preconditioned CG with explicit residual refresh and best-iterate return."""

    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    residual = rhs - apply(x)
    denominator = _safe_norm(rhs)
    relative = float(torch.linalg.vector_norm(residual).item() / denominator.item())
    best_x = x.clone()
    best_relative = relative
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
            residual = rhs - apply(x)
            z = (
                residual.clone()
                if preconditioner is None
                else preconditioner.solve(residual)
            )
            direction = z.clone()
            rz = torch.dot(residual, z)
            continue
        alpha = rz / curvature
        x = x + alpha * direction
        residual = residual - alpha * image
        residual_was_recomputed = (
            iteration % residual_recompute_interval == 0
        )
        if residual_was_recomputed:
            residual = rhs - apply(x)
        relative = float(
            torch.linalg.vector_norm(residual).item() / denominator.item()
        )
        if relative <= relative_tolerance and not residual_was_recomputed:
            residual = rhs - apply(x)
            relative = float(
                torch.linalg.vector_norm(residual).item()
                / denominator.item()
            )
        if relative < best_relative:
            best_relative = relative
            best_x = x.clone()
        if relative <= relative_tolerance:
            converged = True
            break
        z = (
            residual.clone()
            if preconditioner is None
            else preconditioner.solve(residual)
        )
        next_rz = torch.dot(residual, z)
        if iteration % restart_interval == 0:
            direction = z.clone()
        else:
            direction = z + (next_rz / rz) * direction
        rz = next_rz
    final_relative = float(
        torch.linalg.vector_norm(rhs - apply(best_x)).item()
        / denominator.item()
    )
    return PCGResult(
        best_x,
        iteration,
        converged or final_relative <= relative_tolerance,
        final_relative,
    )


def pcg_normal_batch(
    apply,
    rhs: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    preconditioner=None,
    preconditioner_diagonal: torch.Tensor | None = None,
    relative_tolerance: float = 1.0e-10,
    maximum_iterations: int = 500,
    residual_recompute_interval: int = 25,
    restart_interval: int = 100,
) -> BatchedPCGResult:
    """Vectorized independent PCG solves with stable active-set restarts."""

    if rhs.ndim != 2:
        raise ValueError("Batched PCG rhs must have shape (batch, dimension).")
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    if x.shape != rhs.shape:
        raise ValueError("Batched PCG initial shape mismatch.")
    if (
        preconditioner_diagonal is not None
        and preconditioner_diagonal.shape != rhs.shape
    ):
        raise ValueError("Batched PCG preconditioner shape mismatch.")
    if preconditioner is not None and preconditioner_diagonal is not None:
        raise ValueError("Specify one batched PCG preconditioner.")
    if residual_recompute_interval < 1 or restart_interval < 1:
        raise ValueError("PCG refresh and restart intervals must be positive.")

    def precondition(residual: torch.Tensor) -> torch.Tensor:
        if preconditioner is not None:
            return preconditioner.solve(residual)
        if preconditioner_diagonal is None:
            return residual.clone()
        return residual / preconditioner_diagonal.clamp_min(
            torch.finfo(residual.dtype).tiny
        )

    residual = rhs - apply(x)
    operator_calls = 1
    denominator = torch.linalg.vector_norm(rhs, dim=1).clamp_min(
        torch.finfo(rhs.dtype).tiny
    )
    relative = torch.linalg.vector_norm(residual, dim=1) / denominator
    best_relative = relative.clone()
    best_x = x.clone()
    active = relative > relative_tolerance
    iteration_counts = torch.zeros(
        rhs.shape[0], device=rhs.device, dtype=torch.int64
    )
    z = precondition(residual)
    direction = z.clone()
    rz = torch.sum(residual * z, dim=1)
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        if not bool(torch.any(active)):
            break
        image = apply(direction)
        operator_calls += 1
        curvature = torch.sum(direction * image, dim=1)
        scale = (
            torch.linalg.vector_norm(direction, dim=1)
            * torch.linalg.vector_norm(image, dim=1)
        ).clamp_min(torch.finfo(rhs.dtype).tiny)
        positive = curvature > torch.finfo(rhs.dtype).eps * scale
        progressing = active & positive
        alpha = torch.where(
            progressing,
            rz / curvature.clamp_min(torch.finfo(rhs.dtype).tiny),
            torch.zeros_like(rz),
        )
        x = x + alpha[:, None] * direction
        residual = residual - alpha[:, None] * image
        residual_was_recomputed = (
            iteration % residual_recompute_interval == 0
        )
        if residual_was_recomputed:
            residual = rhs - apply(x)
            operator_calls += 1
        relative = torch.linalg.vector_norm(residual, dim=1) / denominator
        apparently_converged = active & (
            relative <= relative_tolerance
        )
        if (
            bool(torch.any(apparently_converged))
            and not residual_was_recomputed
        ):
            residual = rhs - apply(x)
            operator_calls += 1
            relative = (
                torch.linalg.vector_norm(residual, dim=1) / denominator
            )
        improved = active & (relative < best_relative)
        best_x = torch.where(improved[:, None], x, best_x)
        best_relative = torch.where(improved, relative, best_relative)
        finished = active & (relative <= relative_tolerance)
        iteration_counts[
            active & (relative <= relative_tolerance)
        ] = iteration
        next_active = active & (~finished)
        forced_restart = next_active & (~positive)
        if iteration % restart_interval == 0 or bool(torch.any(forced_restart)):
            z = precondition(residual)
            restarted_direction = torch.where(
                next_active[:, None], z, torch.zeros_like(z)
            )
            restarted_rz = torch.sum(residual * z, dim=1)
            if iteration % restart_interval == 0:
                direction = restarted_direction
                rz = restarted_rz
            else:
                next_rz = restarted_rz
                beta = torch.where(
                    next_active & positive,
                    next_rz
                    / rz.clamp_min(torch.finfo(rhs.dtype).tiny),
                    torch.zeros_like(rz),
                )
                continued = z + beta[:, None] * direction
                direction = torch.where(
                    forced_restart[:, None],
                    restarted_direction,
                    continued,
                )
                direction = torch.where(
                    next_active[:, None],
                    direction,
                    torch.zeros_like(direction),
                )
                rz = next_rz
            active = next_active
            continue
        z = precondition(residual)
        next_rz = torch.sum(residual * z, dim=1)
        beta = torch.where(
            next_active,
            next_rz / rz.clamp_min(torch.finfo(rhs.dtype).tiny),
            torch.zeros_like(rz),
        )
        direction = z + beta[:, None] * direction
        direction = torch.where(
            next_active[:, None], direction, torch.zeros_like(direction)
        )
        rz = next_rz
        active = next_active
    iteration_counts[active] = maximum_iterations
    final_residual = rhs - apply(best_x)
    operator_calls += 1
    best_relative = (
        torch.linalg.vector_norm(final_residual, dim=1) / denominator
    )
    converged = best_relative <= relative_tolerance
    return BatchedPCGResult(
        coefficients=best_x,
        iterations=tuple(
            int(value)
            for value in iteration_counts.detach().cpu().tolist()
        ),
        converged=tuple(
            bool(value) for value in converged.detach().cpu().tolist()
        ),
        relative_residuals=tuple(
            float(value)
            for value in best_relative.detach().cpu().tolist()
        ),
        operator_calls=operator_calls,
    )
