"""Variational distributed-lag spline solver.

This experimental alternative exposes the functional-analysis structure

    M(x) = b + sum_j K_{q_j} N_{f_j}(x_j)

with one explicit B-spline expansion per response.  For fixed lag kernels, the
response/intercept subproblem is convex group lasso and is solved by FISTA.
Gamma delay parameters are then updated in a low-dimensional block, producing
a variable-projection-inspired alternating solver.

It is intentionally separate from the two-layer KAN model: users can compare a
neural optimizer with a transparent variational optimizer without changing the
main checkpoint format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from .delay_prior import DiscreteGammaPrior


def make_spline_grid(grid_min: float, grid_max: float, grid_size: int,
                     spline_order: int, *, device=None, dtype=torch.float32):
    if not grid_max > grid_min:
        raise ValueError("grid_max must exceed grid_min")
    h = (grid_max - grid_min) / grid_size
    return (torch.arange(-spline_order, grid_size + spline_order + 1,
                         device=device, dtype=dtype) * h + grid_min)


def bspline_basis_1d(x: torch.Tensor, grid: torch.Tensor,
                     spline_order: int) -> torch.Tensor:
    """Evaluate the same Cox-de Boor basis convention used by KANLinear."""
    value = x.unsqueeze(-1)
    basis = ((value >= grid[:-1]) & (value < grid[1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        basis = (
            (value - grid[:-(k + 1)]) /
            (grid[k:-1] - grid[:-(k + 1)]) * basis[..., :-1]
        ) + (
            (grid[k + 1:] - value) /
            (grid[k + 1:] - grid[1:-k]) * basis[..., 1:]
        )
    return basis


class SplineBasisBank(nn.Module):
    """Fixed per-variable spline bases with trainable linear coefficients."""
    def __init__(self, input_ranges: Iterable[tuple[float, float]],
                 grid_size: int = 12, spline_order: int = 3):
        super().__init__()
        ranges = [tuple(map(float, pair)) for pair in input_ranges]
        self.num_variables = len(ranges)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_basis = grid_size + spline_order
        grids = torch.stack([
            make_spline_grid(lo, hi, grid_size, spline_order)
            for lo, hi in ranges
        ])
        self.register_buffer("grids", grids)

    def forward(self, x_lag: torch.Tensor) -> torch.Tensor:
        """Return bases ``[B,N,L,M]`` for lag-ordered inputs."""
        if x_lag.ndim != 3 or x_lag.shape[1] != self.num_variables:
            raise ValueError("x_lag must have shape [B,N,L]")
        parts = [
            bspline_basis_1d(x_lag[:, j], self.grids[j], self.spline_order)
            for j in range(self.num_variables)
        ]
        return torch.stack(parts, dim=1)


@dataclass
class FISTAResult:
    coefficients: torch.Tensor
    bias: torch.Tensor
    objective_history: list[float]
    iterations: int
    converged: bool
    support: list[int]


def group_soft_threshold(coefficients: torch.Tensor, threshold: float) -> torch.Tensor:
    norms = coefficients.norm(dim=-1, keepdim=True)
    scale = (1.0 - threshold / norms.clamp_min(1e-12)).clamp_min(0.0)
    return coefficients * scale


def _power_lipschitz(design: torch.Tensor, steps: int = 50) -> torch.Tensor:
    """Estimate lambda_max(X^T X / B) without materializing the Gram matrix."""
    b, d = design.shape
    vector = torch.ones(d, device=design.device, dtype=design.dtype)
    vector = vector / vector.norm()
    for _ in range(steps):
        vector = design.transpose(0, 1).matmul(design.matmul(vector)) / b
        vector = vector / vector.norm().clamp_min(1e-12)
    image = design.transpose(0, 1).matmul(design.matmul(vector)) / b
    return torch.dot(vector, image).clamp_min(1e-8)


def solve_fixed_q_group_lasso(
    basis: torch.Tensor,
    q: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_group: float,
    max_iter: int = 2000,
    tolerance: float = 1e-8,
    initial_coefficients: torch.Tensor | None = None,
) -> FISTAResult:
    """Solve the convex response subproblem for fixed q using FISTA.

    ``basis`` has shape ``[B,N,L,M]`` and q has shape ``[N,L]``.
    """
    if basis.ndim != 4 or q.ndim != 2:
        raise ValueError("basis must be [B,N,L,M] and q [N,L]")
    phi = torch.einsum("bnlm,nl->bnm", basis, q)
    b, n, m = phi.shape
    design = phi.reshape(b, n * m)
    y = target.reshape(b).to(design)
    x_mean = design.mean(0, keepdim=True)
    y_mean = y.mean()
    centered_x = design - x_mean
    centered_y = y - y_mean
    lipschitz = _power_lipschitz(centered_x)
    step = 1.0 / lipschitz
    if initial_coefficients is None:
        coefficients = torch.zeros(n, m, device=design.device, dtype=design.dtype)
    else:
        coefficients = initial_coefficients.detach().clone().to(design)
    momentum_point = coefficients.clone()
    acceleration = 1.0
    history: list[float] = []
    converged = False
    for iteration in range(1, max_iter + 1):
        flat = momentum_point.reshape(-1)
        residual = centered_x.matmul(flat) - centered_y
        gradient = centered_x.transpose(0, 1).matmul(residual).reshape(n, m) / b
        candidate = group_soft_threshold(
            momentum_point - step * gradient,
            float(step) * float(lambda_group),
        )
        next_acceleration = (1.0 + (1.0 + 4.0 * acceleration ** 2) ** 0.5) / 2.0
        next_point = candidate + ((acceleration - 1.0) / next_acceleration) * (
            candidate - coefficients)
        delta = (candidate - coefficients).norm() / coefficients.norm().clamp_min(1.0)
        coefficients = candidate
        momentum_point = next_point
        acceleration = next_acceleration
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iter:
            pred_centered = centered_x.matmul(coefficients.reshape(-1))
            data_term = 0.5 * (pred_centered - centered_y).square().mean()
            penalty = float(lambda_group) * coefficients.norm(dim=-1).sum()
            history.append(float((data_term + penalty).detach().cpu()))
        if float(delta.detach().cpu()) < tolerance:
            converged = True
            break
    bias = y_mean - x_mean.reshape(-1).dot(coefficients.reshape(-1))
    support = torch.where(coefficients.norm(dim=-1) > 1e-8)[0].cpu().tolist()
    return FISTAResult(coefficients, bias, history, iteration, converged, support)


class VariationalDistributedLagSpline(nn.Module):
    """One-layer spline response plus StaticGamma lag kernels."""
    def __init__(self, num_variables: int, max_lag: int,
                 input_ranges: Iterable[tuple[float, float]],
                 grid_size: int = 12, spline_order: int = 3):
        super().__init__()
        self.num_variables = num_variables
        self.max_lag = max_lag
        self.basis_bank = SplineBasisBank(input_ranges, grid_size, spline_order)
        self.delay_prior = DiscreteGammaPrior(num_variables, max_lag)
        self.coefficients = nn.Parameter(torch.zeros(
            num_variables, self.basis_bank.num_basis))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("selection_mask", torch.ones(num_variables, dtype=torch.bool))

    def basis(self, windows: torch.Tensor) -> torch.Tensor:
        return self.basis_bank(torch.flip(windows, dims=(-1,)))

    def forward_from_basis(self, basis: torch.Tensor) -> torch.Tensor:
        q = self.delay_prior()
        response = torch.einsum("bnlm,nm->bnl", basis, self.coefficients)
        variable = (response * q.unsqueeze(0)).sum(-1)
        variable = variable * self.selection_mask.to(variable.dtype)
        return (self.bias + variable.sum(-1)).unsqueeze(-1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.forward_from_basis(self.basis(windows))

    @torch.no_grad()
    def install_fista_solution(self, result: FISTAResult) -> None:
        self.coefficients.copy_(result.coefficients)
        self.bias.copy_(result.bias)
        mask = self.coefficients.norm(dim=-1) > 1e-8
        self.selection_mask.copy_(mask)


@dataclass
class AlternatingVariationalResult:
    model: VariationalDistributedLagSpline
    outer_history: list[dict]
    final_fista: FISTAResult


def fit_variational_gamma_spline(
    model: VariationalDistributedLagSpline,
    windows: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_group: float,
    outer_iterations: int = 12,
    delay_steps: int = 100,
    delay_lr: float = 3e-3,
    fista_iterations: int = 1000,
) -> AlternatingVariationalResult:
    """Alternating convex response solve and low-dimensional Gamma update."""
    basis = model.basis(windows).detach()
    delay_optimizer = torch.optim.Adam(model.delay_prior.parameters(), lr=delay_lr)
    coefficients = None
    history: list[dict] = []
    fista = None
    for outer in range(1, outer_iterations + 1):
        with torch.no_grad():
            q_detached = model.delay_prior().detach()
        fista = solve_fixed_q_group_lasso(
            basis, q_detached, target,
            lambda_group=lambda_group,
            max_iter=fista_iterations,
            initial_coefficients=coefficients,
        )
        coefficients = fista.coefficients.detach()
        model.install_fista_solution(fista)
        # Optimize only 2N Gamma coordinates while holding the convex response
        # solution fixed. This is a block-coordinate variable-projection step.
        for _ in range(delay_steps):
            delay_optimizer.zero_grad(set_to_none=True)
            q = model.delay_prior()
            response = torch.einsum("bnlm,nm->bnl", basis, coefficients)
            prediction = model.bias.detach() + (response * q.unsqueeze(0)).sum((-1, -2))
            loss = (prediction - target.reshape(-1)).square().mean()
            loss.backward()
            delay_optimizer.step()
        with torch.no_grad():
            prediction = model.forward_from_basis(basis).squeeze(-1)
            rmse = (prediction - target.reshape(-1)).square().mean().sqrt()
            q_now = model.delay_prior()
            mean = (q_now * model.delay_prior.taus).sum(-1)
        history.append({
            "outer_iteration": outer,
            "rmse": float(rmse.cpu()),
            "support": list(fista.support),
            "fista_iterations": fista.iterations,
            "fista_converged": fista.converged,
            "delay_means": mean.cpu().tolist(),
        })
    # Finish with a convex response solve at the final delay parameters.
    final_fista = solve_fixed_q_group_lasso(
        basis, model.delay_prior().detach(), target,
        lambda_group=lambda_group,
        max_iter=fista_iterations,
        initial_coefficients=coefficients,
    )
    model.install_fista_solution(final_fista)
    return AlternatingVariationalResult(model, history, final_fista)
