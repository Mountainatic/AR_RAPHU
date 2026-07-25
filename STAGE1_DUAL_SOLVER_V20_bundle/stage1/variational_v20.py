"""Formal variational distributed-lag spline solver used by V20.

The model is

    y_t = b + sum_j K_{q_j(theta_j)} N_{f_j}(x_j)(t),
    f_j(x) = sum_m c[j,m] B[j,m](x).

For fixed Gamma lag kernels q, the response problem is convex.  We solve a
scale-normalized group-lasso plus spline-roughness objective with monotone
FISTA.  Only the 2N bounded Gamma coordinates remain in the outer block.

This module intentionally does not depend on the neural KAN response.  PyTorch
is used as a tensor/autodiff backend; CUDA is optional.  At the current S0 size
CPU is valid, while concurrent configuration jobs can use a GPU efficiently.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Iterable

import torch
import torch.nn as nn

from .delay_prior import DiscreteGammaPrior
from .variational_spline import make_spline_grid, bspline_basis_1d


class SplineBasisBankV20(nn.Module):
    def __init__(self, input_ranges: Iterable[tuple[float, float]], *, grid_size: int = 12,
                 spline_order: int = 3, dtype=torch.float32):
        super().__init__()
        ranges = [tuple(map(float, pair)) for pair in input_ranges]
        self.num_variables = len(ranges)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.num_basis = self.grid_size + self.spline_order
        grids = torch.stack([
            make_spline_grid(lo, hi, self.grid_size, self.spline_order, dtype=dtype)
            for lo, hi in ranges
        ])
        self.register_buffer("grids", grids)
        # Second-difference roughness matrix.  It is PSD by construction.
        d2 = torch.zeros(self.num_basis - 2, self.num_basis, dtype=dtype)
        for i in range(self.num_basis - 2):
            d2[i, i:i + 3] = torch.tensor([1.0, -2.0, 1.0], dtype=dtype)
        self.register_buffer("roughness", d2.transpose(0, 1) @ d2)

    def forward(self, lag_ordered_windows: torch.Tensor) -> torch.Tensor:
        if lag_ordered_windows.ndim != 3:
            raise ValueError("expected [B,N,L]")
        if lag_ordered_windows.shape[1] != self.num_variables:
            raise ValueError("variable dimension mismatch")
        return torch.stack([
            bspline_basis_1d(lag_ordered_windows[:, j], self.grids[j], self.spline_order)
            for j in range(self.num_variables)
        ], dim=1)

    def evaluate_grid(self, variable: int, x: torch.Tensor) -> torch.Tensor:
        return bspline_basis_1d(x, self.grids[int(variable)], self.spline_order)

    def sequence_basis(self, raw_sequence: torch.Tensor) -> torch.Tensor:
        """Evaluate B-splines once per unique raw time point: [N,T,M]."""
        if raw_sequence.ndim != 2 or raw_sequence.shape[0] != self.num_variables:
            raise ValueError("raw_sequence must have shape [N,T]")
        return torch.stack([
            bspline_basis_1d(raw_sequence[j], self.grids[j], self.spline_order)
            for j in range(self.num_variables)
        ], dim=0)

    @staticmethod
    def lag_basis_from_sequence(sequence_basis: torch.Tensor, max_lag: int) -> torch.Tensor:
        """Create a current-to-past [B,N,L,M] view from [N,T,M]."""
        if sequence_basis.ndim != 3:
            raise ValueError("sequence_basis must have shape [N,T,M]")
        if sequence_basis.shape[1] < max_lag:
            raise ValueError("sequence shorter than max_lag")
        # unfold time gives [N,B,M,L]; reverse L to current-to-past.
        view = sequence_basis.unfold(1, max_lag, 1).permute(1, 0, 3, 2)
        return torch.flip(view, dims=(2,))


@dataclass
class FISTAHistoryRow:
    iteration: int
    objective: float
    data_loss: float
    group_penalty: float
    smoothness_penalty: float
    prox_gradient_norm: float
    kkt_residual: float
    relative_step: float
    lipschitz: float
    restarted: bool


@dataclass
class FISTAResultV20:
    coefficients: torch.Tensor
    standardized_coefficients: torch.Tensor
    bias: torch.Tensor
    group_scales: torch.Tensor
    support: list[int]
    history: list[dict]
    iterations: int
    converged: bool
    kkt_residual: float
    objective: float


@dataclass
class VariationalFitResult:
    best_state: dict
    best_outer_iteration: int
    best_val_rmse: float
    train_rmse: float
    outer_history: list[dict]
    fista_history: list[dict]
    converged: bool
    selected_support: list[int]


def group_soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    norms = x.norm(dim=-1, keepdim=True)
    return x * (1.0 - float(threshold) / norms.clamp_min(1e-15)).clamp_min(0.0)


def _power_lipschitz(design: torch.Tensor, roughness_blocks: torch.Tensor,
                      lambda_smooth: float, steps: int = 80) -> torch.Tensor:
    b, d = design.shape
    n, m, _ = roughness_blocks.shape
    v = torch.randn(d, device=design.device, dtype=design.dtype)
    v = v / v.norm().clamp_min(1e-15)
    for _ in range(steps):
        matrix = v.reshape(n, m)
        image = design.T @ (design @ v) / b
        if lambda_smooth:
            image = image + float(lambda_smooth) * torch.einsum(
                "nij,nj->ni", roughness_blocks, matrix).reshape(-1)
        norm = image.norm().clamp_min(1e-15)
        v = image / norm
    matrix = v.reshape(n, m)
    image = design.T @ (design @ v) / b
    if lambda_smooth:
        image = image + float(lambda_smooth) * torch.einsum(
            "nij,nj->ni", roughness_blocks, matrix).reshape(-1)
    return torch.dot(v, image).clamp_min(1e-8)


def _objective(centered_x, centered_y, z, lambda_group, lambda_smooth, roughness_blocks):
    residual = centered_x @ z.reshape(-1) - centered_y
    data = 0.5 * residual.square().mean()
    group = float(lambda_group) * z.norm(dim=-1).sum()
    smooth = 0.5 * float(lambda_smooth) * torch.einsum(
        "ni,nij,nj->", z, roughness_blocks, z)
    return data + group + smooth, data, group, smooth


def _gradient(centered_x, centered_y, z, lambda_smooth, roughness_blocks):
    b = centered_x.shape[0]
    residual = centered_x @ z.reshape(-1) - centered_y
    grad = (centered_x.T @ residual / b).reshape_as(z)
    if lambda_smooth:
        grad = grad + float(lambda_smooth) * torch.einsum("nij,nj->ni", roughness_blocks, z)
    return grad


def solve_fixed_q_fista_v20(
    basis: torch.Tensor,
    q: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_group: float,
    lambda_smooth: float = 0.0,
    roughness_matrix: torch.Tensor | None = None,
    max_iter: int = 4000,
    tolerance: float = 1e-7,
    kkt_tolerance: float = 1e-5,
    initial_standardized_coefficients: torch.Tensor | None = None,
    history_interval: int = 10,
    support_threshold: float = 1e-6,
) -> FISTAResultV20:
    """Solve the fixed-delay convex subproblem with monotone restarted FISTA."""
    if basis.ndim != 4 or q.ndim != 2:
        raise ValueError("basis must be [B,N,L,M], q [N,L]")
    phi = torch.einsum("bnlm,nl->bnm", basis, q)
    b, n, m = phi.shape
    # Normalize each variable block to make the group penalty scale-invariant.
    group_scales = phi.square().mean(dim=(0, 2)).sqrt().clamp_min(1e-6)
    standardized_phi = phi / group_scales.view(1, n, 1)
    design = standardized_phi.reshape(b, n * m)
    y = target.reshape(b).to(design)
    x_mean = design.mean(0, keepdim=True)
    y_mean = y.mean()
    cx, cy = design - x_mean, y - y_mean

    if roughness_matrix is None:
        d2 = torch.zeros(m - 2, m, device=design.device, dtype=design.dtype)
        for i in range(m - 2):
            d2[i, i:i + 3] = torch.tensor([1., -2., 1.], device=design.device, dtype=design.dtype)
        roughness_matrix = d2.T @ d2
    roughness_matrix = roughness_matrix.to(design)
    # z is standardized coefficient; physical c = z / group_scale.
    roughness_blocks = roughness_matrix.unsqueeze(0) / group_scales.square().view(n, 1, 1)
    lipschitz = _power_lipschitz(cx, roughness_blocks, lambda_smooth)
    step = 1.0 / lipschitz

    if initial_standardized_coefficients is None:
        current = torch.zeros(n, m, device=design.device, dtype=design.dtype)
    else:
        current = initial_standardized_coefficients.detach().clone().to(design)
    momentum = current.clone()
    acceleration = 1.0
    history: list[dict] = []
    converged = False
    previous_obj = float("inf")
    final_kkt = float("inf")

    for iteration in range(1, max_iter + 1):
        grad = _gradient(cx, cy, momentum, lambda_smooth, roughness_blocks)
        candidate = group_soft_threshold(momentum - step * grad, float(step) * lambda_group)
        obj, data, group, smooth = _objective(
            cx, cy, candidate, lambda_group, lambda_smooth, roughness_blocks)
        restarted = False
        # Monotone restart prevents accelerated oscillation on ill-conditioned bases.
        if float(obj.detach()) > previous_obj + 1e-12:
            restarted = True
            momentum = current
            acceleration = 1.0
            grad = _gradient(cx, cy, momentum, lambda_smooth, roughness_blocks)
            candidate = group_soft_threshold(momentum - step * grad, float(step) * lambda_group)
            obj, data, group, smooth = _objective(
                cx, cy, candidate, lambda_group, lambda_smooth, roughness_blocks)

        grad_candidate = _gradient(cx, cy, candidate, lambda_smooth, roughness_blocks)
        prox_point = group_soft_threshold(candidate - step * grad_candidate,
                                          float(step) * lambda_group)
        prox_gradient = (candidate - prox_point) / step
        final_kkt = float(prox_gradient.norm().detach().cpu())
        relative_step = float(((candidate - current).norm() /
                               current.norm().clamp_min(1.0)).detach().cpu())

        next_acceleration = (1 + (1 + 4 * acceleration * acceleration) ** 0.5) / 2
        next_momentum = candidate + ((acceleration - 1) / next_acceleration) * (candidate - current)
        current = candidate
        momentum = next_momentum
        acceleration = next_acceleration
        previous_obj = float(obj.detach().cpu())

        if iteration == 1 or iteration % history_interval == 0 or iteration == max_iter:
            history.append(asdict(FISTAHistoryRow(
                iteration=iteration,
                objective=previous_obj,
                data_loss=float(data.detach().cpu()),
                group_penalty=float(group.detach().cpu()),
                smoothness_penalty=float(smooth.detach().cpu()),
                prox_gradient_norm=final_kkt,
                kkt_residual=final_kkt,
                relative_step=relative_step,
                lipschitz=float(lipschitz.detach().cpu()),
                restarted=restarted,
            )))
        if relative_step <= tolerance and final_kkt <= kkt_tolerance:
            converged = True
            break

    physical = current / group_scales.view(n, 1)
    bias = y_mean - x_mean.reshape(-1).dot(current.reshape(-1))
    support = torch.where(current.norm(dim=-1) > support_threshold)[0].detach().cpu().tolist()
    return FISTAResultV20(
        coefficients=physical,
        standardized_coefficients=current,
        bias=bias,
        group_scales=group_scales,
        support=support,
        history=history,
        iterations=iteration,
        converged=converged,
        kkt_residual=final_kkt,
        objective=previous_obj,
    )


class VariationalDistributedLagSplineV20(nn.Module):
    def __init__(self, num_variables: int, max_lag: int,
                 input_ranges: Iterable[tuple[float, float]], *,
                 grid_size: int = 12, spline_order: int = 3,
                 dtype=torch.float32):
        super().__init__()
        self.num_variables = int(num_variables)
        self.max_lag = int(max_lag)
        self.basis_bank = SplineBasisBankV20(
            input_ranges, grid_size=grid_size, spline_order=spline_order, dtype=dtype)
        self.delay_prior = DiscreteGammaPrior(num_variables, max_lag)
        self.register_buffer("coefficients", torch.zeros(
            num_variables, self.basis_bank.num_basis, dtype=dtype))
        self.register_buffer("bias", torch.tensor(0.0, dtype=dtype))
        self.register_buffer("selection_mask", torch.ones(num_variables, dtype=torch.bool))

    def basis(self, windows: torch.Tensor) -> torch.Tensor:
        return self.basis_bank(torch.flip(windows, dims=(-1,)))

    def predict_from_basis(self, basis: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
        q = self.delay_prior() if q is None else q
        response = torch.einsum("bnlm,nm->bnl", basis, self.coefficients)
        variable = (response * q.unsqueeze(0)).sum(-1)
        variable = variable * self.selection_mask.to(variable.dtype)
        return self.bias + variable.sum(-1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.predict_from_basis(self.basis(windows)).unsqueeze(-1)

    @torch.no_grad()
    def install(self, result: FISTAResultV20) -> None:
        self.coefficients.copy_(result.coefficients)
        self.bias.copy_(result.bias)
        mask = result.coefficients.norm(dim=-1) > 1e-6
        self.selection_mask.copy_(mask)

    def function_values(self, variable: int, x: torch.Tensor) -> torch.Tensor:
        basis = self.basis_bank.evaluate_grid(variable, x.to(self.coefficients))
        return basis @ self.coefficients[int(variable)]


def initialize_delay_from_seed(model: VariationalDistributedLagSplineV20, seed: int) -> None:
    """Truth-blind, reproducible spread of initial Gamma moments."""
    generator = torch.Generator(device=model.coefficients.device)
    generator.manual_seed(int(seed) + 1771)
    mean = torch.linspace(3.0, model.max_lag - 4.0, model.num_variables,
                          device=model.coefficients.device)
    mean = mean[torch.randperm(model.num_variables, generator=generator,
                               device=model.coefficients.device)]
    jitter = torch.randn(model.num_variables, generator=generator,
                         device=model.coefficients.device) * 1.5
    std = torch.full((model.num_variables,), 4.0, device=model.coefficients.device)
    std = (std + 0.5 * torch.rand(model.num_variables, generator=generator,
                                  device=model.coefficients.device)).clamp(1.0, 8.0)
    model.delay_prior.initialize_from_mean_std((mean + jitter).clamp(1, model.max_lag - 2), std)


def fit_variational_v20(
    model: VariationalDistributedLagSplineV20,
    basis_train: torch.Tensor,
    y_train: torch.Tensor,
    basis_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    lambda_group: float,
    lambda_smooth: float,
    outer_iterations: int = 20,
    outer_patience: int = 6,
    delay_steps: int = 100,
    delay_lr: float = 3e-3,
    delay_solver: str = "adam",
    delay_lbfgs_max_iter: int = 30,
    fista_max_iter: int = 2500,
    fista_tolerance: float = 1e-7,
    fista_kkt_tolerance: float = 1e-5,
) -> VariationalFitResult:
    """Alternating convex response solve and low-dimensional delay update.

    One initial FISTA solve is followed by exactly one warm-started FISTA solve
    per outer delay block.  V19's prototype solved the same response block twice
    per outer iteration; V20 removes that redundancy.
    """
    if delay_solver not in {"adam", "lbfgs"}:
        raise ValueError("delay_solver must be adam or lbfgs")
    optimizer = (torch.optim.Adam(model.delay_prior.parameters(), lr=delay_lr)
                 if delay_solver == "adam" else None)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(1, outer_patience // 2), min_lr=1e-5)
        if optimizer is not None else None)
    best_state = None
    best_val = float("inf")
    best_outer = 0
    stale = 0
    outer_history: list[dict] = []
    all_fista_history: list[dict] = []

    # Initial convex response solution at the truth-blind delay initialization.
    fista = solve_fixed_q_fista_v20(
        basis_train, model.delay_prior().detach(), y_train,
        lambda_group=lambda_group, lambda_smooth=lambda_smooth,
        roughness_matrix=model.basis_bank.roughness,
        max_iter=fista_max_iter, tolerance=fista_tolerance,
        kkt_tolerance=fista_kkt_tolerance,
    )
    standardized = fista.standardized_coefficients.detach()
    model.install(fista)
    for row in fista.history:
        all_fista_history.append({"outer_iteration": 0, **row})

    for outer in range(1, outer_iterations + 1):
        # Optimize only 2N Gamma coordinates with response fixed.
        if delay_solver == "adam":
            for _ in range(delay_steps):
                optimizer.zero_grad(set_to_none=True)
                pred = model.predict_from_basis(basis_train)
                loss = (pred - y_train.reshape(-1)).square().mean()
                loss.backward()
                optimizer.step()
            delay_lr_now = optimizer.param_groups[0]["lr"]
        else:
            # Recreate LBFGS each outer block because the convex response solve
            # changes the reduced objective between outer iterations.
            lbfgs = torch.optim.LBFGS(
                model.delay_prior.parameters(), lr=1.0,
                max_iter=delay_lbfgs_max_iter, tolerance_grad=1e-8,
                tolerance_change=1e-10, line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                pred = model.predict_from_basis(basis_train)
                loss = (pred - y_train.reshape(-1)).square().mean()
                loss.backward()
                return loss
            lbfgs.step(closure)
            delay_lr_now = 1.0

        # One convex response solve at the updated delay.
        fista = solve_fixed_q_fista_v20(
            basis_train, model.delay_prior().detach(), y_train,
            lambda_group=lambda_group, lambda_smooth=lambda_smooth,
            roughness_matrix=model.basis_bank.roughness,
            max_iter=fista_max_iter, tolerance=fista_tolerance,
            kkt_tolerance=fista_kkt_tolerance,
            initial_standardized_coefficients=standardized,
        )
        standardized = fista.standardized_coefficients.detach()
        model.install(fista)
        for row in fista.history:
            all_fista_history.append({"outer_iteration": outer, **row})

        with torch.no_grad():
            train_rmse = float((model.predict_from_basis(basis_train) - y_train).square().mean().sqrt().cpu())
            val_rmse = float((model.predict_from_basis(basis_val) - y_val).square().mean().sqrt().cpu())
            q = model.delay_prior()
            means = (q * model.delay_prior.taus).sum(-1)
            stds = (q * (model.delay_prior.taus - means[:, None]).square()).sum(-1).sqrt()
        if scheduler is not None:
            scheduler.step(val_rmse)
        row = {
            "outer_iteration": outer,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "support": list(fista.support),
            "active_count": len(fista.support),
            "fista_iterations": fista.iterations,
            "fista_converged": fista.converged,
            "kkt_residual": fista.kkt_residual,
            "delay_solver": delay_solver,
            "delay_lr": delay_lr_now,
            "delay_means": means.cpu().tolist(),
            "delay_stds": stds.cpu().tolist(),
        }
        outer_history.append(row)
        if val_rmse < best_val - 1e-9:
            best_val = val_rmse
            best_outer = outer
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= outer_patience:
            break

    if best_state is None:
        raise RuntimeError("variational fitting produced no state")
    model.load_state_dict(best_state)
    with torch.no_grad():
        train_rmse = float((model.predict_from_basis(basis_train) - y_train).square().mean().sqrt().cpu())
    converged = bool(fista.converged and fista.kkt_residual <= fista_kkt_tolerance)
    return VariationalFitResult(
        best_state=best_state,
        best_outer_iteration=best_outer,
        best_val_rmse=best_val,
        train_rmse=train_rmse,
        outer_history=outer_history,
        fista_history=all_fista_history,
        converged=converged,
        selected_support=torch.where(model.selection_mask)[0].cpu().tolist(),
    )

