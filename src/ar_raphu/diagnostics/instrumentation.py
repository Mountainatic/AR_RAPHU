"""Gradient, contribution and lag instrumentation for frozen D6 timelines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum().cpu())
    return float(np.sqrt(total))


def parameter_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        total += float(parameter.detach().square().sum().cpu())
    return float(np.sqrt(total))


def kernel_statistics(q: torch.Tensor) -> dict[str, float]:
    values = q.detach().to(dtype=torch.float64)
    lags = torch.arange(
        values.numel(), device=values.device, dtype=values.dtype
    )
    mean = (lags * values).sum()
    variance = ((lags - mean).square() * values).sum()
    positive = values.clamp_min(1.0e-30)
    return {
        "q_mean": float(mean.cpu()),
        "q_std": float(torch.sqrt(variance).cpu()),
        "q_entropy": float((-(values * positive.log()).sum()).cpu()),
        "q_boundary_mass_last3": float(values[-3:].sum().cpu()),
    }


def detect_starvation(
    rows: list[dict[str, Any]],
    *,
    active_support: tuple[int, ...] = (0, 1, 2),
    gradient_ratio_threshold: float = 0.10,
    contribution_signal_threshold: float = 0.05,
    consecutive_logs: int = 5,
    shrink_threshold: float = 0.99,
) -> list[int]:
    """Return true variables satisfying the preregistered consecutive event."""

    starved: list[int] = []
    for variable in active_support:
        consecutive = 0
        confirmed = False
        for row in rows:
            low_gradient = (
                float(row[f"x_{variable}_gradient_ratio"])
                < gradient_ratio_threshold
            )
            low_signal = (
                float(row[f"x_{variable}_contribution_signal_ratio"])
                < contribution_signal_threshold
            )
            proximal = (
                float(row.get(f"x_{variable}_shrink", 1.0))
                < shrink_threshold
                or variable not in set(row.get("support", range(10)))
            )
            if low_gradient and low_signal and proximal:
                consecutive += 1
                if consecutive >= consecutive_logs:
                    confirmed = True
                    break
            else:
                consecutive = 0
        if confirmed:
            starved.append(variable)
    return starved


def proximal_collapse(
    rows: list[dict[str, Any]],
    *,
    active_support: tuple[int, ...] = (0, 1, 2),
    formed_signal_threshold: float = 0.05,
) -> bool:
    warmup = [row for row in rows if row.get("phase") == "warmup"]
    pruning = [row for row in rows if row.get("phase") == "pruning"]
    if not warmup or not pruning:
        return False
    formed = {
        variable
        for variable in active_support
        if max(
            float(row[f"x_{variable}_contribution_signal_ratio"])
            for row in warmup
        )
        >= formed_signal_threshold
    }
    terminal_support = set(pruning[-1].get("support", range(10)))
    return bool(formed - terminal_support)
