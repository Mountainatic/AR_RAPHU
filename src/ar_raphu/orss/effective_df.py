"""Eligible-only effective degrees-of-freedom calculations."""

from __future__ import annotations

import torch


def reduced_effective_df(
    reduced_system: torch.Tensor, reduced_data_gram: torch.Tensor
) -> float:
    solved = torch.linalg.solve(reduced_system, reduced_data_gram)
    return float(torch.trace(solved).item())


def hutchinson_effective_df(
    *,
    apply_data_normal,
    solve_full,
    dimension: int,
    probes: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[float, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    estimates = []
    for _ in range(probes):
        vector = torch.randint(
            0,
            2,
            (dimension,),
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).to(dtype)
        vector = 2.0 * vector - 1.0
        image = apply_data_normal(vector)
        solved = solve_full(image)
        estimates.append(torch.dot(vector, solved))
    values = torch.stack(estimates).to(torch.float64)
    return float(values.mean().item()), float(
        values.std(unbiased=True).item() / (len(values) ** 0.5)
    )

