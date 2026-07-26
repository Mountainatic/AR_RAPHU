from __future__ import annotations

import torch

from ar_raphu.diagnostics.instrumentation import (
    detect_starvation,
    gradient_norm,
    kernel_statistics,
    parameter_norm,
    proximal_collapse,
)


def test_gradient_and_kernel_instrumentation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    (parameter.square().sum()).backward()
    assert parameter_norm([parameter]) == 5.0
    assert gradient_norm([parameter]) == 10.0
    statistics = kernel_statistics(torch.tensor([0.25, 0.50, 0.25]))
    assert statistics["q_mean"] == 1.0
    assert statistics["q_boundary_mass_last3"] == 1.0
    assert statistics["q_std"] > 0


def test_starvation_requires_five_consecutive_joint_events() -> None:
    rows = []
    for epoch in range(10, 70, 10):
        row = {"epoch": epoch, "phase": "pruning", "support": [0, 1, 2]}
        for variable in (0, 1, 2):
            row[f"x_{variable}_gradient_ratio"] = (
                0.05 if variable in (0, 1) else 0.5
            )
            row[f"x_{variable}_contribution_signal_ratio"] = (
                0.02 if variable in (0, 1) else 0.2
            )
            row[f"x_{variable}_shrink"] = (
                0.98 if variable in (0, 1) else 1.0
            )
        rows.append(row)
    assert detect_starvation(rows) == [0, 1]

    warmup = [
        {
            "phase": "warmup",
            **{
                f"x_{variable}_contribution_signal_ratio": 0.1
                for variable in (0, 1, 2)
            },
        }
    ]
    pruning = [
        {
            "phase": "pruning",
            "support": [0, 2],
            **{
                f"x_{variable}_contribution_signal_ratio": 0.01
                for variable in (0, 1, 2)
            },
        }
    ]
    assert proximal_collapse(warmup + pruning)
