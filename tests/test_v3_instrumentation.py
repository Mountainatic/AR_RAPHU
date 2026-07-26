from __future__ import annotations

import ast
from pathlib import Path
import runpy

import torch

from ar_raphu.diagnostics.config import load_diagnostic_config
from ar_raphu.diagnostics.instrumentation import (
    detect_starvation,
    gradient_norm,
    kernel_statistics,
    parameter_norm,
    proximal_collapse,
)


def test_frozen_v3_config_loads_without_runtime_overrides() -> None:
    config = load_diagnostic_config()
    assert config["status"] == "DIAGNOSTIC_ONLY"
    assert config["common"]["seeds"] == [0, 1, 2, 3, 4]
    assert config["runtime"]["gpu_workers"] == 8


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


def test_v3_runner_sources_are_syntactically_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "tools/run_v3_diagnostic_job.py",
        "tools/run_v3_diagnostic_suite.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        ast.parse(source, filename=relative)
        runpy.run_path(str(root / relative), run_name=f"test_{Path(relative).stem}")
