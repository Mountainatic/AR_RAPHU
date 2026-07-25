"""Evidence extraction helpers for fixed Scheme-A checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .model import ARRAPHURank1
from .sequence_data import PreparedDirectForecastData


@torch.no_grad()
def static_lag_kernels(
    model: ARRAPHURank1,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    external = (
        model.external_branch._static_q().detach().cpu().numpy().copy()
        if model.external_branch is not None
        else None
    )
    ar = (
        model.ar_branch._static_q().detach().cpu().numpy().copy()
        if model.ar_branch is not None
        else None
    )
    return external, ar


@torch.no_grad()
def response_grid_arrays(
    model: ARRAPHURank1,
    x_ranges: list[tuple[float, float]],
    y_range: tuple[float, float],
    *,
    points: int = 401,
) -> dict[str, np.ndarray]:
    if points < 2:
        raise ValueError("At least two response grid points are required.")
    arrays: dict[str, np.ndarray] = {}
    device = model.bias.device
    dtype = model.bias.dtype
    if model.external_branch is not None:
        for variable, (lower, upper) in enumerate(x_ranges):
            grid = torch.linspace(lower, upper, points, device=device, dtype=dtype)
            response = model.external_branch.response_branches.branches[variable](
                grid[:, None]
            ).reshape(-1)
            arrays[f"x_grid_v{variable}"] = grid.cpu().numpy()
            arrays[f"x_response_v{variable}"] = response.cpu().numpy()
    if model.ar_branch is not None:
        grid = torch.linspace(
            y_range[0], y_range[1], points, device=device, dtype=dtype
        )
        response = model.ar_branch.response_branches.branches[0](
            grid[:, None]
        ).reshape(-1)
        arrays["y_grid"] = grid.cpu().numpy()
        arrays["y_response"] = response.cpu().numpy()
    return arrays


def support_frequency(
    supports: list[list[int]], *, variables: int
) -> list[dict[str, float | int]]:
    if not supports:
        raise ValueError("At least one support is required.")
    return [
        {
            "variable": variable,
            "selection_count": sum(variable in support for support in supports),
            "selection_frequency": float(
                np.mean([variable in support for support in supports])
            ),
        }
        for variable in range(variables)
    ]


@torch.no_grad()
def partition_predictions_and_contributions(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    partition: str,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a fixed checkpoint once and retain component evidence."""

    model.eval()
    predicted: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    components: list[np.ndarray] = []
    for batch in data.iter_contiguous_batches(
        partition, batch_size=batch_size, device=device
    ):
        output, auxiliary = model.forward_contiguous(
            batch.get("x_sequence"),
            batch.get("y_sequence"),
            return_aux=True,
        )
        predicted.append(output.squeeze(-1).cpu().numpy())
        observed.append(batch["target"].cpu().numpy())
        indices.append(batch["target_index"].cpu().numpy())
        components.append(
            auxiliary["component_contribution"].cpu().numpy()
        )
    return (
        np.concatenate(predicted),
        np.concatenate(observed),
        np.concatenate(indices),
        np.concatenate(components),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_checksums(root: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    excluded = set(exclude)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded
    }
