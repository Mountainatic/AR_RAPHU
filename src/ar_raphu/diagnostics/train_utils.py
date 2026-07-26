"""Frozen full-split optimization utilities for v3 diagnostic models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ar_raphu.model import ARRAPHURank1
from ar_raphu.training import seed_everything

from .rank2_model import ARRAPHURank2Diagnostic


@dataclass(slots=True)
class DiagnosticTrainResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_rmse: float
    history: list[dict[str, Any]]


def _forward(model: nn.Module, batch: dict[str, torch.Tensor], *, aux: bool):
    if "x_sequence" in batch or "y_sequence" in batch:
        return model.forward_contiguous(
            batch.get("x_sequence"),
            batch.get("y_sequence"),
            return_aux=aux,
        )
    return model(
        batch.get("x_window"),
        batch.get("y_window"),
        return_aux=aux,
    )


def lag_roughness(model: nn.Module) -> torch.Tensor:
    if isinstance(model, ARRAPHURank2Diagnostic):
        return model.lag_roughness()
    if not isinstance(model, ARRAPHURank1) or model.external_branch is None:
        return next(model.parameters()).new_zeros(())
    logits = model.external_branch.delay_logits
    if logits is None:
        return model.bias.new_zeros(())
    second = logits[:, 2:] - 2.0 * logits[:, 1:-1] + logits[:, :-2]
    mask = (
        model.external_branch.active_mask
        & model.external_branch.selection_mask
    ).to(device=logits.device)
    return second[mask].square().sum()


def _unique_trainable(parameters) -> list[nn.Parameter]:
    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    return unique


def optimizer_groups(
    model: nn.Module,
    *,
    response_lr: float,
    lag_lr: float,
    ar_lr: float,
    joint_lr: float | None,
) -> list[dict[str, Any]]:
    if joint_lr is not None:
        return [{"params": _unique_trainable(model.parameters()), "lr": joint_lr}]

    if isinstance(model, ARRAPHURank2Diagnostic):
        groups = [
            {"params": model.response_parameters(), "lr": response_lr},
            {"params": model.lag_parameters(), "lr": lag_lr},
            {"params": model.bias_and_ar_parameters(), "lr": ar_lr},
        ]
        return [group for group in groups if group["params"]]

    if not isinstance(model, ARRAPHURank1):
        raise TypeError(f"Unsupported diagnostic model {type(model)!r}.")

    response: list[nn.Parameter] = []
    lag: list[nn.Parameter] = []
    ar: list[nn.Parameter] = [model.bias]
    if model.external_branch is not None:
        response.extend(model.external_branch.response_branches.parameters())
        if model.external_branch.delay_logits is not None:
            lag.append(model.external_branch.delay_logits)
        elif model.external_branch.delay_prior is not None:
            response.extend(model.external_branch.delay_prior.parameters())
    if model.ar_branch is not None:
        ar.extend(model.ar_branch.parameters())
    groups = [
        {"params": _unique_trainable(response), "lr": response_lr},
        {"params": _unique_trainable(lag), "lr": lag_lr},
        {"params": _unique_trainable(ar), "lr": ar_lr},
    ]
    return [group for group in groups if group["params"]]


@torch.no_grad()
def evaluate_diagnostic(
    model: nn.Module,
    data,
    partition: str,
    *,
    batch_size: int,
    device: torch.device,
    return_contributions: bool = False,
) -> dict[str, np.ndarray | float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    contributions: list[np.ndarray] = []
    for batch in data.iter_contiguous_batches(
        partition, batch_size=batch_size, device=device
    ):
        output, auxiliary = _forward(model, batch, aux=True)
        predictions.append(output.squeeze(-1).detach().cpu().numpy())
        targets.append(batch["target"].detach().cpu().numpy())
        indices.append(batch["target_index"].detach().cpu().numpy())
        if return_contributions:
            contributions.append(
                auxiliary["component_contribution"].detach().cpu().numpy()
            )
    predicted = np.concatenate(predictions)
    observed = np.concatenate(targets)
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    r2 = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    result: dict[str, np.ndarray | float] = {
        "prediction": predicted,
        "target": observed,
        "target_index": np.concatenate(indices),
        "rmse": rmse,
        "r2": r2,
    }
    if return_contributions:
        result["contributions"] = np.concatenate(contributions)
    return result


def train_diagnostic_model(
    model: nn.Module,
    data,
    *,
    max_epochs: int,
    patience: int,
    response_lr: float,
    lag_lr: float,
    ar_lr: float,
    joint_lr: float | None,
    lag_smoothness: float,
    validation_interval: int,
    batch_size: int,
    device: torch.device,
    scheduler_factor: float = 0.5,
    scheduler_patience_validations: int = 20,
    min_learning_rate: float = 1.0e-5,
) -> DiagnosticTrainResult:
    """Optimize one fixed structure; validation RMSE is the only stop signal."""

    if min(max_epochs, patience, validation_interval, batch_size) <= 0:
        raise ValueError("Epoch, patience, interval and batch size must be positive.")
    groups = optimizer_groups(
        model,
        response_lr=response_lr,
        lag_lr=lag_lr,
        ar_lr=ar_lr,
        joint_lr=joint_lr,
    )
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=scheduler_factor,
        patience=scheduler_patience_validations,
        min_lr=min_learning_rate,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = data.target_count("train")
        squared_error_sum = 0.0
        for batch in data.iter_contiguous_batches(
            "train", batch_size=batch_size, device=device
        ):
            prediction = _forward(model, batch, aux=False).squeeze(-1)
            squared_error = (prediction - batch["target"]).square().sum()
            (squared_error / total).backward()
            squared_error_sum += float(squared_error.detach().cpu())
        roughness = lag_roughness(model)
        if lag_smoothness:
            (lag_smoothness * roughness).backward()
        optimizer.step()

        validate = (
            epoch == 1
            or epoch % validation_interval == 0
            or epoch == max_epochs
        )
        if not validate:
            continue
        validation = evaluate_diagnostic(
            model,
            data,
            "validation",
            batch_size=batch_size,
            device=device,
        )
        validation_rmse = float(validation["rmse"])
        scheduler.step(validation_rmse)
        if validation_rmse < best_validation:
            best_validation = validation_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += validation_interval
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_mse": squared_error_sum / total,
            "validation_rmse": validation_rmse,
            "lag_roughness": float(roughness.detach().cpu()),
        }
        for index, group in enumerate(optimizer.param_groups):
            row[f"learning_rate_{index}"] = float(group["lr"])
        history.append(row)
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Diagnostic training produced no validation checkpoint.")
    model.load_state_dict(best_state)
    return DiagnosticTrainResult(
        best_state=best_state,
        best_epoch=best_epoch,
        best_validation_rmse=best_validation,
        history=history,
    )


__all__ = [
    "DiagnosticTrainResult",
    "evaluate_diagnostic",
    "lag_roughness",
    "optimizer_groups",
    "seed_everything",
    "train_diagnostic_model",
]
