"""Deterministic full-split training for the dual-branch V20 adapter."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch

from STAGE1_DUAL_SOLVER_V20_bundle.stage1.proximal import (
    apply_group_proximal_step,
)

from .model import ARRAPHURank1
from .sequence_data import PreparedDirectForecastData


@dataclass(slots=True)
class TrainingStageResult:
    best_state: dict[str, torch.Tensor]
    terminal_state: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_rmse: float
    history: list[dict]
    terminal_support: list[int]
    stable: bool = False
    support_stable_epoch: int | None = None


def free_lag_logit_roughness(model: ARRAPHURank1) -> torch.Tensor:
    """Second-difference penalty for M6 free external lag logits."""

    if model.external_branch is None:
        return model.bias.new_zeros(())
    logits = model.external_branch.delay_logits
    if logits is None:
        return model.bias.new_zeros(())
    second_difference = logits[:, 2:] - 2.0 * logits[:, 1:-1] + logits[:, :-2]
    mask = (
        model.external_branch.active_mask
        & model.external_branch.selection_mask
    ).to(device=logits.device, dtype=logits.dtype)
    return (second_difference.square().sum(dim=1) * mask).sum()


def seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def _forward_batch(
    model: ARRAPHURank1, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    if "x_sequence" in batch or "y_sequence" in batch:
        return model.forward_contiguous(
            batch.get("x_sequence"),
            batch.get("y_sequence"),
            return_aux=False,
        ).squeeze(-1)
    return model(
        batch.get("x_window"),
        batch.get("y_window"),
        return_aux=False,
    ).squeeze(-1)


def _train_full_split_step(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = data.target_count("train")
    squared_error_sum = 0.0
    for batch in data.iter_contiguous_batches(
        "train", batch_size=batch_size, device=device
    ):
        residual = _forward_batch(model, batch) - batch["target"]
        squared_error = residual.square().sum()
        (squared_error / total).backward()
        squared_error_sum += float(squared_error.detach().cpu())
    optimizer.step()
    return squared_error_sum / total


@torch.no_grad()
def evaluate_rmse(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    partition: str,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    squared_error_sum = 0.0
    count = data.target_count(partition)
    for batch in data.iter_contiguous_batches(
        partition, batch_size=batch_size, device=device
    ):
        residual = _forward_batch(model, batch) - batch["target"]
        squared_error_sum += float(residual.square().sum().cpu())
    return float(np.sqrt(squared_error_sum / count))


def external_support(
    model: ARRAPHURank1, *, threshold: float = 1.0e-8
) -> list[int]:
    if model.external_branch is None:
        return []
    norms = model.external_branch.response_branches.compute_branch_norms()
    mask = (
        (norms > threshold)
        & model.external_branch.active_mask
        & model.external_branch.selection_mask
    )
    return torch.where(mask.detach().cpu())[0].tolist()


def train_dense_warmup(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    *,
    epochs: int,
    learning_rate: float,
    patience: int,
    batch_size: int,
    device: torch.device,
    validation_interval: int = 5,
) -> TrainingStageResult:
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=80, min_lr=1.0e-5
    )
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    last_validation = float("nan")
    for epoch in range(1, epochs + 1):
        train_mse = _train_full_split_step(
            model,
            data,
            optimizer,
            batch_size=batch_size,
            device=device,
        )
        validate = (
            epoch == 1
            or epoch % validation_interval == 0
            or epoch == epochs
        )
        if validate:
            last_validation = evaluate_rmse(
                model,
                data,
                "validation",
                batch_size=batch_size,
                device=device,
            )
            scheduler.step(last_validation)
            if last_validation < best_validation:
                best_validation = last_validation
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += validation_interval
        if validate:
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "validation_rmse": last_validation,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "external_support": external_support(model),
                }
            )
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("Dense warmup produced no validation checkpoint.")
    terminal_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return TrainingStageResult(
        best_state=best_state,
        terminal_state=terminal_state,
        best_epoch=best_epoch,
        best_validation_rmse=best_validation,
        history=history,
        terminal_support=external_support(model),
    )


def prune_external_path(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    warmup_state: dict[str, torch.Tensor],
    *,
    requested_scale: float,
    median_warmup_branch_norm: float,
    epochs: int,
    learning_rate: float,
    ramp_epochs: int,
    full_penalty_min_epochs: int,
    stable_epochs: int,
    batch_size: int,
    device: torch.device,
    validation_interval: int = 5,
) -> TrainingStageResult:
    if model.external_branch is None:
        raise ValueError("External pruning requires an X-containing track.")
    model.load_state_dict(copy.deepcopy(warmup_state))
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    lambda_group = (
        requested_scale * median_warmup_branch_norm / learning_rate
    )
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    history: list[dict] = []
    last_support: list[int] | None = None
    unchanged = 0
    stable_epoch = None
    earliest_stop = ramp_epochs + full_penalty_min_epochs + 1
    for epoch in range(1, epochs + 1):
        train_mse = _train_full_split_step(
            model,
            data,
            optimizer,
            batch_size=batch_size,
            device=device,
        )
        penalty_scale = min(1.0, epoch / max(1, ramp_epochs))
        pre_norms, shrink = apply_group_proximal_step(
            model.external_branch.response_branches,
            optimizer.param_groups[0]["lr"],
            lambda_group * penalty_scale,
        )
        norms = (pre_norms * shrink).detach()
        selected = model.external_branch.selection_mask
        active = model.external_branch.active_mask
        support = torch.where(
            ((norms > 1.0e-8) & selected & active).detach().cpu()
        )[0].tolist()
        support_set = set(support)
        for variable in range(model.external_channels):
            if (
                bool(model.external_branch.selection_mask[variable])
                and bool(model.external_branch.active_mask[variable])
                and variable not in support_set
            ):
                model.external_branch.prune_variable(variable, optimizer)

        validate = (
            epoch == 1
            or epoch % validation_interval == 0
            or epoch == epochs
        )
        if validate:
            validation_rmse = evaluate_rmse(
                model,
                data,
                "validation",
                batch_size=batch_size,
                device=device,
            )
            if validation_rmse < best_validation:
                best_validation = validation_rmse
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "validation_rmse": validation_rmse,
                    "penalty_scale": penalty_scale,
                    "external_support": support,
                }
            )

        if epoch >= earliest_stop:
            if support == last_support:
                unchanged += 1
            else:
                last_support = list(support)
                unchanged = 1
            if unchanged >= stable_epochs:
                stable_epoch = epoch
                break
        else:
            last_support = list(support)
            unchanged = 0
    if best_state is None:
        raise RuntimeError("Pruning produced no validation checkpoint.")
    terminal_state = copy.deepcopy(model.state_dict())
    return TrainingStageResult(
        best_state=best_state,
        terminal_state=terminal_state,
        best_epoch=best_epoch,
        best_validation_rmse=best_validation,
        history=history,
        terminal_support=external_support(model),
        stable=stable_epoch is not None,
        support_stable_epoch=stable_epoch,
    )


def refit_fixed_external_support(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    terminal_state: dict[str, torch.Tensor],
    support: list[int],
    *,
    epochs: int,
    learning_rate: float,
    patience: int,
    batch_size: int,
    device: torch.device,
    validation_interval: int = 5,
) -> TrainingStageResult:
    """Refit surviving external branches and the AR branch without penalty."""

    if model.external_branch is None:
        raise ValueError("External refit requires an X-containing track.")
    model.load_state_dict(copy.deepcopy(terminal_state))
    support_set = set(support)
    for variable in range(model.external_channels):
        if (
            bool(model.external_branch.active_mask[variable])
            and variable not in support_set
        ):
            model.external_branch.prune_variable(variable)
    result = train_dense_warmup(
        model,
        data,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        batch_size=batch_size,
        device=device,
        validation_interval=validation_interval,
    )
    result.terminal_support = sorted(support_set)
    return result


def refit_m6_free_lag(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    support: list[int],
    *,
    smoothness_weight: float,
    epochs: int,
    learning_rate: float,
    patience: int,
    batch_size: int,
    device: torch.device,
    validation_interval: int = 5,
) -> TrainingStageResult:
    """Jointly refit a fixed-support M6 model with free external lag logits."""

    if model.external_branch is None:
        raise ValueError("M6 requires an external branch.")
    if model.external_delay_mode != "free_static_logits":
        raise ValueError("M6 requires free_static_logits external delays.")
    if smoothness_weight < 0:
        raise ValueError("M6 smoothness weight must be nonnegative.")
    support_set = set(support)
    if not support_set:
        raise ValueError("M6 requires a non-empty frozen external support.")
    for variable in range(model.external_channels):
        if variable not in support_set:
            model.external_branch.prune_variable(variable)

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=80, min_lr=1.0e-5
    )
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = data.target_count("train")
        squared_error_sum = 0.0
        for batch in data.iter_contiguous_batches(
            "train", batch_size=batch_size, device=device
        ):
            residual = _forward_batch(model, batch) - batch["target"]
            squared_error = residual.square().sum()
            (squared_error / total).backward()
            squared_error_sum += float(squared_error.detach().cpu())
        roughness = free_lag_logit_roughness(model)
        penalty = smoothness_weight * roughness
        penalty.backward()
        optimizer.step()

        validate = (
            epoch == 1
            or epoch % validation_interval == 0
            or epoch == epochs
        )
        if validate:
            validation_rmse = evaluate_rmse(
                model,
                data,
                "validation",
                batch_size=batch_size,
                device=device,
            )
            scheduler.step(validation_rmse)
            if validation_rmse < best_validation:
                best_validation = validation_rmse
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += validation_interval
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": squared_error_sum / total,
                    "lag_logit_roughness": float(roughness.detach().cpu()),
                    "smoothness_penalty": float(penalty.detach().cpu()),
                    "validation_rmse": validation_rmse,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "external_support": sorted(support_set),
                }
            )
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("M6 refit produced no validation checkpoint.")
    terminal_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return TrainingStageResult(
        best_state=best_state,
        terminal_state=terminal_state,
        best_epoch=best_epoch,
        best_validation_rmse=best_validation,
        history=history,
        terminal_support=sorted(support_set),
    )
