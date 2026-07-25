"""Explicit warmup -> pruning -> refit protocols with an optimized fast path.

Backward-compatible tensor inputs remain supported.  Passing a
``ContiguousWindowBatch`` activates the exact sequence-first KAN/convolution
path.  Validation and expensive diagnostics may be sampled less frequently,
which is important for this very small GPU model where Python/CUDA
synchronization can dominate arithmetic.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .proximal import apply_group_proximal_step
from .sequence_ops import ContiguousWindowBatch


def pruning_penalty_scale(epoch: int, ramp_epochs: int) -> float:
    return min(1.0, epoch / max(1, ramp_epochs))


def earliest_stable_stop_epoch(ramp_epochs: int, full_penalty_min_epochs: int,
                               stable_epochs: int) -> int:
    return ramp_epochs + full_penalty_min_epochs + stable_epochs


@dataclass
class StageResult:
    best_state: dict | None
    terminal_state: dict
    best_epoch: int
    best_val_rmse: float
    history: list[dict]
    terminal_support: list[int]
    support_at_best_validation: list[int]
    support_stable_epoch: int | None = None
    stable: bool = False


def _windows(batch):
    return batch.windows if isinstance(batch, ContiguousWindowBatch) else batch


def _target(batch, explicit_target):
    if explicit_target is not None:
        return explicit_target
    if isinstance(batch, ContiguousWindowBatch) and batch.target is not None:
        return batch.target
    raise ValueError("target must be supplied explicitly or cached in the batch")


def _predict(model, batch):
    if isinstance(batch, ContiguousWindowBatch):
        return model.forward_contiguous(batch, return_aux=False)
    return model(batch, return_aux=False)


def _support_from_norms(model, norms: torch.Tensor, threshold: float = 1e-8) -> list[int]:
    selected = getattr(model, "selection_mask", torch.ones_like(norms, dtype=torch.bool))
    mask = (norms > threshold) & selected.to(device=norms.device)
    return torch.where(mask.detach().cpu())[0].tolist()


def _support(model: torch.nn.Module, threshold: float = 1e-8) -> list[int]:
    return _support_from_norms(model, model.response_branches.compute_branch_norms(), threshold)


def _freeze_branch(model: torch.nn.Module, branch: int, optimizer=None) -> None:
    if hasattr(model, "prune_variable"):
        model.prune_variable(branch, optimizer)
        return
    for parameter in model.response_branches.branches[branch].parameters():
        parameter.data.zero_()
        parameter.requires_grad_(False)
        if optimizer is not None:
            optimizer.state.pop(parameter, None)
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if p is not parameter]


def _collect_gradients(model, device) -> dict:
    response_sq = torch.zeros((), device=device)
    for parameter in model.response_branches.parameters():
        if parameter.grad is not None:
            response_sq = response_sq + parameter.grad.detach().square().sum()
    delay_grads = torch.zeros(model.num_variables, device=device)
    delay_parameters = []
    if getattr(model, "delay_prior", None) is not None:
        delay_parameters.extend(model.delay_prior.parameters())
    if getattr(model, "delay_logits", None) is not None:
        delay_parameters.append(model.delay_logits)
    for parameter in delay_parameters:
        if parameter.grad is not None and parameter.grad.ndim:
            delay_grads += parameter.grad.detach().reshape(model.num_variables, -1).norm(dim=1)
    # One synchronization for the vector, rather than N+1 scalar syncs.
    values = torch.cat((response_sq.sqrt().view(1), delay_grads)).cpu().tolist()
    return {
        "response_gradient_norm": values[0],
        "delay_gradient_norm": float(sum(v * v for v in values[1:]) ** 0.5),
        **{f"delay_gradient_{j}": values[j + 1] for j in range(model.num_variables)},
    }


def _train_step(model, optimizer, train_batch, y_train, *, collect_diagnostics=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    target = _target(train_batch, y_train)
    loss = ((_predict(model, train_batch).squeeze(-1) - target.reshape(-1)) ** 2).mean()
    loss.backward()
    diagnostics = _collect_gradients(model, _windows(train_batch).device) if collect_diagnostics else {}
    optimizer.step()
    return loss.detach(), diagnostics


@torch.no_grad()
def _validation_rmse(model, val_batch, y_val) -> float:
    model.eval()
    target = _target(val_batch, y_val)
    mse = ((_predict(model, val_batch).squeeze(-1) - target.reshape(-1)) ** 2).mean()
    return float(mse.sqrt().cpu())


def train_warmup(model, x_train, y_train, x_val, y_val, *, epochs: int,
                 lr: float, patience: int, scheduler_patience: int = 80,
                 validation_interval: int = 1, diagnostic_interval: int = 1,
                 history_interval: int = 1) -> StageResult:
    """Train with no group penalty and return the best validation state."""
    if min(validation_interval, diagnostic_interval, history_interval) < 1:
        raise ValueError("all intervals must be positive")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=scheduler_patience, min_lr=1e-5)
    best_val, best_state, best_epoch, stale_epochs = float("inf"), None, 0, 0
    history = []
    last_val = float("nan")
    for epoch in range(1, epochs + 1):
        collect = epoch == 1 or epoch % diagnostic_interval == 0
        loss, gradients = _train_step(
            model, optimizer, x_train, y_train, collect_diagnostics=collect)
        validate = epoch == 1 or epoch % validation_interval == 0 or epoch == epochs
        if validate:
            last_val = _validation_rmse(model, x_val, y_val)
            scheduler.step(last_val)
            if last_val < best_val:
                best_val = last_val
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += validation_interval
        if epoch == 1 or epoch % history_interval == 0 or validate:
            row = {
                "epoch": epoch,
                "train_mse": float(loss.cpu()),
                "val_rmse": last_val,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if collect:
                row.update(gradients)
            history.append(row)
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("warmup produced no validation checkpoint")
    model.load_state_dict(best_state)
    support = _support(model)
    return StageResult(best_state, copy.deepcopy(model.state_dict()), best_epoch,
                       best_val, history, support, support)


def run_pruning_to_stable_support(model, x_train, y_train, x_val, y_val, *,
                                  warmup_epochs: int = 0, ramp_epochs: int = 200,
                                  full_penalty_min_epochs: int = 100,
                                  stable_epochs: int = 50, max_epochs: int = 1200,
                                  lr: float, lambda_group: float,
                                  threshold: float = 1e-8,
                                  validation_interval: int = 5,
                                  diagnostic_interval: int = 10,
                                  history_interval: int = 5) -> StageResult:
    """Continue proximal pruning; terminal support is authoritative.

    The branch norm vector is computed exactly once per epoch and reused for
    support detection, freezing, and optional logging.
    """
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best_val, best_state, best_support, best_epoch = float("inf"), None, [], 0
    last_support, unchanged, stable_epoch, history = None, 0, None, []
    del warmup_epochs
    earliest_stability_check = ramp_epochs + full_penalty_min_epochs + 1
    last_val = float("nan")
    for epoch in range(1, max_epochs + 1):
        collect = epoch == 1 or epoch % diagnostic_interval == 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        target = _target(x_train, y_train)
        loss = ((_predict(model, x_train).squeeze(-1) - target.reshape(-1)) ** 2).mean()
        loss.backward()
        gradients = _collect_gradients(model, _windows(x_train).device) if collect else {}
        optimizer.step()
        scale = pruning_penalty_scale(epoch, ramp_epochs)
        pre_norms, shrink = apply_group_proximal_step(
            model.response_branches, optimizer.param_groups[0]["lr"], lambda_group * scale)
        # Every tensor in a branch is multiplied by the same nonnegative scale,
        # so the post-proximal norm is exactly pre_norm * scale. No second pass.
        norms = (pre_norms * shrink).detach()
        support = _support_from_norms(model, norms, threshold)
        support_set = set(support)
        selected_before = model.selection_mask.detach().cpu().tolist()
        for j, was_selected in enumerate(selected_before):
            if was_selected and j not in support_set:
                _freeze_branch(model, j, optimizer)

        validate = epoch == 1 or epoch % validation_interval == 0 or epoch == max_epochs
        if validate:
            last_val = _validation_rmse(model, x_val, y_val)
            if last_val < best_val:
                best_val, best_state, best_support, best_epoch = (
                    last_val, copy.deepcopy(model.state_dict()), list(support), epoch)
        if epoch == 1 or epoch % history_interval == 0 or validate:
            row = {
                "epoch": epoch,
                "train_mse": float(loss.detach().cpu()),
                "val_rmse": last_val,
                "support": list(support),
                "penalty_scale": scale,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if collect:
                norm_values = norms.cpu().tolist()
                row.update(gradients)
                row.update({f"branch_norm_{j}": n for j, n in enumerate(norm_values)})
            history.append(row)

        if epoch >= earliest_stability_check:
            if support == last_support:
                unchanged += 1
            else:
                last_support, unchanged = list(support), 1
            if unchanged >= stable_epochs:
                stable_epoch = epoch
                break
        else:
            last_support, unchanged = list(support), 0
    terminal = _support(model, threshold)
    terminal_state = copy.deepcopy(model.state_dict())
    return StageResult(best_state, terminal_state, best_epoch, best_val, history,
                       terminal, best_support, stable_epoch, stable_epoch is not None)


def refit_fixed_support(model, support, x_train, y_train, x_val, y_val, *,
                        epochs: int, lr: float, patience: int,
                        validation_interval: int = 5,
                        diagnostic_interval: int = 20,
                        history_interval: int = 5) -> StageResult:
    """Freeze pruned variables, then refit only survivors and bias."""
    support = set(support)
    for j in range(model.num_variables):
        if j not in support:
            _freeze_branch(model, j)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=max(20, patience // 4), min_lr=1e-5)
    best_val, best_state, best_epoch, stale_epochs, history = float("inf"), None, 0, 0, []
    last_val = float("nan")
    for epoch in range(1, epochs + 1):
        collect = epoch == 1 or epoch % diagnostic_interval == 0
        loss, gradients = _train_step(
            model, optimizer, x_train, y_train, collect_diagnostics=collect)
        validate = epoch == 1 or epoch % validation_interval == 0 or epoch == epochs
        if validate:
            last_val = _validation_rmse(model, x_val, y_val)
            scheduler.step(last_val)
            if last_val < best_val:
                best_val, best_state, best_epoch, stale_epochs = (
                    last_val, copy.deepcopy(model.state_dict()), epoch, 0)
            else:
                stale_epochs += validation_interval
        if epoch == 1 or epoch % history_interval == 0 or validate:
            row = {
                "epoch": epoch,
                "train_mse": float(loss.cpu()),
                "val_rmse": last_val,
                "support": sorted(support),
                "lr": optimizer.param_groups[0]["lr"],
            }
            if collect:
                row.update(gradients)
            history.append(row)
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("refit produced no validation checkpoint")
    model.load_state_dict(best_state)
    return StageResult(best_state, copy.deepcopy(model.state_dict()), best_epoch,
                       best_val, history, sorted(support), sorted(support))
