"""Stable BF16/FP32 training primitives with early stop and resume support."""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .gpu_common import atomic_json, set_global_seed
from .gpu_models import parameter_count, unwrap_output


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    num_workers: int = 4
    dtype: str = "bf16_amp"
    tf32: bool = True
    deterministic: bool = False


@dataclass
class FitResult:
    model: nn.Module
    best_epoch: int
    best_validation_loss: float
    train_seconds: float
    history: list[dict[str, float]]
    dtype_used: str
    fallback_reason: str | None
    peak_memory_bytes: int


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x[indices], dtype=np.float32)),
        torch.from_numpy(np.asarray(y[indices], dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def _autocast(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "fp32":
        return contextlib.nullcontext()
    if dtype == "bf16_amp":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype == "fp16_amp":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"UNKNOWN_TRAINING_DTYPE:{dtype}")


def _configure_precision(device: torch.device, config: TrainConfig) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.tf32)


def _scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int):
    def multiplier(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return max(1e-3, (epoch + 1) / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
) -> float:
    model.train()
    total = 0.0
    count = 0
    use_scaler = device.type == "cuda" and config.dtype == "fp16_amp"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    for inputs, target in loader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.dtype):
            output = unwrap_output(model(inputs))
            prediction = output.prediction.float()
            loss = torch.mean((prediction - target.float()) ** 2)
            if output.auxiliary_loss is not None:
                loss = loss + output.auxiliary_loss.float()
        if not torch.isfinite(loss):
            raise FloatingPointError("NONFINITE_TRAIN_LOSS")
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        total += float(loss.detach()) * len(target)
        count += len(target)
    return total / max(1, count)


@torch.inference_mode()
def _epoch_evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for inputs, target in loader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = unwrap_output(model(inputs))
        loss = torch.mean((output.prediction.float() - target.float()) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("NONFINITE_VALIDATION_LOSS")
        total += float(loss) * len(target)
        count += len(target)
    return total / max(1, count)


def fit_with_early_stopping(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    device: torch.device,
    config: TrainConfig,
    seed: int,
    checkpoint_path: str | Path | None = None,
) -> FitResult:
    set_global_seed(seed, deterministic=config.deterministic)
    _configure_precision(device, config)
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader = _loader(
        x,
        y,
        train_indices,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _loader(
        x,
        y,
        validation_indices,
        batch_size=max(config.batch_size, 256),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = _scheduler(optimizer, config.epochs, config.warmup_epochs)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch_train(model, train_loader, optimizer, device, config)
        validation_loss = _epoch_evaluate(model, validation_loader, device)
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            if checkpoint_path is not None:
                target = Path(checkpoint_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": best_state,
                        "epoch": best_epoch,
                        "validation_loss": best_loss,
                        "parameter_count": parameter_count(model),
                    },
                    target,
                )
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("NO_FINITE_CHECKPOINT")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return FitResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        train_seconds=float(elapsed),
        history=history,
        dtype_used=config.dtype,
        fallback_reason=None,
        peak_memory_bytes=peak,
    )


def fit_fixed_epochs(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    *,
    device: torch.device,
    config: TrainConfig,
    epochs: int,
    seed: int,
    checkpoint_path: str | Path | None = None,
) -> FitResult:
    set_global_seed(seed, deterministic=config.deterministic)
    _configure_precision(device, config)
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loader = _loader(
        x,
        y,
        train_indices,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = _scheduler(optimizer, max(1, epochs), min(config.warmup_epochs, max(0, epochs - 1)))
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = _epoch_train(model, loader, optimizer, device, config)
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if checkpoint_path is not None:
        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epochs,
                "parameter_count": parameter_count(model),
            },
            target,
        )
    return FitResult(
        model=model,
        best_epoch=epochs,
        best_validation_loss=float("nan"),
        train_seconds=float(elapsed),
        history=history,
        dtype_used=config.dtype,
        fallback_reason=None,
        peak_memory_bytes=peak,
    )


@torch.inference_mode()
def predict(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 2,
) -> tuple[np.ndarray, float]:
    indices = np.arange(len(x), dtype=np.int64)
    dummy = np.zeros(len(x), dtype=np.float32)
    loader = _loader(
        x,
        dummy,
        indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    predictions: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for inputs, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        output = unwrap_output(model(inputs))
        predictions.append(output.prediction.float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    values = np.concatenate(predictions).astype(np.float32, copy=False)
    milliseconds_per_1000 = elapsed * 1000.0 * 1000.0 / max(1, len(values))
    return values, float(milliseconds_per_1000)


def safe_fit(
    fit_callable,
    *,
    initial_config: TrainConfig,
    minimum_batch_size: int = 8,
) -> FitResult:
    """Retry OOM by halving batch size, then retry unstable AMP in FP32."""
    config = initial_config
    fallback_reasons: list[str] = []
    while True:
        try:
            result: FitResult = fit_callable(config)
            result.fallback_reason = ";".join(fallback_reasons) or None
            return result
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if config.batch_size <= minimum_batch_size:
                raise
            new_batch = max(minimum_batch_size, config.batch_size // 2)
            fallback_reasons.append(f"OOM_BATCH_{config.batch_size}_TO_{new_batch}")
            config = TrainConfig(**{**config.__dict__, "batch_size": new_batch})
        except FloatingPointError as exc:
            if config.dtype == "fp32":
                raise
            fallback_reasons.append(f"{type(exc).__name__}_AMP_TO_FP32")
            config = TrainConfig(**{**config.__dict__, "dtype": "fp32", "tf32": False})


def write_training_trace(path: str | Path, result: FitResult) -> None:
    atomic_json(
        path,
        {
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "train_seconds": result.train_seconds,
            "dtype_used": result.dtype_used,
            "fallback_reason": result.fallback_reason,
            "peak_memory_bytes": result.peak_memory_bytes,
            "history": result.history,
        },
    )
