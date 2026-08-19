from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    deterministic_subsample,
    input_columns as registered_input_columns,
)
from .v211_support import (
    SUPPORT_CONTRACT,
    load_native_samples,
    require_native_support_contract,
)


LOOKBACK_HOURS = (2, 4, 8)
CAPACITIES = ("SMALL", "MEDIUM")
LEARNING_RATES = (3e-4, 1e-3)
SCREENING_SEED = 20260817
FINAL_SEEDS = (20260817, 20260818, 20260819)
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 12
GRADIENT_CLIP_NORM = 1.0
EFFECTIVE_BATCH_SIZE = 256
PHYSICAL_BATCH_SIZE = 256
NEURAL_FIT_ROW_CAP = 25_000
NEURAL_VALIDATION_ROW_CAP = 5_000
MAX_SEQUENCE_TOKENS = 256
ITRANSFORMER_TEMPORAL_TOKENS = 16
PARAMETER_BUDGET = 250_000
MODEL_FAMILIES = ("LSTM", "iTransformer", "TimeMixer")


@dataclass(frozen=True)
class Candidate:
    model: str
    lookback_hours: int
    history_steps: int
    capacity: str
    learning_rate: float

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.model}__L{self.lookback_hours}h__"
            f"{self.capacity}__lr{self.learning_rate:g}"
        )


@dataclass(frozen=True)
class Scaler:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "fit_scope": "OUTER_TRAINING_ONLY",
        }


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, capacity: str) -> None:
        super().__init__()
        hidden, layers, dropout = (
            (64, 1, 0.0) if capacity == "SMALL" else (96, 2, 0.1)
        )
        self.encoder = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(values)
        return self.head(encoded[:, -1]).squeeze(-1)


class ITransformerModel(nn.Module):
    def __init__(self, input_dim: int, capacity: str) -> None:
        super().__init__()
        d_model, heads, layers, d_ff, dropout = (
            (64, 4, 2, 128, 0.1)
            if capacity == "SMALL"
            else (96, 4, 2, 192, 0.1)
        )
        self.temporal = nn.Linear(ITRANSFORMER_TEMPORAL_TOKENS, d_model)
        self.variable_embedding = nn.Parameter(torch.zeros(1, input_dim, d_model))
        nn.init.normal_(self.variable_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Variables are tokens; time is compressed independently per variable.
        tokens = F.adaptive_avg_pool1d(
            values.transpose(1, 2), ITRANSFORMER_TEMPORAL_TOKENS
        )
        tokens = self.temporal(tokens) + self.variable_embedding
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1)).squeeze(-1)


class TimeMixerModel(nn.Module):
    def __init__(self, input_dim: int, capacity: str) -> None:
        super().__init__()
        d_model = 64 if capacity == "SMALL" else 96
        self.projection = nn.Linear(input_dim, d_model)
        self.past_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(3)
            ]
        )
        self.future_mixer = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Each scale is a causal past summary; no future rows enter pooling.
        channels_first = values.transpose(1, 2)
        scale_vectors = []
        for index, stride in enumerate((1, 2, 4)):
            effective_stride = min(stride, channels_first.shape[-1])
            length = channels_first.shape[-1] - (
                channels_first.shape[-1] % effective_stride
            )
            pooled = F.avg_pool1d(
                channels_first[..., :length],
                kernel_size=effective_stride,
                stride=effective_stride,
            ).transpose(1, 2)
            mixed = self.past_mixers[index](self.projection(pooled))
            scale_vectors.append(mixed.mean(dim=1))
        fused = self.future_mixer(torch.cat(scale_vectors, dim=-1))
        return self.head(fused).squeeze(-1)


def build_model(model_name: str, input_dim: int, capacity: str) -> nn.Module:
    if model_name == "LSTM":
        model = LSTMModel(input_dim, capacity)
    elif model_name == "iTransformer":
        model = ITransformerModel(input_dim, capacity)
    elif model_name == "TimeMixer":
        model = TimeMixerModel(input_dim, capacity)
    else:
        raise ValueError(model_name)
    count = parameter_count(model)
    if count > PARAMETER_BUDGET:
        raise RuntimeError(
            f"{model_name}/{capacity} has {count} parameters > {PARAMETER_BUDGET}"
        )
    return model


def candidate_grid(model_name: str, cadence_seconds: float) -> list[Candidate]:
    result = []
    for hours in LOOKBACK_HOURS:
        history = int(round(hours * 3600.0 / float(cadence_seconds)))
        if history < 1:
            continue
        for capacity in CAPACITIES:
            for learning_rate in LEARNING_RATES:
                result.append(
                    Candidate(
                        model_name,
                        hours,
                        history,
                        capacity,
                        learning_rate,
                    )
                )
    if len(result) > 12:
        raise AssertionError("Neural candidate grid exceeded 12 candidates")
    return result


def native_support(
    samples: pd.DataFrame,
    history_steps: int,
    *,
    dynamic: bool = False,
) -> pd.DataFrame:
    require_native_support_contract(samples)
    origin = samples["origin"].to_numpy(dtype=np.int64)
    floor = samples["causal_history_floor"].to_numpy(dtype=np.int64)
    mask = origin - int(history_steps) >= floor
    if dynamic:
        latest = samples["latest_available_target_index"].to_numpy(dtype=np.int64)
        mask &= latest - int(history_steps) + 1 >= floor
    return samples.loc[mask].copy()
def _cap_after_native_support(
    samples: pd.DataFrame,
    cap: int,
) -> pd.DataFrame:
    if cap < 1:
        raise ValueError("native support row cap must be positive")
    indices = deterministic_subsample(samples, cap)
    return samples.iloc[indices].reset_index(drop=True)


def _partition_candidate_support(
    candidates: list[Candidate],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    dynamic: bool,
) -> tuple[list[tuple[Candidate, pd.DataFrame]], list[dict[str, Any]]]:
    available: list[tuple[Candidate, pd.DataFrame]] = []
    unavailable: list[dict[str, Any]] = []
    for candidate in candidates:
        fit = native_support(
            train, candidate.history_steps, dynamic=dynamic
        )
        score = native_support(
            validation, candidate.history_steps, dynamic=dynamic
        )
        reasons = []
        if fit.empty:
            reasons.append("EMPTY_NATIVE_TRAIN_SUPPORT")
        if score.empty:
            reasons.append("EMPTY_NATIVE_VALIDATION_SUPPORT")
        if reasons:
            unavailable.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "lookback_hours": candidate.lookback_hours,
                    "history_steps": candidate.history_steps,
                    "capacity": candidate.capacity,
                    "learning_rate": candidate.learning_rate,
                    "native_train_rows": int(len(fit)),
                    "native_validation_rows": int(len(score)),
                    "reasons": reasons,
                }
            )
        else:
            available.append((candidate, fit))
    return available, unavailable


def shared_data_fingerprint(shared: Path, view: ViewSpec) -> str:
    paths = [
        shared / "TASK_REGISTRY.json",
        shared / "PROTOCOL.json",
        shared / "dataset_views" / "VIEW_REGISTRY.json",
        shared / "CZ_TASK_REALIZATION.json",
        shared
        / "sample_ids"
        / view.head.task_id
        / view.information_set
        / view.availability_scenario
        / view.proxy_policy
        / "train.parquet",
        shared
        / "sample_ids"
        / view.head.task_id
        / view.information_set
        / view.availability_scenario
        / view.proxy_policy
        / "validation.parquet",
    ]
    digest = hashlib.sha256()
    digest.update(view.relative_root.as_posix().encode("utf-8"))
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(shared)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def support_hash(samples: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in samples["base_origin_id"].astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _blocks(history_steps: int, max_tokens: int) -> list[np.ndarray]:
    if history_steps <= max_tokens:
        return [np.arange(history_steps, dtype=np.int64)]
    return [
        block.astype(np.int64)
        for block in np.array_split(
            np.arange(history_steps, dtype=np.int64), max_tokens
        )
        if len(block)
    ]


def materialize_sequences(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    columns: list[str],
    history_steps: int,
    *,
    dynamic: bool,
) -> np.ndarray:
    if samples.empty:
        return np.empty((0, min(history_steps, MAX_SEQUENCE_TOKENS), len(columns)), dtype=np.float32)
    blocks = _blocks(history_steps, MAX_SEQUENCE_TOKENS)
    origins = samples["origin"].to_numpy(dtype=np.int64)
    output = np.empty(
        (len(samples), len(blocks), len(columns) + (1 if dynamic else 0)),
        dtype=np.float32,
    )
    for token, block in enumerate(blocks):
        indices = origins[:, None] - history_steps + block[None, :]
        values = accessor.gather(samples, columns, indices)
        output[:, token, : len(columns)] = values.reshape(
            len(samples), len(block), len(columns)
        ).mean(axis=1, dtype=np.float64).astype(np.float32)
        if dynamic:
            latest = samples["latest_available_target_index"].to_numpy(dtype=np.int64)
            target_indices = latest[:, None] - history_steps + 1 + block[None, :]
            target = accessor.gather(
                samples, [accessor._target_column], target_indices
            )
            output[:, token, len(columns)] = target.reshape(
                len(samples), len(block)
            ).mean(axis=1, dtype=np.float64).astype(np.float32)
    return output


def _set_target_column(accessor: BaseAccessor, target: str) -> None:
    accessor._target_column = target


def _training_ranges(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    columns: list[str],
    history_steps: int,
) -> list[np.ndarray]:
    ranges: list[np.ndarray] = []
    for entity, group in samples.groupby("entity_id", sort=False):
        origins = group["origin"].to_numpy(dtype=np.int64)
        if len(origins) == 0:
            continue
        start = max(0, int(origins.min()) - history_steps)
        stop = int(origins.max())
        rows, _ = accessor.entities[str(entity)]
        ranges.append(rows[(rows >= start) & (rows < stop)])
    return ranges


def fit_scaler(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    columns: list[str],
    target: str,
    history_steps: int,
) -> Scaler:
    _set_target_column(accessor, target)
    sums = np.zeros(len(columns) + 1, dtype=np.float64)
    squares = np.zeros(len(columns) + 1, dtype=np.float64)
    count = 0
    for entity, group in samples.groupby("entity_id", sort=False):
        origins = group["origin"].to_numpy(dtype=np.int64)
        if len(origins) == 0:
            continue
        start = max(0, int(origins.min()) - history_steps)
        stop = int(origins.max())
        rows, arrays = accessor.entities[str(entity)]
        mask = (rows >= start) & (rows < stop)
        for index, column in enumerate(columns):
            values = np.asarray(arrays[column][mask], dtype=np.float64)
            sums[index] += values.sum(dtype=np.float64)
            squares[index] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
        target_values = np.asarray(arrays[target][mask], dtype=np.float64)
        sums[-1] += target_values.sum(dtype=np.float64)
        squares[-1] += np.square(target_values, dtype=np.float64).sum(dtype=np.float64)
        count += int(mask.sum())
    if count < 1:
        raise RuntimeError("cannot fit neural scaler on empty training support")
    means = sums / count
    variances = np.maximum(squares / count - np.square(means), 1e-12)
    y = samples["y_true"].to_numpy(dtype=np.float64)
    target_mean = float(y.mean(dtype=np.float64))
    target_scale = float(max(y.std(dtype=np.float64), 1e-12))
    return Scaler(
        means[:-1],
        np.sqrt(variances[:-1]),
        target_mean,
        target_scale,
    )


def _scaled_values(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    columns: list[str],
    history_steps: int,
    scaler: Scaler,
    *,
    dynamic: bool,
    device: torch.device,
) -> torch.Tensor:
    values = materialize_sequences(
        accessor, samples, columns, history_steps, dynamic=dynamic
    )
    feature_count = len(columns)
    values[..., :feature_count] = (
        values[..., :feature_count] - scaler.feature_mean.astype(np.float32)
    ) / scaler.feature_scale.astype(np.float32)
    if dynamic:
        values[..., feature_count] = (
            values[..., feature_count] - float(scaler.target_mean)
        ) / float(scaler.target_scale)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _scaled_batch(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    columns: list[str],
    history_steps: int,
    scaler: Scaler,
    *,
    dynamic: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = _scaled_values(
        accessor,
        samples,
        columns,
        history_steps,
        scaler,
        dynamic=dynamic,
        device=device,
    )
    target = (
        samples["y_true"].to_numpy(dtype=np.float32) - scaler.target_mean
    ) / scaler.target_scale
    return values, torch.as_tensor(target, dtype=torch.float32, device=device)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mse = float(np.mean(np.square(error), dtype=np.float64))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    variance = float(np.sum(np.square(y_true - y_true.mean()), dtype=np.float64))
    r2 = float(1.0 - np.sum(np.square(error), dtype=np.float64) / variance) if variance else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def _predict(
    model: nn.Module,
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    columns: list[str],
    history_steps: int,
    scaler: Scaler,
    *,
    dynamic: bool,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(samples), PHYSICAL_BATCH_SIZE):
            batch = samples.iloc[start : start + PHYSICAL_BATCH_SIZE]
            values = _scaled_values(
                accessor,
                batch,
                columns,
                history_steps,
                scaler,
                dynamic=dynamic,
                device=device,
            )
            prediction = model(values).detach().cpu().numpy()
            predictions.append(prediction)
    if not predictions:
        return np.empty(0, dtype=np.float64)
    scaled = np.concatenate(predictions).astype(np.float64)
    return scaled * scaler.target_scale + scaler.target_mean


def _fit_one(
    candidate: Candidate,
    accessor: BaseAccessor,
    fit_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    columns: list[str],
    target: str,
    scaler: Scaler,
    *,
    dynamic: bool,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    model = build_model(
        candidate.model, len(columns) + (1 if dynamic else 0), candidate.capacity
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    accumulation = max(1, EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    permutation_rng = np.random.default_rng(seed)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = permutation_rng.permutation(len(fit_samples))
        optimizer.zero_grad(set_to_none=True)
        for batch_number, start in enumerate(
            range(0, len(order), PHYSICAL_BATCH_SIZE)
        ):
            batch = fit_samples.iloc[order[start : start + PHYSICAL_BATCH_SIZE]]
            values, target_values = _scaled_batch(
                accessor,
                batch,
                columns,
                candidate.history_steps,
                scaler,
                dynamic=dynamic,
                device=device,
            )
            loss = F.mse_loss(model(values), target_values)
            (loss / accumulation).backward()
            if (batch_number + 1) % accumulation == 0 or (
                start + PHYSICAL_BATCH_SIZE >= len(order)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP_NORM
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_prediction = _predict(
            model,
            accessor,
            validation_samples,
            columns,
            candidate.history_steps,
            scaler,
            dynamic=dynamic,
            device=device,
        )
        validation_target = validation_samples["y_true"].to_numpy(dtype=np.float64)
        validation_mse = _metrics(validation_target, validation_prediction)["mse"]
        if validation_mse < best_val:
            best_val = validation_mse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    result = {
        "status": "PASS",
        "candidate_id": candidate.candidate_id,
        "model": candidate.model,
        "lookback_hours": candidate.lookback_hours,
        "history_steps": candidate.history_steps,
        "capacity": candidate.capacity,
        "learning_rate": candidate.learning_rate,
        "parameter_count": parameter_count(model),
        "best_epoch": best_epoch,
        "validation_mse": best_val,
        "validation_rmse": math.sqrt(best_val),
        "training_seconds": time.time() - started,
        "peak_vram_bytes": peak_vram,
        "seed": seed,
    }
    del model, optimizer, best_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result

def _fit_fixed_epochs(
    candidate: Candidate,
    accessor: BaseAccessor,
    fit_samples: pd.DataFrame,
    columns: list[str],
    scaler: Scaler,
    *,
    dynamic: bool,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    if epochs < 1 or epochs > MAX_EPOCHS:
        raise ValueError(f"invalid frozen epoch count: {epochs}")
    if fit_samples.empty:
        raise RuntimeError("cannot fit final neural model on empty support")
    set_seed(seed)
    model = build_model(
        candidate.model,
        len(columns) + (1 if dynamic else 0),
        candidate.capacity,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    accumulation = max(1, EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE)
    permutation_rng = np.random.default_rng(seed)
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _epoch in range(1, epochs + 1):
        model.train()
        order = permutation_rng.permutation(len(fit_samples))
        optimizer.zero_grad(set_to_none=True)
        for batch_number, start in enumerate(
            range(0, len(order), PHYSICAL_BATCH_SIZE)
        ):
            batch = fit_samples.iloc[order[start : start + PHYSICAL_BATCH_SIZE]]
            values, target_values = _scaled_batch(
                accessor,
                batch,
                columns,
                candidate.history_steps,
                scaler,
                dynamic=dynamic,
                device=device,
            )
            loss = F.mse_loss(model(values), target_values)
            (loss / accumulation).backward()
            if (batch_number + 1) % accumulation == 0 or (
                start + PHYSICAL_BATCH_SIZE >= len(order)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP_NORM
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    result = {
        "status": "PASS",
        "candidate_id": candidate.candidate_id,
        "model": candidate.model,
        "lookback_hours": candidate.lookback_hours,
        "history_steps": candidate.history_steps,
        "capacity": candidate.capacity,
        "learning_rate": candidate.learning_rate,
        "parameter_count": parameter_count(model),
        "epochs_run": epochs,
        "best_epoch": epochs,
        "training_seconds": time.time() - started,
        "peak_vram_bytes": peak_vram,
        "seed": seed,
    }
    del optimizer
    gc.collect()
    return model, result




def select_candidate(
    *,
    shared: Path,
    view: ViewSpec,
    model_name: str,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    if model_name not in MODEL_FAMILIES:
        raise ValueError(model_name)
    train = load_native_samples(shared, view, "train")
    validation = load_native_samples(shared, view, "validation")
    require_native_support_contract(train)
    require_native_support_contract(validation)
    dynamic = view.information_set == "dynamic"
    input_columns = registered_input_columns(
        shared, view.head.task_id, view.proxy_policy
    )
    accessor = BaseAccessor(
        shared,
        view.head.dataset,
        "validation",
        [*input_columns, view.head.target],
    )
    _set_target_column(accessor, view.head.target)
    candidates = candidate_grid(model_name, view.head.cadence_seconds)
    native_candidates, unavailable_candidates = _partition_candidate_support(
        candidates, train, validation, dynamic=dynamic
    )
    if not native_candidates:
        raise RuntimeError(f"no valid neural candidates for {view.relative_root}")
    max_history = max(
        candidate.history_steps for candidate, _fit in native_candidates
    )
    common_validation = native_support(
        validation, max_history, dynamic=dynamic
    )
    common_validation_rows_before_cap = len(common_validation)
    common_validation = _cap_after_native_support(
        common_validation, NEURAL_VALIDATION_ROW_CAP
    )
    scaler_fit = native_support(train, max_history, dynamic=dynamic)
    if common_validation.empty or scaler_fit.empty:
        raise AssertionError("available neural candidates lost common support")
    scaler = fit_scaler(
        accessor,
        scaler_fit,
        input_columns,
        view.head.target,
        max_history,
    )
    candidate_results = []
    for candidate, fit in native_candidates:
        native_fit_rows_before_cap = len(fit)
        fit = _cap_after_native_support(fit, NEURAL_FIT_ROW_CAP)
        result = _fit_one(
            candidate,
            accessor,
            fit,
            common_validation,
            input_columns,
            view.head.target,
            scaler,
            dynamic=dynamic,
            seed=SCREENING_SEED,
            device=device,
        )
        result.update(
            {
                "native_fit_rows": int(len(fit)),
                "native_fit_rows_before_cap": int(native_fit_rows_before_cap),
                "native_fit_row_cap": NEURAL_FIT_ROW_CAP,
                "native_fit_support_hash": support_hash(fit),
                "common_validation_rows": int(len(common_validation)),
                "common_validation_rows_before_cap": int(common_validation_rows_before_cap),
                "common_validation_row_cap": NEURAL_VALIDATION_ROW_CAP,
                "common_validation_support_hash": support_hash(common_validation),
                "row_cap_applied_after_native_mask": True,
                "test_metrics_used_for_selection": False,
                "historical_metrics_used_for_selection": False,
            }
        )
        candidate_results.append(result)
    candidate_results.sort(
        key=lambda value: (
            float(value["validation_mse"]),
            int(value["parameter_count"]),
            int(value["lookback_hours"]),
            0 if value["capacity"] == "SMALL" else 1,
            float(value["learning_rate"]),
        )
    )
    selected = candidate_results[0]
    best_mse = float(selected["validation_mse"])
    eligible = [
        value
        for value in candidate_results
        if float(value["validation_mse"]) <= best_mse * 1.01
    ]
    selected = min(
        eligible,
        key=lambda value: (
            int(value["parameter_count"]),
            int(value["lookback_hours"]),
            0 if value["capacity"] == "SMALL" else 1,
            float(value["learning_rate"]),
        ),
    )
    result = {
        "status": "PASS",
        "data_support_fingerprint": shared_data_fingerprint(shared, view),
        "model": model_name,
        "view": view.relative_root.as_posix(),
        "dataset": view.head.dataset,
        "task_id": view.head.task_id,
        "information_set": view.information_set,
        "screening_seed": SCREENING_SEED,
        "candidate_count": len(candidate_results),
        "candidate_grid_count": len(candidates),
        "available_candidate_ids": [
            candidate.candidate_id for candidate, _fit in native_candidates
        ],
        "available_lookback_hours": sorted(
            {
                candidate.lookback_hours
                for candidate, _fit in native_candidates
            }
        ),
        "unavailable_candidates": unavailable_candidates,
        "unavailable_lookback_hours": sorted(
            {item["lookback_hours"] for item in unavailable_candidates}
        ),
        "selected_candidate": selected,
        "selected_profile": {
            "model": selected["model"],
            "lookback_hours": selected["lookback_hours"],
            "history_steps": selected["history_steps"],
            "capacity": selected["capacity"],
            "learning_rate": selected["learning_rate"],
            "parameter_count": selected["parameter_count"],
        },
        "candidate_results": candidate_results,
        "scaler": scaler.to_json(),
        "support_contract": SUPPORT_CONTRACT,
        "native_support": {
            "selected_native_train_rows": selected["native_fit_rows"],
            "selected_native_train_support_hash": selected["native_fit_support_hash"],
            "common_validation_rows": selected["common_validation_rows"],
            "common_validation_support_hash": selected["common_validation_support_hash"],
            "common_support_history_steps": max_history,
            "fit_row_cap": NEURAL_FIT_ROW_CAP,
            "validation_row_cap": NEURAL_VALIDATION_ROW_CAP,
            "cross_candidate_validation_rows_equal": len(
                {value["common_validation_support_hash"] for value in candidate_results}
            )
            == 1,
        },
        "tokenization": {
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "itransformer_temporal_tokens": ITRANSFORMER_TEMPORAL_TOKENS,
            "causal": True,
            "time_pooling": "past_only",
        },
        "test_accessed": False,
        "ood_accessed": False,
    }
    destination = output / model_name / view.relative_root
    destination.mkdir(parents=True, exist_ok=True)
    destination.joinpath("SELECTION.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def materialize_model(
    *,
    shared: Path,
    view: ViewSpec,
    selection: dict[str, Any],
    split: str,
    output: Path,
    device: torch.device,
    final_fit: bool = False,
) -> dict[str, Any]:
    if split not in {"test", "ood"}:
        raise ValueError("final neural materialization is limited to test/OOD")
    if selection.get("status") != "PASS":
        raise RuntimeError("cannot materialize a non-PASS neural selection")
    if selection.get("support_contract") != SUPPORT_CONTRACT:
        raise RuntimeError("neural selection support contract mismatch")
    if selection.get("test_accessed") is not False:
        raise RuntimeError("neural selection accessed test before freeze")

    profile = selection["selected_profile"]
    selected_candidate = selection["selected_candidate"]
    candidate = Candidate(
        profile["model"],
        int(profile["lookback_hours"]),
        int(profile["history_steps"]),
        profile["capacity"],
        float(profile["learning_rate"]),
    )
    if selected_candidate["candidate_id"] != candidate.candidate_id:
        raise RuntimeError("selected profile and candidate disagree")
    frozen_epochs = int(selected_candidate["best_epoch"])
    if frozen_epochs < 1 or frozen_epochs > MAX_EPOCHS:
        raise RuntimeError(f"invalid selected best_epoch: {frozen_epochs}")

    dynamic = view.information_set == "dynamic"
    columns = registered_input_columns(
        shared, view.head.task_id, view.proxy_policy
    )
    train = native_support(
        load_native_samples(shared, view, "train"),
        candidate.history_steps,
        dynamic=dynamic,
    )
    validation = native_support(
        load_native_samples(shared, view, "validation"),
        candidate.history_steps,
        dynamic=dynamic,
    )
    fit_samples = pd.concat([train, validation], ignore_index=True)
    if fit_samples["base_origin_id"].duplicated().any():
        raise RuntimeError("duplicate fit sample IDs across train/validation")
    evaluation = native_support(
        load_native_samples(shared, view, split),
        candidate.history_steps,
        dynamic=dynamic,
    ).reset_index(drop=True)
    if evaluation.empty:
        raise RuntimeError(f"empty final support for {view.relative_root}/{split}")

    accessor = BaseAccessor(
        shared,
        view.head.dataset,
        split,
        [*columns, view.head.target],
    )
    _set_target_column(accessor, view.head.target)
    scaler = Scaler(
        np.asarray(selection["scaler"]["feature_mean"], dtype=np.float64),
        np.asarray(selection["scaler"]["feature_scale"], dtype=np.float64),
        float(selection["scaler"]["target_mean"]),
        float(selection["scaler"]["target_scale"]),
    )
    if len(scaler.feature_mean) != len(columns):
        raise RuntimeError("frozen neural scaler does not match registered inputs")

    seed_results: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    target = evaluation["y_true"].to_numpy(dtype=np.float64)
    for seed in FINAL_SEEDS:
        model, seed_result = _fit_fixed_epochs(
            candidate,
            accessor,
            fit_samples,
            columns,
            scaler,
            dynamic=dynamic,
            seed=seed,
            epochs=frozen_epochs,
            device=device,
        )
        prediction_started = time.time()
        prediction = _predict(
            model,
            accessor,
            evaluation,
            columns,
            candidate.history_steps,
            scaler,
            dynamic=dynamic,
            device=device,
        )
        seed_result["prediction_seconds"] = time.time() - prediction_started
        seed_result["metrics"] = _metrics(target, prediction)
        seed_results.append(seed_result)
        predictions.append(prediction)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prediction_matrix = np.vstack(predictions)
    ensemble_prediction = prediction_matrix.mean(axis=0, dtype=np.float64)
    ensemble_metrics = _metrics(target, ensemble_prediction)
    metric_names = tuple(ensemble_metrics)
    seed_metric_summary = {
        metric: {
            "mean": float(
                np.mean(
                    [item["metrics"][metric] for item in seed_results],
                    dtype=np.float64,
                )
            ),
            "std": float(
                np.std(
                    [item["metrics"][metric] for item in seed_results],
                    dtype=np.float64,
                )
            ),
        }
        for metric in metric_names
    }

    destination = output / "FINAL" / profile["model"] / view.relative_root
    destination.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "sample_id": evaluation["view_sample_id"].astype(str),
            "base_origin_id": evaluation["base_origin_id"].astype(str),
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "split": split,
            "model": profile["model"],
            "y_true": target,
            "y_pred": ensemble_prediction,
            "information_set": view.information_set,
            "entity_id": evaluation["entity_id"].astype(str),
            "profile_id": evaluation["entity_id"].astype(str),
            "origin": evaluation["origin"].to_numpy(dtype=np.int64),
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "seed": -1,
            "dtype": "float64",
            "parameter_count": int(profile["parameter_count"]),
        }
    )
    for row, seed in enumerate(FINAL_SEEDS):
        prediction_frame[f"y_pred_seed_{seed}"] = prediction_matrix[row]
    prediction_path = destination / f"{split}.parquet"
    prediction_frame.to_parquet(
        prediction_path, index=False, compression="zstd"
    )
    prediction_digest = hashlib.sha256()
    with prediction_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            prediction_digest.update(block)

    final = {
        "status": "PASS",
        "support_contract": SUPPORT_CONTRACT,
        "model": profile["model"],
        "view": view.relative_root.as_posix(),
        "split": split,
        "selected_candidate_id": candidate.candidate_id,
        "frozen_best_epoch": frozen_epochs,
        "frozen_best_epoch_source": "DEVELOPMENT_SELECTION",
        "final_seeds": list(FINAL_SEEDS),
        "seed_results": seed_results,
        "seed_metric_summary": seed_metric_summary,
        "ensemble": True,
        "ensemble_contract": "FROZEN_3SEED_ENSEMBLE",
        "ensemble_metrics": ensemble_metrics,
        "parameter_count": int(profile["parameter_count"]),
        "fit_rows": int(len(fit_samples)),
        "fit_train_rows": int(len(train)),
        "fit_validation_rows": int(len(validation)),
        "fit_support_hash": support_hash(fit_samples),
        "evaluation_rows": int(len(evaluation)),
        "evaluation_support_hash": support_hash(evaluation),
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_digest.hexdigest(),
        "final_fit": final_fit,
        "test_rows_used_for_fitting": False,
        "evaluation_targets_used_for_training": False,
        "target_rod_used_for_selection": False,
        "test_accessed": True,
        "ood_accessed": split == "ood",
    }
    destination.joinpath(f"{split.upper()}_RESULT.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final
