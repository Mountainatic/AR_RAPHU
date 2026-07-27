"""Peer-reviewed single-hidden-layer MLP-NARX for PB1."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from ar_raphu.datasets.base import DynamicDataset


@dataclass(frozen=True, slots=True)
class MinMaxScaling:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @staticmethod
    def _scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        width = upper - lower
        if width <= 0.0:
            raise ValueError("MLP-NARX scaler encountered a constant channel.")
        return 2.0 * (values - lower) / width - 1.0

    def x_transform(self, values: np.ndarray) -> np.ndarray:
        return self._scale(np.asarray(values), self.x_min, self.x_max)

    def y_transform(self, values: np.ndarray) -> np.ndarray:
        return self._scale(np.asarray(values), self.y_min, self.y_max)


@dataclass(frozen=True, slots=True)
class MLPWeights:
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: float

    @property
    def width(self) -> int:
        return int(len(self.hidden_bias))


@dataclass(frozen=True, slots=True)
class MLPTrainingResult:
    weights: MLPWeights
    scaling: MinMaxScaling
    final_training_rmse: float
    elapsed_seconds: float


def _records(
    dataset: DynamicDataset, split: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sequence in np.unique(dataset.sequence_id):
        indices = np.flatnonzero(dataset.sequence_id == sequence)
        values = np.unique(dataset.split[indices])
        if len(values) != 1:
            raise ValueError(f"{sequence}: MLP-NARX requires record-atomic splits.")
        if str(values[0]) == split:
            result.append(
                (
                    str(sequence),
                    np.asarray(dataset.x[indices, 0], dtype=np.float64),
                    np.asarray(dataset.y[indices, 0], dtype=np.float64),
                )
            )
    if not result:
        raise ValueError(f"No {split} records.")
    return result


def fit_minmax_scaling(dataset: DynamicDataset) -> MinMaxScaling:
    if np.any(dataset.split == "test"):
        raise PermissionError("MLP-NARX development refuses test rows.")
    train = dataset.split == "train"
    x = np.asarray(dataset.x[train, 0], dtype=np.float64)
    y = np.asarray(dataset.y[train, 0], dtype=np.float64)
    return MinMaxScaling(
        x_min=float(x.min()),
        x_max=float(x.max()),
        y_min=float(y.min()),
        y_max=float(y.max()),
    )


def history_design(
    x: np.ndarray, y: np.ndarray, *, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced histories through t and target y[t+1]."""

    history = max(nx, ny)
    y_windows = np.lib.stride_tricks.sliding_window_view(y, ny)
    x_windows = np.lib.stride_tricks.sliding_window_view(x, nx)
    y_lags = y_windows[history - ny : -1, ::-1]
    x_lags = x_windows[history - nx : -1, ::-1]
    target = y[history:]
    if len(y_lags) != len(target) or len(x_lags) != len(target):
        raise AssertionError("MLP-NARX history alignment failed.")
    return np.column_stack((y_lags, x_lags)), target


def training_arrays(
    dataset: DynamicDataset,
    *,
    nx: int,
    ny: int,
    scaling: MinMaxScaling,
) -> tuple[np.ndarray, np.ndarray]:
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _, x_raw, y_raw in _records(dataset, "train"):
        x = scaling.x_transform(x_raw)
        y = scaling.y_transform(y_raw)
        design, target = history_design(x, y, nx=nx, ny=ny)
        designs.append(np.asarray(design, dtype=np.float32))
        targets.append(np.asarray(target, dtype=np.float32))
    return np.concatenate(designs), np.concatenate(targets)[:, None]


class _MLP(torch.nn.Module):
    def __init__(self, features: int, width: int) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(features, width)
        self.output = torch.nn.Linear(width, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(values)))


def _weights(model: _MLP) -> MLPWeights:
    return MLPWeights(
        hidden_weight=model.hidden.weight.detach().cpu().numpy().astype(
            np.float64
        ),
        hidden_bias=model.hidden.bias.detach().cpu().numpy().astype(np.float64),
        output_weight=model.output.weight.detach().cpu().numpy().reshape(
            -1
        ).astype(np.float64),
        output_bias=float(model.output.bias.detach().cpu().item()),
    )


def train_mlp_narx(
    dataset: DynamicDataset,
    *,
    nx: int,
    ny: int,
    width: int,
    seed: int,
    iterations: int = 20_000,
    learning_rate: float = 1.0e-2,
) -> MLPTrainingResult:
    """Full-training-set Adam optimization with no early stopping."""

    started = time.perf_counter()
    if dataset.n_features != 1 or dataset.n_targets != 1:
        raise ValueError("Champneys 2024 MLP-NARX profile is SISO.")
    if width not in {2, 5, 7, 10} or iterations <= 0:
        raise ValueError("Invalid peer-reviewed MLP-NARX capacity.")
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    scaling = fit_minmax_scaling(dataset)
    design, target = training_arrays(
        dataset, nx=nx, ny=ny, scaling=scaling
    )
    x_tensor = torch.from_numpy(design)
    y_tensor = torch.from_numpy(target)
    model = _MLP(design.shape[1], width)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = float("nan")
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        error = model(x_tensor) - y_tensor
        loss = torch.sqrt(torch.mean(error * error))
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return MLPTrainingResult(
        weights=_weights(model),
        scaling=scaling,
        final_training_rmse=final_loss,
        elapsed_seconds=time.perf_counter() - started,
    )


def predict_mlp(history: np.ndarray, weights: MLPWeights) -> float:
    values = np.asarray(history, dtype=np.float64)
    hidden = np.tanh(weights.hidden_weight @ values + weights.hidden_bias)
    return float(weights.output_weight @ hidden + weights.output_bias)


def simulate_mlp_narx(
    x: np.ndarray,
    y_initialization: np.ndarray,
    *,
    nx: int,
    ny: int,
    weights: MLPWeights,
) -> tuple[np.ndarray, int]:
    x_values = np.asarray(x, dtype=np.float64)
    observed_y = np.asarray(y_initialization, dtype=np.float64)
    burn = max(nx, ny)
    prediction = np.full_like(observed_y, np.nan)
    prediction[:burn] = observed_y[:burn]
    for target_index in range(burn, len(prediction)):
        origin = target_index - 1
        history = np.r_[
            prediction[origin - np.arange(ny)],
            x_values[origin - np.arange(nx)],
        ]
        prediction[target_index] = predict_mlp(history, weights)
        if not np.isfinite(prediction[target_index]):
            break
    return prediction, burn


def validation_aic(
    dataset: DynamicDataset,
    *,
    nx: int,
    ny: int,
    weights: MLPWeights,
    scaling: MinMaxScaling,
) -> tuple[tuple[float, ...], bool]:
    scores: list[float] = []
    stable = True
    parameters = (
        weights.width * (nx + ny)
        + weights.width
        + weights.width
        + 1
    )
    for _, x_raw, y_raw in _records(dataset, "validation"):
        x = scaling.x_transform(x_raw)
        y = scaling.y_transform(y_raw)
        prediction, burn = simulate_mlp_narx(
            x, y, nx=nx, ny=ny, weights=weights
        )
        residual = y[burn:] - prediction[burn:]
        n = len(residual)
        rss = float(residual @ residual)
        if not np.isfinite(rss) or n <= parameters:
            scores.append(float("inf"))
            stable = False
        else:
            scores.append(
                float(
                    2 * parameters
                    + n
                    * np.log(
                        max(rss, np.finfo(np.float64).tiny)
                        / (n - parameters)
                    )
                )
            )
    return tuple(scores), stable
