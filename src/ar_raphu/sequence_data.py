"""Generic chronological direct-forecast data prepared from synthetic/public sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import torch

from .model import Track


Partition = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class SequenceStandardizer:
    fit_stop: int
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float

    @classmethod
    def fit(
        cls, x: np.ndarray, y: np.ndarray, *, train_target_stop: int
    ) -> "SequenceStandardizer":
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Expected x [time, variables] and y [time].")
        if not 0 < train_target_stop <= len(y):
            raise ValueError("Invalid train_target_stop.")
        train_x = x[:train_target_stop].astype(np.float64)
        train_y = y[:train_target_stop].astype(np.float64)
        if not np.isfinite(train_x).all() or not np.isfinite(train_y).all():
            raise ValueError("Training prefix contains non-finite values.")
        x_mean = train_x.mean(axis=0)
        x_scale = train_x.std(axis=0, ddof=0)
        x_scale[x_scale == 0.0] = 1.0
        y_mean = float(train_y.mean())
        y_scale = float(train_y.std(ddof=0))
        if y_scale == 0.0:
            y_scale = 1.0
        return cls(
            fit_stop=train_target_stop,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
        )

    def transform_x(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.x_mean) / self.x_scale

    def transform_y(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.y_mean) / self.y_scale

    def inverse_y(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.y_scale + self.y_mean

    def input_grid_ranges(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pad_fraction: float = 0.05,
        minimum_pad: float = 0.05,
    ) -> tuple[list[tuple[float, float]], tuple[float, float]]:
        train_x = self.transform_x(np.asarray(x)[: self.fit_stop])
        train_y = self.transform_y(np.asarray(y)[: self.fit_stop])

        def padded(values: np.ndarray) -> tuple[float, float]:
            lower, upper = float(values.min()), float(values.max())
            pad = max((upper - lower) * pad_fraction, minimum_pad)
            return lower - pad, upper + pad

        return (
            [padded(train_x[:, variable]) for variable in range(train_x.shape[1])],
            padded(train_y),
        )


@dataclass(slots=True)
class PreparedDirectForecastData:
    track: Track
    horizon: int
    L_x: int
    L_y: int
    scaler: SequenceStandardizer
    split_target_intervals: dict[str, tuple[int, int]]
    x_scaled: np.ndarray
    y_scaled: np.ndarray
    x_window_view: np.ndarray | None
    y_window_view: np.ndarray | None

    @classmethod
    def from_sequence(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        track: Track,
        horizon: int,
        L_x: int,
        L_y: int,
        split_target_intervals: dict[str, tuple[int, int]],
    ) -> "PreparedDirectForecastData":
        if track not in {"X", "AR", "XAR"}:
            raise ValueError("Unknown track.")
        train_stop = split_target_intervals["train"][1]
        scaler = SequenceStandardizer.fit(
            x, y, train_target_stop=train_stop
        )
        x_scaled = scaler.transform_x(x).astype(np.float32)
        y_scaled = scaler.transform_y(y).astype(np.float32)
        x_view = (
            np.lib.stride_tricks.sliding_window_view(
                x_scaled, L_x, axis=0
            )
            if track in {"X", "XAR"}
            else None
        )
        y_view = (
            np.lib.stride_tricks.sliding_window_view(y_scaled, L_y)
            if track in {"AR", "XAR"}
            else None
        )
        prepared = cls(
            track=track,
            horizon=horizon,
            L_x=L_x,
            L_y=L_y,
            scaler=scaler,
            split_target_intervals=split_target_intervals,
            x_scaled=x_scaled,
            y_scaled=y_scaled,
            x_window_view=x_view,
            y_window_view=y_view,
        )
        for partition in ("train", "validation", "test"):
            prepared._validate_partition(partition)
        return prepared

    def _validate_partition(self, partition: Partition) -> None:
        start, stop = self.split_target_intervals[partition]
        if not 0 <= start < stop <= len(self.y_scaled):
            raise ValueError(f"Invalid {partition} target interval.")
        earliest_origin = start - self.horizon
        if earliest_origin - self.L_x + 1 < 0:
            raise ValueError("External history precedes the sequence.")
        if earliest_origin - self.L_y + 1 < 0:
            raise ValueError("Output history precedes the sequence.")

    def target_count(self, partition: Partition) -> int:
        start, stop = self.split_target_intervals[partition]
        return stop - start

    def iter_batches(
        self,
        partition: Partition,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Iterator[dict[str, torch.Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        target_start, target_stop = self.split_target_intervals[partition]
        for batch_start in range(target_start, target_stop, batch_size):
            batch_stop = min(batch_start + batch_size, target_stop)
            targets = np.arange(batch_start, batch_stop, dtype=np.int64)
            origins = targets - self.horizon
            batch: dict[str, torch.Tensor] = {
                "target": torch.as_tensor(
                    self.y_scaled[targets], device=device
                ),
                "target_index": torch.as_tensor(targets, device=device),
                "origin_index": torch.as_tensor(origins, device=device),
            }
            if self.x_window_view is not None:
                starts = origins - self.L_x + 1
                x_window = np.ascontiguousarray(self.x_window_view[starts])
                batch["x_window"] = torch.as_tensor(x_window, device=device)
            if self.y_window_view is not None:
                starts = origins - self.L_y + 1
                y_window = np.ascontiguousarray(self.y_window_view[starts])
                batch["y_window"] = torch.as_tensor(y_window, device=device)
            yield batch

    def iter_contiguous_batches(
        self,
        partition: Partition,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield unique raw sequences for exact V20 sequence-first execution."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        target_start, target_stop = self.split_target_intervals[partition]
        for batch_start in range(target_start, target_stop, batch_size):
            batch_stop = min(batch_start + batch_size, target_stop)
            targets = np.arange(batch_start, batch_stop, dtype=np.int64)
            first_origin = batch_start - self.horizon
            last_origin = batch_stop - 1 - self.horizon
            batch: dict[str, torch.Tensor] = {
                "target": torch.as_tensor(
                    self.y_scaled[targets], device=device
                ),
                "target_index": torch.as_tensor(targets, device=device),
                "origin_index": torch.as_tensor(
                    targets - self.horizon, device=device
                ),
            }
            if self.x_window_view is not None:
                raw = self.x_scaled[
                    first_origin - self.L_x + 1 : last_origin + 1
                ]
                batch["x_sequence"] = torch.as_tensor(
                    np.ascontiguousarray(raw.T), device=device
                )
            if self.y_window_view is not None:
                raw = self.y_scaled[
                    first_origin - self.L_y + 1 : last_origin + 1
                ]
                batch["y_sequence"] = torch.as_tensor(
                    np.ascontiguousarray(raw), device=device
                )
            yield batch
