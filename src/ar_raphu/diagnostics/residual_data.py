"""Causally aligned external-only data for frozen-AR innovation residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from ar_raphu.sequence_data import PreparedDirectForecastData


@dataclass(slots=True)
class PreparedExternalResidualData:
    """Reuse the original X scaling and replace only the scaled target."""

    base: PreparedDirectForecastData
    residual_scaled: np.ndarray

    @classmethod
    def from_ar_predictions(
        cls,
        base: PreparedDirectForecastData,
        predictions: dict[str, np.ndarray],
    ) -> "PreparedExternalResidualData":
        if base.track not in {"XAR", "AR"}:
            raise ValueError("Residual data requires a base task containing AR.")
        residual = np.full_like(base.y_scaled, np.nan, dtype=np.float32)
        for partition in ("train", "validation", "test"):
            start, stop = base.split_target_intervals[partition]
            values = np.asarray(predictions[partition], dtype=np.float32)
            if values.shape != (stop - start,):
                raise ValueError(
                    f"{partition} predictions must have shape {(stop - start,)}."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{partition} predictions contain non-finite values.")
            residual[start:stop] = base.y_scaled[start:stop] - values
        return cls(base=base, residual_scaled=residual)

    @property
    def track(self) -> str:
        return "X"

    @property
    def horizon(self) -> int:
        return self.base.horizon

    @property
    def L_x(self) -> int:
        return self.base.L_x

    @property
    def L_y(self) -> int:
        return self.base.L_y

    @property
    def scaler(self):
        return self.base.scaler

    @property
    def split_target_intervals(self) -> dict[str, tuple[int, int]]:
        return self.base.split_target_intervals

    @property
    def x_scaled(self) -> np.ndarray:
        return self.base.x_scaled

    def target_count(self, partition: str) -> int:
        return self.base.target_count(partition)

    def iter_contiguous_batches(
        self,
        partition: str,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Iterator[dict[str, torch.Tensor]]:
        for batch in self.base.iter_contiguous_batches(
            partition, batch_size=batch_size, device=device
        ):
            indices = batch["target_index"].detach().cpu().numpy()
            yield {
                "target": torch.as_tensor(
                    self.residual_scaled[indices], device=device
                ),
                "target_index": batch["target_index"],
                "origin_index": batch["origin_index"],
                "x_sequence": batch["x_sequence"],
            }

    def iter_batches(
        self,
        partition: str,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Iterator[dict[str, torch.Tensor]]:
        for batch in self.base.iter_batches(
            partition, batch_size=batch_size, device=device
        ):
            indices = batch["target_index"].detach().cpu().numpy()
            yield {
                "target": torch.as_tensor(
                    self.residual_scaled[indices], device=device
                ),
                "target_index": batch["target_index"],
                "origin_index": batch["origin_index"],
                "x_window": batch["x_window"],
            }
