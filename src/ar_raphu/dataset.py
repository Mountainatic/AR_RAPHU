"""Leakage-guarded, lazy CZ direct-forecast datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_protocol import (
    SEQUENCE_LENGTH,
    DirectForecastSample,
    PartitionName,
    iter_direct_forecast_samples,
)
from .model import EXTERNAL_PROCESS_CHANNELS, Track
from .preprocessing import TrainOnlyStandardizer


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset_id: str
    privacy_namespace: Literal["PRIVATE_CZ"]
    track: Track
    horizon: int
    fold: int
    partition: PartitionName
    L_x: int
    L_y: int

    @property
    def cache_key(self) -> str:
        return (
            f"{self.privacy_namespace}__{self.dataset_id}__track-{self.track}"
            f"__h-{self.horizon}__fold-{self.fold}"
            f"__partition-{self.partition}__Lx-{self.L_x}__Ly-{self.L_y}"
        )


class CZDirectForecastDataset(Dataset):
    """Create windows lazily; split membership is determined by target index."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        track: Track,
        fold_number: int,
        partition: PartitionName,
        L_x: int,
        L_y: int,
        horizon: int,
        scaler: TrainOnlyStandardizer,
        protocol_frozen: bool = False,
        dataset_id: str = "cz_private_experiment_data_1",
    ) -> None:
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        if x_array.shape != (SEQUENCE_LENGTH, EXTERNAL_PROCESS_CHANNELS):
            raise ValueError(
                f"x must have shape [{SEQUENCE_LENGTH}, "
                f"{EXTERNAL_PROCESS_CHANNELS}]."
            )
        if y_array.shape != (SEQUENCE_LENGTH,):
            raise ValueError(f"y must have shape [{SEQUENCE_LENGTH}].")
        if not np.issubdtype(x_array.dtype, np.number) or not np.issubdtype(
            y_array.dtype, np.number
        ):
            raise ValueError("CZ source arrays must be numeric.")
        if track not in {"X", "AR", "XAR"}:
            raise ValueError("track must be one of 'X', 'AR', or 'XAR'.")
        scaler._validate_fold(fold_number)

        samples = tuple(
            iter_direct_forecast_samples(
                fold_number,
                partition,
                L_x=L_x,
                L_y=L_y,
                horizon=horizon,
                protocol_frozen=protocol_frozen,
            )
        )
        if not samples:
            raise ValueError("The requested dataset contains no forecast samples.")

        target_indices = np.fromiter(
            (sample.target for sample in samples),
            dtype=np.int64,
            count=len(samples),
        )
        if not np.isfinite(y_array[target_indices]).all():
            raise ValueError("Accessible target values contain non-finite values.")
        if track in {"X", "XAR"}:
            x_start = min(sample.x_start for sample in samples)
            x_stop = max(sample.x_stop for sample in samples)
            if not np.isfinite(x_array[x_start:x_stop]).all():
                raise ValueError(
                    "Accessible external-history values contain non-finite values."
                )
        if track in {"AR", "XAR"}:
            y_start = min(sample.y_start for sample in samples)
            y_stop = max(sample.y_stop for sample in samples)
            if not np.isfinite(y_array[y_start:y_stop]).all():
                raise ValueError(
                    "Accessible output-history values contain non-finite values."
                )

        self.x = x_array
        self.y = y_array
        self.scaler = scaler
        self.identity = DatasetIdentity(
            dataset_id=dataset_id,
            privacy_namespace="PRIVATE_CZ",
            track=track,
            horizon=horizon,
            fold=fold_number,
            partition=partition,
            L_x=L_x,
            L_y=L_y,
        )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def sample_indices(self, index: int) -> DirectForecastSample:
        return self.samples[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        item: dict[str, torch.Tensor] = {
            "target": torch.as_tensor(
                self.scaler.transform_y(
                    self.y[sample.target], fold_number=sample.fold
                ),
                dtype=torch.float32,
            ),
            "origin_index": torch.tensor(sample.origin, dtype=torch.int64),
            "target_index": torch.tensor(sample.target, dtype=torch.int64),
        }
        if self.identity.track in {"X", "XAR"}:
            x_window = self.x[sample.x_start : sample.x_stop]
            x_window = self.scaler.transform_x(
                x_window, fold_number=sample.fold
            )
            item["x_window"] = torch.as_tensor(
                np.ascontiguousarray(x_window.T), dtype=torch.float32
            )
        if self.identity.track in {"AR", "XAR"}:
            y_window = self.scaler.transform_y(
                self.y[sample.y_start : sample.y_stop],
                fold_number=sample.fold,
            )
            item["y_window"] = torch.as_tensor(
                np.ascontiguousarray(y_window), dtype=torch.float32
            )
        return item
