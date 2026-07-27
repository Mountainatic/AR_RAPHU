"""Frozen PB1 dataset registry with explicit source and test-access semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .base import DynamicDataset
from .loaders import (
    load_cascaded_tanks,
    load_pwh,
    load_silverbox,
    load_whpn,
)


@dataclass(frozen=True, slots=True)
class DatasetRegistration:
    dataset_id: str
    loader: Callable[..., DynamicDataset]
    official_doi: str
    license_status: str
    test_access: str = "once_after_protocol_freeze"

    def load(
        self, raw_root: str | Path, *, include_test: bool = False
    ) -> DynamicDataset:
        return self.loader(raw_root, include_test=include_test)


PB1_DATASETS = {
    "pwh": DatasetRegistration(
        "pwh", load_pwh, "10.4121/12950081.v1", "CC BY-SA 4.0"
    ),
    "whpn": DatasetRegistration(
        "whpn", load_whpn, "10.4121/12952124.v2", "CC BY-SA 4.0"
    ),
    "cascaded_tanks": DatasetRegistration(
        "cascaded_tanks",
        load_cascaded_tanks,
        "10.4121/12960104.v1",
        "CC BY-SA 4.0",
    ),
    "silverbox": DatasetRegistration(
        "silverbox",
        load_silverbox,
        "10.23919/ECC.2013.6669201",
        "NOT_STATED_ON_SOURCE",
    ),
}


def get_public_dataset(dataset_id: str) -> DatasetRegistration:
    try:
        return PB1_DATASETS[dataset_id]
    except KeyError as error:
        raise ValueError(f"Unknown PB1 dataset {dataset_id!r}.") from error
