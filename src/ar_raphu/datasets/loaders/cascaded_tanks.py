"""Cascaded Tanks official benchmark adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..base import DynamicDataset
from .common import array_record, records_to_dynamic_dataset


def load_cascaded_tanks(
    raw_root: str | Path, *, include_test: bool = False
) -> DynamicDataset:
    path = (
        Path(raw_root)
        / "Cascaded_Tanks"
        / "CascadedTanksFiles"
        / "dataBenchmark.mat"
    )
    variables = ["uEst", "yEst", "Ts"]
    if include_test:
        variables.extend(("uVal", "yVal"))
    source = loadmat(path, variable_names=variables)
    sampling_time = float(np.asarray(source["Ts"]).squeeze())
    records = [
        (
            array_record(
                u=np.asarray(source["uEst"]).squeeze(),
                y=np.asarray(source["yEst"]).squeeze(),
                name="estimation",
                sampling_time=sampling_time,
            ),
            "train",
            "estimation",
        )
    ]
    if include_test:
        records.append(
            (
                array_record(
                    u=np.asarray(source["uVal"]).squeeze(),
                    y=np.asarray(source["yVal"]).squeeze(),
                    name="official_test",
                    sampling_time=sampling_time,
                ),
                "test",
                "official_test",
            )
        )
    return records_to_dynamic_dataset(
        records,
        dataset_id="cascaded_tanks",
        source_metadata={
            "doi": "10.4121/12960104.v1",
            "official_loader": "nonlinear-benchmarks.Cascaded_Tanks",
            "test_state_initialization_window_length": 50,
            "test_records_exposed": include_test,
            "overflow_mask_status": "NOT_YET_DEFINED",
            "overflow_mask_reason": (
                "Official benchmark describes overflow but gives no numeric "
                "sample-level threshold."
            ),
        },
        feature_names=("pump_input",),
        target_names=("lower_tank_level",),
    )
