"""Parallel Wiener-Hammerstein official benchmark adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..base import DynamicDataset
from .common import array_record, records_to_dynamic_dataset


def load_pwh(
    raw_root: str | Path, *, include_test: bool = False
) -> DynamicDataset:
    path = Path(raw_root) / "ParWH" / "ParWHFiles" / "ParWHData.mat"
    variables = ["uEst", "yEst", "fs"]
    if include_test:
        variables.extend(("uVal", "yVal"))
    source = loadmat(path, variable_names=variables)
    fs = float(np.asarray(source["fs"]).squeeze())
    u_est = source["uEst"].reshape((16384 * 2, 20, 5))
    y_est = source["yEst"].reshape((16384 * 2, 20, 5))
    records = []
    for phase in range(20):
        for amplitude in range(5):
            name = f"Est-phase-{phase}-amp-{amplitude}"
            records.append(
                (
                    array_record(
                        u=u_est[:, phase, amplitude],
                        y=y_est[:, phase, amplitude],
                        name=name,
                        sampling_time=1.0 / fs,
                    ),
                    "train",
                    name,
                )
            )
    if include_test:
        u_val = source["uVal"].reshape((16384 * 2, 1, 5))
        y_val = source["yVal"].reshape((16384 * 2, 1, 5))
        for amplitude in range(5):
            name = f"Val-amp-{amplitude}"
            records.append(
                (
                    array_record(
                        u=u_val[:, 0, amplitude],
                        y=y_val[:, 0, amplitude],
                        name=name,
                        sampling_time=1.0 / fs,
                    ),
                    "test",
                    name,
                )
            )
    return records_to_dynamic_dataset(
        records,
        dataset_id="pwh",
        source_metadata={
            "doi": "10.4121/12950081.v1",
            "loader_semantics_reference": "nonlinear-benchmarks.ParWH@1.0.1",
            "test_records_exposed": include_test,
        },
    )
