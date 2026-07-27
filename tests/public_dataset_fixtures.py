from __future__ import annotations

import numpy as np

from ar_raphu.datasets.base import DynamicDataset


def make_public_dataset(
    *,
    split: np.ndarray | None = None,
    sequence_id: np.ndarray | None = None,
) -> DynamicDataset:
    n_time = 16
    x = np.column_stack(
        [
            np.arange(n_time, dtype=np.float64),
            np.arange(n_time, dtype=np.float64) ** 2,
        ]
    )
    y = (100.0 + np.arange(n_time, dtype=np.float64))[:, None]
    return DynamicDataset(
        x=x,
        y=y,
        timestamps=np.arange(n_time),
        sequence_id=(
            np.zeros(n_time, dtype=np.int64)
            if sequence_id is None
            else sequence_id
        ),
        split=(
            np.array(["train"] * 10 + ["validation"] * 3 + ["test"] * 3)
            if split is None
            else split
        ),
        label_mask=np.ones_like(y, dtype=bool),
        quality_mask=np.ones((n_time, 3), dtype=bool),
        feature_names=("x0", "x1"),
        target_names=("y",),
        metadata={
            "dataset_id": "unit_test",
            "official_split": True,
            "truth_available": False,
        },
    )
