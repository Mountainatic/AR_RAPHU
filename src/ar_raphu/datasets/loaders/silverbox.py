"""Silverbox official benchmark adapter."""

from __future__ import annotations

from pathlib import Path

import nonlinear_benchmarks as nb

from ..base import DynamicDataset
from .common import records_to_dynamic_dataset


def load_silverbox(
    raw_root: str | Path, *, include_test: bool = False
) -> DynamicDataset:
    train_record, test_records = nb.Silverbox(
        train_test_split=True,
        dir_placement=str(Path(raw_root)),
        atleast_2d=True,
    )
    test_names = (
        "test_multisine",
        "test_arrow_full",
        "test_arrow_no_extrapolation",
    )
    records = [(train_record, "train", "multisine_estimation")]
    if include_test:
        records.extend(
            (record, "test", name) for record, name in zip(test_records, test_names)
        )
    return records_to_dynamic_dataset(
        records,
        dataset_id="silverbox",
        source_metadata={
            "citation_doi": "10.23919/ECC.2013.6669201",
            "official_loader": "nonlinear-benchmarks.Silverbox",
            "test_state_initialization_window_length": 50,
            "test_records_exposed": include_test,
            "test_arrow_no_extrapolation_is_subset_of_test_arrow_full": True,
        },
        feature_names=("V1",),
        target_names=("V2",),
    )
