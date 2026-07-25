import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.model import ARRAPHURank1
from ar_raphu.phase1_evidence import (
    artifact_checksums,
    response_grid_arrays,
    partition_predictions_and_contributions,
    static_lag_kernels,
    support_frequency,
)
from ar_raphu.sequence_data import PreparedDirectForecastData


def build_model() -> ARRAPHURank1:
    return ARRAPHURank1(
        track="XAR",
        horizon=1,
        external_channels=3,
        inactive_external_channels=(),
        L_x=8,
        L_y=4,
        input_grid_ranges_x=[(-2.0, 2.0)] * 3,
        input_grid_range_y=(-3.0, 3.0),
        hidden_kan=4,
        grid_size=5,
    )


def test_evidence_extracts_normalized_separate_kernels_and_response_grids() -> None:
    model = build_model()
    external, ar = static_lag_kernels(model)
    assert external.shape == (3, 8)
    assert ar.shape == (1, 4)
    np.testing.assert_allclose(external.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(ar.sum(axis=1), 1.0, atol=1e-6)
    arrays = response_grid_arrays(
        model, [(-2.0, 2.0)] * 3, (-3.0, 3.0), points=17
    )
    assert arrays["x_grid_v0"].shape == (17,)
    assert arrays["x_response_v2"].shape == (17,)
    assert arrays["y_response"].shape == (17,)
    assert all(np.isfinite(value).all() for value in arrays.values())


def test_support_frequency_and_checksums_are_deterministic(tmp_path) -> None:
    rows = support_frequency([[0, 2], [0], [1, 2]], variables=3)
    assert rows[0]["selection_frequency"] == 2 / 3
    assert rows[1]["selection_count"] == 1
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    first = artifact_checksums(tmp_path)
    second = artifact_checksums(tmp_path)
    assert first == second
    assert list(first) == ["a.txt", "b.txt"]


def test_partition_evidence_contributions_close_prediction() -> None:
    model = build_model()
    rng = np.random.default_rng(5)
    x = rng.normal(size=(180, 3))
    y = rng.normal(size=180)
    data = PreparedDirectForecastData.from_sequence(
        x,
        y,
        track="XAR",
        horizon=1,
        L_x=8,
        L_y=4,
        split_target_intervals={
            "train": (20, 100),
            "validation": (100, 140),
            "test": (140, 180),
        },
    )
    predicted, observed, indices, components = (
        partition_predictions_and_contributions(
            model,
            data,
            "test",
            batch_size=13,
            device=torch.device("cpu"),
        )
    )
    assert observed.shape == predicted.shape == indices.shape == (40,)
    assert components.shape == (40, 4)
    np.testing.assert_allclose(
        predicted, float(model.bias.detach()) + components.sum(axis=1), atol=2e-6
    )
