from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.stagewise_ablation_reporting import (
    _moving_block_indices,
    holm_adjust,
    paired_moving_block_gain,
)


def _frame(prediction: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "entity_id": ["a"] * 3 + ["b"] * 3,
            "origin": [0, 1, 2, 0, 1, 2],
            "y_true": np.zeros(6),
            "y_pred": prediction,
        }
    )


def test_paired_moving_block_gain_is_deterministic_and_positive() -> None:
    earlier = _frame([2, 2, 2, 2, 2, 2])
    later = _frame([1, 1, 1, 1, 1, 1])
    first = paired_moving_block_gain(
        earlier, later, block_length=2, replicates=20, seed=7
    )
    second = paired_moving_block_gain(
        earlier, later, block_length=2, replicates=20, seed=7
    )
    assert first == second
    assert first["delta_rmse"] == 1.0
    assert first["ci95_low"] == 1.0
    assert first["ci95_high"] == 1.0


def test_vectorized_moving_block_indices_match_scalar_reference() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["a"] * 7 + ["b"] * 4,
            "origin": list(range(7)) + list(range(4)),
        }
    )

    def scalar_reference(block_length: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sampled: list[np.ndarray] = []
        for _, part in frame.groupby("entity_id", sort=False):
            indices = part.index.to_numpy(dtype=np.int64)
            count = len(indices)
            width = min(max(1, block_length), count)
            starts = np.arange(count - width + 1, dtype=np.int64)
            blocks = [
                indices[int(rng.choice(starts)) :][:width]
                for _ in range((count + width - 1) // width)
            ]
            sampled.append(np.concatenate(blocks)[:count])
        return np.concatenate(sampled)

    for block_length in (1, 2, 3, 10):
        for seed in (0, 7, 20260904):
            observed = _moving_block_indices(
                frame, block_length, np.random.default_rng(seed)
            )
            assert np.array_equal(observed, scalar_reference(block_length, seed))


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    observed = holm_adjust([0.04, 0.01, 0.03])
    assert observed == [0.06, 0.03, 0.06]
