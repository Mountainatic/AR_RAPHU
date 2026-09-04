from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.stagewise_ablation_reporting import (
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


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    observed = holm_adjust([0.04, 0.01, 0.03])
    assert observed == [0.06, 0.03, 0.06]
