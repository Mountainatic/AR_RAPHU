from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.tim_validation.e6_robustness.n1_gaussian import (
    perturb_gaussian_process_only,
)


def test_gaussian_levels_are_nested_and_target_missingness_is_preserved() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, 4.0],
            "target": [10.0, 11.0, 12.0, 13.0],
            "entity_id": ["a"] * 4,
            "row_in_entity": range(4),
        }
    )
    sigma = {"x": 2.0}
    low = perturb_gaussian_process_only(
        frame, task="T", seed=7, alpha=0.05, sigma=sigma
    )
    high = perturb_gaussian_process_only(
        frame, task="T", seed=7, alpha=0.10, sigma=sigma
    )
    finite = frame["x"].notna().to_numpy()
    low_delta = low.loc[finite, "x"].to_numpy() - frame.loc[finite, "x"].to_numpy()
    high_delta = high.loc[finite, "x"].to_numpy() - frame.loc[finite, "x"].to_numpy()
    assert np.allclose(high_delta, 2.0 * low_delta)
    assert low["x"].isna().tolist() == frame["x"].isna().tolist()
    assert low["target"].tolist() == frame["target"].tolist()


def test_alpha_zero_preserves_values_exactly() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "target": [3.0, 4.0]})
    observed = perturb_gaussian_process_only(
        frame, task="T", seed=1, alpha=0.0, sigma={"x": 1.0}
    )
    assert observed.equals(frame)
