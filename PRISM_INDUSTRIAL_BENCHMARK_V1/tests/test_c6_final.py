from __future__ import annotations

import numpy as np

from prism_benchmark.c6_final import _holm, _paired_bootstrap


def test_paired_bootstrap_preserves_positive_difference() -> None:
    diff = np.ones(100, dtype=np.float64)
    entities = np.array(["a"] * 50 + ["b"] * 50)
    draws = _paired_bootstrap(diff, entities, 10, 50, 7)
    np.testing.assert_allclose(draws, 1.0)


def test_holm_is_monotone_and_bounded() -> None:
    rows = [{"p_value": 0.01}, {"p_value": 0.03}, {"p_value": 0.2}]
    _holm(rows, 0.05)
    adjusted = [row["holm_adjusted_p"] for row in rows]
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
