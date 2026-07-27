from __future__ import annotations

import numpy as np

from ar_raphu.baselines.mlp_narx_champneys2024 import (
    MLPWeights,
    history_design,
    simulate_mlp_narx,
)


def test_history_design_uses_no_future_input() -> None:
    x = np.arange(10, dtype=np.float64)
    y = 10.0 + x
    design, target = history_design(x, y, nx=2, ny=2)
    np.testing.assert_array_equal(design[0], [11.0, 10.0, 1.0, 0.0])
    assert target[0] == 12.0


def test_mlp_free_run_alignment_matches_linearized_network() -> None:
    x = np.linspace(-0.1, 0.1, 20)
    y = np.zeros(20)
    weights = MLPWeights(
        hidden_weight=np.array([[0.0, 1.0]]),
        hidden_bias=np.zeros(1),
        output_weight=np.ones(1),
        output_bias=0.0,
    )
    prediction, burn = simulate_mlp_narx(
        x, y, nx=1, ny=1, weights=weights
    )
    np.testing.assert_allclose(prediction[burn:], np.tanh(x[:-1]))
