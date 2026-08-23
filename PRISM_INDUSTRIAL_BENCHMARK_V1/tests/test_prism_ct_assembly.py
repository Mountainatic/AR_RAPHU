import numpy as np

from prism_benchmark.prism_ct_assembly import (
    fit_simplex_assembly,
    predict_simplex_assembly,
    project_probability_simplex,
)


def test_simplex_projection_is_nonnegative_and_normalized():
    projected = project_probability_simplex(np.array([2.0, -1.0, 0.5]))
    assert np.all(projected >= -1e-12)
    assert np.isclose(projected.sum(), 1.0)


def test_simplex_assembly_recovers_good_branch():
    target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    branches = {
        "good": target.copy(),
        "bad": -target,
    }
    state = fit_simplex_assembly(branches, target, ridge=1e-6)
    assert np.all(state.weights >= -1e-12)
    assert state.persistence_weight >= -1e-12
    assert np.isclose(state.weights.sum() + state.persistence_weight, 1.0)
    prediction = predict_simplex_assembly(state, branches)
    assert np.mean((target - prediction) ** 2) < 1e-8


def test_persistence_anchor_can_take_all_weight():
    target = np.zeros(10)
    branches = {
        "positive": np.ones(10),
        "negative": -np.ones(10),
    }
    state = fit_simplex_assembly(branches, target, ridge=1e-3)
    prediction = predict_simplex_assembly(state, branches)
    assert np.mean(prediction ** 2) < 1e-10
    assert np.isclose(state.weights.sum() + state.persistence_weight, 1.0)
