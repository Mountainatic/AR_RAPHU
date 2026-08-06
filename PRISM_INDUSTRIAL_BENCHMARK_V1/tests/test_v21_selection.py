import numpy as np

from prism_benchmark.v21_selection import (
    assert_final_prediction_contract,
    guarded_local_one_se_select,
)


def test_guarded_local_one_se_keeps_stable_active_when_neutral_is_acceptable():
    losses = {
        "ZERO": [1.00, 1.01, 0.99, 1.00],
        "ACTIVE": [0.86, 0.99, 0.98, 1.05],
    }
    selected = guarded_local_one_se_select(
        losses,
        lambda value: (0 if value == "ZERO" else 1,),
        neutral="ZERO",
        minimum_relative_improvement=0.01,
        minimum_positive_fraction=0.75,
    )
    assert selected.final_selected_candidate == "ACTIVE"
    assert "ZERO" in selected.acceptable_candidates


def test_guarded_local_one_se_returns_neutral_without_practical_gain():
    selected = guarded_local_one_se_select(
        {"ZERO": [1.0, 1.0, 1.0, 1.0], "ACTIVE": [0.999, 0.999, 0.999, 0.999]},
        lambda value: (0 if value == "ZERO" else 1,),
        neutral="ZERO",
    )
    assert selected.final_selected_candidate == "ZERO"


def test_final_prediction_contract_requires_same_materialized_selection():
    result = {
        "final_selected_candidate": "ACTIVE",
        "final_selected_fold_losses": [1.0, 0.9, 0.8],
        "final_selected_prediction_path": "validation.parquet",
        "final_selected_contract": {"family": "ACTIVE"},
        "final_prediction_loss": 0.75,
    }
    assert_final_prediction_contract(result, recomputed_loss=0.75)
    with np.testing.assert_raises(RuntimeError):
        assert_final_prediction_contract(result, recomputed_loss=0.8)
