import numpy as np

from ar_raphu.spectral.rank_bootstrap import (
    circular_block_indices,
    rank_interval,
)


def test_circular_blocks_are_reproducible_and_interval_is_bounded():
    first = circular_block_indices(
        100, block_length=16, rng=np.random.default_rng(340)
    )
    second = circular_block_indices(
        100, block_length=16, rng=np.random.default_rng(340)
    )
    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0 and first.max() < 100
    assert rank_interval(np.array([2] * 180 + [3] * 20)) == (2, 2, 3)


def test_bootstrap_contract_does_not_reselect_smoothing():
    from ar_raphu.spectral.contracts import ExperimentContract

    contract = ExperimentContract(
        scientific_question="Bootstrap effective rank stability.",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="single_oracle_variable",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        experiment_role="BOOTSTRAP_RANK",
        model_class="M2",
        resolution_role="STRUCTURAL",
        evaluation_distribution="SPACE",
        rank_budget_grid=(0.10, 0.05, 0.02),
        rank_max=12,
        smoothing_reselected=False,
    )
    assert contract.to_dict()["smoothing_reselected"] is False
