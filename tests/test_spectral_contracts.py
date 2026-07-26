import pytest

from ar_raphu.spectral.contracts import ExperimentContract


def _contract(**overrides):
    values = {
        "scientific_question": "conditional X kernel",
        "target_contains_ar": True,
        "model_contains_ar": True,
        "target_contains_x": True,
        "model_contains_x": True,
        "truth_used_for_training": False,
        "truth_used_for_evaluation": True,
        "support_used_for_training": "all",
        "hyperparameter_selection_metric": "validation_prediction_loss_only",
        "rank_inputs_used_for_selection": False,
        "test_used_for_selection": False,
    }
    values.update(overrides)
    return ExperimentContract(**values)


def test_formal_contract_accepts_matched_target_and_model():
    _contract().validate()


def test_ar_free_model_rejected_for_ar_target():
    with pytest.raises(ValueError, match="AR-free"):
        _contract(model_contains_ar=False).validate()


def test_oracle_component_diagnostic_can_remove_ar_from_model():
    _contract(
        model_contains_ar=False,
        target_contains_ar=True,
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
        support_used_for_training="oracle",
    ).validate()


@pytest.mark.parametrize(
    "field", ["truth_used_for_training", "rank_inputs_used_for_selection", "test_used_for_selection"]
)
def test_forbidden_selection_inputs_fail(field):
    with pytest.raises(ValueError):
        _contract(**{field: True}).validate()
