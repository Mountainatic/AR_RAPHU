from pathlib import Path

import pytest
import json

from ar_raphu.spectral.contracts import ExperimentContract
from ar_raphu.spectral.resolution_roles import role_from_config


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads(
        (ROOT / "configs" / "spectral_v033.yaml").read_text(encoding="utf-8")
    )


def test_resolution_roles_are_frozen_and_cannot_be_overridden():
    config = _config()
    assert role_from_config(config, "PREDICTIVE").lag_count == 32
    assert role_from_config(config, "STRUCTURAL").lag_count == 48
    assert role_from_config(config, "MOTHER").lag_type == "discrete_identity"
    config["e1b"]["predictive_role"]["lag_count"] = 40
    with pytest.raises(ValueError, match="cannot be overridden"):
        role_from_config(config, "PREDICTIVE")


def test_e1b_contract_truth_basis_selection_is_explicitly_allowed():
    contract = ExperimentContract(
        scientific_question="Certify frozen resolution roles.",
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
        experiment_role="ORACLE_REPRESENTATION_DIAGNOSTIC",
        evaluation_distribution="ORACLE_GRID",
        basis_selection_uses_truth=True,
        smoothing_selection_metric="validation_prediction_mse",
    )
    payload = contract.to_dict()
    assert payload["basis_selection_uses_truth"] is True
    assert payload["evaluation_distribution"] == "ORACLE_GRID"
