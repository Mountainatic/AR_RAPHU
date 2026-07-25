import pytest

from ar_raphu.model_selection import validation_one_se_select


def test_one_se_uses_minimum_configuration_se_and_complexity_order() -> None:
    configs = ["g8_l0", "g8_l1", "g12_l0"]
    losses = {
        "g8_l0": [1.00, 1.10, 0.90],
        "g8_l1": [1.01, 1.11, 0.91],
        "g12_l0": [0.99, 1.09, 0.89],
    }
    rows = [
        {
            "config_id": config_id,
            "unit_id": f"seed_{seed}",
            "validation_loss": loss,
        }
        for config_id, values in losses.items()
        for seed, loss in enumerate(values)
    ]
    complexity = {
        "g8_l0": (8, 0),
        "g8_l1": (8, -1),
        "g12_l0": (12, 0),
    }
    selected = validation_one_se_select(
        rows,
        declared_config_order=configs,
        complexity_key=complexity.__getitem__,
    )
    assert selected["minimum_config_id"] == "g12_l0"
    assert selected["selected_config_id"] == "g8_l1"
    assert selected["test_used"] is False


def test_one_se_rejects_inconsistent_units() -> None:
    rows = [
        {"config_id": "a", "unit_id": "0", "validation_loss": 1.0},
        {"config_id": "a", "unit_id": "1", "validation_loss": 1.1},
        {"config_id": "b", "unit_id": "0", "validation_loss": 0.9},
    ]
    with pytest.raises(ValueError, match="same selection units"):
        validation_one_se_select(
            rows,
            declared_config_order=["a", "b"],
            complexity_key=lambda config_id: (config_id,),
        )
