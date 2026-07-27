from __future__ import annotations

import copy

from ar_raphu.datasets.pb1_preflight import (
    development_preregistration_gaps,
    development_preflight_status,
)


def _complete_config() -> dict:
    return {
        "dataset": {
            "id": "pwh",
            "development_split": {"status": "FROZEN"},
        },
        "task": {"xar_history_selection": "predeclared"},
        "models": [
            "ar_ridge",
            "arx_ridge",
            "narx_mlp",
            "rank1_ar_raphu",
            "fixed_rank2_ar_raphu",
            "full_spectral_ar_raphu",
            "adaptive_spectral_ar_raphu",
        ],
        "baselines": {
            "ridge_weight_grid": [0.0, 0.1],
            "narx_mlp": {
                "hidden_width_grid": [8],
                "depth_grid": [2],
                "learning_rate": 0.001,
                "batch_size": 64,
                "max_epochs": 10,
                "early_stopping_patience": 3,
                "seeds": [0],
            },
        },
        "selection": {
            "regularization_grid": {
                "lag_smoothness": [0.001],
                "amplitude_smoothness": [0.001],
                "ridge_weight": [1e-08],
            }
        },
        "bootstrap": {
            "block_length": 64,
            "development_replicates": 100,
            "seed": 0,
        },
    }


def test_complete_preflight_is_ready() -> None:
    config = _complete_config()
    assert development_preregistration_gaps(config) == []
    assert development_preflight_status(config) == "READY_FOR_DEVELOPMENT"


def test_missing_material_choices_are_machine_visible() -> None:
    config = _complete_config()
    del config["selection"]["regularization_grid"]["lag_smoothness"]
    del config["baselines"]["narx_mlp"]["seeds"]
    gaps = development_preregistration_gaps(config)
    assert "selection.regularization_grid.lag_smoothness" in gaps
    assert "baselines.narx_mlp.seeds" in gaps
    assert development_preflight_status(config).startswith("BLOCKED")


def test_whpn_huber_delta_is_required() -> None:
    config = copy.deepcopy(_complete_config())
    config["dataset"]["id"] = "whpn"
    config["selection"]["loss_sensitivity"] = ["mse", "huber"]
    assert "selection.huber_delta" in development_preregistration_gaps(config)
