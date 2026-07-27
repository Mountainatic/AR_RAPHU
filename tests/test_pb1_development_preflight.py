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
        "task": {
            "xar_history_selection": {
                "H1_baseline_faithful": {},
                "H2_native_model": {},
                "H3_shared_history_fairness": {},
            }
        },
        "models": [
            "arx_qr",
            "mlpnarx_champneys2024",
            "rank1_ar_raphu",
            "fixed_rank2_ar_raphu",
            "full_spectral_ar_raphu",
            "adaptive_spectral_ar_raphu",
        ],
        "literature_profiles": {
            "path": "profiles.json",
            "paper_profile": "CHAMPNEYS2024_PEER_REVIEWED",
            "companion_code_commit": "d9c4972",
            "use_companion_internal_split": False,
        },
        "baselines": {
            "arx_primary": {
                "solver": "PIVOTED_QR_OR_SVD_MINIMUM_NORM",
                "scientific_ridge": 0.0,
            },
            "mlpnarx_primary": {
                "profile": "MLPNARX_CHAMPNEYS2024",
                "hidden_layers": 1,
                "activation": "tanh",
                "widths": [2, 5, 7, 10],
                "optimizer": "Adam",
                "learning_rate": 0.01,
                "iterations": 20000,
                "early_stopping": False,
                "initializations": 5,
                "history": "ARX_AIC_SELECTED",
                "scaling": "TRAIN_ONLY_MINMAX_MINUS1_PLUS1",
                "selection": "VALIDATION_AIC",
            },
        },
        "selection": {
            "spectral_penalty": {
                "normalization": (
                    "POSITIVE_GENERALIZED_EIGENVALUE_MEDIAN_RELATIVE_TO_TRAIN_GRAM"
                ),
                "interval": "AUTOMATIC_SHRINKAGE_COVERAGE",
                "near_unpenalized_shrinkage": 0.999,
                "near_zero_shrinkage": 0.001,
                "grid_points_per_axis": 7,
                "boundary_expansions_max": 2,
                "risk": "GROUPED",
                "one_se_tie": "LOWEST_EFFECTIVE_DF",
            }
        },
        "bootstrap": {
            "primary_unit": "WHOLE_RECORD",
            "within_record_block": "TRIGGERED_ONLY",
            "development_replicates": 250,
            "confirmation_replicates": 1000,
            "shared_fixed_block_length": None,
        },
        "solver": {"numerical_jitter_is_scientific_ridge": False},
    }


def test_complete_preflight_is_ready() -> None:
    config = _complete_config()
    assert development_preregistration_gaps(config) == []
    assert development_preflight_status(config) == "READY_FOR_DEVELOPMENT"


def test_missing_material_choices_are_machine_visible() -> None:
    config = _complete_config()
    del config["selection"]["spectral_penalty"]["risk"]
    config["baselines"]["mlpnarx_primary"]["early_stopping"] = True
    gaps = development_preregistration_gaps(config)
    assert "selection.spectral_penalty.risk" in gaps
    assert "baselines.mlpnarx_primary.early_stopping" in gaps
    assert development_preflight_status(config).startswith("BLOCKED")


def test_whpn_huber_is_appendix_only_and_does_not_block_primary() -> None:
    config = copy.deepcopy(_complete_config())
    config["dataset"]["id"] = "whpn"
    config["dataset"]["preserve_process_noise"] = True
    config["selection"]["primary_loss"] = "MSE"
    config["selection"]["loss_sensitivity"] = {
        "role": "APPENDIX_ONLY",
        "participates_in_primary_selection": False,
    }
    assert development_preregistration_gaps(config) == []
