from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from prism_benchmark.cz_authority_pure_k_checkpoint import (
    PURE_K_CODEC,
    predict_pure_k_from_compressed,
    pure_k_contract_from_c_result,
    verify_pure_k_checkpoint_reload,
)
from prism_benchmark.portable_checkpoints import write_portable_checkpoint


def test_pure_k_contract_uses_frozen_best_active_channel() -> None:
    result = {
        "active_channels": ["joint_lift"],
        "best_active_k_channel": "joint_lift",
        "selected_family": "BEST_ACTIVE_K_CHANNEL",
    }
    contract = pure_k_contract_from_c_result(
        result, ["main_heater_power", "joint_lift"]
    )
    assert contract == {
        "family": "BEST_ACTIVE_K",
        "channel": "joint_lift",
        "intercept": 0.0,
        "coefficient": [1.0],
        "parameter_count": 1,
    }


def test_pure_k_prediction_selects_only_registered_channel() -> None:
    compressed = np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64)
    observed = predict_pure_k_from_compressed(
        compressed,
        {"family": "BEST_ACTIVE_K", "channel": "joint_lift"},
        ["main_heater_power", "joint_lift"],
    )
    np.testing.assert_array_equal(observed, np.asarray([2.0, 5.0]))


def test_pure_k_rejects_unfrozen_channel() -> None:
    with pytest.raises(RuntimeError, match="BEST_ACTIVE_CHANNEL_NOT_FROZEN"):
        pure_k_contract_from_c_result(
            {
                "active_channels": ["joint_lift"],
                "best_active_k_channel": "main_heater_power",
            },
            ["main_heater_power", "joint_lift"],
        )


def test_pure_k_exact_zero_requires_frozen_intercept() -> None:
    result = {"active_channels": [], "selected_family": "K_EXACT_ZERO"}
    contract = pure_k_contract_from_c_result(
        result,
        [],
        {"family": "K_EXACT_ZERO", "intercept": 0.125},
    )
    assert contract["intercept"] == 0.125
    with pytest.raises(RuntimeError, match="EXACT_ZERO_INTERCEPT_NOT_FROZEN"):
        pure_k_contract_from_c_result(result, [])


def test_pure_k_checkpoint_reload_is_exact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pure-k"
    compressed = np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64)
    contract = {"family": "BEST_ACTIVE_K", "channel": "joint_lift"}
    prediction = predict_pure_k_from_compressed(
        compressed, contract, ["main_heater_power", "joint_lift"]
    )
    write_portable_checkpoint(
        checkpoint,
        {
            "codec": PURE_K_CODEC,
            "refit_performed": False,
            "physical": {"channels": ["main_heater_power", "joint_lift"]},
            "k_contract": contract,
            "reload_prediction_tolerance": 1e-10,
        },
        {"compressed": compressed, "pure_k_prediction": prediction},
    )
    previous = os.environ.get("PRISM_FORMAL_INFERENCE_ONLY")
    try:
        audit = verify_pure_k_checkpoint_reload(checkpoint)
    finally:
        if previous is None:
            os.environ.pop("PRISM_FORMAL_INFERENCE_ONLY", None)
        else:
            os.environ["PRISM_FORMAL_INFERENCE_ONLY"] = previous
    assert audit["status"] == "PASS"
    assert audit["maximum_absolute_prediction_error"] == 0.0
