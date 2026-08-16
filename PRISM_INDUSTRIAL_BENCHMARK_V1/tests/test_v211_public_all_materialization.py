from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.cpu_data import HeadSpec, ViewSpec
from prism_benchmark.v211_joint import J_KA
from prism_benchmark.v211_joint_stability_config import (
    CHANNEL_COMPRESSED,
    FULL_BASIS,
    JOINT_ESTIMATOR_SEMANTICS,
)
from prism_benchmark.v211_public_all_materialization import (
    _fit_frozen_joint_candidate,
    _joint_evaluation_k_block,
    _validate_joint_materialization_contract,
    preflight_public_all_materialization,
)


def _contract(representation: str = CHANNEL_COMPRESSED) -> dict[str, object]:
    return {
        "family": J_KA,
        "joint_estimator_semantics": JOINT_ESTIMATOR_SEMANTICS,
        "k_representation": representation,
        "numerical_alpha": 0.0,
        "predictive_eta": 1.0,
        "raw_k_support": ["k1", "k2"],
    }


def _view() -> ViewSpec:
    return ViewSpec(
        head=HeadSpec(
            head_id="TEST_HEAD__H1__W1",
            task_id="TEST_HEAD",
            dataset="synthetic",
            target="y",
            cadence_seconds=1.0,
            h_steps=1,
            w_steps=1,
            w0_steps=1,
            primary=True,
        ),
        information_set="dynamic",
        availability_scenario="record_time",
        proxy_policy="primary",
    )


@pytest.mark.parametrize(
    ("representation", "expected_columns", "expected_value"),
    [
        (CHANNEL_COMPRESSED, 2, 7.0),
        (FULL_BASIS, 5, 11.0),
    ],
)
def test_joint_materialization_uses_frozen_k_representation(
    representation: str, expected_columns: int, expected_value: float
) -> None:
    features = {
        "channels": ["k1", "k2"],
        "compressed_train": np.full((4, 2), 3.0),
        "compressed_evaluation": np.full((3, 2), 7.0),
        "joint_train": np.full((4, 5), 5.0),
        "joint_evaluation": np.full((3, 5), 11.0),
    }
    observed = _joint_evaluation_k_block(features, _contract(representation))
    assert observed.shape == (3, expected_columns)
    np.testing.assert_array_equal(observed, expected_value)


@pytest.mark.parametrize("representation", [CHANNEL_COMPRESSED, FULL_BASIS])
def test_joint_materialization_refits_predictive_stability_contract(
    representation: str,
) -> None:
    axis = np.linspace(-1.0, 1.0, 80, dtype=np.float64)
    train = {
        "K": np.column_stack([axis, axis**2]),
        "W": np.empty((len(axis), 0), dtype=np.float64),
        "A": np.column_stack([np.roll(axis, 1), np.roll(axis, 2)]),
    }
    target = 0.7 * train["K"][:, 0] - 0.2 * train["A"][:, 1]
    evaluation = {name: values[::4] for name, values in train.items()}
    prediction, fitted, _ = _fit_frozen_joint_candidate(
        train, target, evaluation, _contract(representation)
    )
    assert prediction.shape == (20,)
    assert fitted["joint_estimator_semantics"] == JOINT_ESTIMATOR_SEMANTICS
    assert fitted["k_representation"] == representation
    assert fitted["numerical_alpha"] == 0.0
    assert fitted["predictive_eta"] == 1.0
    assert fitted["raw_k_support"] == ["k1", "k2"]


def test_joint_materialization_rejects_legacy_contract_schema() -> None:
    with pytest.raises(RuntimeError, match="missing stability fields"):
        _validate_joint_materialization_contract(
            {
                "family": J_KA,
                "alpha": 1e-4,
                "k_over_a_ratio": 1.0,
                "w_over_a_ratio": 1.0,
            }
        )


def test_materialization_preflight_validates_frozen_joint_fields(
    tmp_path: Path,
) -> None:
    view = _view()
    result_path = (
        tmp_path
        / "DEVELOPMENT"
        / "JOINT"
        / view.head.head_id
        / view.availability_scenario
        / view.proxy_policy
        / "RESULT.json"
    )
    result_path.parent.mkdir(parents=True)
    contract = _contract()
    result_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "ar_profile": [1, 2],
                "selected_k_representation": contract["k_representation"],
                "selected_predictive_eta": contract["predictive_eta"],
                "selected_numerical_alpha": contract["numerical_alpha"],
                "joint_contract": contract,
            }
        ),
        encoding="utf-8",
    )
    paths = SimpleNamespace(output=tmp_path)
    observed = preflight_public_all_materialization(paths, [view])
    assert observed["status"] == "PASS"
    assert observed["formal_joint_views"] == 1
    assert observed["test_accessed"] is False

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["selected_predictive_eta"] = 0.1
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen field mismatch"):
        preflight_public_all_materialization(paths, [view])
