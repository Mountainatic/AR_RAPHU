from __future__ import annotations

import numpy as np

from prism_benchmark.v2_final import _j_selection, _linear_contract_prediction, _physical_for_assembly


def test_joint_selection_and_linear_contract_round_trip() -> None:
    assert _j_selection({"selected_candidate": "('JOINT_K_STATE_LINEAR', 0.1, 2.0)"}) == (
        "JOINT_K_STATE_LINEAR",
        0.1,
        2.0,
    )
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    contract = {
        "family": "JOINT_K_STATE_LINEAR",
        "mean": [1.0, 2.0],
        "scale": [2.0, 4.0],
        "coefficient": [2.0, 4.0],
        "intercept": 1.0,
    }
    np.testing.assert_allclose(_linear_contract_prediction(features, contract), [1.0, 5.0])


def test_wiener_assembly_uses_selected_pairwise_base() -> None:
    context = {
        "c_result": {"selected_family": "SPARSE_PAIRWISE_ANOVA_MAX3"},
        "w_contract": {"family": "IDENTITY"},
    }
    values = {
        "K_COMPRESSED": np.asarray([1.0]),
        "K_JOINT_BASIS": np.asarray([2.0]),
        "K_PAIRWISE": np.asarray([3.0]),
    }
    np.testing.assert_allclose(_physical_for_assembly(context, values, "K_JOINT_BASIS_W"), [3.0])
