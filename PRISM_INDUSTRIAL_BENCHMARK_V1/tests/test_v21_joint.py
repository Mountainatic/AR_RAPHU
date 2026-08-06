import numpy as np
import pandas as pd

import prism_benchmark.v21_joint as v21_joint

from prism_benchmark.v21_joint import (
    J_KWA,
    fit_joint_candidate,
    input_path_gate,
    registered_joint_candidates,
)


def test_joint_candidates_forbid_k_zero_and_ar_only():
    candidates = registered_joint_candidates()
    assert candidates == ("J_K", "J_KW", "J_KA", "J_KWA")
    assert all("ZERO" not in value and value != "J_A" for value in candidates)


def test_joint_runner_exposes_dataframe_reader_dependency():
    assert v21_joint.pd is pd


def test_joint_has_full_w_basis_block_and_prediction_components_close():
    rng = np.random.default_rng(4)
    k = rng.normal(size=(80, 2))
    w = rng.normal(size=(80, 3))
    a = rng.normal(size=(80, 2))
    y = k[:, 0] + 0.5 * w[:, 1] + 0.2 * a[:, 0]
    prediction, contract, components = fit_joint_candidate(
        {"K": k, "W": w, "A": a}, y,
        {"K": k, "W": w, "A": a}, candidate=J_KWA,
        alpha=1e-6, k_over_a_ratio=1.0, w_over_a_ratio=1.0,
    )
    assert contract["blocks"]["W"]["columns"] == 3
    np.testing.assert_allclose(
        prediction,
        contract["intercept"] + components["K"] + components["W"] + components["A"],
        atol=1e-10,
    )


def test_input_path_collapse_is_reported_not_replaced_by_ar():
    gate = input_path_gate([1.0, 1.0, 1.0, 1.0], [0.8, 0.8, 0.8, 0.8], np.zeros(20))
    assert gate["status"] == "JOINT_INPUT_PATH_COLLAPSED"
