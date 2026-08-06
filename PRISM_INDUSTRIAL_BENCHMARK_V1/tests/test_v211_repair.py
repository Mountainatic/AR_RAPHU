from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.cpu_selection import mse
from prism_benchmark.v21_selection import assert_final_prediction_contract
from prism_benchmark.v211_assembly import pf_and_joint_input_status_match
from prism_benchmark.v211_c import (
    BEST_ACTIVE_K,
    COMPRESSED,
    JOINT_BASIS,
    select_c_family_with_fallback,
)
from prism_benchmark.v211_config import V211Paths, require_v211_test_freeze
from prism_benchmark.v211_joint import registered_joint_candidates
from prism_benchmark.v211_selection import (
    input_path_preservation_gate,
    profile_one_se_regret_guard,
    select_smallest_stable,
)
from prism_benchmark.v211_w import IDENTITY, w_candidate_scope


def _profile_complexity(profile: tuple[int, int]) -> tuple[int, int]:
    return profile[1], -profile[0]


def test_profile_one_se_regret_guard():
    best = (2, 40)
    over_simplified = (8, 10)
    selection = profile_one_se_regret_guard(
        {
            best: [0.4, 0.4, 1.6, 1.6],
            over_simplified: [1.18, 1.18, 1.18, 1.18],
        },
        _profile_complexity,
    )
    assert over_simplified in selection.one_se_candidates
    assert over_simplified not in selection.regret_guarded_candidates
    assert selection.retained_profiles == (best,)


def test_profile_retains_best_and_near_simple():
    best = (2, 40)
    near = (4, 20)
    simplest_near = (8, 10)
    selection = profile_one_se_regret_guard(
        {
            best: [0.8, 0.8, 1.2, 1.2],
            near: [1.015] * 4,
            simplest_near: [1.019] * 4,
        },
        _profile_complexity,
    )
    assert selection.retained_profiles == (best, simplest_near)
    assert len(selection.retained_profiles) == 2


def test_minimal_stabilizing_ridge_prefers_zero():
    def fit_candidate(value: float):
        return {
            "coefficient": [1.0],
            "alpha": value,
            "numerical_certificate": {"status": "PASS"},
        }

    selected, contract, audit = select_smallest_stable(
        [0.0, 1e-4, 1e8], fit_candidate
    )
    assert selected == 0.0
    assert contract["alpha"] == 0.0
    assert len(audit) == 1


def test_c_cannot_erase_active_k():
    target = np.linspace(-1.0, 1.0, 200)
    best_k = target * 0.9
    erased = np.full_like(target, 1e-14)
    gate = input_path_preservation_gate(
        target,
        erased,
        best_k,
        nonintercept_coefficients=[1e-14],
        numerical_certificate_passed=True,
    )
    assert gate["pass"] is False
    assert gate["checks"]["variance"] is False
    assert gate["checks"]["coefficient"] is False


def test_c_fallback_to_best_active_k():
    selected = select_c_family_with_fallback(
        JOINT_BASIS,
        {
            JOINT_BASIS: {"pass": False},
            COMPRESSED: {"pass": False},
        },
    )
    assert selected == BEST_ACTIVE_K


def _w_configs():
    v211 = {
        "W": {
            "minimum_usable_folds": 3,
            "minimum_distinct_latent_values_per_fold": 20,
            "minimum_rank_of_intercept_and_latent": 2,
            "relative_scale_floor_multiplier_float64_eps": 64,
            "soft_overlap_mu": [0.0, 0.03, 0.3, 3.0, 30.0],
        }
    }
    v21 = {"W": {"monotone_knots": [4], "natural_cubic_knots": [4]}}
    v2 = {"W_module": {"smoothness_penalties": [0.0]}}
    return v211, v21, v2


def test_w_three_of_four_usable_folds():
    v211, v21, v2 = _w_configs()
    varying = [
        np.linspace(0.0, 1.0e-6, 100),
        np.linspace(1.0, 1.0 + 2.0e-6, 100),
        np.linspace(-2.0e-6, 2.0e-6, 100),
    ]
    candidates, audit = w_candidate_scope(
        v211,
        v21,
        v2,
        1,
        True,
        input_path_preserved=True,
        k_exact_zero=False,
        fold_train_latents=[*varying, np.ones(100)],
    )
    assert audit["usable_fold_count"] == 3
    assert audit["nonlinear_candidates_allowed"] is True
    assert len(candidates) > 1


def test_w_exact_zero_k_forces_identity():
    v211, v21, v2 = _w_configs()
    candidates, audit = w_candidate_scope(
        v211,
        v21,
        v2,
        1,
        True,
        input_path_preserved=False,
        k_exact_zero=True,
        fold_train_latents=[np.ones(100) for _ in range(4)],
    )
    assert candidates == [IDENTITY]
    assert audit["identity_forced"] is True
    assert audit["reason"] == "K_EXACT_ZERO"


def test_pf_and_joint_share_input_path_gate():
    target = np.linspace(-1.0, 1.0, 100)
    best_k = target * 0.8
    common = dict(
        target=target,
        candidate_prediction=best_k,
        best_active_k_prediction=best_k,
        input_prediction=best_k,
        nonintercept_coefficients=[1.0],
        numerical_certificate_passed=True,
    )
    pf_gate = input_path_preservation_gate(**common)
    joint_gate = input_path_preservation_gate(**common)
    assert pf_gate == joint_gate
    assert pf_and_joint_input_status_match(
        {"input_path_preservation": pf_gate},
        {"input_path_preservation": joint_gate},
    )


def test_final_loss_matches_materialized_prediction(tmp_path):
    frame = pd.DataFrame({"y_true": [0.0, 1.0, 2.0], "y_pred": [0.1, 0.9, 2.2]})
    path = tmp_path / "validation.parquet"
    frame.to_parquet(path, index=False)
    stored = mse(
        frame["y_true"].to_numpy(dtype=np.float64),
        frame["y_pred"].to_numpy(dtype=np.float64),
    )
    result = {
        "final_selected_candidate": "candidate",
        "final_selected_fold_losses": [0.2, 0.1, 0.15],
        "final_selected_prediction_path": str(path),
        "final_selected_contract": {"family": "candidate"},
        "final_prediction_loss": stored,
    }
    materialized = pd.read_parquet(path)
    recomputed = mse(
        materialized["y_true"].to_numpy(dtype=np.float64),
        materialized["y_pred"].to_numpy(dtype=np.float64),
    )
    assert_final_prediction_contract(result, recomputed_loss=recomputed)
    with pytest.raises(RuntimeError, match="stored loss disagree"):
        assert_final_prediction_contract(result, recomputed_loss=recomputed + 0.1)


def test_joint_has_no_ar_only_candidate():
    candidates = registered_joint_candidates()
    assert candidates == ("J_K", "J_KW", "J_KA", "J_KWA")
    assert "AR_ONLY" not in candidates
    assert all(candidate.startswith("J_K") for candidate in candidates)


def test_no_test_access_before_v211_freeze(tmp_path):
    paths = V211Paths(
        project=tmp_path / "project",
        shared=tmp_path / "shared",
        output=tmp_path / "results_prism_v2_1_1_sru",
    )
    with pytest.raises(RuntimeError, match="requires V211_SRU_FINAL_FREEZE_MANIFEST"):
        require_v211_test_freeze(paths)
