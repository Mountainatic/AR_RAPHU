from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from prism_benchmark.v211_joint import (
    J_K,
    J_KA,
    J_KW,
    J_KWA,
    fit_joint_candidate,
)
from prism_benchmark.v211_metro_final import materialize_view
from prism_benchmark.v22_config import (
    CHANNEL_COMPRESSED,
    ETA_PRED_GRID,
    FULL_BASIS,
)
from prism_benchmark.v22_joint import (
    V22Candidate,
    evaluate_v22_candidates_ordered,
    fit_joint_candidate_v22,
    k_representation_blocks,
    predictive_penalty_scale,
    prepare_joint_representation,
    registered_v22_joint_candidates,
    select_k_representation,
    select_predictive_eta,
    select_smallest_numerical_alpha,
    solve_prepared_legacy_anchor,
    solve_prepared_v22,
    v22_candidate_id,
)


def _blocks(rows: int = 120) -> tuple[dict[str, np.ndarray], np.ndarray]:
    axis = np.linspace(-1.5, 1.5, rows, dtype=np.float64)
    blocks = {
        "K": np.column_stack([axis, axis**2, np.sin(axis)]),
        "W": np.column_stack([np.cos(axis), np.sin(2.0 * axis)]),
        "A": np.column_stack([np.roll(axis, 1), np.roll(axis, 2)]),
    }
    target = (
        0.6 * blocks["K"][:, 0]
        - 0.15 * blocks["K"][:, 1]
        + 0.2 * blocks["W"][:, 0]
        - 0.1 * blocks["A"][:, 1]
    )
    return blocks, target


def _prepared(rows: int = 120, representation: str = FULL_BASIS):
    blocks, target = _blocks(rows)
    evaluation = {name: values[::2] for name, values in blocks.items()}
    return (
        prepare_joint_representation(
            blocks,
            target,
            evaluation,
            k_representation=representation,
            raw_k_support=("k1", "k2", "k3"),
        ),
        target[::2],
    )


def test_full_basis_legacy_anchor_matches_v212_solver() -> None:
    blocks, target = _blocks()
    evaluation = {name: values[::2] for name, values in blocks.items()}
    expected, _, _ = fit_joint_candidate(
        blocks,
        target,
        evaluation,
        candidate=J_KWA,
        alpha=1e-4,
        k_over_a_ratio=4.0,
        w_over_a_ratio=0.3,
    )
    prepared = prepare_joint_representation(
        blocks,
        target,
        evaluation,
        k_representation=FULL_BASIS,
        raw_k_support=("k1", "k2", "k3"),
    )
    observed, _ = solve_prepared_legacy_anchor(
        prepared,
        route=J_KWA,
        alpha=1e-4,
        k_over_a_ratio=4.0,
        w_over_a_ratio=0.3,
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-12)


def test_channel_compressed_columns_equal_active_k_count() -> None:
    rows = 20
    features = {
        "channels": ["a", "b", "c"],
        "compressed_train": np.ones((rows, 3)),
        "compressed_evaluation": np.ones((5, 3)),
        "joint_train": np.ones((rows, 12)),
        "joint_evaluation": np.ones((5, 12)),
    }
    blocks, audit = k_representation_blocks(features, ("a", "b", "c"))
    assert blocks[CHANNEL_COMPRESSED][0].shape[1] == 3
    assert audit["channel_compressed_columns"] == 3


def test_compressed_and_full_share_exact_raw_k_support() -> None:
    features = {
        "channels": ["a", "b"],
        "compressed_train": np.ones((8, 2)),
        "compressed_evaluation": np.ones((4, 2)),
        "joint_train": np.ones((8, 9)),
        "joint_evaluation": np.ones((4, 9)),
    }
    _, audit = k_representation_blocks(features, ("a", "b"))
    assert audit["same_raw_support"] is True
    assert audit["raw_k_support"] == ["a", "b"]


def test_zeroed_raw_channel_cannot_be_resurrected() -> None:
    features = {
        "channels": ["active", "zeroed"],
        "compressed_train": np.ones((8, 2)),
        "compressed_evaluation": np.ones((4, 2)),
        "joint_train": np.ones((8, 5)),
        "joint_evaluation": np.ones((4, 5)),
    }
    with pytest.raises(RuntimeError, match="raw support"):
        k_representation_blocks(features, ("active",))


def test_eta_zero_is_exact_predictive_ridge_boundary() -> None:
    prepared, _ = _prepared()
    _, contract, _ = solve_prepared_v22(
        prepared, route=J_K, numerical_alpha=1e-4, predictive_eta=0.0
    )
    assert contract["predictive_eta"] == 0.0
    assert contract["predictive_penalty_scale"] == 0.0


def test_predictive_penalty_uses_n_times_eta() -> None:
    assert predictive_penalty_scale(100, 0.01) == pytest.approx(1.0)
    assert predictive_penalty_scale(250, 0.01) == pytest.approx(2.5)


def test_numerical_alpha_selection_ignores_validation_loss() -> None:
    failed = {"coefficient": [1.0], "numerical_certificate": {"status": "FAILED"}}
    passed = {"coefficient": [1.0], "numerical_certificate": {"status": "PASS"}}
    selected, audit = select_smallest_numerical_alpha(
        [0.0, 1e-8, 1e-4],
        {0.0: [failed] * 4, 1e-8: [passed] * 4, 1e-4: [passed] * 4},
        required_folds=4,
    )
    assert selected == 1e-8
    assert audit["selection_loss_used"] is False


def test_eta_selection_uses_oof_risk_and_prefers_stronger_one_se_eta() -> None:
    selected, audit = select_predictive_eta(
        {
            0.0: [1.0, 1.0, 1.0, 1.0],
            0.1: [0.90, 1.00, 0.90, 1.00],
            1.0: [0.91, 1.01, 0.91, 1.01],
        }
    )
    assert selected == 1.0
    assert audit["best"] == "0.1"


def test_representation_one_se_prefers_compressed_when_equal() -> None:
    selected, _ = select_k_representation(
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        minimum_relative_improvement=0.01,
        minimum_positive_fraction=0.75,
    )
    assert selected == CHANNEL_COMPRESSED


def test_full_requires_practical_and_fold_consistency_guard() -> None:
    rejected, _ = select_k_representation(
        [1.0, 1.0, 1.0, 1.0],
        [0.995, 0.995, 0.995, 0.995],
        minimum_relative_improvement=0.01,
        minimum_positive_fraction=0.75,
    )
    accepted, _ = select_k_representation(
        [1.0, 1.0, 1.0, 1.0],
        [0.98, 0.98, 0.98, 0.98],
        minimum_relative_improvement=0.01,
        minimum_positive_fraction=0.75,
    )
    assert rejected == CHANNEL_COMPRESSED
    assert accepted == FULL_BASIS


def test_joint_route_set_remains_exact() -> None:
    assert registered_v22_joint_candidates() == (J_K, J_KW, J_KA, J_KWA)


def test_ar_only_and_k_zero_cannot_be_fitted() -> None:
    blocks, target = _blocks()
    evaluation = {name: values[::2] for name, values in blocks.items()}
    for forbidden in ("AR_ONLY", "J_A", "K_ZERO", "BOTH_ZERO"):
        with pytest.raises(ValueError):
            fit_joint_candidate_v22(
                blocks,
                target,
                evaluation,
                candidate=forbidden,
                k_representation=FULL_BASIS,
                numerical_alpha=1e-4,
                predictive_eta=0.0,
            )


def test_w_coefficients_remain_jointly_fitted() -> None:
    prepared, _ = _prepared()
    for route in (J_KW, J_KWA):
        _, contract, _ = solve_prepared_v22(
            prepared, route=route, numerical_alpha=1e-4, predictive_eta=0.01
        )
        assert contract["blocks"]["W"]["columns"] > 0
        assert "kw_scalar" not in repr(contract)


def test_candidate_id_binds_all_v22_hyperparameters() -> None:
    base = V22Candidate(J_K, CHANNEL_COMPRESSED, 0.0, 0.0)
    identifiers = {
        v22_candidate_id("view", base),
        v22_candidate_id("view", V22Candidate(J_KW, CHANNEL_COMPRESSED, 0.0, 0.0)),
        v22_candidate_id("view", V22Candidate(J_K, FULL_BASIS, 0.0, 0.0)),
        v22_candidate_id("view", V22Candidate(J_K, CHANNEL_COMPRESSED, 1e-8, 0.0)),
        v22_candidate_id("view", V22Candidate(J_K, CHANNEL_COMPRESSED, 0.0, 1e-3)),
    }
    assert len(identifiers) == 5
    assert set(base.descriptor()) == {
        "route",
        "k_representation",
        "numerical_alpha",
        "predictive_eta",
    }


def test_serial_and_fork_candidate_selection_are_identical() -> None:
    prepared, evaluation_target = _prepared()
    prepared_folds = [{FULL_BASIS: prepared, CHANNEL_COMPRESSED: prepared}] * 4
    targets = [evaluation_target] * 4
    candidates = [
        V22Candidate(J_K, FULL_BASIS, 1e-4, eta)
        for eta in ETA_PRED_GRID[:4]
    ]
    serial = evaluate_v22_candidates_ordered(
        prepared_folds, targets, candidates, workers=1
    )
    parallel = evaluate_v22_candidates_ordered(
        prepared_folds,
        targets,
        candidates,
        workers=min(3, os.cpu_count() or 1),
    )
    assert serial == parallel
    serial_eta, _ = select_predictive_eta(
        {candidate.predictive_eta: item["fold_losses"] for candidate, item in zip(candidates, serial, strict=True)}
    )
    parallel_eta, _ = select_predictive_eta(
        {candidate.predictive_eta: item["fold_losses"] for candidate, item in zip(candidates, parallel, strict=True)}
    )
    assert serial_eta == parallel_eta


def test_original_four_fold_provenance_code_is_retained() -> None:
    source = inspect.getsource(__import__("prism_benchmark.v22_joint", fromlist=["run_joint_v22_view"]).run_joint_v22_view)
    assert "registered_joint_inner_fold_frames" in source
    assert "audit_joint_fold_protocol" in source
    assert "len(prepared_folds) != 4" in source
    assert "w_physical_oof_used_as_training_pool" in source


def test_pf_estimators_are_unchanged_from_parent_commit() -> None:
    project = Path(__file__).resolve().parents[1]
    paths = [
        "src/prism_benchmark/v211_k.py",
        "src/prism_benchmark/v211_c.py",
        "src/prism_benchmark/v211_w.py",
        "src/prism_benchmark/v211_a.py",
    ]
    diff = subprocess.run(
        ["git", "-C", str(project), "diff", "6ebcac898a75b6c1aa05c920a3a39847db052957", "--", *paths],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert diff == ""


def test_m5_never_accesses_test_or_ood_and_m7_is_contract_driven() -> None:
    import prism_benchmark.v22_joint as module

    source = inspect.getsource(module.run_joint_v22_view)
    assert 'load_samples(shared, view, "test")' not in source
    assert 'load_samples(shared, view, "ood")' not in source
    assert '"test_accessed": False' in source
    assert '"ood_accessed": False' in source
    materialization_source = inspect.getsource(materialize_view)
    assert "k_representation" in materialization_source
    assert "predict_joint_candidate" in materialization_source
    assert "predictive_eta" in materialization_source
