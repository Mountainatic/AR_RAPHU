from __future__ import annotations

import ast
import inspect
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
from prism_benchmark.v211_joint_stability_config import (
    CHANNEL_COMPRESSED,
    ETA_PRED_GRID,
    FULL_BASIS,
)
from prism_benchmark.v211_joint_stability import (
    StabilityCandidate,
    evaluate_stability_candidates_ordered,
    fit_joint_candidate_stability,
    k_representation_blocks,
    predictive_penalty_scale,
    prepare_joint_representation,
    registered_joint_stability_candidates,
    _registered_stability_rejection,
    select_k_representation,
    select_predictive_eta,
    select_smallest_numerical_alpha,
    solve_prepared_legacy_anchor,
    solve_prepared_stability,
    stability_candidate_id,
    stability_guarded_selection_json,
)
from prism_benchmark.v21_selection import guarded_local_one_se_select


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


def test_c_input_path_failure_is_a_legal_joint_stability_rejection() -> None:
    view = SimpleNamespace(
        head=SimpleNamespace(dataset="metropt", head_id="TEST_HEAD"),
        availability_scenario="record_time",
        proxy_policy="primary",
    )
    result = _registered_stability_rejection(
        view,
        {
            "pass": False,
            "input_path_failure_class": "INPUT_PATH_PRESERVATION_PERFORMANCE_GATE_FAILED",
        },
        started=0.0,
    )
    assert result["status"] == "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT"
    assert result["selected_candidate"] is None
    assert result["input_path_preservation"]["reason"] == "C_INPUT_PATH_NOT_PRESERVED"
    assert result["test_accessed"] is False
    assert result["ood_accessed"] is False


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
    import prism_benchmark.v211_joint_stability as module

    assert "fit_joint_candidate(" in inspect.getsource(module.run_joint_stability_view)


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


def test_k_representation_preserves_frozen_k_column_order() -> None:
    features = {
        "channels": ["k_artifact_first", "k_artifact_second"],
        "compressed_train": np.ones((8, 2)),
        "compressed_evaluation": np.ones((4, 2)),
        "joint_train": np.ones((8, 7)),
        "joint_evaluation": np.ones((4, 7)),
    }
    _, audit = k_representation_blocks(
        features, ("k_artifact_first", "k_artifact_second")
    )
    assert audit["raw_k_support"] == [
        "k_artifact_first",
        "k_artifact_second",
    ]


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
    _, contract, _ = solve_prepared_stability(
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
    assert registered_joint_stability_candidates() == (J_K, J_KW, J_KA, J_KWA)


def test_ar_only_and_k_zero_cannot_be_fitted() -> None:
    blocks, target = _blocks()
    evaluation = {name: values[::2] for name, values in blocks.items()}
    for forbidden in ("AR_ONLY", "J_A", "K_ZERO", "BOTH_ZERO"):
        with pytest.raises(ValueError):
            fit_joint_candidate_stability(
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
        _, contract, _ = solve_prepared_stability(
            prepared, route=route, numerical_alpha=1e-4, predictive_eta=0.01
        )
        assert contract["blocks"]["W"]["columns"] > 0
        assert "kw_scalar" not in repr(contract)


def test_candidate_id_binds_all_stability_hyperparameters() -> None:
    base = StabilityCandidate(J_K, CHANNEL_COMPRESSED, 0.0, 0.0)
    identifiers = {
        stability_candidate_id("view", base),
        stability_candidate_id("view", StabilityCandidate(J_KW, CHANNEL_COMPRESSED, 0.0, 0.0)),
        stability_candidate_id("view", StabilityCandidate(J_K, FULL_BASIS, 0.0, 0.0)),
        stability_candidate_id("view", StabilityCandidate(J_K, CHANNEL_COMPRESSED, 1e-8, 0.0)),
        stability_candidate_id("view", StabilityCandidate(J_K, CHANNEL_COMPRESSED, 0.0, 1e-3)),
    }
    assert len(identifiers) == 5
    assert set(base.descriptor()) == {
        "route",
        "k_representation",
        "numerical_alpha",
        "predictive_eta",
    }


def test_stability_route_selection_serialization_preserves_candidate_identity() -> None:
    neutral = StabilityCandidate(J_K, CHANNEL_COMPRESSED, 0.0, 0.1)
    active = StabilityCandidate(J_KW, FULL_BASIS, 1e-8, 0.01)
    selection = guarded_local_one_se_select(
        {neutral: [1.0, 1.0, 1.0, 1.0], active: [0.95, 0.95, 0.95, 0.95]},
        lambda candidate: (0 if candidate == neutral else 1,),
        neutral=neutral,
        minimum_relative_improvement=0.01,
        minimum_positive_fraction=0.75,
        minimum_usable_folds=4,
    )
    payload = stability_guarded_selection_json(selection)
    assert payload["final_selected_candidate"]["candidate_key"] == active.key()
    assert set(payload["means"]) == {neutral.key(), active.key()}


def test_serial_and_fork_candidate_selection_are_identical() -> None:
    prepared, evaluation_target = _prepared()
    prepared_folds = [{FULL_BASIS: prepared, CHANNEL_COMPRESSED: prepared}] * 4
    targets = [evaluation_target] * 4
    candidates = [
        StabilityCandidate(J_K, FULL_BASIS, 1e-4, eta)
        for eta in ETA_PRED_GRID[:4]
    ]
    serial = evaluate_stability_candidates_ordered(
        prepared_folds, targets, candidates, workers=1
    )
    parallel = evaluate_stability_candidates_ordered(
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
    source = inspect.getsource(__import__("prism_benchmark.v211_joint_stability", fromlist=["run_joint_stability_view"]).run_joint_stability_view)
    assert "registered_joint_inner_fold_frames" in source
    assert "audit_joint_fold_protocol" in source
    assert "len(prepared_folds) != 4" in source
    assert "w_physical_oof_used_as_training_pool" in source


def _source_at_commit(project: Path, commit: str, relative_path: str) -> str:
    repository = Path(
        subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    ).resolve()
    repository_path = (project / relative_path).resolve().relative_to(repository)
    return subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{repository_path.as_posix()}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def _normalized_function(source: str, name: str) -> str:
    matches = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1
    return ast.dump(matches[0], include_attributes=False)


def test_pf_core_estimators_are_unchanged_from_native_support_source_commit() -> None:
    project = Path(__file__).resolve().parents[1]
    commit = "e47542a319640bc045ca0d31ae9b40763182dde8"
    unchanged_paths = [
        "src/prism_benchmark/v211_c.py",
        "src/prism_benchmark/v2_urysohn.py",
    ]
    diff = subprocess.run(
        ["git", "-C", str(project), "diff", commit, "--", *unchanged_paths],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert diff == ""

    k_path = "src/prism_benchmark/v211_k.py"
    current_k = (project / k_path).read_text(encoding="utf-8")
    original_k = _source_at_commit(project, commit, k_path)
    for function in (
        "oof_replay_audit",
        "select_smallest_stable_full_and_folds",
        "_profile_complexity",
        "_structural_complexity",
        "evaluate_candidate",
    ):
        assert _normalized_function(current_k, function) == _normalized_function(
            original_k, function
        )

    v2_k_path = "src/prism_benchmark/v2_k.py"
    current_v2_k = (project / v2_k_path).read_text(encoding="utf-8")
    original_v2_k = _source_at_commit(project, commit, v2_k_path)
    assert _normalized_function(current_v2_k, "profile_values") == _normalized_function(
        original_v2_k, "profile_values"
    )


def test_m5_never_accesses_test_or_ood_and_m7_is_contract_driven() -> None:
    import prism_benchmark.v211_joint_stability as module

    source = inspect.getsource(module.run_joint_stability_view)
    assert 'load_samples(shared, view, "test")' not in source
    assert 'load_samples(shared, view, "ood")' not in source
    assert '"test_accessed": False' in source
    assert '"ood_accessed": False' in source
    materialization_source = inspect.getsource(materialize_view)
    assert "k_representation" in materialization_source
    assert "predict_joint_candidate" in materialization_source
    assert "predictive_eta" in materialization_source
