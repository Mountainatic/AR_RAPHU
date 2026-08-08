from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import inner_folds
from prism_benchmark.v21_selection import assert_final_prediction_contract
from prism_benchmark.v211_joint import (
    J_K,
    J_KA,
    J_KW,
    J_KWA,
    audit_joint_fold_protocol,
    build_w_design,
    evaluate_joint_candidates_ordered,
    fit_joint_candidate,
    registered_joint_candidates,
    registered_joint_inner_fold_frames,
    run_joint_view,
)
from prism_benchmark.v211_metro_config import OUTPUT_DIRECTORY
from prism_benchmark.v211_metro_runner import (
    STAGES,
    run_m0,
    run_m1,
    run_m2,
    run_m3,
    run_m4,
    run_m5,
    run_m6,
)
from prism_benchmark.v211_selection import (
    INPUT_PATH_PRESERVATION_PERFORMANCE_GATE_FAILED,
    input_path_failure_class,
    input_path_preservation_gate,
)
from prism_benchmark.v211_w import NATURAL_CUBIC
from prism_benchmark.v2_k import _cap


def _train_frame(rows: int = 900, *, information_set: str = "dynamic") -> pd.DataFrame:
    origin = np.arange(rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "base_origin_id": [f"base-{value:05d}" for value in origin],
            "view_sample_id": [
                f"{information_set}-sample-{value:05d}" for value in origin
            ],
            "dataset": ["metropt"] * rows,
            "entity_id": ["metro-one"] * rows,
            "origin": origin,
            "dependency_start": origin,
            "dependency_stop_exclusive": origin + 1,
            "latest_available_target_index": np.maximum(origin - 1, 0),
            "y_true": np.sin(origin / 50.0),
        }
    )


def _joint_blocks(rows: int = 240) -> tuple[dict[str, np.ndarray], np.ndarray]:
    axis = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    blocks = {
        "K": np.column_stack([axis, axis * axis]),
        "W": np.column_stack([np.sin(axis), np.cos(axis)]),
        "A": np.column_stack([np.roll(axis, 1), np.roll(axis, 2)]),
    }
    target = (
        0.5 * blocks["K"][:, 0]
        + 0.2 * blocks["W"][:, 0]
        - 0.1 * blocks["A"][:, 1]
    )
    return blocks, target


def test_joint_uses_original_inner_fold_fit_support() -> None:
    train = _train_frame()
    frames = registered_joint_inner_fold_frames(
        train, fold_count=4, fit_cap=10000, evaluation_cap=10000
    )
    raw_folds = inner_folds(train, 4)
    prior_evaluation_ids: set[str] = set()
    for fold_index, ((fit_index, evaluation_index), frame) in enumerate(
        zip(raw_folds, frames, strict=True)
    ):
        assert frame["fold_index"] == fold_index
        expected_fit = _cap(train.iloc[fit_index], 10_000).reset_index(drop=True)
        assert frame["fit"]["base_origin_id"].tolist() == expected_fit[
            "base_origin_id"
        ].tolist()
        if fold_index:
            assert set(frame["fit"]["base_origin_id"]) != prior_evaluation_ids
        prior_evaluation_ids.update(
            train.iloc[evaluation_index]["base_origin_id"].astype(str)
        )


def test_joint_evaluation_fold_matches_c_and_w_registered_ids() -> None:
    dynamic = _train_frame(information_set="dynamic")
    input_only = _train_frame(information_set="input")
    joint_folds = registered_joint_inner_fold_frames(
        dynamic, fold_count=4, fit_cap=10000, evaluation_cap=10000
    )
    input_folds = registered_joint_inner_fold_frames(
        input_only, fold_count=4, fit_cap=10000, evaluation_cap=10000
    )
    for joint_fold, input_fold in zip(joint_folds, input_folds, strict=True):
        expected = input_fold["evaluation"]
        audit = audit_joint_fold_protocol(
            joint_fold,
            input_fold,
            expected[["base_origin_id", "view_sample_id"]].copy(),
            expected[["base_origin_id", "view_sample_id"]].copy(),
        )
        assert audit["pass"] is True
    corrupt = input_folds[0]["evaluation"][
        ["base_origin_id", "view_sample_id"]
    ].copy()
    corrupt.loc[0, "view_sample_id"] = "wrong"
    audit = audit_joint_fold_protocol(
        joint_folds[0],
        input_folds[0],
        corrupt,
        input_folds[0]["evaluation"][["base_origin_id", "view_sample_id"]],
    )
    assert audit["pass"] is False


def test_joint_does_not_skip_fold_zero() -> None:
    frames = registered_joint_inner_fold_frames(
        _train_frame(), fold_count=4, fit_cap=10000, evaluation_cap=10000
    )
    assert [item["fold_index"] for item in frames] == [0, 1, 2, 3]


def test_w_physical_oof_cannot_be_joint_fit_pool() -> None:
    source = inspect.getsource(run_joint_view)
    assert 'fit = oof[oof["oof_fold"] < fold]' not in source
    assert 'fit["physical_oof"]' not in source
    assert 'w_result["oof_path"]' in source
    assert "CROSS_STAGE_FOLD_ID_AND_PROVENANCE_AUDIT_ONLY" in source


def test_joint_w_basis_is_fold_local() -> None:
    fit = np.linspace(-2.0, 1.0, 180, dtype=np.float64)
    early = np.linspace(-1.0, 0.5, 50, dtype=np.float64)
    future = np.linspace(10.0, 100.0, 80, dtype=np.float64)
    first = build_w_design(fit, early, NATURAL_CUBIC, 5)
    second = build_w_design(
        fit, np.concatenate([early, future]), NATURAL_CUBIC, 5
    )
    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.evaluation, second.evaluation[: len(early)])
    assert first.metadata == second.metadata


def test_joint_candidate_set_remains_exact() -> None:
    assert registered_joint_candidates() == (J_K, J_KW, J_KA, J_KWA)
    assert not {"J_A", "AR_ONLY", "K_ZERO", "BOTH_ZERO"}.intersection(
        registered_joint_candidates()
    )


def test_joint_w_columns_are_jointly_solved_without_kw_scalar() -> None:
    blocks, target = _joint_blocks()
    evaluation = {name: value[::2] for name, value in blocks.items()}
    for route in (J_KW, J_KWA):
        _, contract, _ = fit_joint_candidate(
            blocks,
            target,
            evaluation,
            candidate=route,
            alpha=1e-4,
        )
        assert contract["blocks"]["W"]["columns"] > 0
        assert "kw_scalar" not in repr(contract)


def test_joint_serial_and_fork_candidate_results_are_exact() -> None:
    blocks, target = _joint_blocks()
    evaluation = {name: value[::2] for name, value in blocks.items()}
    evaluation_target = target[::2]
    candidates = [
        (J_K, 1e-4, 1.0, 1.0),
        (J_KW, 1e-4, 1.0, 1.0),
        (J_KA, 1e-3, 0.3, 1.0),
        (J_KWA, 1e-3, 3.0, 0.3),
    ]
    serial = evaluate_joint_candidates_ordered(
        blocks,
        target,
        evaluation,
        evaluation_target,
        candidates,
        workers=1,
    )
    parallel = evaluate_joint_candidates_ordered(
        blocks,
        target,
        evaluation,
        evaluation_target,
        candidates,
        workers=min(3, os.cpu_count() or 1),
    )
    assert serial == parallel


def test_joint_candidate_binding_is_one_route() -> None:
    blocks, target = _joint_blocks()
    evaluation = {name: value[::2] for name, value in blocks.items()}
    prediction, contract, _ = fit_joint_candidate(
        blocks,
        target,
        evaluation,
        candidate=J_KWA,
        alpha=1e-4,
    )
    result = {
        "final_selected_candidate": J_KWA,
        "final_selected_fold_losses": [1.0, 0.9, 0.8, 0.7],
        "final_selected_prediction_path": "validation_J_KWA.parquet",
        "final_selected_contract": contract,
        "final_selected_prediction_candidate": J_KWA,
    }
    assert prediction.shape == (len(target[::2]),)
    assert contract["family"] == result["final_selected_candidate"]
    assert_final_prediction_contract(result)


def test_mse_only_input_gate_failure_has_performance_name() -> None:
    target = np.linspace(-1.0, 1.0, 200, dtype=np.float64)
    best_k = target.copy()
    candidate = target + 1.0
    gate = input_path_preservation_gate(
        target,
        candidate,
        best_k,
        input_prediction=target,
        nonintercept_coefficients=[1.0],
        numerical_certificate_passed=True,
    )
    assert gate["checks"] == {
        "variance": True,
        "mse": False,
        "coefficient": True,
        "numerical_certificate": True,
    }
    assert gate["input_path_failure_class"] == (
        INPUT_PATH_PRESERVATION_PERFORMANCE_GATE_FAILED
    )
    assert input_path_failure_class(gate["checks"]) == (
        INPUT_PATH_PRESERVATION_PERFORMANCE_GATE_FAILED
    )


def test_m0_through_m6_precede_any_test_or_ood_stage() -> None:
    assert STAGES[:7] == ("m0", "m1", "m2", "m3", "m4", "m5", "m6")
    assert STAGES[7:] == ("m7", "m8")
    for stage_runner in (run_m0, run_m1, run_m2, run_m3, run_m4, run_m5, run_m6):
        source = inspect.getsource(stage_runner)
        assert '"test_accessed": False' in source
        assert '"ood_accessed": False' in source


def test_v212_result_namespace_is_ignored_before_m0_clean_check() -> None:
    project = Path(__file__).resolve().parents[1]
    ignore_lines = (project / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert f"{OUTPUT_DIRECTORY}/" in ignore_lines
