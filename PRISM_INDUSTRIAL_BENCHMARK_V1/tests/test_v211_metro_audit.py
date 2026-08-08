from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.v2_k import _cap
from prism_benchmark.v2_runtime import ordered_fork_map, run_parallel
from prism_benchmark.v211_assembly import (
    build_joint_card,
    pf_and_joint_input_status_match,
    pf_joint_input_gate_inconsistent,
)
from prism_benchmark.v211_a import _evaluate_a_candidate, _merge_w_oof_for_a
from prism_benchmark.v211_joint import (
    J_K,
    J_KA,
    J_KW,
    J_KWA,
    _evaluate_joint_candidate,
    registered_joint_candidates,
)
from prism_benchmark.v211_k import (
    _ordered_parallel_map,
    oof_replay_audit,
    select_smallest_stable_full_and_folds,
)
from prism_benchmark.v211_metro_config import (
    MetroV211Paths,
    effective_worker_count,
    require_metro_test_freeze,
    runtime_parallelism_audit,
)
from prism_benchmark.v211_metro_contracts import (
    assert_candidate_id_binding,
    bind_result_candidate_ids,
)
from prism_benchmark.v211_metro_final import (
    _candidate_ids,
    _development_prerequisites,
    _formal_candidate_names,
)
from prism_benchmark.v211_metro_reporting import (
    _formal_comparisons,
    moving_block_matrix_means,
)
from prism_benchmark.v211_metro_runner import (
    compare_shared_data_audits,
    full_shared_data_audit,
    hierarchical_route_freeze_decision,
)
from prism_benchmark.v211_w import (
    IDENTITY,
    NATURAL_CUBIC,
    _evaluate_w_candidate,
    fit_w_correction,
    predict_w_correction,
)


def _deterministic_affine(value: int) -> int:
    return 3 * value + 1


def _nested_ordered_fork(value: int) -> list[int]:
    return ordered_fork_map(
        _deterministic_affine,
        [(value + offset,) for offset in range(4)],
        workers=2,
        label="METRO_TEST_NESTED_FORK_INNER",
    )


def test_identity_w_is_exactly_skip_w() -> None:
    latent = np.linspace(-2.0, 2.0, 257)
    correction, contract = fit_w_correction(
        latent,
        np.sin(latent),
        latent,
        family=IDENTITY,
    )
    assert np.array_equal(correction, np.zeros_like(latent))
    assert np.array_equal(latent + correction, latent)
    assert np.array_equal(predict_w_correction(latent, contract), correction)


def test_nonlinear_w_chunked_prediction_matches_full() -> None:
    latent = np.linspace(-2.0, 2.0, 400)
    target = 0.2 * latent * latent + 0.1 * latent
    _, contract = fit_w_correction(
        latent,
        target,
        latent,
        family=NATURAL_CUBIC,
        knot_count=5,
        smoothness=1e-3,
        mu=0.0,
        direction=1,
    )
    full = predict_w_correction(latent, contract)
    chunked = np.concatenate(
        [predict_w_correction(latent[start : start + 37], contract) for start in range(0, len(latent), 37)]
    )
    np.testing.assert_allclose(chunked, full, rtol=1e-13, atol=1e-13)


def test_cap_is_deterministic_and_cannot_import_nonfit_rows() -> None:
    frame = pd.DataFrame(
        {
            "base_origin_id": [f"row-{index:04d}" for index in range(100)],
            "marker": np.arange(100),
        }
    )
    fit = frame.iloc[:80].copy()
    first = _cap(fit, 23)
    second = _cap(fit.sample(frac=1.0, random_state=7), 23)
    assert set(first["base_origin_id"]) == set(second["base_origin_id"])
    assert set(first["base_origin_id"]).issubset(set(fit["base_origin_id"]))
    assert not set(first["base_origin_id"]) & set(frame.iloc[80:]["base_origin_id"])


def test_lockbox_rejects_test_access_without_freeze(tmp_path: Path) -> None:
    paths = MetroV211Paths(project=tmp_path, shared=tmp_path, output=tmp_path / "output")
    with pytest.raises(RuntimeError, match="requires development freeze"):
        require_metro_test_freeze(paths)


def _gate(pass_: bool, prediction: str) -> dict[str, object]:
    return {
        "input_path_preservation": {
            "pass": pass_,
            "gate_evaluation_identity": {
                "gate_version": "INPUT_GATE_V1",
                "gate_parameters_sha256": "parameters",
                "input_prediction_sha256": prediction,
                "target_sha256": "target",
                "best_k_comparator_sha256": "best-k",
            },
        }
    }


def test_shared_gate_contract_does_not_require_same_outcome_for_different_predictions() -> None:
    pf = _gate(True, "pf-prediction")
    joint = _gate(False, "joint-prediction")
    assert not pf_joint_input_gate_inconsistent(pf, joint)
    assert pf_and_joint_input_status_match(pf, joint)


def test_same_gate_evaluation_with_different_outcome_is_inconsistent() -> None:
    pf = _gate(True, "same-prediction")
    joint = _gate(False, "same-prediction")
    assert pf_joint_input_gate_inconsistent(pf, joint)
    assert not pf_and_joint_input_status_match(pf, joint)


def test_joint_registry_has_no_ar_only_or_k_zero() -> None:
    assert registered_joint_candidates() == ("J_K", "J_KW", "J_KA", "J_KWA")
    assert all("AR_ONLY" not in value and "K_ZERO" not in value for value in registered_joint_candidates())


def test_candidate_id_binds_selected_loss_prediction_and_contract(tmp_path: Path) -> None:
    output = tmp_path / "output"
    prediction_path = output / "development" / "validation.parquet"
    prediction_path.parent.mkdir(parents=True)
    prediction = pd.DataFrame(
        {"y_true": [0.0, 1.0, 2.0], "y_pred": [0.1, 0.9, 2.1]}
    )
    prediction.to_parquet(prediction_path, index=False)
    loss = float(np.mean(np.square(prediction["y_true"] - prediction["y_pred"])))
    result_path = output / "development" / "RESULT.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stage": "UNIT_W",
                "target_head": "METRO_P60__H6__W1",
                "proxy_policy": "proxy_excluded",
                "final_selected_candidate": "ACTIVE_W",
                "final_selected_fold_losses": [1.0, 1.1, 0.9],
                "final_selected_prediction_path": str(prediction_path.relative_to(output)),
                "final_selected_contract": {"family": "ACTIVE_W"},
                "final_prediction_loss": loss,
                "candidate_fold_losses": {"ACTIVE_W": [1.0, 1.1, 0.9]},
            }
        ),
        encoding="utf-8",
    )
    bound = bind_result_candidate_ids(output, result_path)
    assert_candidate_id_binding(bound)
    assert bound["candidate_id_binding"]["prediction_loss_recomputed"] == pytest.approx(loss)


def test_selected_alias_candidate_ids_follow_frozen_routes() -> None:
    class Head:
        head_id = "METRO_P60__H6__W1"
        dataset = "metropt"

    class View:
        head = Head()
        relative_root = Path("dynamic/record_time/proxy_excluded")

    ids = _candidate_ids(View(), "KCA", "J_KW", "freeze")
    assert ids["PF_SELECTED"] == ids["KCA"]
    assert ids["J_SELECTED"] == ids["J_KW"]
    assert ids["KCW"] != ids["KC"]


def _passing_pf_checks() -> dict[str, bool]:
    return {
        "data_hash_unchanged": True,
        "k_c_input_path_noncollapsed": True,
        "w_candidates_actually_compared": True,
        "identity_equivalence_pass": True,
        "all_a_pass": True,
        "pf_assembly_card_valid": True,
        "pf_candidate_binding_pass": True,
        "test_accessed_false": True,
        "ood_accessed_false": True,
        "code_tree_clean": True,
    }


def _passing_joint_checks() -> dict[str, bool]:
    return {
        "joint_fold_protocol_all_pass": True,
        "joint_uses_original_registered_inner_support": True,
        "all_registered_joint_folds_present": True,
        "joint_candidate_set_complete": True,
        "joint_w_jointly_fit": True,
        "joint_candidate_binding_pass": True,
        "joint_numerical_solver_valid": True,
        "pf_joint_same_evaluation_not_inconsistent": True,
    }


def test_pf_and_joint_both_freeze_when_both_pass() -> None:
    decision = hierarchical_route_freeze_decision(
        _passing_pf_checks(), _passing_joint_checks(), joint_model_gate_pass=True
    )
    assert decision["status"] == "PASS_PF_AND_JOINT"
    assert decision["formal_routes"] == ["PHYSICS_FIRST", "JOINT"]


def test_pf_freezes_independently_when_only_joint_model_gate_fails() -> None:
    decision = hierarchical_route_freeze_decision(
        _passing_pf_checks(), _passing_joint_checks(), joint_model_gate_pass=False
    )
    assert decision["status"] == "PASS_PF_ONLY"
    assert decision["development_frozen"] is True
    assert decision["joint_status"] == "JOINT_NOT_SUPPORTED_ON_DEVELOPMENT"
    assert decision["joint_formal_test_eligible"] is False


def test_joint_protocol_mismatch_remains_a_hard_stop() -> None:
    joint = _passing_joint_checks()
    joint["joint_fold_protocol_all_pass"] = False
    decision = hierarchical_route_freeze_decision(
        _passing_pf_checks(), joint, joint_model_gate_pass=False
    )
    assert decision["hard_stop"] is True
    assert decision["status"] == "STOP_JOINT_FOLD_PROTOCOL_MISMATCH"


def test_pf_failure_remains_a_hard_stop_even_if_joint_passes() -> None:
    pf = _passing_pf_checks()
    pf["pf_assembly_card_valid"] = False
    decision = hierarchical_route_freeze_decision(
        pf, _passing_joint_checks(), joint_model_gate_pass=True
    )
    assert decision["hard_stop"] is True
    assert decision["status"] == "PHYSICS_ROUTE_NOT_SUPPORTED"


def test_pf_only_candidate_registry_contains_no_joint_ids() -> None:
    class Head:
        head_id = "METRO_P60__H6__W1"
        dataset = "metropt"

    class View:
        head = Head()
        relative_root = Path("dynamic/record_time/proxy_excluded")

    ids = _candidate_ids(
        View(),
        "KC",
        None,
        "freeze",
        formal_routes=["PHYSICS_FIRST"],
    )
    assert set(ids) == {"KC", "KCW", "KCA", "KCWA", "PF_SELECTED"}
    assert _formal_candidate_names(["PHYSICS_FIRST"]) == (
        "KC",
        "KCW",
        "KCA",
        "KCWA",
        "PF_SELECTED",
    )
    assert all(
        "J_" not in candidate and "J_" not in comparator
        for candidate, comparator, _ in _formal_comparisons(["PHYSICS_FIRST"])
    )


def test_pf_only_m7_prerequisites_do_not_read_joint_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Head:
        head_id = "METRO_P60__H6__W1"
        dataset = "metropt"

    class View:
        head = Head()
        availability_scenario = "record_time"
        proxy_policy = "proxy_excluded"

    paths = MetroV211Paths(project=tmp_path, shared=tmp_path, output=tmp_path / "output")
    read_paths: list[Path] = []

    def fake_read(path: Path) -> dict[str, str]:
        read_paths.append(path)
        if "JOINT" in path.parts:
            raise AssertionError("PF-only M7 attempted to read Joint RESULT")
        return {"status": "PASS"}

    monkeypatch.setattr("prism_benchmark.v211_metro_final._read", fake_read)
    prerequisites = _development_prerequisites(
        paths, View(), ["PHYSICS_FIRST"]
    )
    assert set(prerequisites) == {"C", "W", "A"}
    assert all("JOINT" not in path.parts for path in read_paths)


def test_joint_gate_failure_card_is_diagnostic_only() -> None:
    card = build_joint_card(
        {
            "status": "JOINT_OOF_PROTOCOL_CORRECTED_BUT_MODEL_GATE_FAILED",
            "final_selected_candidate": "J_KW",
            "input_path_preservation": {
                "pass": False,
                "input_path_failure_class": (
                    "INPUT_PATH_PRESERVATION_PERFORMANCE_GATE_FAILED"
                ),
            },
        }
    )
    assert card["status"] == "JOINT_NOT_SUPPORTED_ON_DEVELOPMENT"
    assert card["assembly"] is None
    assert card["formal_test_eligible"] is False
    assert card["evidence_role"] == "DEVELOPMENT_DIAGNOSTIC_ONLY"


def test_shared_data_full_hash_detects_and_accepts_expected_state(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"alpha")
    before = full_shared_data_audit(tmp_path)
    same = full_shared_data_audit(tmp_path)
    assert compare_shared_data_audits(before, same)["status"] == "PASS"
    (tmp_path / "a.bin").write_bytes(b"beta")
    after = full_shared_data_audit(tmp_path)
    comparison = compare_shared_data_audits(before, after)
    assert comparison["status"] == "STOP_DATA_BASE_MUTATED"
    assert comparison["changed"] == ["a.bin"]


def test_vectorized_moving_block_bootstrap_is_deterministic_and_shared() -> None:
    rows = 37
    entity = np.asarray(["one"] * rows)
    origin = np.arange(rows, dtype=np.int64)
    first = np.linspace(-1.0, 1.0, rows)
    values = np.column_stack([first, 2.0 * first])
    left = moving_block_matrix_means(
        entity, origin, values, block_length=5, replicates=25, seed=123
    )
    right = moving_block_matrix_means(
        entity, origin, values, block_length=5, replicates=25, seed=123
    )
    np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(left[:, 1], 2.0 * left[:, 0], rtol=1e-14, atol=1e-14)


def test_worker_override_changes_only_task_parallelism(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = {"resource": {"workers": 2, "blas_threads": 1}}
    requested = min(4, __import__("os").cpu_count() or 1)
    monkeypatch.setenv("PRISM_V211_METRO_WORKERS", str(requested))
    monkeypatch.setenv("PRISM_V211_K_MEMORY_GIB_PER_WORKER", "1.75")
    assert effective_worker_count(configured) == requested
    audit = runtime_parallelism_audit(configured)
    assert audit["scientific_contract_unchanged"] is True
    assert audit["override_scope"] == "TASK_LEVEL_THROUGHPUT_ONLY"
    jobs = [(value,) for value in range(8)]
    serial = sorted(
        run_parallel(
            _deterministic_affine,
            jobs,
            1,
            per_worker_gib=0.01,
            label="METRO_TEST_SERIAL",
        )
    )
    parallel = sorted(
        run_parallel(
            _deterministic_affine,
            jobs,
            requested,
            per_worker_gib=0.01,
            label="METRO_TEST_PARALLEL",
        )
    )
    assert parallel == serial


def test_k_oof_replay_difference_is_recorded_without_changing_candidate() -> None:
    audit = oof_replay_audit(
        [1.0, 0.9, 0.8, 0.7],
        [1.0 + 1e-8, 0.9, 0.8 - 2e-8, 0.7],
    )
    assert audit["strict_1e_12_replay_match"] is False
    assert audit["candidate_unchanged"] is True
    assert audit["selection_use"] is False
    assert audit["maximum_absolute_loss_difference"] == pytest.approx(2e-8)


def test_k_ridge_requires_full_refit_and_every_inner_fold_certificate() -> None:
    attempted_folds: list[float] = []

    def fit_full(value: float) -> dict[str, object]:
        return {
            "ridge": value,
            "certificate": {"status": "PASS", "scope": "full"},
        }

    def fit_folds(value: float) -> list[dict[str, object]]:
        attempted_folds.append(value)
        return [
            {
                "fold": fold,
                "fit_rows": 100,
                "contract": {
                    "ridge": value,
                    "certificate": {
                        "status": (
                            "NUMERICALLY_INVALID"
                            if value == 0.0 and fold == 2
                            else "PASS"
                        ),
                        "scope": f"fold-{fold}",
                    },
                },
                "prediction": np.full(3, value + fold, dtype=np.float64),
                "loss": float(value + fold),
            }
            for fold in range(4)
        ]

    selected, contract, fold_payloads, audit = (
        select_smallest_stable_full_and_folds(
            [0.0, 1e-4, 1e-3],
            fit_full,
            fit_folds,
            valid_full=lambda candidate: (
                candidate["certificate"]["status"] == "PASS"
            ),
            valid_fold=lambda payload: (
                payload["contract"]["certificate"]["status"] == "PASS"
            ),
            parallel_workers=2,
        )
    )

    assert selected == pytest.approx(1e-4)
    assert contract["ridge"] == pytest.approx(1e-4)
    assert sorted(attempted_folds) == [0.0, 1e-4]
    assert len(fold_payloads) == 4
    assert [entry["pass"] for entry in audit] == [False, True]
    assert audit[0]["full_refit_pass"] is True
    assert audit[0]["all_inner_folds_pass"] is False
    assert audit[0]["inner_fold_certificates"][2]["pass"] is False
    assert audit[0]["inner_fold_certificates"][2]["certificate"]["status"] == (
        "NUMERICALLY_INVALID"
    )
    assert audit[1]["all_inner_folds_pass"] is True


def test_k_inner_parallel_map_matches_serial_and_preserves_order() -> None:
    jobs = [(value,) for value in range(17)]

    def transform(value: int) -> tuple[int, int]:
        return value, value * value

    serial = _ordered_parallel_map(transform, jobs, workers=1)
    parallel = _ordered_parallel_map(transform, jobs, workers=4)
    assert parallel == serial


def test_ordered_fork_map_matches_serial_and_preserves_order() -> None:
    jobs = [(value,) for value in range(12)]
    serial = [_deterministic_affine(*job) for job in jobs]
    parallel = ordered_fork_map(
        _deterministic_affine,
        jobs,
        workers=4,
        label="METRO_TEST_FORK_PARALLEL",
    )
    assert parallel == serial


def test_ordered_fork_map_runs_inside_outer_process_pool() -> None:
    parallel = run_parallel(
        _nested_ordered_fork,
        [(0,), (10,)],
        requested_workers=2,
        per_worker_gib=0.01,
        label="METRO_TEST_NESTED_FORK_OUTER",
    )
    assert sorted(parallel) == sorted(
        [
            [_deterministic_affine(value) for value in range(4)],
            [_deterministic_affine(value) for value in range(10, 14)],
        ]
    )


def _assert_nested_exact(left, right) -> None:
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_exact(left_item, right_item)
    elif isinstance(left, float) and np.isnan(left):
        assert np.isnan(right)
    else:
        assert left == right


def test_w_candidate_parallel_map_is_exact_and_ordered() -> None:
    latent = np.linspace(-1.5, 1.5, 160, dtype=np.float64)
    target = latent + 0.12 * latent * latent
    upstream = np.column_stack([latent, latent * latent])
    candidates = [
        IDENTITY,
        (NATURAL_CUBIC, 4, 0.0, 0.0, 1),
        (NATURAL_CUBIC, 5, 1e-3, 0.03, 1),
    ]
    jobs = [
        (candidate, latent, target, latent, upstream, target, 0, True)
        for candidate in candidates
    ]
    serial = _ordered_parallel_map(_evaluate_w_candidate, jobs, workers=1)
    parallel = _ordered_parallel_map(_evaluate_w_candidate, jobs, workers=3)
    _assert_nested_exact(serial, parallel)


def test_a_candidate_parallel_map_is_exact_and_ordered() -> None:
    x_fit = np.column_stack(
        [np.ones(180), np.linspace(-1.0, 1.0, 180), np.linspace(1.0, 2.0, 180)]
    )
    x_evaluation = x_fit[::2].copy()
    y_fit = 0.2 + 0.4 * x_fit[:, 1] - 0.1 * x_fit[:, 2]
    y_evaluation = 0.2 + 0.4 * x_evaluation[:, 1] - 0.1 * x_evaluation[:, 2]
    upstream = x_fit[:, 1:].copy()
    jobs = [
        (x_fit, y_fit, x_evaluation, y_evaluation, alpha, mu, upstream)
        for alpha, mu in ((0.0, 0.0), (1e-4, 0.03), (1e-3, 0.3))
    ]
    serial = _ordered_parallel_map(_evaluate_a_candidate, jobs, workers=1)
    parallel = _ordered_parallel_map(_evaluate_a_candidate, jobs, workers=3)
    _assert_nested_exact(serial, parallel)


def test_a_oof_merge_preserves_both_registered_w_routes() -> None:
    train = pd.DataFrame(
        {
            "base_origin_id": ["a", "b"],
            "y_true": [1.0, 2.0],
        }
    )
    w_oof = pd.DataFrame(
        {
            "base_origin_id": ["a", "b"],
            "physical_oof": [0.8, 1.9],
            "delta_w_oof": [0.1, 0.1],
            "physical_w_oof": [0.9, 2.0],
            "delta_w_ablation_oof": [0.2, 0.3],
            "physical_w_ablation_oof": [1.0, 2.2],
            "oof_fold": [0, 1],
            "k_channel_contribution_000": [0.4, 0.5],
        }
    )

    merged = _merge_w_oof_for_a(
        train, w_oof, ["k_channel_contribution_000"]
    )

    assert merged["physical_w_ablation_oof"].tolist() == [1.0, 2.2]
    assert merged["delta_w_ablation_oof"].tolist() == [0.2, 0.3]


def test_joint_candidate_parallel_map_is_exact_and_ordered() -> None:
    axis = np.linspace(-1.0, 1.0, 200, dtype=np.float64)
    train_blocks = {
        "K": np.column_stack([axis, axis * axis]),
        "W": np.column_stack([np.sin(axis), np.cos(axis)]),
        "A": np.column_stack([np.roll(axis, 1), np.roll(axis, 2)]),
    }
    evaluation_blocks = {
        key: value[::2].copy() for key, value in train_blocks.items()
    }
    target = (
        0.5 * train_blocks["K"][:, 0]
        + 0.2 * train_blocks["W"][:, 0]
        - 0.1 * train_blocks["A"][:, 1]
    )
    evaluation_target = target[::2].copy()
    candidates = [
        (J_K, 1e-4, 1.0, 1.0),
        (J_KW, 1e-4, 1.0, 1.0),
        (J_KA, 1e-3, 0.3, 1.0),
        (J_KWA, 1e-3, 3.0, 0.3),
    ]
    jobs = [
        (
            train_blocks,
            target,
            evaluation_blocks,
            evaluation_target,
            candidate,
        )
        for candidate in candidates
    ]
    serial = _ordered_parallel_map(
        _evaluate_joint_candidate, jobs, workers=1
    )
    parallel = _ordered_parallel_map(
        _evaluate_joint_candidate, jobs, workers=4
    )
    _assert_nested_exact(serial, parallel)


def test_k_parallel_ridge_selection_matches_serial() -> None:
    def fit_full(value: float) -> dict[str, object]:
        return {"ridge": value, "certificate": {"status": "PASS"}}

    def fit_folds(value: float) -> list[dict[str, object]]:
        return [
            {
                "fold": fold,
                "fit_rows": 100,
                "contract": {
                    "certificate": {
                        "status": (
                            "NUMERICALLY_INVALID"
                            if value == 0.0 and fold == 1
                            else "PASS"
                        )
                    }
                },
                "prediction": [value + fold],
                "loss": value + fold,
            }
            for fold in range(4)
        ]

    def run(workers: int):
        return select_smallest_stable_full_and_folds(
            [0.0, 1e-4, 1e-3],
            fit_full,
            fit_folds,
            valid_full=lambda candidate: candidate["certificate"]["status"] == "PASS",
            valid_fold=lambda payload: payload["contract"]["certificate"]["status"] == "PASS",
            parallel_workers=workers,
        )

    serial = run(1)
    parallel = run(2)
    assert parallel[0] == serial[0] == pytest.approx(1e-4)
    assert parallel[1] == serial[1]
    assert parallel[3] == serial[3]
    assert [item["loss"] for item in parallel[2]] == [
        item["loss"] for item in serial[2]
    ]
