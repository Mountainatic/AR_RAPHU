from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.v2_k import _cap
from prism_benchmark.v2_runtime import run_parallel
from prism_benchmark.v211_assembly import pf_and_joint_input_status_match
from prism_benchmark.v211_joint import registered_joint_candidates
from prism_benchmark.v211_k import oof_replay_audit
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
from prism_benchmark.v211_metro_final import _candidate_ids
from prism_benchmark.v211_metro_reporting import moving_block_matrix_means
from prism_benchmark.v211_metro_runner import (
    compare_shared_data_audits,
    full_shared_data_audit,
)
from prism_benchmark.v211_w import (
    IDENTITY,
    NATURAL_CUBIC,
    fit_w_correction,
    predict_w_correction,
)


def _deterministic_affine(value: int) -> int:
    return 3 * value + 1


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


def test_pf_joint_read_the_same_input_gate() -> None:
    passed = {"input_path_preservation": {"pass": True}}
    failed = {"input_path_preservation": {"pass": False}}
    assert pf_and_joint_input_status_match(passed, passed)
    assert pf_and_joint_input_status_match(failed, failed)
    assert not pf_and_joint_input_status_match(passed, failed)


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
