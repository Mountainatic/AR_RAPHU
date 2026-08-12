from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prism_benchmark.neurobem_data import (
    NeuroBEMProtocolError,
    SegmentData,
    SegmentRecord,
    frozen_parent_partitions,
    generalized_targets,
    legal_target_rows,
    load_segment,
    sample_id,
)
from prism_benchmark.neurobem_linear import (
    era_from_markov,
    fit_numerical_ridge,
    guarded_one_se,
    k_design,
    predict_ridge,
    simulate_era,
)
from prism_benchmark.neurobem_experiment import aggregate_predictions, run_k_development


def record(partition: str = "train", segment: str = "2021-01-01-00-00-00_seg_1") -> SegmentRecord:
    return SegmentRecord(
        flight_id=segment.split("_seg_")[0],
        segment_id=segment,
        filename=f"processed_data/{segment}.csv",
        partition=partition,
        inner_fold=0 if partition == "train" else None,
        zip_uncompressed_bytes=1,
        zip_crc32="00000000",
    )


def segment(rows: int = 12, partition: str = "train", offset: float = 0.0) -> SegmentData:
    values = np.zeros((rows, 29), dtype=np.float64)
    values[:, 0] = offset + np.arange(rows) * 0.0025
    values[:, 20:24] = np.arange(rows)[:, None] + np.arange(4)[None, :] + 1.0
    return SegmentData(record(partition), values)


def test_parent_flight_is_atomic_across_splits():
    flights = [f"2021-01-01-00-00-{value:02d}" for value in range(30)]
    official = [f"{flights[3]}_seg_2", f"{flights[7]}_seg_1"]
    split = frozen_parent_partitions(flights, official, 5, "salt")
    assert split[flights[3]] == "test"
    assert split[flights[7]] == "test"
    assert sum(value == "validation" for value in split.values()) == 5
    assert set(split) == set(flights)


def test_locked_segment_cannot_be_read(tmp_path: Path):
    with pytest.raises(NeuroBEMProtocolError, match="TEST_LOCKBOX_ACCESS_FORBIDDEN"):
        load_segment(tmp_path, record("test"))


def test_k_design_is_strictly_lagged_and_segment_local():
    item = segment(rows=10)
    x, _, rows = k_design(item, history_steps=3)
    assert rows[0] == 3
    # The first lag is row t-1 and no value from another segment is available.
    expected_latest = np.square(item.values[2, 20:24])
    np.testing.assert_array_equal(x[0, :4], expected_latest)
    assert x.shape == (7, 12)


def test_native_support_rows_start_at_candidate_history():
    np.testing.assert_array_equal(legal_target_rows(8, 3), np.array([3, 4, 5, 6, 7]))
    np.testing.assert_array_equal(legal_target_rows(8, 6), np.array([6, 7]))


def test_generalized_torque_includes_rigid_body_cross_term():
    item = segment(rows=2)
    item.values[:, 1:4] = np.array([1.0, 2.0, 3.0])
    item.values[:, 4:7] = np.array([2.0, 3.0, 4.0])
    item.values[:, 13] = 5.0
    inertia = np.array([0.0025, 0.0021, 0.0043])
    expected_torque = inertia * np.array([1.0, 2.0, 3.0]) + np.cross(
        np.array([2.0, 3.0, 4.0]), inertia * np.array([2.0, 3.0, 4.0])
    )
    target = generalized_targets(item, 0.772, inertia)
    np.testing.assert_allclose(target[0, :3], expected_torque)
    assert target[0, 3] == pytest.approx(0.772 * 5.0)


def test_numerical_ridge_uses_smallest_certified_alpha_and_roundtrips():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 5))
    coefficient = rng.normal(size=(5, 2))
    y = 1.2 + x @ coefficient
    contract = fit_numerical_ridge(x, y, [0.0, 1e-8], 1e10, 1e-8)
    assert contract.alpha == 0.0
    np.testing.assert_allclose(predict_ridge(contract, x), y, atol=1e-10)


def test_one_se_prefers_simpler_registered_candidate():
    losses = {4: [1.0, 1.1, 0.9, 1.0], 8: [0.99, 1.09, 0.89, 0.99]}
    selected = guarded_one_se(losses, [4, 8], maximum_relative_regret=0.02)
    assert selected["selected"] == 4


def test_era_recovers_stable_markov_sequence():
    a = np.array([[0.8, 0.0], [0.0, 0.5]])
    b = np.array([[1.0], [0.4]])
    c = np.array([[0.7, -0.2]])
    markov = []
    state_map = b.copy()
    for _ in range(12):
        markov.append(c @ state_map)
        state_map = a @ state_map
    contract = era_from_markov(np.asarray(markov), order=2, block_rows=6)
    assert contract.spectral_radius < 1.0
    impulse = np.zeros((12, 1))
    impulse[0, 0] = 1.0
    prediction = simulate_era(contract, impulse)
    # With D=0, the first Markov response appears one step after the impulse.
    np.testing.assert_allclose(prediction[1:11, 0], np.asarray(markov)[:10, 0, 0], atol=1e-8)


def test_sample_identity_is_segment_aware_and_stable():
    a = record(segment="2021-01-01-00-00-00_seg_1")
    b = record(segment="2021-01-01-00-00-00_seg_2")
    assert sample_id(a, 10) == sample_id(a, 10)
    assert sample_id(a, 10) != sample_id(b, 10)


def test_pf_selected_and_kwa_are_same_materialized_prediction():
    y = np.arange(20, dtype=float).reshape(5, 4)
    prediction = y + 0.2
    value = {
        "y": y,
        "K": y + 1.0,
        "KW": y + 0.5,
        "KWA": prediction,
        "PF_SELECTED": prediction,
        "ERA_K": None,
        "speed": np.ones(5),
    }
    metrics = aggregate_predictions([value])
    assert metrics["KWA"] == metrics["PF_SELECTED"]


def test_small_grouped_k_audit_uses_all_original_folds():
    rng = np.random.default_rng(9)
    inertia = np.array([0.0025, 0.0021, 0.0043])
    coefficient = rng.normal(scale=1e-7, size=(4, 4))
    items = []
    for index in range(10):
        rows = 96
        values = np.zeros((rows, 29), dtype=np.float64)
        values[:, 0] = np.arange(rows) * 0.0025
        values[:, 20:24] = rng.uniform(200.0, 900.0, size=(rows, 4))
        target = np.zeros((rows, 4))
        target[1:] = np.square(values[:-1, 20:24]) @ coefficient
        values[:, 1:4] = target[:, :3] / inertia
        values[:, 13] = target[:, 3] / 0.772
        partition = "train" if index < 8 else "validation"
        rec = SegmentRecord(
            flight_id=f"flight-{index}",
            segment_id=f"flight-{index}_seg_1",
            filename=f"flight-{index}_seg_1.csv",
            partition=partition,
            inner_fold=index % 4 if partition == "train" else None,
            zip_uncompressed_bytes=1,
            zip_crc32="0",
        )
        items.append(SegmentData(rec, values))
    config = {
        "entity_contract": {"inner_folds": 4},
        "runtime": {"workers": 2},
        "rigid_body_targets": {"mass_kg": 0.772, "inertia_diagonal_kg_m2": inertia.tolist()},
        "K": {
            "candidate_fir_histories_samples": [4, 8],
            "local_common_scoring_history_samples": 8,
            "numerical_ridge_grid": [0.0, 1e-12, 1e-8],
            "maximum_condition_number": 1e12,
            "maximum_relative_kkt_residual": 1e-8,
            "maximum_relative_regret_vs_best": 0.02,
            "minimum_relative_improvement_vs_zero": 0.01,
            "minimum_positive_fold_fraction": 0.75,
            "input_gate": {
                "minimum_prediction_variance_to_target_variance_ratio": 1e-8,
                "minimum_nonintercept_coefficient_abs": 1e-12,
                "maximum_mse_ratio_vs_axis_intercept_baseline": 1.02,
            },
        },
    }
    result, _, route_folds, contracts = run_k_development(items[:8], items[8:], config)
    assert result["status"] == "PASS"
    assert set(contracts) == {0, 1, 2, 3}
    assert set(route_folds) == {0, 1, 2, 3}
    for fold, roles in route_folds.items():
        assert {frame["segment"].record.inner_fold for frame in roles["evaluation"]} == {fold}
        assert all(frame["segment"].record.inner_fold != fold for frame in roles["fit"])
    assert all(len(value) == 4 for value in result["candidate_fold_losses"].values())
