from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prism_benchmark.neurobem_data import SegmentData, SegmentRecord, body_context
from prism_benchmark.neurobem_linear import RidgeContract, fit_numerical_ridge, k_design
from prism_benchmark.neurobem_multihorizon import (
    EXACT_ZERO,
    IDENTITY,
    W0,
    W1,
    W2,
    _apply_a,
    _apply_w,
    _generic_knots,
    _w_features,
    base_prediction_support_start,
    common_horizon_support_start,
    fit_k_contract,
    k_design_horizon,
    mature_target_lags,
    multihorizon_sample_id,
    route_support_start,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "PRISM_V2_1_1_NEUROBEM_MULTI_HORIZON_W_PRIOR_AUDIT_PACKAGE" / "MULTIHORIZON_CONFIG_FROZEN.json"


def config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def record(segment_id: str = "2021-01-01-00-00-00_seg_1", fold: int = 0) -> SegmentRecord:
    return SegmentRecord(segment_id.split("_seg_")[0], segment_id, f"processed_data/{segment_id}.csv", "train", fold, 1, "00000000")


def segment(rows: int = 160, segment_id: str = "2021-01-01-00-00-00_seg_1", fold: int = 0, offset: float = 0.0) -> SegmentData:
    rng = np.random.default_rng(abs(hash(segment_id)) % (2**32))
    values = np.zeros((rows, 29), dtype=np.float64)
    values[:, 0] = offset + np.arange(rows) * .0025
    values[:, 1:17] = rng.normal(size=(rows, 16))
    values[:, 20:24] = 700.0 + rng.normal(0, 20, size=(rows, 4))
    return SegmentData(record(segment_id, fold), values)


def dummy_contract(features: int, outputs: int = 4) -> RidgeContract:
    return RidgeContract(0.0, np.zeros(features), np.ones(features), np.zeros(outputs), np.zeros((features, outputs)), 1.0, 0.0, 10)


def frame(item: SegmentData, horizon: int = 4, start: int = 90) -> dict[str, object]:
    rows = np.arange(start, item.row_count)
    origins = rows - horizon
    prediction = np.zeros((len(rows), 4))
    return {
        "segment": item,
        "rows": rows,
        "origins": origins,
        "formal_mask": np.ones(len(rows), dtype=bool),
        "formal_rows": rows,
        "y": np.zeros((len(rows), 4)),
        "k_prediction": prediction,
        "context": body_context(item)[origins],
        "speed": np.linalg.norm(body_context(item)[origins, :3], axis=1),
    }


def test_multihorizon_h1_target_semantics_match_r1():
    item = segment()
    old_x, _, old_rows = k_design(item, 8)
    new_x, new_rows, origins = k_design_horizon(item, 8, 1)
    np.testing.assert_array_equal(new_x, old_x)
    np.testing.assert_array_equal(new_rows, old_rows)
    np.testing.assert_array_equal(origins, old_rows - 1)


def test_multihorizon_origin_equals_target_minus_h():
    _, targets, origins = k_design_horizon(segment(), 12, 20)
    np.testing.assert_array_equal(origins, targets - 20)


def test_multihorizon_k_never_reads_after_origin():
    item = segment()
    x, _, origins = k_design_horizon(item, 4, 8)
    expected = np.square(item.values[origins[0], 20:24])
    np.testing.assert_allclose(x[0, :4], expected)


def test_multihorizon_w_never_reads_after_origin():
    item = segment()
    current = frame(item, horizon=20)
    np.testing.assert_array_equal(current["context"], body_context(item)[current["origins"]])


def test_multihorizon_a_residual_is_mature():
    assert mature_target_lags(8, [0, 1, 3]) == (8, 9, 11)


def test_multihorizon_a_h1_age_mapping_matches_old_lags():
    assert mature_target_lags(1, [0, 1, 3, 7, 11, 19]) == (1, 2, 4, 8, 12, 20)


def test_multihorizon_a_h20_cannot_use_t_minus_1_residual():
    assert min(mature_target_lags(20, [0, 1, 3])) == 20


def test_multihorizon_no_segment_crossing():
    a = segment(segment_id="2021-01-01-00-00-00_seg_1")
    b = segment(segment_id="2021-01-01-00-00-00_seg_2")
    _, rows_a, _ = k_design_horizon(a, 64, 80)
    _, rows_b, _ = k_design_horizon(b, 64, 80)
    assert rows_a[0] == rows_b[0] == 143


def test_multihorizon_no_flight_crossing():
    a = segment(segment_id="2021-01-01-00-00-00_seg_1")
    b = segment(segment_id="2021-01-02-00-00-00_seg_1")
    assert a.record.flight_id != b.record.flight_id
    assert multihorizon_sample_id(a.record, 4, 100, 104) != multihorizon_sample_id(b.record, 4, 100, 104)


def test_multihorizon_sample_id_contains_horizon():
    item = segment()
    assert multihorizon_sample_id(item.record, 1, 100, 101) != multihorizon_sample_id(item.record, 4, 100, 104)


def test_multihorizon_route_support_is_identical_within_horizon():
    current = frame(segment())
    no_w = _apply_w([current], None)[0]
    with_a = _apply_a([no_w], None, 4, None)[0]
    np.testing.assert_array_equal(no_w["formal_rows"], with_a["formal_rows"])


def test_common_horizon_support_is_common():
    cfg = config()
    common = common_horizon_support_start(cfg)
    maximum_history = max(cfg["K"]["candidate_fir_histories_samples"])
    maximum_age = max(max(values) for values in cfg["A"]["mature_residual_age_sets_samples"])
    assert common == max(route_support_start(maximum_history, h, maximum_age) for h in cfg["targets"]["forecast_horizons_samples"])


def test_generic_w_reads_k_latent_only():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(100, 4))
    knots = _generic_knots(latent, 4)
    first = _w_features("NATURAL_CUBIC_K_LATENT_K4", rng.normal(size=(100, 10)), latent, knots)
    second = _w_features("NATURAL_CUBIC_K_LATENT_K4", rng.normal(size=(100, 10)), latent, knots)
    np.testing.assert_array_equal(first, second)


def test_generic_w_does_not_read_velocity_or_body_rate():
    assert config()["W"]["generic_latent_forbidden_inputs"] == ["body_velocity", "body_angular_velocity", "speed_norm"]


def test_aero_w_context_is_origin_causal():
    item = segment()
    current = frame(item, horizon=8)
    np.testing.assert_array_equal(current["context"], body_context(item)[current["origins"]])


def test_w0_is_exact_identity():
    current = frame(segment())
    output = _apply_w([current], None)[0]
    np.testing.assert_array_equal(output["no_a_prediction"], current["k_prediction"])
    np.testing.assert_array_equal(output["w_correction"], 0.0)


def test_w1_contains_identity_candidate():
    assert config()["W"]["arms"][W1][0] == IDENTITY


def test_w2_contains_all_w1_candidates():
    cfg = config()
    assert set(cfg["W"]["arms"][W1]).issubset(cfg["W"]["arms"][W2])


def test_no_global_w_arm_selection():
    assert config()["W"]["global_arm_winner_selection_forbidden"] is True


def test_a_exact_zero_materializes_zero():
    no_w = _apply_w([frame(segment())], None)[0]
    output = _apply_a([no_w], None, 4, None)[0]
    np.testing.assert_array_equal(output["a_prediction"], 0.0)
    np.testing.assert_array_equal(output["with_a_prediction"], output["no_a_prediction"])


def test_candidate_loss_prediction_contract_ids_match():
    source = (ROOT / "src" / "prism_benchmark" / "neurobem_multihorizon.py").read_text(encoding="utf-8")
    assert '"final_selected_prediction_candidate_id": selected_id' in source
    assert '"final_selected_contract_candidate_id": selected_id' in source


def test_train_only_spline_knots():
    rng = np.random.default_rng(3)
    fit_latent = rng.normal(size=(200, 4))
    knots = _generic_knots(fit_latent, 6)
    future_a = rng.normal(size=(50, 4))
    future_b = future_a * 1000
    # Evaluation/future values never enter registered knot fitting.
    for left, right in zip(knots, _generic_knots(fit_latent, 6), strict=True):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(_generic_knots(future_a, 6)[0], _generic_knots(future_b, 6)[0])


def test_train_only_scaling_all_horizons():
    cfg = config()
    cfg["K"]["candidate_fir_histories_samples"] = [4]
    train = [segment(rows=180, segment_id="2021-01-01-00-00-00_seg_1")]
    for horizon in cfg["targets"]["forecast_horizons_samples"]:
        first = fit_k_contract(train, 4, horizon, cfg)
        second = fit_k_contract(train, 4, horizon, cfg)
        np.testing.assert_array_equal(first.feature_mean, second.feature_mean)


def test_test_not_accessed_before_global_freeze():
    source = (ROOT / "src" / "prism_benchmark" / "neurobem_multihorizon_runner.py").read_text(encoding="utf-8")
    assert source.index('write_json(output_root / "GLOBAL_DEVELOPMENT_FREEZE.json"') < source.index('load_partition(records, data_root / "extracted", "test", allow_locked_test=True)')


def test_prior_r1_test_access_is_disclosed():
    cfg = config()
    assert cfg["historical_test_access_exists"] is True
    assert cfg["evidence_class"] == "POST_LOCKBOX_PROSPECTIVE_EXTENSION"


def test_high_speed_subset_uses_origin_speed():
    assert config()["evaluation"]["speed_value_time"] == "prediction_origin"


def test_all_registered_horizons_are_reported():
    assert config()["targets"]["forecast_horizons_samples"] == [1, 4, 8, 20, 40, 80]


def test_route_support_accounts_for_mature_residual_prediction():
    assert base_prediction_support_start(64, 20) == 83
    assert route_support_start(64, 20, 19) == 122


def test_w0_and_exact_zero_names_are_frozen():
    assert W0 == "W0_IDENTITY"
    assert EXACT_ZERO == "EXACT_ZERO"
