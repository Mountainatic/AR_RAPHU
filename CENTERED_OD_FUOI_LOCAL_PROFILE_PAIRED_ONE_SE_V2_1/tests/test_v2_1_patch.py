from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.v2_1_packaging import PACKAGE, validate_v2_1_package
from src.v2_1_selection import LocalPairedProfile, inverse_log_excess, log_excess_coordinate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/frozen_protocol_v2_1.yaml").read_text())


class SyntheticMap:
    def __init__(self, target: np.ndarray, loss, phase: float = 0.0):
        self.target = target
        self.loss = loss
        pattern = np.sin(np.linspace(phase, phase + 7.0, len(target))) + 1.5
        self.pattern = pattern / np.sqrt(np.mean(pattern**2))

    def predict_at_df(self, d: float, relative_tolerance: float):
        prediction = self.target + self.pattern * np.sqrt(max(float(self.loss(d)), 0.0))
        return prediction, 1.0 / max(d, 1e-12), d


def profile(loss, *, lower: float = 1.0, upper: float = 40.0, points: int = 41, reps: int = 40):
    targets = [np.zeros(80) for _ in range(4)]
    rows = [{"edf": float(d), "mean_mse": float(loss(d))} for d in np.linspace(lower, upper, points)]
    return LocalPairedProfile(
        [SyntheticMap(target, loss, index) for index, target in enumerate(targets)], targets, rows,
        lower=lower, upper=upper, max_evaluations=120, d_tolerance=.05,
        inversion_tolerance=1e-8, bootstrap_replicates=reps, bootstrap_seed=20260731,
    )


def test_01_v2_profile_cache_imported():
    current = profile(lambda d: 1 + (d - 7) ** 2 / 50)
    assert current.imported_count == 41 and current.new_evaluations == 0


def test_02_log_excess_mapping_monotone_and_invertible():
    values = np.array([1.0, 2.0, 5.0, 20.0]); epsilon = 1e-4
    mapped = [log_excess_coordinate(value, 1.0, epsilon) for value in values]
    assert np.all(np.diff(mapped) > 0)
    assert np.max(np.abs([inverse_log_excess(value, 1.0, epsilon) for value in mapped] - values)) < 1e-12


def test_03_far_field_high_loss_is_pruned():
    result = profile(lambda d: 1 + (d - 7) ** 2 / 20).discover_and_refine()
    assert any(row["classification"] == "FAR_FIELD_PRUNED" for row in result["far_field"])


def test_04_far_field_interpolation_not_local_gate():
    result = profile(lambda d: 1 + (d - 7) ** 2 / 20).discover_and_refine()
    assert result["far_field_interpolation_resolved"] is False
    assert result["local_minimum_resolved"] is True


def test_05_synthetic_double_basin_is_observed():
    loss = lambda d: 1 + min((d - 7) ** 2 / 20, 0.2 + (d - 27) ** 2 / 15)
    result = profile(loss, points=81).discover_and_refine()
    assert len(result["candidate_basins"]) >= 2


def test_06_three_local_refinements_agree():
    result = profile(lambda d: 1 + (d - 9) ** 2 / 30).discover_and_refine()
    best = min(result["candidate_basins"], key=lambda row: row["loss"])
    assert len(best["independent_runs"]) == 3 and best["independent_spread"] <= .05


def test_07_fold_predictions_cached_for_paired_errors():
    current = profile(lambda d: 1 + (d - 8) ** 2 / 40)
    current.paired_difference(6, 8, 20)
    assert len(current.prediction_cache) == 2 and all(len(value) == 4 for value in current.prediction_cache.values())


def test_08_bootstrap_indices_never_cross_folds():
    current = profile(lambda d: 1 + (d - 8) ** 2 / 40)
    indices = current._bootstrap_indices(20)
    assert len(indices) == 4 and all(np.min(value) >= 0 and np.max(value) < 80 for value in indices)


def test_09_each_fold_has_independent_block_draws():
    current = profile(lambda d: 1 + (d - 8) ** 2 / 40)
    indices = current._bootstrap_indices(20)
    assert not np.array_equal(indices[0], indices[1])


def test_10_common_resamples_reused_across_d():
    current = profile(lambda d: 1 + (d - 8) ** 2 / 40)
    first = current._bootstrap_indices(20)
    second = current._bootstrap_indices(20)
    assert first is second


def test_11_delta_at_reference_is_exact_zero():
    current = profile(lambda d: 1 + (d - 8) ** 2 / 40)
    result = current.paired_difference(8, 8, 20)
    assert result.mean_delta == 0 and result.se_delta == 0 and result.g == 0


def test_12_paired_se_is_finite():
    current = profile(lambda d: 1 + (d - 8) ** 2 / 40)
    result = current.paired_difference(5, 8, 20)
    assert np.isfinite(result.se_delta) and result.se_delta >= 0


def test_13_paired_one_se_selects_left_connected_boundary():
    current = profile(lambda d: .3 + (d - 8) ** 2 / 100)
    selected = current.paired_one_se(8, primary_block_rows=20, sensitivity_block_rows=(10, 30))
    assert selected["paired_one_se_boundary_resolved"] and selected["d_paired_1se"] <= 8


def test_14_paired_profile_reports_root_bracket_or_lower_hit():
    current = profile(lambda d: .3 + (d - 8) ** 2 / 100)
    selected = current.paired_one_se(8, primary_block_rows=20, sensitivity_block_rows=(10, 30))
    assert selected["root_bracket"] is not None or selected["paired_one_se_hits_lower_bound"]


def test_15_lower_bound_hit_is_explicit():
    current = profile(lambda d: 1.0)
    selected = current.paired_one_se(8, primary_block_rows=20, sensitivity_block_rows=(10, 30))
    assert selected["paired_one_se_hits_lower_bound"] is True


def test_16_upper_bound_hit_is_separate():
    result = profile(lambda d: 50 - d).discover_and_refine()
    assert result["upper_bound_hit"] or result["global_basin_discovery"] == "UNRESOLVED"


def test_17_selection_change_threshold_is_frozen():
    assert CONFIG["continuous_edf_v2_1"]["local_d_tolerance"] == .05


def test_18_old_surface_reuse_guard_is_in_runner():
    source = (ROOT / "src/v2_1_runner.py").read_text(encoding="utf-8")
    assert "OLD_V2_SURFACE_REUSED_AFTER_SELECTION_CHANGE" in source


def test_19_estimator_gate_ignores_far_field_interpolation():
    source = (ROOT / "src/v2_1_runner.py").read_text(encoding="utf-8")
    gate = source[source.index("selection_resolved = all("):source.index("if selection_resolved:")]
    assert "far_field_interpolation" not in gate


def test_20_package_roundtrip_contract(tmp_path: Path):
    (tmp_path / "safe.txt").write_text("ok")
    validate_v2_1_package(tmp_path)
    assert PACKAGE.endswith("V2_1_RESULTS")
