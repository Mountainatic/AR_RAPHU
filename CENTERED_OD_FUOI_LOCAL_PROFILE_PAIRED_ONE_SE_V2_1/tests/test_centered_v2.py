from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.centered import centered_increment, support_audit
from src.edf import ContinuousProfile, prepare_edf_map
from src.io_data import moving_block_bootstrap, sha256_array
from src.packaging import PACKAGE, validate_package
from src.v2_model import diagnostic_smoothing_curve, fit_prepared, prepare_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/frozen_protocol.yaml").read_text())


def _ridge_map(seed: int = 1, n: int = 160, p: int = 12):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, p))
    target = matrix @ rng.normal(size=p) + rng.normal(size=n) * 0.2
    validation = rng.normal(size=(40, p))
    return prepare_edf_map(matrix, target, np.eye(p), validation)


class _SyntheticMap:
    def __init__(self, target: np.ndarray, loss):
        self.target = target
        self.loss = loss

    def predict_at_df(self, d: float, relative_tolerance: float):
        error = np.sqrt(max(float(self.loss(d)), 0.0))
        return self.target + error, 1.0 / d, d


def _profile(loss, *, upper: float = 20.0):
    target = np.zeros(20)
    return ContinuousProfile(
        [_SyntheticMap(target, loss) for _ in range(4)], [target] * 4, [20] * 4,
        lower=1.001, upper=upper, max_evaluations=80, d_tolerance=0.05,
        interpolation_tolerance=1e-4, inversion_tolerance=1e-8,
    )


def test_01_protocol_is_v2_and_frozen():
    assert CONFIG["schema"].endswith("V2") and CONFIG["protocol_frozen"] is True


def test_02_all_four_bundle_hashes_are_frozen():
    assert all(len(CONFIG[key]) == 64 for key in ("shared_dataset_sha256", "cpu_baseline_bundle_sha256", "gpu_baseline_bundle_sha256", "v1_results_sha256"))


def test_03_centered_formula_current_to_past():
    value = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
    delta = centered_increment(value)
    assert np.array_equal(delta[:, 3], value[:, 3] - value[:, 0])


def test_04_centered_lag_zero_is_exact_zero():
    rng = np.random.default_rng(2)
    delta = centered_increment(rng.normal(size=(8, 12, 4)))
    assert np.array_equal(delta[:, 0], np.zeros((8, 4)))


def test_05_support_audit_detects_extension():
    train = np.zeros((2, 3, 1)); test = train.copy(); test[0, 2, 0] = 1
    assert support_audit(train, test)[0]["extension_ratio"] > 0


def test_06_centering_uses_no_future_sample():
    sequence = np.arange(30.0).reshape(2, 5, 3)
    changed = sequence.copy(); changed[1] += 1000
    assert np.array_equal(centered_increment(sequence)[0], centered_increment(changed)[0])


def test_07_tensor_preparation_is_fp64():
    rng = np.random.default_rng(3)
    sequence = rng.normal(size=(100, 12, 4)).astype(np.float32)
    config = {**CONFIG, "lag_basis": {"number_of_basis": 6, "degree": 3, "warp": "sqrt"}, "amplitude_basis": {"degree": 3, "quantiles": [0, 10, 30, 50, 70, 90, 100]}}
    prepared = prepare_model(sequence, rng.normal(size=100), sequence[:10], config=config, lag_count=6, amplitude_quantiles=config["amplitude_basis"]["quantiles"])
    assert prepared.fit_design.dtype == np.float64


def test_08_penalty_is_spd():
    current = _ridge_map()
    assert np.min(np.linalg.eigvalsh(current.penalty)) > 0


def test_09_generalized_eigenspectrum_is_nonnegative():
    assert np.min(_ridge_map().eigenvalues) >= 0


def test_10_edf_is_strictly_decreasing_in_lambda():
    current = _ridge_map()
    values = [current.df_at_lambda(value) for value in (1e-8, 1e-4, 1.0, 1e4)]
    assert np.all(np.diff(values) < 0)


def test_11_brent_inversion_matches_target():
    current = _ridge_map(); target = 5.25
    lam = current.lambda_for_df(target)
    assert abs(current.df_at_lambda(lam) - target) < 1e-7


def test_12_stable_interval_nonempty():
    interval = _ridge_map().stable_interval(condition_epsilon_limit=1e-6, lower_excess=1e-6)
    assert interval["lower_df"] < interval["upper_df"]


def test_13_equal_target_df_across_folds():
    maps = [_ridge_map(seed) for seed in range(4)]
    target = 4.5
    attained = [current.predict_at_df(target)[2] for current in maps]
    assert max(attained) - min(attained) < 1e-7


def test_14_synthetic_unimodal_profile():
    profile = _profile(lambda d: 1 + (d - 7) ** 2 / 100)
    resolved = profile.resolve()
    assert abs(float(resolved["d_min"]) - 7) < 0.1


def test_15_synthetic_bimodal_profile():
    profile = _profile(lambda d: 1 + min((d - 5) ** 2 / 30, 0.1 + (d - 14) ** 2 / 20))
    resolved = profile.resolve()
    assert len(resolved["minima"]) >= 1 and abs(float(resolved["d_min"]) - 5) < 0.2


def test_16_all_detected_minima_have_brackets():
    resolved = _profile(lambda d: 1 + (d - 8) ** 2 / 50).resolve()
    assert all(row["bracket_left"] <= row["edf"] <= row["bracket_right"] for row in resolved["minima"])


def test_17_continuous_one_se_uses_connected_left_component():
    profile = _profile(lambda d: 1 + (d - 8) ** 2 / 50)
    resolved = profile.resolve(); selected = profile.one_se(resolved)
    assert selected["d_1se"] <= selected["d_min"]


def test_18_upper_bound_is_reported_unresolved():
    resolved = _profile(lambda d: 30 - d, upper=10).resolve()
    assert resolved["upper_bound_hit"] is True and resolved["resolved"] is False


def test_19_full_refit_matches_target_df():
    current = _ridge_map(); fit = current.fit_at_df(6.5)
    assert abs(fit.attained_df - 6.5) < 1e-7


def test_20_diagnostic_curve_never_selects():
    assert all(row["used_for_selection"] == 0 for row in diagnostic_smoothing_curve(_ridge_map(), 7))


def _small_fit():
    rng = np.random.default_rng(9)
    sequence = rng.normal(size=(180, 12, 4))
    target = rng.normal(size=180)
    config = {**CONFIG, "lag_basis": {"number_of_basis": 6, "degree": 3, "warp": "sqrt"}, "amplitude_basis": {"degree": 3, "quantiles": [0, 10, 30, 50, 70, 90, 100]}}
    prepared = prepare_model(sequence, target, sequence[:30], config=config, lag_count=6, amplitude_quantiles=config["amplitude_basis"]["quantiles"])
    return fit_prepared(prepared, 10.0, config=config, include_surfaces=True)


def test_21_projection_reconstruction():
    assert _small_fit()["reconstruction_error"] < 1e-10


def test_22_per_lag_nonlinear_orthogonality():
    fit = _small_fit()
    assert fit["constant_constraint_residual"] < 1e-10 and fit["linear_constraint_residual"] < 1e-10


def test_23_rank1_is_exact_linear_subset():
    fit = _small_fit()
    assert fit["prediction_rank1"].shape == fit["prediction_linear"].shape and fit["rank1_energy_ratio"] <= 1 + 1e-12


def test_24_surface_contains_centered_axis():
    fit = _small_fit()
    assert all("amplitude_value" in surface for surface in fit["surfaces"].values())


def test_25_prediction_hash_is_deterministic():
    fit = _small_fit()
    assert fit["prediction_sha256"] == sha256_array(fit["prediction_full"])


def test_26_paired_bootstrap_is_deterministic():
    a = moving_block_bootstrap([np.arange(100.0)], [np.arange(100.0) * .9], block_rows=10, replicates=20, seed=2)
    b = moving_block_bootstrap([np.arange(100.0)], [np.arange(100.0) * .9], block_rows=10, replicates=20, seed=2)
    assert a == b


def test_27_mesh_refinement_is_audit_only():
    assert CONFIG["mesh_refinement"]["lag_basis"] == 28 and "mesh_refinement" not in CONFIG["continuous_edf"]


def test_28_fp64_kkt():
    assert _ridge_map().fit_at_df(6.0).kkt_residual < 1e-8


def test_29_package_privacy(tmp_path: Path):
    (tmp_path / "safe.txt").write_text("ok")
    validate_package(tmp_path)


def test_30_package_schema_is_v2():
    assert PACKAGE == "CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS"
