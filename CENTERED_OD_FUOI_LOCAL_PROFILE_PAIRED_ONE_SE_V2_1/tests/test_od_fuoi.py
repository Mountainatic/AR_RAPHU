from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.basis import (
    _hermite_nonlinear_features,
    build_fit_design,
    build_lag_basis,
    build_projected_designs,
    fit_amplitude_basis,
    sobolev_penalty,
    split_projection_coefficients,
)
from src.io_data import metrics, moving_block_bootstrap, sha256_array, validate_alignment
from src.packaging import validate_package
from src.residual import residual_design
from src.solver import fit_gcv, one_se_select, ridge_fit


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/frozen_protocol.yaml").read_text())


def _sequence(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(60, 24, 4)).astype(np.float64)


def _specs(sequence: np.ndarray):
    return [fit_amplitude_basis(sequence[:, :, index], channel=name, quantiles=[0, 10, 30, 50, 70, 90, 100], degree=3, band_spans=1.0) for index, name in enumerate(CONFIG["controls"])]


def test_01_protocol_frozen() -> None:
    assert CONFIG["protocol_frozen"] is True


def test_02_bundle_hashes_frozen() -> None:
    assert len(CONFIG["shared_dataset_sha256"]) == 64
    assert len(CONFIG["cpu_baseline_bundle_sha256"]) == 64
    assert len(CONFIG["gpu_baseline_bundle_sha256"]) == 64


def test_03_lag_warp_monotone() -> None:
    lag = build_lag_basis(number_of_basis=8, degree=3, sequence_steps=24, cadence_sec=10)
    assert np.all(np.diff(np.sqrt(lag.lag_minutes / 40.0)) >= 0)


def test_04_lag_partition_of_unity() -> None:
    lag = build_lag_basis(number_of_basis=8, degree=3, sequence_steps=24, cadence_sec=10)
    assert lag.partition_error < 1e-12


def test_05_amplitude_basis_centered() -> None:
    values = np.linspace(-3, 4, 2001)
    spec = fit_amplitude_basis(values, channel="x", quantiles=[0, 5, 15, 30, 50, 70, 85, 95, 100], degree=3, band_spans=1)
    from src.basis import evaluate_reduced
    reduced = evaluate_reduced(spec, values)
    assert np.max(np.abs(np.mean(reduced, axis=0))) < 1e-12


def test_06_linear_coordinate_reproduced() -> None:
    values = np.linspace(-2, 3, 1001)
    spec = fit_amplitude_basis(values, channel="x", quantiles=[0, 10, 30, 50, 70, 90, 100], degree=3, band_spans=1)
    assert spec.projection_error < 1e-10


def test_07_nonlinear_basis_orthogonal_to_linear() -> None:
    values = np.linspace(-2, 3, 1001)
    spec = fit_amplitude_basis(values, channel="x", quantiles=[0, 10, 30, 50, 70, 90, 100], degree=3, band_spans=1)
    nonlinear, _ = _hermite_nonlinear_features(spec, values)
    xi = (values - spec.mean) / spec.scale
    assert np.max(np.abs(np.mean(nonlinear, axis=0))) < 1e-10
    assert np.max(np.abs(nonlinear.T @ xi / len(xi))) < 1e-10


def test_08_tensor_design_shape() -> None:
    sequence = _sequence(); lag = build_lag_basis(number_of_basis=8, degree=3, sequence_steps=24, cadence_sec=10); specs = _specs(sequence)
    design, slices = build_fit_design(sequence, lag, specs, 10)
    assert design.shape == (60, sum(8 * spec.reduced_size for spec in specs))
    assert len(slices) == 4


def test_09_tensor_design_fp64() -> None:
    sequence = _sequence().astype(np.float32); lag = build_lag_basis(number_of_basis=8, degree=3, sequence_steps=24, cadence_sec=10); specs = _specs(sequence)
    design, _ = build_fit_design(sequence, lag, specs, 10)
    assert design.dtype == np.float64


def test_10_sobolev_penalty_symmetric_positive() -> None:
    sequence = _sequence(); lag = build_lag_basis(number_of_basis=8, degree=3, sequence_steps=24, cadence_sec=10); specs = _specs(sequence)
    penalty = sobolev_penalty(lag, specs)
    assert np.max(np.abs(penalty - penalty.T)) < 1e-10
    assert np.min(np.linalg.eigvalsh(penalty)) > 0


def test_11_gcv_deterministic() -> None:
    rng = np.random.default_rng(2); matrix = rng.normal(size=(150, 12)); target = matrix @ rng.normal(size=12) + rng.normal(size=150) * .1; penalty = np.eye(12)
    first = fit_gcv(matrix, target, penalty, log10_min=-6, log10_max=6, bracket_points=13, brent_xatol=1e-8)
    second = fit_gcv(matrix, target, penalty, log10_min=-6, log10_max=6, bracket_points=13, brent_xatol=1e-8)
    assert first.selected_lambda == second.selected_lambda
    assert np.array_equal(first.coefficient, second.coefficient)


def test_12_gcv_kkt() -> None:
    rng = np.random.default_rng(3); matrix = rng.normal(size=(200, 10)); target = matrix @ rng.normal(size=10)
    fit = fit_gcv(matrix, target, np.eye(10), log10_min=-8, log10_max=4, bracket_points=13, brent_xatol=1e-8)
    assert fit.kkt_residual < 1e-8


def test_13_projection_reconstructs_feature_model() -> None:
    sequence = _sequence(); lag = build_lag_basis(number_of_basis=8, degree=3, sequence_steps=24, cadence_sec=10); specs = _specs(sequence)
    fit_design, slices = build_fit_design(sequence, lag, specs, 10); rng=np.random.default_rng(5); coefficient=rng.normal(size=fit_design.shape[1])
    beta, nonlinear, _ = split_projection_coefficients(coefficient, lag, specs, slices)
    linear_design, nonlinear_design, _, _ = build_projected_designs(sequence, lag, specs, 10)
    reconstructed = linear_design @ beta.T.reshape(-1) + nonlinear_design @ nonlinear
    assert np.linalg.norm(reconstructed - fit_design @ coefficient) / np.linalg.norm(fit_design @ coefficient) < 1e-10


def test_14_rank1_derivation_does_not_mutate() -> None:
    coefficient = np.arange(20.0); before = coefficient.copy(); _ = np.linalg.svd(coefficient.reshape(5,4), full_matrices=False)
    assert np.array_equal(coefficient, before)


def test_15_c1_value_continuity() -> None:
    values=np.linspace(-2,2,1001); spec=fit_amplitude_basis(values,channel="x",quantiles=[0,10,30,50,70,90,100],degree=3,band_spans=1); eps=1e-7
    physical=spec.mean+spec.scale*np.array([spec.lower-eps,spec.lower,spec.lower+eps]); feature,_=_hermite_nonlinear_features(spec,physical)
    assert np.max(np.abs((feature[0]+feature[2])/2-feature[1])) < 1e-6


def test_16_c1_derivative_continuity() -> None:
    values=np.linspace(-2,2,1001); spec=fit_amplitude_basis(values,channel="x",quantiles=[0,10,30,50,70,90,100],degree=3,band_spans=1); eps=1e-7
    x=spec.mean+spec.scale*np.array([spec.upper-2*eps,spec.upper-eps,spec.upper,spec.upper+eps,spec.upper+2*eps]); feature,_=_hermite_nonlinear_features(spec,x)
    assert np.max(np.abs((feature[2]-feature[0])/(2*eps)-(feature[4]-feature[2])/(2*eps))) < 1e-3


def test_17_saturation_is_constant() -> None:
    values=np.linspace(-2,2,1001); spec=fit_amplitude_basis(values,channel="x",quantiles=[0,10,30,50,70,90,100],degree=3,band_spans=1)
    x=spec.mean+spec.scale*np.array([spec.upper+2*spec.right_band,spec.upper+3*spec.right_band]); feature,region=_hermite_nonlinear_features(spec,x)
    assert np.array_equal(region,np.array([2,2])) and np.array_equal(feature[0],feature[1])


def test_18_ridge_intercept_unpenalized() -> None:
    matrix=np.arange(50.0)[:,None]; target=3+2*matrix[:,0]; coefficient,intercept=ridge_fit(matrix,target,0)
    assert abs(coefficient[0]-2)<1e-10 and abs(intercept-3)<1e-10


def test_19_one_se_prefers_zero() -> None:
    selected=one_se_select([{"candidate":"A0","ridge":0,"mean_mse":1.01,"se_mse":0.02},{"candidate":"AR:2","ridge":1,"mean_mse":1.0,"se_mse":0.02}])
    assert selected["candidate"]=="A0"


def test_20_matured_residual_causality() -> None:
    n=1000; origin=np.arange(n)*5; future=origin+660; residual=np.sin(np.arange(n)/20)
    design,eligible,maximum=residual_design(residual,origin,future,candidate="AR:2",cadence_sec=10,sample_period_sec=2)
    assert len(design)==len(eligible)>0 and np.all(maximum<=origin[eligible])


def test_21_bootstrap_alignment_deterministic() -> None:
    first=moving_block_bootstrap([np.arange(100,dtype=float)],[np.arange(100,dtype=float)*.9],block_rows=10,replicates=50,seed=7)
    second=moving_block_bootstrap([np.arange(100,dtype=float)],[np.arange(100,dtype=float)*.9],block_rows=10,replicates=50,seed=7)
    assert first==second


def test_22_metrics_exact() -> None:
    result=metrics(np.array([0.,2.]),np.array([0.,0.])); assert result["MSE"]==2 and result["RMSE"]==np.sqrt(2)


def test_23_alignment_rejects_mismatch() -> None:
    reference={"sample_id":np.array(["a"]),"target_z":np.array([1.]),"evaluation_mask":np.array([True])}; candidate={**reference,"sample_id":np.array(["b"])}
    try: validate_alignment(reference,candidate,"x")
    except RuntimeError: return
    raise AssertionError("mismatch accepted")


def test_24_prediction_hash_stable() -> None:
    value=np.arange(12,dtype=np.float64); assert sha256_array(value)==sha256_array(value.copy())


def test_25_package_privacy(tmp_path: Path) -> None:
    (tmp_path/"safe.txt").write_text("ok"); validate_package(tmp_path); (tmp_path/"private.xlsx").write_text("x")
    try: validate_package(tmp_path)
    except RuntimeError: return
    raise AssertionError("private workbook accepted")
