from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .basis import (
    AmplitudeBasis,
    LagBasis,
    _hermite_nonlinear_features,
    _spline_matrix,
    amplitude_manifest,
    build_fit_design,
    build_lag_basis,
    build_projected_designs,
    fit_amplitude_basis,
    sobolev_penalty,
    split_projection_coefficients,
)
from .io_data import load_direction, metrics, sha256_array
from .solver import fit_gcv


def _matrix_sqrt(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    floor = max(float(np.max(values)) * 1e-14, 1e-15)
    values = np.maximum(values, floor)
    return (vectors * np.sqrt(values)) @ vectors.T, (vectors / np.sqrt(values)) @ vectors.T


def _fit_specs(
    sequence: np.ndarray,
    channels: list[str],
    quantiles: list[float],
    degree: int,
    band_spans: float,
) -> list[AmplitudeBasis]:
    return [
        fit_amplitude_basis(
            sequence[:, :, channel],
            channel=name,
            quantiles=quantiles,
            degree=degree,
            band_spans=band_spans,
        )
        for channel, name in enumerate(channels)
    ]


def _surface_arrays(
    lag: LagBasis,
    amplitude: list[AmplitudeBasis],
    beta: np.ndarray,
    nonlinear_coefficient: np.ndarray,
    nonlinear_slices: list[slice],
) -> dict[str, dict[str, np.ndarray]]:
    lag_minutes = np.linspace(0.0, 40.0, 161, dtype=np.float64)
    lag_grid = _spline_matrix(lag.knots, lag.degree, np.sqrt(lag_minutes / 40.0))
    output: dict[str, dict[str, np.ndarray]] = {}
    for channel, (spec, current_slice) in enumerate(zip(amplitude, nonlinear_slices)):
        xi_grid = np.linspace(spec.lower, spec.upper, 121, dtype=np.float64)
        value_grid = spec.mean + spec.scale * xi_grid
        nonlinear_basis, _ = _hermite_nonlinear_features(spec, value_grid)
        nonlinear_matrix = nonlinear_coefficient[current_slice].reshape(lag.number_of_basis, spec.reduced_size)
        beta_grid = lag_grid @ beta[:, channel]
        nonlinear_surface = lag_grid @ nonlinear_matrix @ nonlinear_basis.T
        linear_surface = beta_grid[:, None] * xi_grid[None, :]
        output[spec.channel] = {
            "lag_minutes": lag_minutes,
            "amplitude_value": value_grid,
            "amplitude_xi": xi_grid,
            "beta": beta_grid,
            "linear": linear_surface,
            "nonlinear": nonlinear_surface,
            "full": linear_surface + nonlinear_surface,
        }
    return output


def fit_model(
    sequence_train: np.ndarray,
    target_train: np.ndarray,
    sequence_predict: np.ndarray,
    *,
    config: dict[str, Any],
    lag_count: int,
    amplitude_quantiles: list[float],
    include_surfaces: bool,
) -> dict[str, Any]:
    sequence_train = np.asarray(sequence_train, dtype=np.float64)
    sequence_predict = np.asarray(sequence_predict, dtype=np.float64)
    target_train = np.asarray(target_train, dtype=np.float64)
    lag = build_lag_basis(
        number_of_basis=int(lag_count),
        degree=int(config["lag_basis"]["degree"]),
        sequence_steps=sequence_train.shape[1],
        cadence_sec=float(config["cadence_sec"]),
    )
    amplitude = _fit_specs(
        sequence_train,
        list(config["controls"]),
        amplitude_quantiles,
        int(config["amplitude_basis"]["degree"]),
        float(config["extension_band_knot_spans"]),
    )
    fit_design, coefficient_slices = build_fit_design(sequence_train, lag, amplitude, float(config["cadence_sec"]))
    penalty = sobolev_penalty(lag, amplitude)
    gcv = fit_gcv(fit_design, target_train, penalty, **{
        "log10_min": config["gcv"]["log10_min"],
        "log10_max": config["gcv"]["log10_max"],
        "bracket_points": config["gcv"]["bracket_points"],
        "brent_xatol": config["gcv"]["brent_xatol"],
    })
    beta, nonlinear_coefficient, nonlinear_slices = split_projection_coefficients(
        gcv.coefficient, lag, amplitude, coefficient_slices
    )
    train_linear_design, train_nonlinear_design, _, train_region = build_projected_designs(
        sequence_train, lag, amplitude, float(config["cadence_sec"])
    )
    predict_linear_design, predict_nonlinear_design, _, predict_region = build_projected_designs(
        sequence_predict, lag, amplitude, float(config["cadence_sec"])
    )
    beta_vector = beta.T.reshape(-1)
    train_linear = gcv.intercept + train_linear_design @ beta_vector
    train_full = train_linear + train_nonlinear_design @ nonlinear_coefficient
    predict_linear = gcv.intercept + predict_linear_design @ beta_vector
    predict_full = predict_linear + predict_nonlinear_design @ nonlinear_coefficient
    reconstruction_error = float(np.linalg.norm(train_full - gcv.prediction) / max(np.linalg.norm(gcv.prediction), 1e-30))

    gram_sqrt, gram_inv_sqrt = _matrix_sqrt(lag.gram)
    whitened = gram_sqrt @ beta
    u, singular, vt = np.linalg.svd(whitened, full_matrices=False)
    whitened_rank1 = singular[0] * np.outer(u[:, 0], vt[0]) if len(singular) else np.zeros_like(whitened)
    beta_rank1 = gram_inv_sqrt @ whitened_rank1
    rank1_vector = beta_rank1.T.reshape(-1)
    train_rank1 = gcv.intercept + train_linear_design @ rank1_vector
    predict_rank1 = gcv.intercept + predict_linear_design @ rank1_vector
    energy_ratio = float(singular[0] ** 2 / max(np.sum(singular**2), 1e-30)) if len(singular) else 0.0

    constant_residual = 0.0
    linear_residual = 0.0
    nonlinear_energy: dict[str, float] = {}
    linear_energy: dict[str, float] = {}
    channel_prediction: dict[str, np.ndarray] = {}
    for channel, (spec, current_slice) in enumerate(zip(amplitude, nonlinear_slices)):
        matrix = nonlinear_coefficient[current_slice].reshape(lag.number_of_basis, spec.reduced_size)
        constant_residual = max(constant_residual, float(np.max(np.abs(matrix @ np.mean(evaluate_training_nonlinear(spec, sequence_train[:, :, channel]), axis=0)))))
        linear_residual = max(linear_residual, float(np.max(np.abs(matrix @ spec.linear_coordinate))))
        linear_energy[spec.channel] = float(beta[:, channel] @ lag.gram @ beta[:, channel])
        nonlinear_energy[spec.channel] = float(np.trace(matrix.T @ lag.gram @ matrix))
        linear_block = predict_linear_design[:, channel * lag.number_of_basis : (channel + 1) * lag.number_of_basis]
        nonlinear_block = predict_nonlinear_design[:, current_slice]
        channel_prediction[spec.channel] = linear_block @ beta[:, channel] + nonlinear_block @ nonlinear_coefficient[current_slice]

    artifact: dict[str, Any] = {
        "lag": lag,
        "amplitude": amplitude,
        "coefficient_slices": coefficient_slices,
        "nonlinear_slices": nonlinear_slices,
        "coefficient": gcv.coefficient,
        "intercept": gcv.intercept,
        "beta": beta,
        "beta_rank1": beta_rank1,
        "nonlinear_coefficient": nonlinear_coefficient,
        "train_prediction_full": train_full,
        "train_prediction_linear": train_linear,
        "train_prediction_rank1": train_rank1,
        "prediction_full": predict_full,
        "prediction_linear": predict_linear,
        "prediction_rank1": predict_rank1,
        "predict_region": predict_region,
        "train_region": train_region,
        "channel_prediction": channel_prediction,
        "selected_lambda": gcv.selected_lambda,
        "effective_df": gcv.effective_df,
        "gcv_value": gcv.gcv,
        "gcv_curve": gcv.curve,
        "kkt_residual": gcv.kkt_residual,
        "condition_number": gcv.condition_number,
        "generalized_eigenvalues": gcv.generalized_eigenvalues,
        "singular_values": singular,
        "rank1_energy_ratio": energy_ratio,
        "rank1_time_shape": u[:, 0] if len(singular) else np.zeros(lag.number_of_basis),
        "rank1_channel_coordinates": vt[0] if len(singular) else np.zeros(len(amplitude)),
        "reconstruction_error": reconstruction_error,
        "constant_constraint_residual": constant_residual,
        "linear_constraint_residual": linear_residual,
        "linear_energy": linear_energy,
        "nonlinear_energy": nonlinear_energy,
        "amplitude_manifest": [amplitude_manifest(spec) for spec in amplitude],
        "partition_error": lag.partition_error,
        "coefficient_sha256": sha256_array(gcv.coefficient),
        "prediction_sha256": sha256_array(predict_full),
    }
    if include_surfaces:
        artifact["surfaces"] = _surface_arrays(lag, amplitude, beta, nonlinear_coefficient, nonlinear_slices)
    return artifact


def evaluate_training_nonlinear(spec: AmplitudeBasis, values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    xi = (flattened - spec.mean) / spec.scale
    from .basis import evaluate_reduced
    return evaluate_reduced(spec, flattened) - xi[:, None] * spec.linear_coordinate[None, :]


def fit_task(payload: dict[str, Any]) -> dict[str, Any]:
    shared_root = Path(payload["shared_root"])
    direction = str(payload["direction"])
    mode = str(payload["mode"])
    config = payload["config"]
    data = load_direction(shared_root, direction)
    if mode == "main":
        artifact = fit_model(
            data.train["sequence_u"], data.train["target_z"], data.test["sequence_u"],
            config=config,
            lag_count=int(config["lag_basis"]["number_of_basis"]),
            amplitude_quantiles=list(config["amplitude_basis"]["quantiles"]),
            include_surfaces=True,
        )
        artifact.update({
            "direction": direction,
            "sample_id": data.test["sample_id"].astype("U"),
            "target_z": data.test["target_z"].astype(np.float64),
            "evaluation_mask": data.test["evaluation_mask"].astype(bool),
            "train_sample_id": data.train["sample_id"].astype("U"),
            "train_target_z": data.train["target_z"].astype(np.float64),
        })
        test_standardized = [
            (np.asarray(data.test["sequence_u"][:, :, channel], dtype=np.float64) - spec.mean) / spec.scale
            for channel, spec in enumerate(artifact["amplitude"])
        ]
        common_mask = np.ones(len(data.test["target_z"]), dtype=bool)
        for channel, (spec, values) in enumerate(zip(artifact["amplitude"], test_standardized)):
            test_lower, test_upper = float(np.min(values)), float(np.max(values))
            lower, upper = max(spec.lower, test_lower), min(spec.upper, test_upper)
            common_mask &= np.all((values >= lower) & (values <= upper), axis=1)
        artifact["common_support_mask"] = common_mask
        return artifact
    if mode == "refine":
        artifact = fit_model(
            data.train["sequence_u"], data.train["target_z"], data.test["sequence_u"],
            config=config,
            lag_count=int(config["mesh_refinement"]["lag_basis"]),
            amplitude_quantiles=list(config["mesh_refinement"]["amplitude_quantiles"]),
            include_surfaces=False,
        )
        return {"direction": direction, "prediction_full": artifact["prediction_full"], "selected_lambda": artifact["selected_lambda"], "kkt_residual": artifact["kkt_residual"], "condition_number": artifact["condition_number"]}
    if mode == "fold":
        training = np.asarray(payload["training"], dtype=np.int64)
        validation = np.asarray(payload["validation"], dtype=np.int64)
        artifact = fit_model(
            data.train["sequence_u"][training], data.train["target_z"][training], data.train["sequence_u"][validation],
            config=config,
            lag_count=int(config["lag_basis"]["number_of_basis"]),
            amplitude_quantiles=list(config["amplitude_basis"]["quantiles"]),
            include_surfaces=False,
        )
        return {"direction": direction, "fold": int(payload["fold"]), "validation": validation, "prediction_full": artifact["prediction_full"], "selected_lambda": artifact["selected_lambda"], "kkt_residual": artifact["kkt_residual"]}
    raise ValueError(mode)


def direction_metrics(artifact: dict[str, Any], key: str) -> dict[str, float | int]:
    mask = artifact["evaluation_mask"]
    return metrics(artifact["target_z"][mask], artifact[key][mask])
