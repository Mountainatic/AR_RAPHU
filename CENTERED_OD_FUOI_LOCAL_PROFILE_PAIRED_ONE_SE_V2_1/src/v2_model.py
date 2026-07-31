from __future__ import annotations

from dataclasses import dataclass
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
from .centered import centered_increment
from .edf import EDFMap, prepare_edf_map
from .io_data import sha256_array


@dataclass
class PreparedModel:
    train_delta: np.ndarray
    predict_delta: np.ndarray
    lag: LagBasis
    amplitude: list[AmplitudeBasis]
    fit_design: np.ndarray
    predict_design: np.ndarray
    penalty: np.ndarray
    coefficient_slices: list[slice]
    predict_region: np.ndarray
    edf_map: EDFMap


def _fit_specs(sequence: np.ndarray, config: dict[str, Any], quantiles: list[float]) -> list[AmplitudeBasis]:
    return [
        fit_amplitude_basis(
            sequence[:, :, index],
            channel=name,
            quantiles=quantiles,
            degree=int(config["amplitude_basis"]["degree"]),
            band_spans=float(config["extension_band_knot_spans"]),
        )
        for index, name in enumerate(config["controls"])
    ]


def _extended_design(
    sequence: np.ndarray,
    lag: LagBasis,
    amplitude: list[AmplitudeBasis],
    cadence_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    linear, nonlinear, nonlinear_slices, region = build_projected_designs(sequence, lag, amplitude, cadence_sec)
    blocks: list[np.ndarray] = []
    for channel, (spec, current_slice) in enumerate(zip(amplitude, nonlinear_slices)):
        linear_block = linear[:, channel * lag.number_of_basis : (channel + 1) * lag.number_of_basis]
        nonlinear_block = nonlinear[:, current_slice]
        q = spec.linear_coordinate
        linear_lift = np.einsum("nt,m->ntm", linear_block, q, optimize=True).reshape(len(sequence), -1)
        blocks.append(linear_lift + nonlinear_block)
    return np.concatenate(blocks, axis=1), region


def prepare_model(
    sequence_train: np.ndarray,
    target_train: np.ndarray,
    sequence_predict: np.ndarray,
    *,
    config: dict[str, Any],
    lag_count: int,
    amplitude_quantiles: list[float],
) -> PreparedModel:
    train_delta = centered_increment(sequence_train)
    predict_delta = centered_increment(sequence_predict)
    lag = build_lag_basis(
        number_of_basis=int(lag_count),
        degree=int(config["lag_basis"]["degree"]),
        sequence_steps=train_delta.shape[1],
        cadence_sec=float(config["cadence_sec"]),
    )
    amplitude = _fit_specs(train_delta, config, amplitude_quantiles)
    fit_design, coefficient_slices = build_fit_design(train_delta, lag, amplitude, float(config["cadence_sec"]))
    predict_design, predict_region = _extended_design(predict_delta, lag, amplitude, float(config["cadence_sec"]))
    penalty = sobolev_penalty(lag, amplitude)
    current_map = prepare_edf_map(fit_design, target_train, penalty, predict_design)
    return PreparedModel(
        train_delta=train_delta,
        predict_delta=predict_delta,
        lag=lag,
        amplitude=amplitude,
        fit_design=fit_design,
        predict_design=predict_design,
        penalty=penalty,
        coefficient_slices=coefficient_slices,
        predict_region=predict_region,
        edf_map=current_map,
    )


def _full_features(spec: AmplitudeBasis, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    xi = (values - spec.mean) / spec.scale
    nonlinear, _ = _hermite_nonlinear_features(spec, values)
    return nonlinear + xi[..., None] * spec.linear_coordinate


def _per_lag_projection(
    prepared: PreparedModel,
    coefficient: np.ndarray,
) -> dict[str, Any]:
    lag = prepared.lag
    n_lags = prepared.train_delta.shape[1]
    channels = prepared.train_delta.shape[2]
    m = np.zeros((n_lags, channels), dtype=np.float64)
    beta = np.zeros((n_lags, channels), dtype=np.float64)
    train_nonlinear = np.zeros_like(prepared.train_delta)
    predict_nonlinear = np.zeros_like(prepared.predict_delta)
    train_full_values = np.zeros_like(prepared.train_delta)
    predict_full_values = np.zeros_like(prepared.predict_delta)
    constant_residual = 0.0
    linear_residual = 0.0
    linear_energy: dict[str, float] = {}
    nonlinear_energy: dict[str, float] = {}
    channel_prediction: dict[str, np.ndarray] = {}
    dt = float(prepared.lag.lag_minutes[1] - prepared.lag.lag_minutes[0]) if n_lags > 1 else 1.0

    for channel, (spec, current_slice) in enumerate(zip(prepared.amplitude, prepared.coefficient_slices)):
        coefficient_matrix = coefficient[current_slice].reshape(lag.number_of_basis, spec.reduced_size)
        lag_coefficients = lag.sample_design @ coefficient_matrix
        train_features = _full_features(spec, prepared.train_delta[:, :, channel])
        predict_features = _full_features(spec, prepared.predict_delta[:, :, channel])
        train_k = np.einsum("nlm,lm->nl", train_features, lag_coefficients, optimize=True)
        predict_k = np.einsum("nlm,lm->nl", predict_features, lag_coefficients, optimize=True)
        train_full_values[:, :, channel] = train_k
        predict_full_values[:, :, channel] = predict_k
        for lag_index in range(n_lags):
            x = prepared.train_delta[:, lag_index, channel]
            y = train_k[:, lag_index]
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            centered = x - x_mean
            variance = float(centered @ centered)
            slope = float(centered @ (y - y_mean) / variance) if variance > 1e-24 else 0.0
            intercept = y_mean - slope * x_mean
            m[lag_index, channel] = intercept
            beta[lag_index, channel] = slope
        train_n = train_k - m[:, channel][None, :] - beta[:, channel][None, :] * prepared.train_delta[:, :, channel]
        predict_n = predict_k - m[:, channel][None, :] - beta[:, channel][None, :] * prepared.predict_delta[:, :, channel]
        train_nonlinear[:, :, channel] = train_n
        predict_nonlinear[:, :, channel] = predict_n
        constant_residual = max(constant_residual, float(np.max(np.abs(np.mean(train_n, axis=0)))))
        linear_residual = max(linear_residual, float(np.max(np.abs(np.mean(train_n * prepared.train_delta[:, :, channel], axis=0)))))
        linear_energy[spec.channel] = float(np.sum(beta[:, channel] ** 2) * dt)
        nonlinear_energy[spec.channel] = float(np.mean(np.sum(train_n**2, axis=1) * dt))
        channel_prediction[spec.channel] = np.sum(predict_k, axis=1) * dt

    return {
        "m": m,
        "beta": beta,
        "train_nonlinear_values": train_nonlinear,
        "predict_nonlinear_values": predict_nonlinear,
        "train_full_values": train_full_values,
        "predict_full_values": predict_full_values,
        "constant_constraint_residual": constant_residual,
        "linear_constraint_residual": linear_residual,
        "linear_energy": linear_energy,
        "nonlinear_energy": nonlinear_energy,
        "channel_prediction": channel_prediction,
        "dt_minutes": dt,
    }


def _surfaces(prepared: PreparedModel, coefficient: np.ndarray, projection: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    lag_minutes = np.linspace(0.0, 40.0, 161, dtype=np.float64)
    lag_design = _spline_matrix(prepared.lag.knots, prepared.lag.degree, np.sqrt(lag_minutes / 40.0))
    output: dict[str, dict[str, np.ndarray]] = {}
    for channel, (spec, current_slice) in enumerate(zip(prepared.amplitude, prepared.coefficient_slices)):
        delta_grid = np.linspace(spec.mean + spec.scale * spec.lower, spec.mean + spec.scale * spec.upper, 121)
        features = _full_features(spec, delta_grid)
        coefficient_matrix = coefficient[current_slice].reshape(prepared.lag.number_of_basis, spec.reduced_size)
        full = lag_design @ coefficient_matrix @ features.T
        m_grid = np.interp(lag_minutes, prepared.lag.lag_minutes, projection["m"][:, channel])
        beta_grid = np.interp(lag_minutes, prepared.lag.lag_minutes, projection["beta"][:, channel])
        linear = beta_grid[:, None] * delta_grid[None, :]
        nonlinear = full - m_grid[:, None] - linear
        output[spec.channel] = {
            "lag_minutes": lag_minutes,
            "amplitude_value": delta_grid,
            "amplitude_xi": (delta_grid - spec.mean) / spec.scale,
            "m": m_grid,
            "beta": beta_grid,
            "linear": linear,
            "nonlinear": nonlinear,
            "full": linear + nonlinear,
            "raw_full_with_m": full,
        }
    return output


def fit_prepared(prepared: PreparedModel, target_df: float, *, config: dict[str, Any], include_surfaces: bool) -> dict[str, Any]:
    fit = prepared.edf_map.fit_at_df(target_df, float(config["continuous_edf"]["inversion_relative_tolerance"]))
    projection = _per_lag_projection(prepared, fit.coefficient)
    dt = projection["dt_minutes"]
    adjusted_intercept = fit.intercept + float(np.sum(projection["m"]) * dt)
    train_linear = adjusted_intercept + np.sum(projection["beta"][None, :, :] * prepared.train_delta, axis=(1, 2)) * dt
    predict_linear = adjusted_intercept + np.sum(projection["beta"][None, :, :] * prepared.predict_delta, axis=(1, 2)) * dt
    train_full = train_linear + np.sum(projection["train_nonlinear_values"], axis=(1, 2)) * dt
    predict_full = predict_linear + np.sum(projection["predict_nonlinear_values"], axis=(1, 2)) * dt
    direct_train = fit.intercept + prepared.fit_design @ fit.coefficient
    reconstruction_error = float(np.linalg.norm(train_full - direct_train) / max(np.linalg.norm(direct_train), 1e-30))

    u, singular, vt = np.linalg.svd(projection["beta"], full_matrices=False)
    beta_rank1 = singular[0] * np.outer(u[:, 0], vt[0]) if len(singular) else np.zeros_like(projection["beta"])
    train_rank1 = adjusted_intercept + np.sum(beta_rank1[None, :, :] * prepared.train_delta, axis=(1, 2)) * dt
    predict_rank1 = adjusted_intercept + np.sum(beta_rank1[None, :, :] * prepared.predict_delta, axis=(1, 2)) * dt
    rank1_energy = float(singular[0] ** 2 / max(np.sum(singular**2), 1e-30)) if len(singular) else 0.0

    artifact: dict[str, Any] = {
        "lag": prepared.lag,
        "amplitude": prepared.amplitude,
        "coefficient_slices": prepared.coefficient_slices,
        "coefficient": fit.coefficient,
        "intercept": fit.intercept,
        "centered_intercept": adjusted_intercept,
        "m": projection["m"],
        "beta": projection["beta"],
        "beta_rank1": beta_rank1,
        "train_prediction_full": train_full,
        "train_prediction_linear": train_linear,
        "train_prediction_rank1": train_rank1,
        "prediction_full": predict_full,
        "prediction_linear": predict_linear,
        "prediction_rank1": predict_rank1,
        "predict_region": prepared.predict_region,
        "channel_prediction": projection["channel_prediction"],
        "selected_lambda": fit.selected_lambda,
        "effective_df": fit.attained_df,
        "target_edf": target_df,
        "kkt_residual": fit.kkt_residual,
        "condition_number": fit.condition_number,
        "generalized_eigenvalues": prepared.edf_map.eigenvalues,
        "singular_values": singular,
        "rank1_energy_ratio": rank1_energy,
        "rank1_time_shape": u[:, 0] if len(singular) else np.zeros(prepared.train_delta.shape[1]),
        "rank1_channel_coordinates": vt[0] if len(singular) else np.zeros(prepared.train_delta.shape[2]),
        "reconstruction_error": reconstruction_error,
        "constant_constraint_residual": projection["constant_constraint_residual"],
        "linear_constraint_residual": projection["linear_constraint_residual"],
        "linear_energy": projection["linear_energy"],
        "nonlinear_energy": projection["nonlinear_energy"],
        "amplitude_manifest": [amplitude_manifest(spec) for spec in prepared.amplitude],
        "partition_error": prepared.lag.partition_error,
        "coefficient_sha256": sha256_array(fit.coefficient),
        "prediction_sha256": sha256_array(predict_full),
        "train_delta": prepared.train_delta,
        "predict_delta": prepared.predict_delta,
    }
    if include_surfaces:
        artifact["surfaces"] = _surfaces(prepared, fit.coefficient, projection)
    return artifact


def diagnostic_smoothing_curve(current_map: EDFMap, points: int = 25) -> list[dict[str, float]]:
    center = current_map._scale_log10()
    logs = np.linspace(center - 12.0, center + 12.0, points)
    centered_y = current_map.target - current_map.y_mean
    total_energy = float(centered_y @ centered_y)
    rows: list[dict[str, float]] = []
    for log_value in logs:
        lam = float(10.0**log_value)
        denominator = current_map.eigenvalues + current_map.n_rows * lam
        coefficient_eigen = current_map.projected_rhs / denominator
        prediction = current_map.y_mean + (current_map.matrix - current_map.x_mean) @ current_map.eigenvectors @ coefficient_eigen
        rss = float(np.sum((current_map.target - prediction) ** 2))
        df = current_map.df_at_lambda(lam)
        gcv = rss / max((current_map.n_rows - df) ** 2, 1e-30) * current_map.n_rows
        penalty_norm = float(np.sum(coefficient_eigen**2))
        reml = float(current_map.n_rows * np.log(max(rss / current_map.n_rows, 1e-30)) + np.sum(np.log(np.maximum(denominator, 1e-300))))
        rows.append({
            "log10_lambda": float(log_value),
            "lambda": lam,
            "effective_df": df,
            "gcv": gcv,
            "reml_diagnostic": reml,
            "rss": rss,
            "penalty_norm": penalty_norm,
            "target_energy": total_energy,
            "used_for_selection": 0.0,
        })
    return rows


def predict_from_artifact(artifact: dict[str, Any], sequence_u: np.ndarray, cadence_sec: float) -> dict[str, np.ndarray]:
    delta = centered_increment(sequence_u)
    design, region = _extended_design(delta, artifact["lag"], artifact["amplitude"], cadence_sec)
    full = artifact["intercept"] + design @ artifact["coefficient"]
    dt = float(cadence_sec) / 60.0
    linear = artifact["centered_intercept"] + np.sum(artifact["beta"][None, :, :] * delta, axis=(1, 2)) * dt
    return {"full": np.asarray(full), "linear": np.asarray(linear), "region": region, "delta": delta}
