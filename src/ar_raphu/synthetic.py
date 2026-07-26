"""Pre-registered AR-S0--AR-S7 synthetic truth generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data_protocol import PREDICTION_HORIZONS
from .protocol_config import load_protocol_config


SCENARIOS = tuple(f"AR-S{index}" for index in range(8))


def _normalized_gamma(length: int, shape: float, scale: float) -> np.ndarray:
    lag = np.arange(length, dtype=np.float64)
    logits = (shape - 1.0) * np.log(lag + 1.0e-3)
    logits -= (lag + 1.0e-3) / scale
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def _normalized_gaussian(
    length: int, center: float, standard_deviation: float
) -> np.ndarray:
    lag = np.arange(length, dtype=np.float64)
    weights = np.exp(
        -0.5 * ((lag - float(center)) / float(standard_deviation)) ** 2
    )
    return weights / weights.sum()


def _truth_response(index: int, values: np.ndarray) -> np.ndarray:
    function = index % 3
    if function == 0:
        return np.tanh(values)
    if function == 1:
        return 0.5 * values**2 - 0.5
    return np.sin(1.5 * values)


def _second_truth_response(index: int, values: np.ndarray) -> np.ndarray:
    function = index % 3
    if function == 0:
        return np.sin(0.75 * values)
    if function == 1:
        return np.tanh(1.5 * values)
    return 0.25 * values**3


def truth_response(index: int, values: np.ndarray) -> np.ndarray:
    """Read-only public view of the frozen primary synthetic response."""

    return _truth_response(index, np.asarray(values))


def second_truth_response(index: int, values: np.ndarray) -> np.ndarray:
    """Read-only public view of the frozen secondary synthetic response."""

    return _second_truth_response(index, np.asarray(values))


def _ar_response(values: np.ndarray) -> np.ndarray:
    return 0.85 * values + 0.10 * np.tanh(values)


@dataclass(frozen=True, slots=True)
class SyntheticSequence:
    scenario: str
    seed: int
    x: np.ndarray
    y_observed: np.ndarray
    y_measurement_clean: np.ndarray
    y_latent: np.ndarray
    target_start: int
    target_stop: int
    split_target_intervals: dict[str, tuple[int, int]]
    truth: dict[str, Any]

    @property
    def usable_target_count(self) -> int:
        return self.target_stop - self.target_start


def _stationary_var1(
    rng: np.random.Generator,
    length: int,
    variables: int,
    *,
    coefficient: float,
    cross_correlation: float,
) -> np.ndarray:
    covariance = (
        (1.0 - cross_correlation) * np.eye(variables)
        + cross_correlation * np.ones((variables, variables))
    )
    cholesky = np.linalg.cholesky(covariance)
    innovations = rng.standard_normal((length, variables)) @ cholesky.T
    values = np.zeros((length, variables), dtype=np.float64)
    innovation_scale = np.sqrt(1.0 - coefficient**2)
    for time in range(1, length):
        values[time] = (
            coefficient * values[time - 1]
            + innovation_scale * innovations[time]
        )
    return values


def generate_synthetic_sequence(
    scenario: str,
    *,
    seed: int,
    n_samples: int | None = None,
    external_variables: int | None = None,
    snr_db: float | None = None,
) -> SyntheticSequence:
    """Generate one chronological sequence with direct-forecast-safe history."""

    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}.")
    config = load_protocol_config(require_phase1_frozen=True)
    synthetic = config["phase1_synthetic"]
    common = synthetic["scenario_parameter_grids"]["common"]
    n_samples = (
        int(synthetic["sample_sizes"]["core"])
        if n_samples is None
        else int(n_samples)
    )
    variables = (
        int(synthetic["external_variable_count"]["core"])
        if external_variables is None
        else int(external_variables)
    )
    snr_db = (
        float(synthetic["noise_or_SNR_levels"]["core_db"])
        if snr_db is None
        else float(snr_db)
    )
    if n_samples <= 0 or variables < 3:
        raise ValueError("n_samples must be positive and variables must be at least 3.")

    L_x = int(common["L_x"])
    L_y = int(synthetic["AR_kernel_and_response"]["L_y"])
    history = max(L_x, L_y) + max(PREDICTION_HORIZONS) - 1
    burn_in = int(common["burn_in"])
    total = burn_in + history + n_samples
    rng = np.random.default_rng(seed)
    x_full = _stationary_var1(
        rng,
        total,
        variables,
        coefficient=float(common["input_AR_coefficient"]),
        cross_correlation=float(common["weak_cross_correlation"]),
    )

    q_y = _normalized_gamma(L_y, shape=1.5, scale=1.5)
    generator_version = int(synthetic["generator_version"])
    if generator_version != 2:
        raise ValueError("Only frozen synthetic generator version 2 is supported.")
    process_innovation_sd = float(
        synthetic["latent_process_innovation"]["standard_deviation"]
    )
    q_primary = np.zeros((variables, L_x), dtype=np.float64)
    q_secondary = np.zeros_like(q_primary)
    active = [] if scenario == "AR-S0" else [0, 1, 2]
    scenario_config = synthetic["scenario_parameter_grids"][scenario]

    if scenario in {"AR-S1", "AR-S5", "AR-S6", "AR-S7"}:
        gamma_parameters = synthetic["scenario_parameter_grids"]["AR-S1"][
            "shape_scale_by_active_variable"
        ]
        for variable, (shape, scale) in zip(active, gamma_parameters, strict=True):
            q_primary[variable] = _normalized_gamma(
                L_x, float(shape), float(scale)
            )
    elif scenario == "AR-S2":
        for variable, centers in zip(
            active, scenario_config["mode_pairs"], strict=True
        ):
            q_primary[variable] = 0.5 * (
                _normalized_gaussian(
                    L_x,
                    centers[0],
                    scenario_config["mode_standard_deviation"],
                )
                + _normalized_gaussian(
                    L_x,
                    centers[1],
                    scenario_config["mode_standard_deviation"],
                )
            )
    elif scenario == "AR-S3":
        for variable in active:
            q_primary[variable] = _normalized_gaussian(
                L_x,
                scenario_config["early_mode_centers"][variable],
                scenario_config["mode_standard_deviation"],
            )
            q_secondary[variable] = _normalized_gaussian(
                L_x,
                scenario_config["late_mode_centers"][variable],
                scenario_config["mode_standard_deviation"],
            )

    latent = np.zeros(total, dtype=np.float64)
    dynamic_delay_means = np.full(
        (total, len(active)), np.nan, dtype=np.float64
    )
    start_generation = max(L_x, L_y)
    innovation_scale = np.sqrt(1.0 - 0.7**2)
    for time in range(start_generation, total):
        if scenario == "AR-S7":
            x_full[time, 0] = (
                0.7 * x_full[time - 1, 0]
                + 0.3 * np.tanh(latent[time - 1])
                + innovation_scale * rng.standard_normal()
            )
            x_full[time, 3] = x_full[time, 0] + 0.1 * rng.standard_normal()

        past_y = latent[time - 1 - np.arange(L_y)]
        value = float(np.dot(q_y, _ar_response(past_y)))
        lagged_x = x_full[time - 1 - np.arange(L_x)]
        if scenario == "AR-S4":
            means = []
            for variable in active:
                amplitude = x_full[time - 1, variable]
                center = 8.0 + 12.0 / (1.0 + np.exp(-2.0 * amplitude))
                q_dynamic = _normalized_gaussian(L_x, center, 2.0)
                value += float(
                    np.dot(q_dynamic, _truth_response(variable, lagged_x[:, variable]))
                )
                means.append(center)
            dynamic_delay_means[time] = means
        elif scenario == "AR-S3":
            for variable in active:
                primary = np.dot(
                    q_primary[variable],
                    _truth_response(variable, lagged_x[:, variable]),
                )
                secondary = np.dot(
                    q_secondary[variable],
                    _second_truth_response(variable, lagged_x[:, variable]),
                )
                value += 0.6 * float(primary) + 0.4 * float(secondary)
        else:
            for variable in active:
                value += float(
                    np.dot(
                        q_primary[variable],
                        _truth_response(variable, lagged_x[:, variable]),
                    )
                )
        latent[time] = value + process_innovation_sd * rng.standard_normal()

    measurement_clean = latent.copy()
    if scenario == "AR-S5":
        filter_weights = np.asarray(
            scenario_config["causal_filter_weights"], dtype=np.float64
        )
        for time in range(len(filter_weights) - 1, total):
            measurement_clean[time] = np.dot(
                filter_weights,
                latent[time - np.arange(len(filter_weights))],
            )
    elif scenario == "AR-S6":
        delay = int(scenario_config["delay_samples"])
        measurement_clean[delay:] = latent[:-delay]
        measurement_clean[:delay] = 0.0

    target_start_full = burn_in + history
    target_stop_full = target_start_full + n_samples
    signal = measurement_clean[target_start_full:target_stop_full]
    signal_variance = float(np.var(signal))
    noise_standard_deviation = np.sqrt(
        signal_variance / (10.0 ** (snr_db / 10.0))
    )
    noise = rng.normal(0.0, noise_standard_deviation, size=total)
    observed = measurement_clean + noise

    x = x_full[burn_in:].copy()
    y_observed = observed[burn_in:].copy()
    y_measurement_clean = measurement_clean[burn_in:].copy()
    y_latent = latent[burn_in:].copy()
    target_start = history
    target_stop = history + n_samples
    train_stop = target_start + int(np.floor(0.6 * n_samples))
    validation_stop = target_start + int(np.floor(0.8 * n_samples))
    splits = {
        "train": (target_start, train_stop),
        "validation": (train_stop, validation_stop),
        "test": (validation_stop, target_stop),
    }

    truth: dict[str, Any] = {
        "scenario": scenario,
        "active_support": active,
        "L_x": L_x,
        "L_y": L_y,
        "q_y": q_y,
        "q_primary": q_primary,
        "q_secondary": q_secondary,
        "rank_by_variable": {
            str(variable): (2 if scenario == "AR-S3" else 1)
            for variable in active
        },
        "snr_db": snr_db,
        "generator_version": generator_version,
        "latent_process_innovation_standard_deviation": process_innovation_sd,
        "noise_standard_deviation": noise_standard_deviation,
        "lag_semantics": (
            "kernel_index_0_multiplies_value_at_target_time_minus_1"
        ),
        "current_target_in_input": False,
        "future_X_in_primary_input": False,
    }
    if scenario == "AR-S4":
        truth["dynamic_delay_means"] = dynamic_delay_means[burn_in:].copy()
    if scenario == "AR-S5":
        truth["measurement_filter_weights"] = np.asarray(
            scenario_config["causal_filter_weights"], dtype=np.float64
        )
    if scenario == "AR-S6":
        truth["measurement_delay_samples"] = int(
            scenario_config["delay_samples"]
        )

    return SyntheticSequence(
        scenario=scenario,
        seed=seed,
        x=x,
        y_observed=y_observed,
        y_measurement_clean=y_measurement_clean,
        y_latent=y_latent,
        target_start=target_start,
        target_stop=target_stop,
        split_target_intervals=splits,
        truth=truth,
    )
