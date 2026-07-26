"""Exact replay of frozen generator components for target semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ar_raphu.synthetic import (
    SyntheticSequence,
    _ar_response,
    _normalized_gaussian,
    second_truth_response,
    truth_response,
)


@dataclass(frozen=True, slots=True)
class SyntheticComponents:
    ar_contribution: np.ndarray
    x_contribution_by_variable: np.ndarray
    x_total_contribution: np.ndarray
    process_innovation: np.ndarray
    measurement_noise: np.ndarray


def true_kernel_surface(
    sequence: SyntheticSequence,
    variable: int,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Evaluate the frozen lag/amplitude truth kernel for one variable."""

    values = np.asarray(amplitudes, dtype=np.float64)
    truth = sequence.truth
    q_primary = np.asarray(truth["q_primary"], dtype=np.float64)
    q_secondary = np.asarray(truth["q_secondary"], dtype=np.float64)
    L_x = q_primary.shape[1]
    if variable not in truth["active_support"]:
        return np.zeros((L_x, len(values)), dtype=np.float64)
    if sequence.scenario == "AR-S4":
        result = np.empty((L_x, len(values)), dtype=np.float64)
        response = truth_response(variable, values)
        for index, amplitude in enumerate(values):
            center = 8.0 + 12.0 / (1.0 + np.exp(-2.0 * amplitude))
            result[:, index] = (
                _normalized_gaussian(L_x, center, 2.0) * response[index]
            )
        return result
    primary = q_primary[variable, :, None] * truth_response(variable, values)[None, :]
    if sequence.scenario == "AR-S3":
        secondary = q_secondary[variable, :, None] * second_truth_response(
            variable, values
        )[None, :]
        return 0.6 * primary + 0.4 * secondary
    return primary


def replay_synthetic_components(
    sequence: SyntheticSequence,
) -> SyntheticComponents:
    """Replay latent AR/X terms with the generator's target-minus-one lag semantics."""

    x = np.asarray(sequence.x, dtype=np.float64)
    latent = np.asarray(sequence.y_latent, dtype=np.float64)
    truth = sequence.truth
    q_y = np.asarray(truth["q_y"], dtype=np.float64)
    q_primary = np.asarray(truth["q_primary"], dtype=np.float64)
    q_secondary = np.asarray(truth["q_secondary"], dtype=np.float64)
    variables, L_x = q_primary.shape
    L_y = len(q_y)
    ar = np.zeros(len(latent), dtype=np.float64)
    by_variable = np.zeros((len(latent), variables), dtype=np.float64)
    start = max(L_x, L_y)

    for time in range(start, len(latent)):
        past_y = latent[time - 1 - np.arange(L_y)]
        ar[time] = float(np.dot(q_y, _ar_response(past_y)))
        lagged_x = x[time - 1 - np.arange(L_x)]
        if sequence.scenario == "AR-S4":
            for variable in truth["active_support"]:
                amplitude = x[time - 1, variable]
                center = 8.0 + 12.0 / (1.0 + np.exp(-2.0 * amplitude))
                q_dynamic = _normalized_gaussian(L_x, center, 2.0)
                by_variable[time, variable] = float(
                    np.dot(q_dynamic, truth_response(variable, lagged_x[:, variable]))
                )
        elif sequence.scenario == "AR-S3":
            for variable in truth["active_support"]:
                primary = np.dot(
                    q_primary[variable],
                    truth_response(variable, lagged_x[:, variable]),
                )
                secondary = np.dot(
                    q_secondary[variable],
                    second_truth_response(variable, lagged_x[:, variable]),
                )
                by_variable[time, variable] = 0.6 * primary + 0.4 * secondary
        else:
            for variable in truth["active_support"]:
                by_variable[time, variable] = float(
                    np.dot(
                        q_primary[variable],
                        truth_response(variable, lagged_x[:, variable]),
                    )
                )

    x_total = by_variable.sum(axis=1)
    innovation = latent - ar - x_total
    measurement_noise = (
        np.asarray(sequence.y_observed, dtype=np.float64)
        - np.asarray(sequence.y_measurement_clean, dtype=np.float64)
    )
    return SyntheticComponents(
        ar_contribution=ar,
        x_contribution_by_variable=by_variable,
        x_total_contribution=x_total,
        process_innovation=innovation,
        measurement_noise=measurement_noise,
    )
