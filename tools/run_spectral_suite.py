#!/usr/bin/env python3
"""Frozen Spectral v0.3 through v0.3.3 staged experiment runner."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.spectral.contracts import ExperimentContract
from ar_raphu.spectral.amplitude_domain import (
    AmplitudeDomain,
    AmplitudeOutOfDomainError,
)
from ar_raphu.spectral.capacity_diagnostics import (
    direct_apply_projected_kernel,
    direct_apply_truth_kernel,
)
from ar_raphu.spectral.capacity_matrix import (
    build_matrix_from_histories,
    build_single_variable_matrix,
)
from ar_raphu.spectral.capacity_matrix import (
    select_minimum_validation_mse,
    smoothing_pairs,
)
from ar_raphu.spectral.excitation import (
    chronological_split_indices,
    permuted_marginal_excitation,
    space_filling_core_excitation,
    space_filling_history_excitation,
)
from ar_raphu.spectral.operator_metrics import empirical_operator_nrmse
from ar_raphu.spectral.representation_certificate import (
    certify_resolution_roles,
)
from ar_raphu.spectral.resolution_roles import role_from_config
from ar_raphu.spectral.design import build_spectral_design
from ar_raphu.spectral.gram_svd import gram_whitened_svd
from ar_raphu.spectral.metrics import normalized_root_mean_square_error, r2
from ar_raphu.spectral.metrics import mse, rmse
from ar_raphu.spectral.penalties import tensor_penalty
from ar_raphu.spectral.projection import (
    identity_lag_basis,
    project_tensor_surface,
)
from ar_raphu.spectral.scenario_registry import (
    SCENARIO_REGISTRY,
    s4c_mixed_difference,
)
from ar_raphu.spectral.spline_basis import (
    CenteredSplineBasis,
    clamped_knots,
    evaluate_basis,
)
from ar_raphu.spectral.synthetic_components import (
    e2a_component_target,
    replay_synthetic_components,
    true_kernel_surface,
)
from ar_raphu.spectral.truth_spectrum import truth_spectrum
from ar_raphu.spectral.weighted_projection import (
    normalized_trapezoidal_weights,
    weighted_tensor_projection,
)
from ar_raphu.spectral.solver import solve_full_kernel
from ar_raphu.spectral.solver import solve_full_kernel_pcg
from ar_raphu.synthetic import (
    generate_synthetic_sequence,
    second_truth_response,
    truth_response,
)


CONFIG_PATH = ROOT / "configs" / "spectral_v03.yaml"
RESULT_ROOT = ROOT / "results" / "spectral_v03"


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty metrics table.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_e0(config: dict[str, object]) -> str:
    common = config["common"]
    output = RESULT_ROOT / "E0"
    contract = ExperimentContract(
        scientific_question="Do frozen generator components exactly reconstruct targets?",
        target_contains_ar=True,
        model_contains_ar=True,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="all",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)
    rows: list[dict[str, object]] = []
    identity: list[dict[str, object]] = []
    scenarios = ["AR-S0", "AR-S1", "AR-S2", "AR-S3", "AR-S4", "AR-S7"]
    for scenario in scenarios:
        for seed in common["development_seeds"]:
            sequence = generate_synthetic_sequence(
                scenario,
                seed=int(seed),
                n_samples=int(common["n_samples"]),
                external_variables=int(common["external_variables"]),
            )
            components = replay_synthetic_components(sequence)
            latent_reconstructed = (
                components.ar_contribution
                + components.x_total_contribution
                + components.process_innovation
            )
            latent_error = float(
                np.max(np.abs(sequence.y_latent - latent_reconstructed))
            )
            observed_error = float(
                np.max(
                    np.abs(
                        sequence.y_observed
                        - sequence.y_measurement_clean
                        - components.measurement_noise
                    )
                )
            )
            identity.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "latent_identity_max_abs": latent_error,
                    "measurement_identity_max_abs": observed_error,
                }
            )
            if latent_error > 1e-10 or observed_error > 1e-10:
                write_json(output / "component_identity.json", identity)
                raise RuntimeError("E0 component replay identity failed.")
            for partition, (start, stop) in sequence.split_target_intervals.items():
                target = sequence.y_observed[start:stop]
                g_x = components.x_total_contribution[start:stop]
                g_ar = components.ar_contribution[start:stop]
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "partition": partition,
                        "r2_y_x_truth": r2(target, g_x),
                        "r2_y_ar_truth": r2(target, g_ar),
                        "r2_y_xar_truth": r2(target, g_x + g_ar),
                        "r2_x_self": r2(g_x, g_x),
                        "target_variance": float(np.var(target)),
                        "x_variance_ratio": float(
                            np.var(g_x) / max(np.var(target), np.finfo(float).eps)
                        ),
                        "ar_variance_ratio": float(
                            np.var(g_ar) / max(np.var(target), np.finfo(float).eps)
                        ),
                        "process_variance_ratio": float(
                            np.var(components.process_innovation[start:stop])
                            / max(np.var(target), np.finfo(float).eps)
                        ),
                        "noise_variance_ratio": float(
                            np.var(components.measurement_noise[start:stop])
                            / max(np.var(target), np.finfo(float).eps)
                        ),
                        "active_contribution_variances": json.dumps(
                            np.var(
                                components.x_contribution_by_variable[start:stop],
                                axis=0,
                            ).tolist()
                        ),
                    }
                )
    write_csv(output / "oracle_ceiling.csv", rows)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "component_identity.json", identity)
    write_json(
        output / "summary.json",
        {
            "status": "E0_COMPONENT_IDENTITY_PASS",
            "jobs": len(identity),
            "maximum_latent_identity_error": max(
                row["latent_identity_max_abs"] for row in identity
            ),
            "maximum_measurement_identity_error": max(
                row["measurement_identity_max_abs"] for row in identity
            ),
        },
    )
    return "E0_COMPONENT_IDENTITY_PASS"


def _project_surface(
    sequence,
    variable: int,
    *,
    lag_basis_count: int,
    amplitude_basis_count: int,
    degree: int,
    quantiles: tuple[float, float],
) -> float:
    train_start, train_stop = sequence.split_target_intervals["train"]
    train_values = sequence.x[:train_stop, variable]
    lower, upper = np.quantile(train_values, quantiles)
    amplitudes = np.linspace(lower, upper, 401)
    amplitude_basis = CenteredSplineBasis.fit(
        train_values,
        n_basis=amplitude_basis_count,
        degree=degree,
        quantiles=quantiles,
    )
    lag_knots = clamped_knots(
        0.0, float(sequence.truth["L_x"] - 1), lag_basis_count, degree
    )
    lag_basis = evaluate_basis(
        np.arange(sequence.truth["L_x"]), lag_knots, degree
    )
    amp_eval = amplitude_basis.transform(amplitudes)
    design = np.einsum("la,ub->luab", lag_basis, amp_eval).reshape(
        len(lag_basis) * len(amplitudes), -1
    )
    truth = true_kernel_surface(sequence, variable, amplitudes)
    empirical_truth = true_kernel_surface(
        sequence, variable, train_values
    ).mean(axis=1, keepdims=True)
    centered_truth = truth - empirical_truth
    coefficients = np.linalg.lstsq(
        design, centered_truth.reshape(-1), rcond=None
    )[0]
    estimate = (design @ coefficients).reshape(centered_truth.shape)
    return normalized_root_mean_square_error(centered_truth, estimate)


def run_e1(config: dict[str, object]) -> str:
    common = config["common"]
    basis = config["external_basis"]
    output = RESULT_ROOT / "E1"
    contract = ExperimentContract(
        scientific_question="Can the frozen tensor spline spaces represent each truth kernel?",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="oracle",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)
    rows: list[dict[str, object]] = []
    scenarios = ["AR-S1", "AR-S2", "AR-S3", "AR-S4"]
    for scenario in scenarios:
        for seed in common["development_seeds"]:
            sequence = generate_synthetic_sequence(
                scenario,
                seed=int(seed),
                n_samples=int(common["n_samples"]),
                external_variables=int(common["external_variables"]),
            )
            for variable in sequence.truth["active_support"]:
                for m_lag in basis["lag_basis_candidates"]:
                    for m_amp in basis["amplitude_basis_candidates"]:
                        rows.append(
                            {
                                "scenario": scenario,
                                "seed": seed,
                                "variable": variable,
                                "lag_basis": m_lag,
                                "amplitude_basis": m_amp,
                                "is_fallback": False,
                                "projection_surface_nrmse": _project_surface(
                                    sequence,
                                    variable,
                                    lag_basis_count=int(m_lag),
                                    amplitude_basis_count=int(m_amp),
                                    degree=int(basis["degree"]),
                                    quantiles=tuple(basis["amplitude_quantiles"]),
                                ),
                            }
                        )
    selected: dict[str, dict[str, object]] = {}
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        keys = {
            (int(row["lag_basis"]), int(row["amplitude_basis"]))
            for row in scenario_rows
        }
        best_key = min(
            keys,
            key=lambda key: np.mean(
                [
                    row["projection_surface_nrmse"]
                    for row in scenario_rows
                    if (row["lag_basis"], row["amplitude_basis"]) == key
                ]
            ),
        )
        worst_active_seed = max(
            min(
                row["projection_surface_nrmse"]
                for row in scenario_rows
                if row["seed"] == seed and row["variable"] == variable
            )
            for seed in common["development_seeds"]
            for variable in (0, 1, 2)
        )
        use_fallback = worst_active_seed > 0.10
        fallback_worst = None
        if use_fallback:
            fallback_rows: list[dict[str, object]] = []
            for seed in common["development_seeds"]:
                sequence = generate_synthetic_sequence(
                    scenario,
                    seed=int(seed),
                    n_samples=int(common["n_samples"]),
                    external_variables=int(common["external_variables"]),
                )
                for variable in sequence.truth["active_support"]:
                    fallback_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "variable": variable,
                            "lag_basis": int(basis["fallback_lag_basis"]),
                            "amplitude_basis": int(
                                basis["fallback_amplitude_basis"]
                            ),
                            "is_fallback": True,
                            "projection_surface_nrmse": _project_surface(
                                sequence,
                                variable,
                                lag_basis_count=int(
                                    basis["fallback_lag_basis"]
                                ),
                                amplitude_basis_count=int(
                                    basis["fallback_amplitude_basis"]
                                ),
                                degree=int(basis["degree"]),
                                quantiles=tuple(basis["amplitude_quantiles"]),
                            ),
                        }
                    )
            rows.extend(fallback_rows)
            fallback_worst = max(
                float(row["projection_surface_nrmse"])
                for row in fallback_rows
            )
        selected[scenario] = {
            "use_fallback": use_fallback,
            "lag_basis": (
                int(basis["fallback_lag_basis"]) if use_fallback else best_key[0]
            ),
            "amplitude_basis": (
                int(basis["fallback_amplitude_basis"])
                if use_fallback
                else best_key[1]
            ),
            "best_main_worst_active_seed_nrmse": worst_active_seed,
            "fallback_worst_active_seed_nrmse": fallback_worst,
            "capacity_pass": (
                worst_active_seed <= 0.10
                if not use_fallback
                else fallback_worst is not None and fallback_worst <= 0.10
            ),
        }
    all_pass = all(item["capacity_pass"] for item in selected.values())
    status = (
        "E1_PROJECTION_CAPACITY_PASS"
        if all_pass
        else "E1_PROJECTION_FALLBACK_FAIL"
    )
    write_csv(output / "projection_oracle.csv", rows)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "selected_capacity_basis.json", selected)
    write_json(
        output / "summary.json",
        {
            "status": status,
            "selected_capacity_basis": selected,
            "next_allowed_experiment": "E2" if all_pass else "STOP",
        },
    )
    return status


def _e1r_projection(
    sequence,
    variable: int,
    *,
    lag_basis_count: int,
    identity_reference: bool,
    config: dict[str, object],
) -> tuple[float, str]:
    common = config["common"]
    amplitude_config = config["amplitude_basis"]
    _, train_stop = sequence.split_target_intervals["train"]
    train_values = sequence.x[:train_stop, variable]
    lower, upper = np.quantile(
        train_values, tuple(amplitude_config["quantiles"])
    )
    amplitudes = np.linspace(
        lower,
        upper,
        int(amplitude_config["evaluation_grid_points"]),
        dtype=np.float64,
    )
    amplitude_basis = CenteredSplineBasis.fit(
        train_values,
        n_basis=int(amplitude_config["basis_count"]),
        degree=int(amplitude_config["degree"]),
        quantiles=tuple(amplitude_config["quantiles"]),
    )
    amplitude_eval = amplitude_basis.transform(amplitudes)
    truth = true_kernel_surface(sequence, variable, amplitudes)
    empirical_mean = true_kernel_surface(
        sequence, variable, train_values
    ).mean(axis=1, keepdims=True)
    centered_truth = truth - empirical_mean
    if identity_reference:
        lag_basis = identity_lag_basis(int(common["L_x"]))
        basis_type = "discrete_identity"
    else:
        lag_knots = clamped_knots(
            0.0,
            float(int(common["L_x"]) - 1),
            lag_basis_count,
            int(amplitude_config["degree"]),
        )
        lag_basis = evaluate_basis(
            np.arange(int(common["L_x"])),
            lag_knots,
            int(amplitude_config["degree"]),
        )
        basis_type = "cubic_bspline"
    result = project_tensor_surface(centered_truth, lag_basis, amplitude_eval)
    return result.nrmse, basis_type


def run_e1r(config: dict[str, object], result_root: Path) -> str:
    common = config["common"]
    repair = config["lag_representation_repair"]
    amplitude_config = config["amplitude_basis"]
    output = result_root / "E1R"
    contract = ExperimentContract(
        scientific_question=(
            "Which compressed lag space first has structural-grade capacity "
            "for the frozen synthetic kernel family?"
        ),
        target_semantics="centered_true_external_kernel_surface",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="oracle",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        basis_selection_uses_truth=False,
        smoothing_selection_metric="validation_prediction_mse",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        allowed_next_experiment="E2A",
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)
    rows: list[dict[str, object]] = []
    scenarios = ["AR-S1", "AR-S2", "AR-S3", "AR-S4"]
    lag_counts = [
        *[int(value) for value in repair["compressed_candidates"]],
        int(repair["identity_reference"]),
    ]
    for scenario in scenarios:
        for seed in common["development_seeds"]:
            sequence = generate_synthetic_sequence(
                scenario,
                seed=int(seed),
                n_samples=int(common["n_samples"]),
                external_variables=int(common["external_variables"]),
            )
            for variable in sequence.truth["active_support"]:
                per_lag: dict[int, tuple[float, str]] = {}
                for lag_count in lag_counts:
                    per_lag[lag_count] = _e1r_projection(
                        sequence,
                        int(variable),
                        lag_basis_count=lag_count,
                        identity_reference=(
                            lag_count == int(repair["identity_reference"])
                        ),
                        config=config,
                    )
                reference = per_lag[int(repair["identity_reference"])][0]
                for lag_count in lag_counts:
                    nrmse, basis_type = per_lag[lag_count]
                    reference_ratio = nrmse / max(
                        reference, np.finfo(np.float64).eps
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "variable": int(variable),
                            "lag_basis_type": basis_type,
                            "lag_basis_count": lag_count,
                            "amplitude_basis_count": int(
                                amplitude_config["basis_count"]
                            ),
                            "projection_surface_nrmse": nrmse,
                            "identity_reference_nrmse": reference,
                            "reference_ratio": reference_ratio,
                            "prediction_grade_pass": (
                                nrmse
                                <= float(repair["prediction_grade_max_nrmse"])
                            ),
                            "structural_grade_pass": (
                                nrmse
                                <= float(repair["structural_grade_max_nrmse"])
                                and reference_ratio
                                <= float(
                                    repair[
                                        "structural_grade_max_reference_ratio"
                                    ]
                                )
                            ),
                        }
                    )
    compressed = [int(value) for value in repair["compressed_candidates"]]
    scenario_worst: dict[str, dict[str, float]] = {}
    for scenario in scenarios:
        scenario_worst[scenario] = {}
        for lag_count in lag_counts:
            scenario_worst[scenario][str(lag_count)] = max(
                float(row["projection_surface_nrmse"])
                for row in rows
                if row["scenario"] == scenario
                and row["lag_basis_count"] == lag_count
            )
    certified: list[int] = []
    for lag_count in compressed:
        if max(
            scenario_worst[scenario][str(lag_count)] for scenario in scenarios
        ) > float(repair["structural_grade_max_nrmse"]):
            continue
        if all(
            scenario_worst[scenario][str(lag_count)]
            <= float(repair["structural_grade_max_reference_ratio"])
            * scenario_worst[scenario][str(repair["identity_reference"])]
            for scenario in scenarios
        ):
            certified.append(lag_count)
    selected = min(certified) if certified else None
    regression_errors = {
        scenario: {
            lag_count: abs(
                scenario_worst[scenario][lag_count]
                - float(repair["expected_worst_nrmse"][scenario][lag_count])
            )
            for lag_count in repair["expected_worst_nrmse"][scenario]
        }
        for scenario in scenarios
    }
    maximum_regression_error = max(
        value
        for scenario_errors in regression_errors.values()
        for value in scenario_errors.values()
    )
    implementation_match = (
        maximum_regression_error <= float(repair["regression_tolerance"])
    )
    selected_match = selected == int(repair["frozen_structural_basis"])
    status = (
        "E1R_REPRESENTATION_CERTIFIED_32x16"
        if implementation_match and selected_match
        else "E1R_IMPLEMENTATION_MISMATCH"
    )
    certificate = {
        "status": status,
        "selected_structural_lag_basis": selected,
        "amplitude_basis_count": int(amplitude_config["basis_count"]),
        "certified_lag_bases": certified,
        "scenario_worst_nrmse": scenario_worst,
        "maximum_regression_error": maximum_regression_error,
        "regression_errors": regression_errors,
        "next_allowed_experiment": "E2A" if "CERTIFIED" in status else "STOP",
    }
    write_csv(output / "projection_repair.csv", rows)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "representation_certificate.json", certificate)
    write_json(output / "summary.json", certificate)
    np.savez(
        output / "fit.npz",
        selected_structural_lag_basis=np.array(
            [-1 if selected is None else selected], dtype=np.int64
        ),
    )
    return status


def _select_one_se(
    validation_scores: dict[tuple[float, float], list[float]],
    configuration_order: list[tuple[float, float]],
) -> tuple[float, float]:
    means = {
        key: float(np.mean(values)) for key, values in validation_scores.items()
    }
    best = min(configuration_order, key=lambda key: means[key])
    values = np.asarray(validation_scores[best], dtype=np.float64)
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    threshold = means[best] + standard_error
    eligible = [key for key in configuration_order if means[key] <= threshold]
    order_index = {key: index for index, key in enumerate(configuration_order)}
    return min(
        eligible,
        key=lambda key: (-key[0] * key[1], order_index[key]),
    )


def _split_positions(sequence) -> dict[str, np.ndarray]:
    start = sequence.split_target_intervals["train"][0]
    stop = sequence.split_target_intervals["test"][1]
    targets = np.arange(start, stop, dtype=np.int64)
    return {
        name: np.flatnonzero(
            (targets >= interval_start) & (targets < interval_stop)
        )
        for name, (interval_start, interval_stop) in (
            sequence.split_target_intervals.items()
        )
    }


def _read_e1r_projection_lookup(result_root: Path) -> dict[tuple[str, int, int], float]:
    path = result_root / "E1R" / "projection_repair.csv"
    if not path.exists():
        raise RuntimeError("E2A requires a completed E1R projection table.")
    lookup: dict[tuple[str, int, int], float] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["lag_basis_count"]) == 32:
                lookup[
                    (
                        row["scenario"],
                        int(row["seed"]),
                        int(row["variable"]),
                    )
                ] = float(row["projection_surface_nrmse"])
    return lookup


def run_e2a(config: dict[str, object], result_root: Path) -> str:
    e1r_summary = json.loads(
        (result_root / "E1R" / "summary.json").read_text(encoding="utf-8")
    )
    if e1r_summary["status"] != "E1R_REPRESENTATION_CERTIFIED_32x16":
        raise RuntimeError("E2A is blocked until E1R certifies 32x16.")
    output = result_root / "E2A"
    common = config["common"]
    amplitude_config = config["amplitude_basis"]
    regularization = config["spectral_regularization"]
    lag_count = int(config["lag_representation_repair"]["frozen_structural_basis"])
    amplitude_count = int(amplitude_config["basis_count"])
    contract = ExperimentContract(
        scientific_question=(
            "Can a 32x16 full kernel recover one true variable contribution "
            "when its identity is known?"
        ),
        target_semantics="single_true_external_variable_contribution",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="single_oracle_variable",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        basis_selection_uses_truth=False,
        smoothing_selection_metric="validation_prediction_mse",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        allowed_next_experiment="E2B",
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)
    configurations = [
        (float(lag_weight), float(amplitude_weight))
        for lag_weight in regularization["lag_smoothness_candidates"]
        for amplitude_weight in regularization["amplitude_smoothness_candidates"]
    ]
    projection_lookup = _read_e1r_projection_lookup(result_root)
    rows: list[dict[str, object]] = []
    saved_coefficients: list[np.ndarray] = []
    saved_identifiers: list[tuple[str, int, int]] = []
    selected_by_group: dict[str, dict[str, float]] = {}
    scenarios = ["AR-S1", "AR-S2", "AR-S3", "AR-S4"]
    for scenario in scenarios:
        for variable in (0, 1, 2):
            jobs: list[dict[str, object]] = []
            scores = {key: [] for key in configurations}
            for seed in common["development_seeds"]:
                sequence = generate_synthetic_sequence(
                    scenario,
                    seed=int(seed),
                    n_samples=int(common["n_samples"]),
                    external_variables=int(common["external_variables"]),
                )
                if variable not in sequence.truth["active_support"]:
                    raise RuntimeError("Frozen E2A variable is not active.")
                components = replay_synthetic_components(sequence)
                start = sequence.split_target_intervals["train"][0]
                stop = sequence.split_target_intervals["test"][1]
                targets = np.arange(start, stop, dtype=np.int64)
                positions = _split_positions(sequence)
                design = build_spectral_design(
                    sequence.x[:, [variable]],
                    target_indices=targets,
                    train_target_stop=sequence.split_target_intervals["train"][1],
                    horizon=int(common["primary_horizon"]),
                    L_x=int(common["L_x"]),
                    lag_basis_count=lag_count,
                    amplitude_basis_count=amplitude_count,
                    degree=int(amplitude_config["degree"]),
                    amplitude_quantiles=tuple(amplitude_config["quantiles"]),
                )
                target = e2a_component_target(components, variable, targets)
                train = positions["train"]
                validation = positions["validation"]
                fits = {}
                for configuration in configurations:
                    lag_weight, amplitude_weight = configuration
                    penalty = tensor_penalty(
                        design.lag_gram,
                        design.amplitude_grams,
                        lag_smoothness=lag_weight,
                        amplitude_smoothness=amplitude_weight,
                        ridge_weight=float(regularization["ridge_weight"]),
                    )
                    fit = solve_full_kernel(
                        design.matrix[train],
                        target[train],
                        penalty,
                        numerical_jitter_relative=float(
                            common["numerical_jitter_relative"]
                        ),
                        fit_intercept=True,
                        compute_condition_number=False,
                    )
                    fits[configuration] = fit
                    prediction = (
                        design.matrix[validation] @ fit.coefficients
                        + fit.intercept
                    )
                    scores[configuration].append(
                        mse(target[validation], prediction)
                    )
                jobs.append(
                    {
                        "seed": int(seed),
                        "sequence": sequence,
                        "design": design,
                        "target": target,
                        "positions": positions,
                        "fits": fits,
                    }
                )
            selected = _select_one_se(scores, configurations)
            selected_by_group[f"{scenario}:x{variable}"] = {
                "lag_smoothness": selected[0],
                "amplitude_smoothness": selected[1],
            }
            for job in jobs:
                seed = int(job["seed"])
                sequence = job["sequence"]
                design = job["design"]
                target = job["target"]
                positions = job["positions"]
                fit = job["fits"][selected]
                train = positions["train"]
                validation = positions["validation"]
                test = positions["test"]
                selected_penalty = tensor_penalty(
                    design.lag_gram,
                    design.amplitude_grams,
                    lag_smoothness=selected[0],
                    amplitude_smoothness=selected[1],
                    ridge_weight=float(regularization["ridge_weight"]),
                )
                checked_fit = solve_full_kernel(
                    design.matrix[train],
                    target[train],
                    selected_penalty,
                    numerical_jitter_relative=float(
                        common["numerical_jitter_relative"]
                    ),
                    fit_intercept=True,
                    compute_condition_number=True,
                )
                coefficients = checked_fit.coefficients.reshape(
                    lag_count, amplitude_count
                )
                basis = design.amplitude_bases[0]
                amplitudes = np.linspace(
                    basis.lower,
                    basis.upper,
                    int(amplitude_config["evaluation_grid_points"]),
                )
                amplitude_eval = basis.transform(amplitudes)
                truth = true_kernel_surface(sequence, variable, amplitudes)
                train_values = sequence.x[
                    : sequence.split_target_intervals["train"][1], variable
                ]
                empirical_mean = true_kernel_surface(
                    sequence, variable, train_values
                ).mean(axis=1, keepdims=True)
                centered_truth = truth - empirical_mean
                estimated_surface = (
                    design.lag_basis @ coefficients @ amplitude_eval.T
                )
                surface_nrmse = normalized_root_mean_square_error(
                    centered_truth, estimated_surface
                )
                projection_nrmse = projection_lookup[
                    (scenario, seed, variable)
                ]
                spectrum = gram_whitened_svd(
                    coefficients,
                    design.lag_gram,
                    design.amplitude_grams[0],
                )
                full_validation_prediction = (
                    design.matrix[validation] @ checked_fit.coefficients
                    + checked_fit.intercept
                )
                rank_validation_mse: dict[int, float] = {}
                for rank in (1, 2):
                    truncated = spectrum.truncate(rank).reshape(-1)
                    truncated_intercept = float(
                        np.mean(
                            target[train]
                            - design.matrix[train] @ truncated
                        )
                    )
                    rank_validation_mse[rank] = mse(
                        target[validation],
                        design.matrix[validation] @ truncated
                        + truncated_intercept,
                    )
                row: dict[str, object] = {
                    "scenario": scenario,
                    "seed": seed,
                    "variable": variable,
                    "lag_basis_count": lag_count,
                    "amplitude_basis_count": amplitude_count,
                    "lag_smoothness": selected[0],
                    "amplitude_smoothness": selected[1],
                    "train_contribution_rmse": rmse(
                        target[train], checked_fit.predictions
                    ),
                    "train_contribution_r2": r2(
                        target[train], checked_fit.predictions
                    ),
                    "validation_contribution_rmse": rmse(
                        target[validation], full_validation_prediction
                    ),
                    "validation_contribution_r2": r2(
                        target[validation], full_validation_prediction
                    ),
                    "test_contribution_rmse": rmse(
                        target[test],
                        design.matrix[test] @ checked_fit.coefficients
                        + checked_fit.intercept,
                    ),
                    "test_contribution_r2": r2(
                        target[test],
                        design.matrix[test] @ checked_fit.coefficients
                        + checked_fit.intercept,
                    ),
                    "surface_nrmse": surface_nrmse,
                    "e1r_projection_nrmse": projection_nrmse,
                    "excess_surface_error": surface_nrmse - projection_nrmse,
                    "relative_kkt_residual": checked_fit.relative_kkt_residual,
                    "condition_number": checked_fit.condition_number,
                    "full_validation_mse": mse(
                        target[validation], full_validation_prediction
                    ),
                    "rank1_validation_mse": rank_validation_mse[1],
                    "rank2_validation_mse": rank_validation_mse[2],
                    "eta1": spectrum.tail_energy_ratio(1),
                    "eta2": spectrum.tail_energy_ratio(2),
                    "eta3": spectrum.tail_energy_ratio(3),
                }
                row["capacity_pass"] = (
                    row["validation_contribution_r2"] >= 0.995
                    and row["surface_nrmse"]
                    <= max(0.03, 1.5 * projection_nrmse)
                    and row["relative_kkt_residual"] <= 1e-8
                )
                rows.append(row)
                saved_coefficients.append(checked_fit.coefficients)
                saved_identifiers.append((scenario, seed, variable))
    scenario_seed_passes: dict[str, dict[str, bool]] = {}
    scenario_pass = {}
    for scenario in scenarios:
        scenario_seed_passes[scenario] = {}
        for seed in common["development_seeds"]:
            selected_rows = [
                row
                for row in rows
                if row["scenario"] == scenario and row["seed"] == seed
            ]
            scenario_seed_passes[scenario][str(seed)] = (
                len(selected_rows) == 3
                and all(bool(row["capacity_pass"]) for row in selected_rows)
            )
        scenario_pass[scenario] = (
            sum(scenario_seed_passes[scenario].values()) >= 4
        )
    all_pass = all(scenario_pass.values())
    status = (
        "E2A_SINGLE_KERNEL_CAPACITY_PASS"
        if all_pass
        else "E2A_ESTIMATOR_OR_DATA_EXCITATION_FAIL"
    )
    summary = {
        "status": status,
        "selected_smoothing_by_scenario_variable": selected_by_group,
        "scenario_seed_passes": scenario_seed_passes,
        "scenario_pass": scenario_pass,
        "passing_rows": sum(bool(row["capacity_pass"]) for row in rows),
        "total_rows": len(rows),
        "next_allowed_experiment": "E2B" if all_pass else "STOP",
    }
    write_csv(output / "metrics.csv", rows)
    write_json(output / "summary.json", summary)
    np.savez(
        output / "fit.npz",
        coefficients=np.stack(saved_coefficients),
        identifiers=np.asarray(saved_identifiers, dtype="U16"),
    )
    return status


def run_r1(config: dict[str, object], result_root: Path) -> str:
    output = result_root / "R1"
    common = config["common"]
    domain_config = config["amplitude_domain"]
    contract = ExperimentContract(
        scientific_question=(
            "Did the old amplitude protocol clip histories and is AR-S4 "
            "outside the 2D additive Urysohn model class?"
        ),
        target_semantics="protocol_domain_and_model_class_audit",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="oracle",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        basis_selection_uses_truth=False,
        smoothing_selection_metric="validation_prediction_mse",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        allowed_next_experiment="E1A",
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)
    rows: list[dict[str, object]] = []
    L_x = int(common["L_x"])
    for scenario in ("AR-S1", "AR-S2", "AR-S3", "AR-S4"):
        for seed in common["development_seeds"]:
            sequence = generate_synthetic_sequence(
                scenario,
                seed=int(seed),
                n_samples=int(common["n_samples_natural"]),
                external_variables=int(common["external_variables"]),
            )
            train_stop = sequence.split_target_intervals["train"][1]
            for variable in (0, 1, 2):
                train_values = sequence.x[:train_stop, variable]
                lower, upper = np.quantile(
                    train_values,
                    tuple(domain_config["old_audit_quantiles"]),
                )
                for partition, (start, stop) in (
                    sequence.split_target_intervals.items()
                ):
                    point_values = sequence.x[start:stop, variable]
                    point_ood = (point_values < lower) | (point_values > upper)
                    targets = np.arange(start, stop, dtype=np.int64)
                    origins = targets - int(common["primary_horizon"])
                    offsets = np.arange(L_x, dtype=np.int64)
                    windows = sequence.x[
                        origins[:, None] - offsets[None, :], variable
                    ]
                    window_ood = np.any(
                        (windows < lower) | (windows > upper), axis=1
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": int(seed),
                            "variable": variable,
                            "partition": partition,
                            "old_lower": float(lower),
                            "old_upper": float(upper),
                            "point_clipping_rate": float(np.mean(point_ood)),
                            "window_clipping_rate": float(np.mean(window_ood)),
                            "point_count": len(point_values),
                            "window_count": len(windows),
                        }
                    )
    classification = {
        name: spec.to_dict() for name, spec in SCENARIO_REGISTRY.items()
    }
    classification["AR-S4"]["mixed_difference"] = s4c_mixed_difference()
    registry_valid = (
        not classification["AR-S4"]["eligible_for_2d_capacity"]
        and classification["AR-S4U"]["eligible_for_2d_capacity"]
        and abs(classification["AR-S4"]["mixed_difference"]) > 1e-8
    )
    status = (
        "R1_DOMAIN_AND_MODEL_CLASS_AUDIT_PASS"
        if registry_valid
        else "R1_SCENARIO_OR_DOMAIN_AUDIT_FAIL"
    )
    summary = {
        "status": status,
        "maximum_point_clipping_rate": max(
            float(row["point_clipping_rate"]) for row in rows
        ),
        "maximum_window_clipping_rate": max(
            float(row["window_clipping_rate"]) for row in rows
        ),
        "median_window_clipping_rate": float(
            np.median([row["window_clipping_rate"] for row in rows])
        ),
        "model_class_registry_valid": registry_valid,
        "next_allowed_experiment": "E1A" if registry_valid else "STOP",
    }
    write_csv(output / "old_clipping_audit.csv", rows)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "scenario_classification.json", classification)
    write_json(output / "summary.json", summary)
    np.savez(output / "fit.npz", audit_only=np.array([1], dtype=np.int8))
    return status


def _v032_surface_projection(
    sequence,
    variable: int,
    *,
    lag_basis_count: int,
    amplitude_basis_count: int,
    domain: AmplitudeDomain,
    evaluation_domain: str,
    config: dict[str, object],
):
    basis_config = config["basis_recertification"]
    train_stop = sequence.split_target_intervals["train"][1]
    train_values = sequence.x[:train_stop, variable]
    amplitude_basis = CenteredSplineBasis.fit(
        train_values,
        n_basis=amplitude_basis_count,
        degree=int(basis_config["degree"]),
        domain=domain,
    )
    if evaluation_domain == "core":
        lower, upper = domain.core_lower, domain.core_upper
    elif evaluation_domain == "fit":
        lower, upper = domain.fit_lower, domain.fit_upper
    else:
        raise ValueError("evaluation_domain must be core or fit.")
    amplitudes = np.linspace(
        lower,
        upper,
        int(basis_config["evaluation_grid_points"]),
    )
    amplitude_eval = amplitude_basis.transform(amplitudes)
    truth = true_kernel_surface(sequence, variable, amplitudes)
    empirical_mean = true_kernel_surface(
        sequence, variable, train_values
    ).mean(axis=1, keepdims=True)
    centered_truth = truth - empirical_mean
    if lag_basis_count == int(basis_config["identity_lag_reference"]):
        lag_basis = identity_lag_basis(int(config["common"]["L_x"]))
        basis_type = "discrete_identity"
    else:
        knots = clamped_knots(
            0.0,
            float(int(config["common"]["L_x"]) - 1),
            lag_basis_count,
            int(basis_config["degree"]),
        )
        lag_basis = evaluate_basis(
            np.arange(int(config["common"]["L_x"])),
            knots,
            int(basis_config["degree"]),
        )
        basis_type = "cubic_bspline"
    result = project_tensor_surface(centered_truth, lag_basis, amplitude_eval)
    return result, amplitude_basis, lag_basis, basis_type


def run_e1a(config: dict[str, object], result_root: Path) -> str:
    r1_path = result_root / "R1" / "summary.json"
    if not r1_path.exists() or json.loads(
        r1_path.read_text(encoding="utf-8")
    )["status"] != "R1_DOMAIN_AND_MODEL_CLASS_AUDIT_PASS":
        raise RuntimeError("E1A is blocked until R1 passes.")
    output = result_root / "E1A"
    common = config["common"]
    basis_config = config["basis_recertification"]
    domain_config = config["amplitude_domain"]
    contract = ExperimentContract(
        scientific_question=(
            "Which frozen amplitude basis first represents the four core "
            "2D Urysohn scenarios on both core and fit domains?"
        ),
        target_semantics="centered_true_2d_external_kernel_surface",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="oracle",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        basis_selection_uses_truth=False,
        smoothing_selection_metric="validation_prediction_mse",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        allowed_next_experiment="E2A0",
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)
    scenarios = list(config["scenario_sets"]["core_2d_urysohn"])
    core_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    sequences: dict[tuple[str, int], object] = {}
    e0u_errors: list[float] = []
    for scenario in scenarios:
        for seed in common["development_seeds"]:
            sequence = generate_synthetic_sequence(
                scenario,
                seed=int(seed),
                n_samples=int(common["n_samples_natural"]),
                external_variables=int(common["external_variables"]),
            )
            sequences[(scenario, int(seed))] = sequence
            if scenario == "AR-S4U":
                components = replay_synthetic_components(sequence)
                reconstructed = (
                    components.ar_contribution
                    + components.x_total_contribution
                    + components.process_innovation
                )
                e0u_errors.append(
                    float(
                        np.max(np.abs(sequence.y_latent - reconstructed))
                    )
                )
            train_stop = sequence.split_target_intervals["train"][1]
            for variable in sequence.truth["active_support"]:
                train_values = sequence.x[:train_stop, variable]
                domain = AmplitudeDomain.fit(
                    train_values,
                    padding_fraction=float(domain_config["padding_fraction"]),
                    core_quantiles=tuple(domain_config["core_quantiles"]),
                )
                for amplitude_count in basis_config[
                    "amplitude_basis_candidates"
                ]:
                    for lag_count in (
                        int(basis_config["lag_basis_count"]),
                        int(basis_config["identity_lag_reference"]),
                    ):
                        for evaluation_domain, rows in (
                            ("core", core_rows),
                            ("fit", fit_rows),
                        ):
                            result, _, _, basis_type = _v032_surface_projection(
                                sequence,
                                int(variable),
                                lag_basis_count=lag_count,
                                amplitude_basis_count=int(amplitude_count),
                                domain=domain,
                                evaluation_domain=evaluation_domain,
                                config=config,
                            )
                            rows.append(
                                {
                                    "scenario": scenario,
                                    "seed": int(seed),
                                    "variable": int(variable),
                                    "lag_basis_type": basis_type,
                                    "lag_basis_count": lag_count,
                                    "amplitude_basis_count": int(
                                        amplitude_count
                                    ),
                                    "surface_nrmse": result.nrmse,
                                    "domain_lower": (
                                        domain.core_lower
                                        if evaluation_domain == "core"
                                        else domain.fit_lower
                                    ),
                                    "domain_upper": (
                                        domain.core_upper
                                        if evaluation_domain == "core"
                                        else domain.fit_upper
                                    ),
                                }
                            )
    selected = None
    candidate_diagnostics: dict[str, object] = {}
    main_lag = int(basis_config["lag_basis_count"])
    identity_lag = int(basis_config["identity_lag_reference"])
    for amplitude_count in basis_config["amplitude_basis_candidates"]:
        core_main = [
            row
            for row in core_rows
            if row["lag_basis_count"] == main_lag
            and row["amplitude_basis_count"] == amplitude_count
        ]
        fit_main = [
            row
            for row in fit_rows
            if row["lag_basis_count"] == main_lag
            and row["amplitude_basis_count"] == amplitude_count
        ]
        ratios_valid = True
        scenario_ratios = {}
        for scenario in scenarios:
            main_core = max(
                row["surface_nrmse"]
                for row in core_main
                if row["scenario"] == scenario
            )
            identity_core = max(
                row["surface_nrmse"]
                for row in core_rows
                if row["scenario"] == scenario
                and row["lag_basis_count"] == identity_lag
                and row["amplitude_basis_count"] == amplitude_count
            )
            main_fit = max(
                row["surface_nrmse"]
                for row in fit_main
                if row["scenario"] == scenario
            )
            identity_fit = max(
                row["surface_nrmse"]
                for row in fit_rows
                if row["scenario"] == scenario
                and row["lag_basis_count"] == identity_lag
                and row["amplitude_basis_count"] == amplitude_count
            )
            core_ratio = main_core / max(identity_core, np.finfo(float).eps)
            fit_ratio = main_fit / max(identity_fit, np.finfo(float).eps)
            scenario_ratios[scenario] = {
                "core": core_ratio,
                "fit": fit_ratio,
            }
            ratios_valid &= (
                core_ratio <= float(basis_config["lag_reference_ratio_max"])
                and fit_ratio
                <= float(basis_config["lag_reference_ratio_max"])
            )
        passes = (
            max(row["surface_nrmse"] for row in core_main)
            <= float(basis_config["core_surface_max_nrmse"])
            and max(row["surface_nrmse"] for row in fit_main)
            <= float(basis_config["fit_surface_max_nrmse"])
            and ratios_valid
        )
        candidate_diagnostics[str(amplitude_count)] = {
            "worst_core_nrmse": max(
                row["surface_nrmse"] for row in core_main
            ),
            "worst_fit_nrmse": max(
                row["surface_nrmse"] for row in fit_main
            ),
            "scenario_reference_ratios": scenario_ratios,
            "passes": bool(passes),
        }
        if selected is None and passes:
            selected = int(amplitude_count)
    e0u_max = max(e0u_errors) if e0u_errors else float("inf")
    e0u_status = (
        "E0U_COMPONENT_IDENTITY_PASS"
        if e0u_max <= 1e-12
        else "E0U_COMPONENT_IDENTITY_FAIL"
    )
    if selected is not None and e0u_status.endswith("_PASS"):
        upper_lag = int(basis_config["upper_lag_neighbor"])
        for scenario in scenarios:
            for seed in common["development_seeds"]:
                sequence = sequences[(scenario, int(seed))]
                train_stop = sequence.split_target_intervals["train"][1]
                for variable in sequence.truth["active_support"]:
                    domain = AmplitudeDomain.fit(
                        sequence.x[:train_stop, variable],
                        padding_fraction=float(
                            domain_config["padding_fraction"]
                        ),
                        core_quantiles=tuple(
                            domain_config["core_quantiles"]
                        ),
                    )
                    for evaluation_domain, rows in (
                        ("core", core_rows),
                        ("fit", fit_rows),
                    ):
                        result, _, _, basis_type = _v032_surface_projection(
                            sequence,
                            int(variable),
                            lag_basis_count=upper_lag,
                            amplitude_basis_count=selected,
                            domain=domain,
                            evaluation_domain=evaluation_domain,
                            config=config,
                        )
                        rows.append(
                            {
                                "scenario": scenario,
                                "seed": int(seed),
                                "variable": int(variable),
                                "lag_basis_type": basis_type,
                                "lag_basis_count": upper_lag,
                                "amplitude_basis_count": selected,
                                "surface_nrmse": result.nrmse,
                                "domain_lower": (
                                    domain.core_lower
                                    if evaluation_domain == "core"
                                    else domain.fit_lower
                                ),
                                "domain_upper": (
                                    domain.core_upper
                                    if evaluation_domain == "core"
                                    else domain.fit_upper
                                ),
                            }
                        )
    status = (
        "E1A_DOMAIN_SAFE_REPRESENTATION_PASS"
        if selected is not None and e0u_status.endswith("_PASS")
        else "E1A_AMPLITUDE_REPRESENTATION_FAIL"
    )
    certificate = {
        "status": status,
        "e0u_status": e0u_status,
        "e0u_maximum_identity_error": e0u_max,
        "selected_amplitude_basis_count": selected,
        "lag_basis_count": main_lag,
        "candidate_diagnostics": candidate_diagnostics,
        "next_allowed_experiment": "E2A0" if status.endswith("_PASS") else "STOP",
    }
    write_csv(output / "projection_core.csv", core_rows)
    write_csv(output / "projection_fit.csv", fit_rows)
    write_csv(output / "metrics.csv", core_rows + fit_rows)
    write_json(output / "representation_certificate.json", certificate)
    write_json(output / "summary.json", certificate)
    np.savez(
        output / "fit.npz",
        selected_amplitude_basis_count=np.array(
            [-1 if selected is None else selected], dtype=np.int64
        ),
    )
    return status


def run_e2a0(config: dict[str, object], result_root: Path) -> str:
    e1a = json.loads(
        (result_root / "E1A" / "summary.json").read_text(encoding="utf-8")
    )
    if e1a["status"] != "E1A_DOMAIN_SAFE_REPRESENTATION_PASS":
        raise RuntimeError("E2A0 is blocked until E1A passes.")
    output = result_root / "E2A0"
    common = config["common"]
    domain_config = config["amplitude_domain"]
    basis_config = config["basis_recertification"]
    tolerances = config["e2a0"]
    amplitude_count = int(e1a["selected_amplitude_basis_count"])
    lag_count = int(basis_config["lag_basis_count"])
    contract = ExperimentContract(
        scientific_question=(
            "Do truth replay, tensor design, strict domains, and FP64 solvers "
            "close before scientific capacity evaluation?"
        ),
        target_semantics="implementation_and_operator_closure",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="single_oracle_variable",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        basis_selection_uses_truth=False,
        smoothing_selection_metric="validation_prediction_mse",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        allowed_next_experiment="E2A_NAT",
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)
    rows: list[dict[str, object]] = []
    maximum_target_error = 0.0
    maximum_design_error = 0.0
    all_train_in_domain = True
    strict_ood_raised = True
    for scenario in config["scenario_sets"]["core_2d_urysohn"]:
        for seed in common["development_seeds"]:
            sequence = generate_synthetic_sequence(
                scenario,
                seed=int(seed),
                n_samples=int(common["n_samples_natural"]),
                external_variables=int(common["external_variables"]),
            )
            components = replay_synthetic_components(sequence)
            start = sequence.split_target_intervals["train"][0]
            stop = sequence.split_target_intervals["test"][1]
            targets = np.arange(start, stop, dtype=np.int64)
            train_stop = sequence.split_target_intervals["train"][1]
            for variable in sequence.truth["active_support"]:
                direct_target = direct_apply_truth_kernel(
                    sequence, int(variable), targets
                )
                replayed_target = components.x_contribution_by_variable[
                    targets, variable
                ]
                target_error = float(
                    np.max(np.abs(direct_target - replayed_target))
                )
                maximum_target_error = max(maximum_target_error, target_error)
                train_values = sequence.x[:train_stop, variable]
                domain = AmplitudeDomain.fit(
                    train_values,
                    padding_fraction=float(domain_config["padding_fraction"]),
                    core_quantiles=tuple(domain_config["core_quantiles"]),
                )
                all_train_in_domain &= bool(
                    np.all(domain.in_domain_mask(train_values))
                )
                projection, amplitude_basis, lag_basis, _ = (
                    _v032_surface_projection(
                        sequence,
                        int(variable),
                        lag_basis_count=lag_count,
                        amplitude_basis_count=amplitude_count,
                        domain=domain,
                        evaluation_domain="fit",
                        config=config,
                    )
                )
                design = build_spectral_design(
                    sequence.x[:, [variable]],
                    target_indices=targets,
                    train_target_stop=train_stop,
                    horizon=int(common["primary_horizon"]),
                    L_x=int(common["L_x"]),
                    lag_basis_count=lag_count,
                    amplitude_basis_count=amplitude_count,
                    degree=int(basis_config["degree"]),
                    amplitude_domains=[domain],
                )
                via_matrix = design.matrix @ projection.coefficients.reshape(-1)
                via_direct = direct_apply_projected_kernel(
                    sequence.x,
                    variable=int(variable),
                    target_indices=targets,
                    horizon=int(common["primary_horizon"]),
                    lag_basis=lag_basis,
                    amplitude_basis=amplitude_basis,
                    coefficients=projection.coefficients,
                )
                design_error = float(
                    np.max(np.abs(via_matrix - via_direct))
                )
                maximum_design_error = max(maximum_design_error, design_error)
                try:
                    amplitude_basis.transform(
                        np.array([domain.fit_upper + 1e-6])
                    )
                    strict_ood_raised = False
                except AmplitudeOutOfDomainError:
                    pass
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": int(seed),
                        "variable": int(variable),
                        "target_replay_max_abs_error": target_error,
                        "design_forward_max_abs_error": design_error,
                        "train_domain_coverage": float(
                            np.mean(domain.in_domain_mask(train_values))
                        ),
                    }
                )
    rng = np.random.default_rng(320)
    matrix = rng.normal(size=(700, 200))
    target = rng.normal(size=700)
    raw = rng.normal(size=(200, 200))
    penalty = raw.T @ raw / 200 + 0.1 * np.eye(200)
    direct = solve_full_kernel(
        matrix, target, penalty, compute_condition_number=False
    )
    pcg = solve_full_kernel_pcg(
        matrix,
        target,
        penalty,
        relative_tolerance=1e-10,
        max_iterations=2000,
        block_slices=tuple(
            slice(start, start + 50) for start in range(0, 200, 50)
        ),
    )
    system = matrix.T @ matrix / len(matrix) + penalty
    scale = max(float(np.trace(system) / len(system)), 1.0)
    system = system + float(common["numerical_jitter_relative"]) * scale * np.eye(200)
    rhs = matrix.T @ target / len(matrix)
    reference = np.linalg.solve(system, rhs)
    direct_error = float(
        np.linalg.norm(direct.coefficients - reference)
        / np.linalg.norm(reference)
    )
    pcg_error = float(
        np.linalg.norm(pcg.coefficients - reference)
        / np.linalg.norm(reference)
    )
    prediction_error = float(
        np.linalg.norm(matrix @ pcg.coefficients - matrix @ reference)
        / np.linalg.norm(matrix @ reference)
    )
    passes = (
        maximum_target_error
        <= float(tolerances["target_replay_tolerance"])
        and maximum_design_error
        <= float(tolerances["forward_operator_tolerance"])
        and direct_error
        <= float(tolerances["random_solver_relative_error"])
        and pcg_error
        <= float(tolerances["direct_vs_matrix_free_relative_error"])
        and prediction_error <= 1e-9
        and pcg.converged
        and all_train_in_domain
        and strict_ood_raised
    )
    status = (
        "E2A0_IMPLEMENTATION_CONSISTENCY_PASS"
        if passes
        else "E2A0_IMPLEMENTATION_CONSISTENCY_FAIL"
    )
    summary = {
        "status": status,
        "maximum_target_replay_error": maximum_target_error,
        "maximum_design_forward_error": maximum_design_error,
        "direct_reference_relative_error": direct_error,
        "pcg_reference_relative_error": pcg_error,
        "pcg_prediction_relative_error": prediction_error,
        "pcg_converged": pcg.converged,
        "all_train_values_in_domain": all_train_in_domain,
        "strict_ood_raised": strict_ood_raised,
        "next_allowed_experiment": "E2A_NAT" if passes else "STOP",
    }
    write_csv(output / "metrics.csv", rows)
    write_json(output / "summary.json", summary)
    np.savez(
        output / "fit.npz",
        direct_coefficients=direct.coefficients,
        pcg_coefficients=pcg.coefficients,
        reference_coefficients=reference,
    )
    return status


def _v033_lag_basis(
    *,
    lag_type: str,
    lag_count: int,
    length: int,
    degree: int,
) -> np.ndarray:
    if lag_type == "discrete_identity":
        if lag_count != length:
            raise ValueError("The identity lag basis must span every discrete lag.")
        return identity_lag_basis(length)
    if lag_type != "cubic_bspline":
        raise ValueError(f"Unknown lag basis type: {lag_type}.")
    knots = clamped_knots(0.0, float(length - 1), lag_count, degree)
    return evaluate_basis(np.arange(length), knots, degree)


def run_e1b(config: dict[str, object], result_root: Path) -> str:
    """Run the unconditional 1,440-row v0.3.3 representation certificate."""

    output = result_root / "E1B"
    common = config["common"]
    domain_config = config["domain"]
    e1b = config["e1b"]
    for name in ("PREDICTIVE", "STRUCTURAL", "MOTHER"):
        role_from_config(config, name)
    contract = ExperimentContract(
        scientific_question=(
            "How much weighted representation error is attributable to lag and "
            "amplitude resolution, and do the frozen P/S/M roles pass?"
        ),
        target_semantics="centered_true_2d_external_kernel_surface",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="single_oracle_variable",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        basis_selection_uses_truth=True,
        smoothing_selection_metric="validation_prediction_mse",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        allowed_next_experiment="E2A0",
        experiment_role="ORACLE_REPRESENTATION_DIAGNOSTIC",
        model_class="M2",
        evaluation_distribution="ORACLE_GRID",
        resolution_role="NONE",
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)

    rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    projection_arrays: dict[str, np.ndarray] = {}
    lag_weights = np.full(
        int(common["L_x"]), 1.0 / int(common["L_x"]), dtype=np.float64
    )
    degree = int(e1b["degree"])
    strong = e1b["strong_rank2"]
    for scenario in common["scenarios_2d"]:
        for seed in common["development_seeds"]:
            sequence = generate_synthetic_sequence(
                str(scenario),
                seed=int(seed),
                n_samples=int(common["n_samples_natural"]),
                external_variables=10,
            )
            train_stop = sequence.split_target_intervals["train"][1]
            for variable in common["active_variables"]:
                variable = int(variable)
                if variable not in sequence.truth["active_support"]:
                    raise RuntimeError(
                        f"{scenario} variable {variable} is not oracle-active."
                    )
                train_values = sequence.x[:train_stop, variable]
                amplitude_domain = AmplitudeDomain.fit(
                    train_values,
                    padding_fraction=float(domain_config["padding_fraction"]),
                    core_quantiles=tuple(domain_config["core_quantiles"]),
                )
                truth_mean = true_kernel_surface(
                    sequence, variable, train_values
                ).mean(axis=1, keepdims=True)
                for domain_name, lower, upper in (
                    (
                        "core",
                        amplitude_domain.core_lower,
                        amplitude_domain.core_upper,
                    ),
                    (
                        "fit",
                        amplitude_domain.fit_lower,
                        amplitude_domain.fit_upper,
                    ),
                ):
                    amplitude_grid = np.linspace(
                        lower,
                        upper,
                        int(domain_config["grid_points"]),
                        dtype=np.float64,
                    )
                    amplitude_weights = normalized_trapezoidal_weights(
                        amplitude_grid
                    )
                    centered_truth = (
                        true_kernel_surface(sequence, variable, amplitude_grid)
                        - truth_mean
                    )
                    whitened_truth = (
                        np.sqrt(lag_weights)[:, None]
                        * centered_truth
                        * np.sqrt(amplitude_weights)[None, :]
                    )
                    spectrum = truth_spectrum(
                        whitened_truth,
                        tail_energy_min=float(
                            strong["truth_tail_energy_min"]
                        ),
                        sigma2_sigma1_min=float(
                            strong["truth_sigma2_sigma1_min"]
                        ),
                    )
                    singular = np.pad(
                        spectrum.singular_values,
                        (0, max(0, 3 - len(spectrum.singular_values))),
                    )
                    truth_rows.append(
                        {
                            "scenario": scenario,
                            "seed": int(seed),
                            "variable": variable,
                            "domain": domain_name,
                            "sigma1_truth": float(singular[0]),
                            "sigma2_truth": float(singular[1]),
                            "sigma3_truth": float(singular[2]),
                            "truth_tail_energy": spectrum.tail_energy,
                            "truth_sigma2_sigma1": spectrum.sigma2_sigma1,
                            "truth_spectral_gap": spectrum.spectral_gap,
                            "truth_rank_class": spectrum.rank_class,
                        }
                    )
                    for amplitude_count in e1b["amplitude_basis_candidates"]:
                        amplitude_basis = CenteredSplineBasis.fit(
                            train_values,
                            n_basis=int(amplitude_count),
                            degree=degree,
                            domain=amplitude_domain,
                        )
                        amplitude_evaluation = amplitude_basis.transform(
                            amplitude_grid
                        )
                        for lag_spec in e1b["lag_basis_candidates"]:
                            lag_type = str(lag_spec["type"])
                            lag_count = int(lag_spec["count"])
                            lag_basis = _v033_lag_basis(
                                lag_type=lag_type,
                                lag_count=lag_count,
                                length=int(common["L_x"]),
                                degree=degree,
                            )
                            projection = weighted_tensor_projection(
                                centered_truth,
                                lag_basis,
                                amplitude_evaluation,
                                lag_weights=lag_weights,
                                amplitude_weights=amplitude_weights,
                            )
                            sigma2_denominator = max(
                                float(singular[1]), np.finfo(np.float64).eps
                            )
                            gap_denominator = max(
                                spectrum.spectral_gap,
                                np.finfo(np.float64).eps,
                            )
                            rows.append(
                                {
                                    "scenario": scenario,
                                    "seed": int(seed),
                                    "variable": variable,
                                    "domain": domain_name,
                                    "lag_basis_type": lag_type,
                                    "lag_basis_count": lag_count,
                                    "amplitude_basis_count": int(
                                        amplitude_count
                                    ),
                                    "epsilon_lag": projection.epsilon_lag,
                                    "epsilon_amplitude": (
                                        projection.epsilon_amplitude
                                    ),
                                    "epsilon_amplitude_given_lag": (
                                        projection.epsilon_amplitude_given_lag
                                    ),
                                    "epsilon_lag_given_amplitude": (
                                        projection.epsilon_lag_given_amplitude
                                    ),
                                    "epsilon_joint": projection.epsilon_joint,
                                    "operator_error": projection.operator_error,
                                    "sigma1_truth": float(singular[0]),
                                    "sigma2_truth": float(singular[1]),
                                    "sigma3_truth": float(singular[2]),
                                    "truth_tail_energy": spectrum.tail_energy,
                                    "truth_sigma2_sigma1": (
                                        spectrum.sigma2_sigma1
                                    ),
                                    "operator_error_over_sigma2": (
                                        projection.operator_error
                                        / sigma2_denominator
                                    ),
                                    "operator_error_over_gap": (
                                        projection.operator_error
                                        / gap_denominator
                                    ),
                                    "truth_rank_class": spectrum.rank_class,
                                }
                            )
                            if (
                                int(amplitude_count) == 28
                                and (
                                    (lag_type, lag_count)
                                    in {
                                        ("cubic_bspline", 32),
                                        ("cubic_bspline", 48),
                                        ("discrete_identity", 64),
                                    }
                                )
                            ):
                                key = (
                                    f"{scenario}_seed{seed}_var{variable}_"
                                    f"{domain_name}_{lag_type}{lag_count}_x28"
                                )
                                projection_arrays[key + "_coefficients"] = (
                                    projection.coefficients
                                )
                                projection_arrays[key + "_amplitude_grid"] = (
                                    amplitude_grid
                                )
    expected_rows = (
        len(common["scenarios_2d"])
        * len(common["development_seeds"])
        * len(common["active_variables"])
        * 2
        * len(e1b["lag_basis_candidates"])
        * len(e1b["amplitude_basis_candidates"])
    )
    if len(rows) != expected_rows or expected_rows != 1440:
        raise RuntimeError(
            f"E1B full-grid violation: got {len(rows)}, expected 1440."
        )
    certificate = certify_resolution_roles(rows, config)
    certificate["full_grid_row_count"] = len(rows)
    certificate["all_lag_counts_present"] = sorted(
        {int(row["lag_basis_count"]) for row in rows}
    )
    certificate["all_amplitude_counts_present"] = sorted(
        {int(row["amplitude_basis_count"]) for row in rows}
    )
    write_csv(output / "representation_grid.csv", rows)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "truth_rank_profile.csv", truth_rows)
    write_json(output / "role_certificate.json", certificate)
    write_json(output / "summary.json", certificate)
    np.savez_compressed(output / "projection_arrays.npz", **projection_arrays)
    return str(certificate["status"])


def run_e2a0_v033(config: dict[str, object], result_root: Path) -> str:
    """Close truth replay, projected design, and direct/PCG solves."""

    e1b_summary = json.loads(
        (result_root / "E1B" / "summary.json").read_text(encoding="utf-8")
    )
    if e1b_summary["status"] != "E1B_RESOLUTION_ROLES_CERTIFIED":
        raise RuntimeError("E2A0 is blocked until E1B certifies all roles.")
    output = result_root / "E2A0"
    common = config["common"]
    domain_config = config["domain"]
    tolerances = config["e2a0"]
    contract = ExperimentContract(
        scientific_question=(
            "Do truth replay, projected tensor design, and FP64 direct/PCG "
            "solvers form a numerically closed implementation?"
        ),
        target_semantics="implementation_and_projected_operator_closure",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="single_oracle_variable",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        basis_selection_uses_truth=False,
        smoothing_selection_metric="validation_prediction_mse",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        allowed_next_experiment="E2A_M_SPACE",
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
        model_class="M2",
        evaluation_distribution="ORACLE_GRID",
        resolution_role="NONE",
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)
    rows: list[dict[str, object]] = []
    maximum_replay = 0.0
    maximum_matrix_direct = 0.0
    lag_weights = np.full(
        int(common["L_x"]), 1.0 / int(common["L_x"]), dtype=np.float64
    )
    for scenario in common["scenarios_2d"]:
        for seed in common["development_seeds"]:
            sequence = generate_synthetic_sequence(
                str(scenario),
                seed=int(seed),
                n_samples=int(common["n_samples_natural"]),
                external_variables=10,
            )
            components = replay_synthetic_components(sequence)
            train_stop = sequence.split_target_intervals["train"][1]
            targets = np.arange(int(common["L_x"]), train_stop, dtype=np.int64)
            for variable in common["active_variables"]:
                variable = int(variable)
                direct_truth = direct_apply_truth_kernel(
                    sequence, variable, targets
                )
                replay_truth = components.x_contribution_by_variable[
                    targets, variable
                ]
                replay_error = float(
                    np.max(np.abs(direct_truth - replay_truth))
                )
                maximum_replay = max(maximum_replay, replay_error)
                train_values = sequence.x[:train_stop, variable]
                amplitude_domain = AmplitudeDomain.fit(
                    train_values,
                    padding_fraction=float(domain_config["padding_fraction"]),
                    core_quantiles=tuple(domain_config["core_quantiles"]),
                )
                truth_mean = true_kernel_surface(
                    sequence, variable, train_values
                ).mean(axis=1, keepdims=True)
                amplitude_grid = np.linspace(
                    amplitude_domain.fit_lower,
                    amplitude_domain.fit_upper,
                    int(domain_config["grid_points"]),
                    dtype=np.float64,
                )
                amplitude_weights = normalized_trapezoidal_weights(
                    amplitude_grid
                )
                centered_truth = (
                    true_kernel_surface(sequence, variable, amplitude_grid)
                    - truth_mean
                )
                for role_name in ("PREDICTIVE", "STRUCTURAL", "MOTHER"):
                    role = role_from_config(config, role_name)
                    lag_basis = _v033_lag_basis(
                        lag_type=role.lag_type,
                        lag_count=role.lag_count,
                        length=int(common["L_x"]),
                        degree=int(config["e1b"]["degree"]),
                    )
                    amplitude_basis = CenteredSplineBasis.fit(
                        train_values,
                        n_basis=role.amplitude_count,
                        degree=int(config["e1b"]["degree"]),
                        domain=amplitude_domain,
                    )
                    projection = weighted_tensor_projection(
                        centered_truth,
                        lag_basis,
                        amplitude_basis.transform(amplitude_grid),
                        lag_weights=lag_weights,
                        amplitude_weights=amplitude_weights,
                    )
                    matrix = build_single_variable_matrix(
                        sequence.x[:, variable],
                        origin_indices=targets - 1,
                        lag_basis=lag_basis,
                        amplitude_basis=amplitude_basis,
                    )
                    via_matrix = (
                        matrix @ projection.coefficients.reshape(-1)
                    )
                    via_direct = direct_apply_projected_kernel(
                        sequence.x,
                        variable=variable,
                        target_indices=targets,
                        horizon=1,
                        lag_basis=lag_basis,
                        amplitude_basis=amplitude_basis,
                        coefficients=projection.coefficients,
                    )
                    closure_error = float(
                        np.max(np.abs(via_matrix - via_direct))
                    )
                    maximum_matrix_direct = max(
                        maximum_matrix_direct, closure_error
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": int(seed),
                            "variable": variable,
                            "resolution_role": role_name,
                            "truth_replay_max_abs_error": replay_error,
                            "matrix_direct_max_abs_error": closure_error,
                        }
                    )

    rng = np.random.default_rng(330)
    matrix = rng.normal(size=(800, 160))
    theta = rng.normal(size=160)
    projected_target = matrix @ theta
    penalty = np.zeros((160, 160), dtype=np.float64)
    direct = solve_full_kernel(
        matrix,
        projected_target,
        penalty,
        numerical_jitter_relative=1.0e-12,
        compute_condition_number=False,
    )
    pcg = solve_full_kernel_pcg(
        matrix,
        projected_target,
        penalty,
        relative_tolerance=1.0e-10,
        max_iterations=int(config["solver"]["pcg_max_iterations"]),
        block_slices=tuple(
            slice(start, min(start + 40, 160)) for start in range(0, 160, 40)
        ),
        numerical_jitter_relative=1.0e-12,
    )
    target_norm = max(
        float(np.linalg.norm(projected_target)), np.finfo(np.float64).eps
    )
    projected_relative_error = float(
        np.linalg.norm(direct.predictions - projected_target) / target_norm
    )
    direct_pcg_difference = float(
        np.linalg.norm(direct.predictions - pcg.predictions) / target_norm
    )
    passed = (
        maximum_replay <= float(tolerances["replay_tolerance"])
        and maximum_matrix_direct
        <= float(tolerances["matrix_direct_tolerance"])
        and projected_relative_error
        <= float(tolerances["projected_target_prediction_relative_error"])
        and direct.relative_kkt_residual
        <= float(config["solver"]["kkt_relative_residual"])
        and direct_pcg_difference
        <= float(tolerances["direct_pcg_relative_difference"])
        and pcg.converged
    )
    status = (
        "E2A0_IMPLEMENTATION_CLOSURE_PASS"
        if passed
        else "E2A0_IMPLEMENTATION_CLOSURE_FAIL"
    )
    summary = {
        "status": status,
        "maximum_truth_replay_error": maximum_replay,
        "maximum_matrix_direct_error": maximum_matrix_direct,
        "projected_target_prediction_relative_error": projected_relative_error,
        "direct_kkt_relative_residual": direct.relative_kkt_residual,
        "pcg_kkt_relative_residual": pcg.relative_kkt_residual,
        "direct_pcg_prediction_relative_difference": direct_pcg_difference,
        "pcg_converged": pcg.converged,
        "next_allowed_experiment": "E2A_M_SPACE" if passed else "STOP",
    }
    write_csv(output / "metrics.csv", rows)
    write_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "fit.npz",
        direct_coefficients=direct.coefficients,
        pcg_coefficients=pcg.coefficients,
        projected_target=projected_target,
    )
    return status


def _capacity_split_and_values(
    sequence,
    *,
    variable: int,
    distribution: str,
    config: dict[str, object],
) -> tuple[object, np.ndarray, dict[str, np.ndarray]]:
    """Create one-variable NAT/PERM/SPACE inputs and target-index splits."""

    common = config["common"]
    natural_train_stop = sequence.split_target_intervals["train"][1]
    natural_train_values = sequence.x[:natural_train_stop, variable]
    natural_domain = AmplitudeDomain.fit(
        natural_train_values,
        padding_fraction=float(config["domain"]["padding_fraction"]),
        core_quantiles=tuple(config["domain"]["core_quantiles"]),
    )
    if distribution == "NAT":
        values = sequence.x[:, variable].copy()
        splits = {
            name: np.arange(start, stop, dtype=np.int64)
            for name, (start, stop) in sequence.split_target_intervals.items()
        }
        splits = {
            name: indices[indices >= int(common["L_x"])]
            for name, indices in splits.items()
        }
        return sequence, values, splits
    length = int(common["n_samples_excitation"])
    if distribution == "SPACE":
        values = space_filling_core_excitation(
            natural_domain,
            length=length,
            seed=int(sequence.seed) + int(config["space"]["seed_offset"]),
        )
    elif distribution == "PERM":
        values = permuted_marginal_excitation(
            natural_train_values,
            length=length,
            seed=int(sequence.seed) + int(config["perm"]["seed_offset"]),
        )
    else:
        raise ValueError(f"Unknown capacity distribution: {distribution}.")
    x = np.zeros((length, sequence.x.shape[1]), dtype=np.float64)
    x[:, variable] = values
    capacity_sequence = replace(sequence, x=x)
    splits = chronological_split_indices(
        length,
        burn_in=int(common["L_x"]),
        fractions=tuple(config["space"]["split_fractions"]),
    )
    return capacity_sequence, values, splits


def _capacity_task(payload: dict[str, object]) -> dict[str, object]:
    """Fit one scenario/seed/variable capacity task in an isolated process."""

    os.environ["OMP_NUM_THREADS"] = "3"
    os.environ["MKL_NUM_THREADS"] = "3"
    os.environ["OPENBLAS_NUM_THREADS"] = "3"
    os.environ["NUMEXPR_NUM_THREADS"] = "3"
    config = payload["config"]
    scenario = str(payload["scenario"])
    seed = int(payload["seed"])
    variable = int(payload["variable"])
    distribution = str(payload["distribution"])
    role_name = str(payload["role"])
    e1b_projection_core = float(payload["e1b_projection_core"])
    truth_rank_class = str(payload["truth_rank_class"])
    common = config["common"]
    sequence = generate_synthetic_sequence(
        scenario,
        seed=seed,
        n_samples=int(common["n_samples_natural"]),
        external_variables=10,
    )
    natural_train_stop = sequence.split_target_intervals["train"][1]
    natural_domain = AmplitudeDomain.fit(
        sequence.x[:natural_train_stop, variable],
        padding_fraction=float(config["domain"]["padding_fraction"]),
        core_quantiles=tuple(config["domain"]["core_quantiles"]),
    )
    role = role_from_config(config, role_name)
    lag_basis = _v033_lag_basis(
        lag_type=role.lag_type,
        lag_count=role.lag_count,
        length=int(common["L_x"]),
        degree=int(config["e1b"]["degree"]),
    )
    if distribution == "SPACE":
        histories = space_filling_history_excitation(
            natural_domain,
            sample_count=int(common["n_samples_excitation"]),
            lag_count=int(common["L_x"]),
            seed=seed + int(config["space"]["seed_offset"]),
        )
        splits = chronological_split_indices(
            len(histories),
            burn_in=0,
            fractions=tuple(config["space"]["split_fractions"]),
        )
        train_amplitude_values = histories[splits["train"]].reshape(-1)
        amplitude_basis = CenteredSplineBasis.fit(
            train_amplitude_values,
            n_basis=role.amplitude_count,
            degree=int(config["e1b"]["degree"]),
            domain=natural_domain,
        )
        matrices = {
            name: build_matrix_from_histories(
                histories[indices],
                lag_basis=lag_basis,
                amplitude_basis=amplitude_basis,
            )
            for name, indices in splits.items()
        }

        def history_kernel_values(selected: np.ndarray) -> np.ndarray:
            length = int(common["L_x"])
            primary = np.asarray(
                sequence.truth["q_primary"], dtype=np.float64
            )[variable]
            primary_response = truth_response(variable, selected)
            if scenario == "AR-S3":
                secondary = np.asarray(
                    sequence.truth["q_secondary"], dtype=np.float64
                )[variable]
                return (
                    0.6 * primary_response * primary[None, :]
                    + 0.4
                    * second_truth_response(variable, selected)
                    * secondary[None, :]
                )
            if scenario == "AR-S4U":
                kernel = np.empty_like(selected)
                lag_axis = np.arange(length, dtype=np.float64)[:, None]
                for lag in range(length):
                    amplitudes = selected[:, lag]
                    centers = 8.0 + 12.0 / (
                        1.0 + np.exp(-2.0 * amplitudes)
                    )
                    unnormalized = np.exp(
                        -0.5 * ((lag_axis - centers[None, :]) / 2.0) ** 2
                    )
                    weights = unnormalized[lag] / unnormalized.sum(axis=0)
                    kernel[:, lag] = weights * primary_response[:, lag]
                return kernel
            return primary_response * primary[None, :]

        targets = {
            name: history_kernel_values(histories[indices]).sum(axis=1)
            for name, indices in splits.items()
        }
        truth_mean = history_kernel_values(
            histories[splits["train"]]
        ).mean(axis=0)[:, None]
    else:
        capacity_sequence, values, splits = _capacity_split_and_values(
            sequence,
            variable=variable,
            distribution=distribution,
            config=config,
        )
        train_stop = int(splits["train"][-1] + 1)
        train_amplitude_values = values[:train_stop]
        amplitude_basis = CenteredSplineBasis.fit(
            train_amplitude_values,
            n_basis=role.amplitude_count,
            degree=int(config["e1b"]["degree"]),
            domain=natural_domain,
        )
        matrices = {
            name: build_single_variable_matrix(
                values,
                origin_indices=indices - 1,
                lag_basis=lag_basis,
                amplitude_basis=amplitude_basis,
            )
            for name, indices in splits.items()
        }
        targets = {
            name: direct_apply_truth_kernel(
                capacity_sequence, variable, indices
            )
            for name, indices in splits.items()
        }
        truth_mean = true_kernel_surface(
            capacity_sequence, variable, train_amplitude_values
        ).mean(axis=1, keepdims=True)
    lag_gram = lag_basis.T @ lag_basis / len(lag_basis)
    train_amplitude = amplitude_basis.transform(train_amplitude_values)
    amplitude_gram = train_amplitude.T @ train_amplitude / len(train_amplitude)
    candidate_rows: list[dict[str, object]] = []
    candidate_fits = []
    for order, (lag_smoothing, amplitude_smoothing) in enumerate(
        smoothing_pairs(config)
    ):
        penalty = tensor_penalty(
            lag_gram,
            [amplitude_gram],
            lag_smoothness=lag_smoothing,
            amplitude_smoothness=amplitude_smoothing,
            ridge_weight=float(config["capacity"]["numerical_ridge"]),
        )
        fit = solve_full_kernel(
            matrices["train"],
            targets["train"],
            penalty,
            numerical_jitter_relative=float(config["capacity"]["numerical_ridge"]),
            fit_intercept=True,
            compute_condition_number=False,
        )
        validation_prediction = (
            matrices["validation"] @ fit.coefficients + fit.intercept
        )
        candidate_rows.append(
            {
                "configuration_order": order,
                "lag_smoothing": lag_smoothing,
                "amplitude_smoothing": amplitude_smoothing,
                "validation_contribution_mse": mse(
                    targets["validation"], validation_prediction
                ),
            }
        )
        candidate_fits.append(fit)
    selected = select_minimum_validation_mse(candidate_rows)
    selected_fit = candidate_fits[int(selected["configuration_order"])]
    predictions = {
        name: matrix @ selected_fit.coefficients + selected_fit.intercept
        for name, matrix in matrices.items()
    }

    amplitude_grid_core = np.linspace(
        natural_domain.core_lower,
        natural_domain.core_upper,
        int(config["domain"]["grid_points"]),
        dtype=np.float64,
    )
    amplitude_grid_fit = np.linspace(
        natural_domain.fit_lower,
        natural_domain.fit_upper,
        int(config["domain"]["grid_points"]),
        dtype=np.float64,
    )
    def surface_metrics(amplitude_grid: np.ndarray) -> tuple[float, np.ndarray]:
        truth = (
            true_kernel_surface(sequence, variable, amplitude_grid)
            - truth_mean
        )
        estimate = (
            lag_basis
            @ selected_fit.coefficients.reshape(
                role.lag_count, role.amplitude_count
            )
            @ amplitude_basis.transform(amplitude_grid).T
        )
        return normalized_root_mean_square_error(truth, estimate), estimate

    core_surface_nrmse, core_estimate = surface_metrics(amplitude_grid_core)
    fit_surface_nrmse, _ = surface_metrics(amplitude_grid_fit)
    covariance = matrices["train"].T @ matrices["train"] / len(
        matrices["train"]
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive = eigenvalues[
        eigenvalues > max(float(eigenvalues[-1]), 1.0) * 1.0e-12
    ]
    design_condition = (
        float(np.sqrt(positive[-1] / positive[0]))
        if positive.size
        else float("inf")
    )
    probabilities = np.maximum(eigenvalues, 0)
    probabilities /= max(float(probabilities.sum()), np.finfo(float).eps)
    probabilities = probabilities[probabilities > 0]
    effective_rank = float(
        np.exp(-np.sum(probabilities * np.log(probabilities)))
    )

    lag_surface_gram = lag_basis.T @ (
        np.full(len(lag_basis), 1.0 / len(lag_basis))[:, None] * lag_basis
    )
    amplitude_weights = normalized_trapezoidal_weights(amplitude_grid_core)
    amplitude_evaluation = amplitude_basis.transform(amplitude_grid_core)
    amplitude_surface_gram = amplitude_evaluation.T @ (
        amplitude_weights[:, None] * amplitude_evaluation
    )
    spectrum = gram_whitened_svd(
        selected_fit.coefficients.reshape(
            role.lag_count, role.amplitude_count
        ),
        lag_surface_gram,
        amplitude_surface_gram,
    )
    validation_full_mse = mse(
        targets["validation"], predictions["validation"]
    )
    rank_mses: dict[int, float] = {}
    for rank in (1, 2):
        if rank <= min(role.lag_count, role.amplitude_count):
            truncated = spectrum.truncate(rank).reshape(-1)
            rank_prediction = (
                matrices["validation"] @ truncated + selected_fit.intercept
            )
            rank_mses[rank] = mse(targets["validation"], rank_prediction)
    reducible_gap_capture = float("nan")
    if validation_full_mse < rank_mses[1]:
        reducible_gap_capture = (
            rank_mses[1] - rank_mses[2]
        ) / max(rank_mses[1] - validation_full_mse, np.finfo(float).eps)

    row = {
        "scenario": scenario,
        "seed": seed,
        "variable": variable,
        "distribution": distribution,
        "resolution_role": role_name,
        "lag_basis_type": role.lag_type,
        "lag_basis_count": role.lag_count,
        "amplitude_basis_count": role.amplitude_count,
        "selected_lag_smoothing": selected["lag_smoothing"],
        "selected_amplitude_smoothing": selected["amplitude_smoothing"],
        "validation_contribution_mse": validation_full_mse,
        "validation_contribution_r2": r2(
            targets["validation"], predictions["validation"]
        ),
        "test_contribution_mse": mse(
            targets["test"], predictions["test"]
        ),
        "test_contribution_r2": r2(
            targets["test"], predictions["test"]
        ),
        "empirical_operator_nrmse": empirical_operator_nrmse(
            targets["validation"], predictions["validation"]
        ),
        "core_surface_nrmse": core_surface_nrmse,
        "fit_surface_nrmse": fit_surface_nrmse,
        "e1b_projection_core_nrmse": e1b_projection_core,
        "estimator_to_projection_core_nrmse": float("nan"),
        "kkt_relative_residual": selected_fit.relative_kkt_residual,
        "design_condition_number": design_condition,
        "effective_rank": effective_rank,
        "truth_rank_class": truth_rank_class,
        "estimated_sigma1": float(spectrum.singular_values[0]),
        "estimated_sigma2": float(spectrum.singular_values[1]),
        "rank1_validation_mse": rank_mses[1],
        "rank2_validation_mse": rank_mses[2],
        "rank2_reducible_gap_capture": reducible_gap_capture,
        "candidate_count": len(candidate_rows),
    }
    return {
        "row": row,
        "coefficients": selected_fit.coefficients,
        "intercept": selected_fit.intercept,
        "core_estimate": core_estimate,
        "candidate_rows": candidate_rows,
    }


def _e1b_lookup(
    result_root: Path,
    *,
    role_name: str,
) -> dict[tuple[str, int, int], tuple[float, str]]:
    role = {
        "MOTHER": ("discrete_identity", 64),
        "STRUCTURAL": ("cubic_bspline", 48),
        "PREDICTIVE": ("cubic_bspline", 32),
    }[role_name]
    lookup = {}
    with (result_root / "E1B" / "representation_grid.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            if (
                row["domain"] == "core"
                and row["lag_basis_type"] == role[0]
                and int(row["lag_basis_count"]) == role[1]
                and int(row["amplitude_basis_count"]) == 28
            ):
                lookup[
                    (row["scenario"], int(row["seed"]), int(row["variable"]))
                ] = (
                    float(row["epsilon_joint"]),
                    row["truth_rank_class"],
                )
    return lookup


def run_capacity_v033(
    config: dict[str, object],
    result_root: Path,
    *,
    experiment: str,
) -> str:
    specifications = {
        "E2A_M_SPACE": (
            "MOTHER",
            "SPACE",
            "mother_space",
            "E2A_M_SPACE_CAPACITY_PASS",
            "E2A_M_SPACE_CAPACITY_FAIL",
            "E2A_S_SPACE",
        ),
        "E2A_S_SPACE": (
            "STRUCTURAL",
            "SPACE",
            "structural_space",
            "E2A_S_SPACE_CAPACITY_PASS",
            "E2A_S_SPACE_CAPACITY_FAIL",
            "E2A_P_NAT",
        ),
        "E2A_P_NAT": (
            "PREDICTIVE",
            "NAT",
            "predictive_natural",
            "E2A_P_NAT_CAPACITY_PASS",
            "E2A_P_NAT_CAPACITY_FAIL",
            "E2A_P_PERM",
        ),
        "E2A_P_PERM": (
            "PREDICTIVE",
            "PERM",
            "predictive_permuted",
            "E2A_P_PERM_CAPACITY_PASS",
            "E2A_P_PERM_CAPACITY_FAIL",
            "DECISION",
        ),
    }
    role_name, distribution, gate_name, pass_status, fail_status, next_stage = (
        specifications[experiment]
    )
    dependency = {
        "E2A_M_SPACE": ("E2A0", "E2A0_IMPLEMENTATION_CLOSURE_PASS"),
        "E2A_S_SPACE": ("E2A_M_SPACE", "E2A_M_SPACE_CAPACITY_PASS"),
        "E2A_P_NAT": ("E2A_S_SPACE", "E2A_S_SPACE_CAPACITY_PASS"),
        "E2A_P_PERM": ("E2A_S_SPACE", "E2A_S_SPACE_CAPACITY_PASS"),
    }[experiment]
    dependency_summary = result_root / dependency[0] / "summary.json"
    if not dependency_summary.exists() or json.loads(
        dependency_summary.read_text(encoding="utf-8")
    )["status"] != dependency[1]:
        raise RuntimeError(f"{experiment} is blocked by {dependency[0]}.")
    output = result_root / experiment
    contract = ExperimentContract(
        scientific_question=(
            f"Does the frozen {role_name} space recover one-variable "
            f"contributions under {distribution} excitation?"
        ),
        target_semantics="single_oracle_variable_true_external_contribution",
        target_contains_ar=False,
        model_contains_ar=False,
        target_contains_x=True,
        model_contains_x=True,
        truth_used_for_training=False,
        truth_used_for_evaluation=True,
        support_used_for_training="single_oracle_variable",
        hyperparameter_selection_metric="validation_prediction_loss_only",
        basis_selection_uses_truth=False,
        smoothing_selection_metric="validation_prediction_mse",
        rank_inputs_used_for_selection=False,
        test_used_for_selection=False,
        allowed_next_experiment=next_stage,
        experiment_role="ORACLE_COMPONENT_DIAGNOSTIC",
        model_class="M2",
        evaluation_distribution=distribution,
        resolution_role=role_name,
    )
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "config.json", config)
    lookup = _e1b_lookup(result_root, role_name=role_name)
    payloads = []
    for scenario in config["common"]["scenarios_2d"]:
        for seed in config["common"]["development_seeds"]:
            for variable in config["common"]["active_variables"]:
                projection, rank_class = lookup[
                    (str(scenario), int(seed), int(variable))
                ]
                payloads.append(
                    {
                        "config": config,
                        "scenario": scenario,
                        "seed": seed,
                        "variable": variable,
                        "distribution": distribution,
                        "role": role_name,
                        "e1b_projection_core": projection,
                        "truth_rank_class": rank_class,
                    }
                )
    results = []
    workers = min(8, len(payloads))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_capacity_task, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print(
                f"{experiment}_PROGRESS={index}/{len(payloads)}",
                flush=True,
            )
    results.sort(
        key=lambda item: (
            item["row"]["scenario"],
            int(item["row"]["seed"]),
            int(item["row"]["variable"]),
        )
    )
    rows = [item["row"] for item in results]
    gate = config["gates"][gate_name]
    for row in rows:
        base_pass = (
            float(row["validation_contribution_r2"])
            >= float(gate["validation_contribution_r2"])
            and float(row["empirical_operator_nrmse"])
            <= float(gate["empirical_operator_nrmse"])
            and float(row["kkt_relative_residual"])
            <= float(config["solver"]["kkt_relative_residual"])
        )
        if role_name in {"MOTHER", "STRUCTURAL"}:
            surface_limit = max(
                float(gate["surface_nrmse_absolute_max"]),
                float(gate["surface_nrmse_projection_multiplier"])
                * float(row["e1b_projection_core_nrmse"]),
            )
            base_pass &= float(row["core_surface_nrmse"]) <= surface_limit
        if role_name == "STRUCTURAL":
            if row["truth_rank_class"] == "rank1":
                base_pass &= float(row["rank1_validation_mse"]) <= (
                    float(gate["rank1_full_mse_ratio_max"])
                    * float(row["validation_contribution_mse"])
                )
            elif row["truth_rank_class"] == "strong_rank2":
                base_pass &= float(row["rank2_validation_mse"]) <= (
                    1.05 * float(row["validation_contribution_mse"])
                ) and float(row["rank2_reducible_gap_capture"]) >= float(
                    gate["strong_rank2_reducible_gap_capture"]
                )
        row["seed_gate_pass"] = bool(base_pass)
    group_passes = {}
    for scenario in config["common"]["scenarios_2d"]:
        for variable in config["common"]["active_variables"]:
            group = [
                row
                for row in rows
                if row["scenario"] == scenario
                and int(row["variable"]) == int(variable)
            ]
            passed_seeds = sum(bool(row["seed_gate_pass"]) for row in group)
            group_passes[f"{scenario}:var{variable}"] = {
                "passed_seeds": passed_seeds,
                "required_seeds": 4,
                "passed": passed_seeds >= 4,
            }
    passed = all(item["passed"] for item in group_passes.values())
    status = pass_status if passed else fail_status
    write_csv(output / "metrics.csv", rows)
    write_json(
        output / "summary.json",
        {
            "status": status,
            "task_count": len(rows),
            "group_results": group_passes,
            "next_allowed_experiment": next_stage if passed else "STOP",
        },
    )
    arrays = {}
    for item in results:
        row = item["row"]
        key = f"{row['scenario']}_seed{row['seed']}_var{row['variable']}"
        arrays[key + "_coefficients"] = item["coefficients"]
        arrays[key + "_intercept"] = np.array([item["intercept"]])
    np.savez_compressed(output / "fit.npz", **arrays)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--experiment",
        choices=[
            "E0",
            "E1",
            "E1R",
            "E2A",
            "E2B",
            "E3",
            "R1",
            "E1A",
            "E2A0",
            "E2A_NAT",
            "E2A_PERM",
            "E2A_SPACE",
            "E1B",
            "E2A_M_SPACE",
            "E2A_S_SPACE",
            "E2A_P_NAT",
            "E2A_P_PERM",
        ],
        required=True,
    )
    parser.add_argument("--stage", choices=["development", "confirmation"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage != "development":
        raise SystemExit("Confirmation is not unlocked.")
    config_path = args.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    schema_version = int(config.get("schema_version", 1))
    if schema_version == 4:
        result_root = ROOT / "results" / "spectral_v033"
        runners = {
            "E1B": run_e1b,
            "E2A0": run_e2a0_v033,
            "E2A_M_SPACE": lambda cfg, root: run_capacity_v033(
                cfg, root, experiment="E2A_M_SPACE"
            ),
            "E2A_S_SPACE": lambda cfg, root: run_capacity_v033(
                cfg, root, experiment="E2A_S_SPACE"
            ),
            "E2A_P_NAT": lambda cfg, root: run_capacity_v033(
                cfg, root, experiment="E2A_P_NAT"
            ),
            "E2A_P_PERM": lambda cfg, root: run_capacity_v033(
                cfg, root, experiment="E2A_P_PERM"
            ),
        }
        if args.experiment not in runners:
            raise SystemExit(
                f"{args.experiment} is not implemented or not unlocked for v0.3.3."
            )
        status = runners[args.experiment](config, result_root)
    elif schema_version == 3:
        result_root = ROOT / "results" / "spectral_v032"
        runners = {
            "R1": run_r1,
            "E1A": run_e1a,
            "E2A0": run_e2a0,
        }
        if args.experiment not in runners:
            raise SystemExit(
                f"{args.experiment} is not implemented or not unlocked for v0.3.2."
            )
        status = runners[args.experiment](config, result_root)
    elif schema_version == 2:
        result_root = ROOT / "results" / "spectral_v031"
        if args.experiment not in {"E1R", "E2A"}:
            raise SystemExit(
                f"{args.experiment} is not implemented or not unlocked for v0.3.1."
            )
        status = (
            run_e1r(config, result_root)
            if args.experiment == "E1R"
            else run_e2a(config, result_root)
        )
    else:
        if args.experiment not in {"E0", "E1"}:
            raise SystemExit("Only E0/E1 belong to the v0.3 runner.")
        status = run_e0(config) if args.experiment == "E0" else run_e1(config)
    print(status)


if __name__ == "__main__":
    main()
