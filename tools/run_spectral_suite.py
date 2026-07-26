#!/usr/bin/env python3
"""Frozen Spectral v0.3 E0/E1 development runner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.spectral.contracts import ExperimentContract
from ar_raphu.spectral.metrics import normalized_root_mean_square_error, r2
from ar_raphu.spectral.spline_basis import (
    CenteredSplineBasis,
    clamped_knots,
    evaluate_basis,
)
from ar_raphu.spectral.synthetic_components import (
    replay_synthetic_components,
    true_kernel_surface,
)
from ar_raphu.synthetic import generate_synthetic_sequence


CONFIG_PATH = ROOT / "configs" / "spectral_v03.yaml"
RESULT_ROOT = ROOT / "results" / "spectral_v03"


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["E0", "E1"], required=True)
    parser.add_argument("--stage", choices=["development", "confirmation"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage != "development":
        raise SystemExit("E0/E1 confirmation is not unlocked.")
    config = load_config()
    status = run_e0(config) if args.experiment == "E0" else run_e1(config)
    print(status)


if __name__ == "__main__":
    main()
