#!/usr/bin/env python3
"""Run one preregistered Predictive-State AR-RAPHU v3 diagnostic job."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from STAGE1_DUAL_SOLVER_V20_bundle.stage1.proximal import (  # noqa: E402
    apply_group_proximal_step,
)
from ar_raphu.diagnostics.config import (  # noqa: E402
    EXPERIMENT_VARIANTS,
    load_diagnostic_config,
    validate_job,
)
from ar_raphu.diagnostics.gate_fista import solve_gate_path  # noqa: E402
from ar_raphu.diagnostics.instrumentation import (  # noqa: E402
    detect_starvation,
    gradient_norm,
    kernel_statistics,
    parameter_norm,
    proximal_collapse,
)
from ar_raphu.diagnostics.rank2_model import (  # noqa: E402
    ACTIVE_SUPPORT,
    ARRAPHURank2Diagnostic,
)
from ar_raphu.diagnostics.residual_data import (  # noqa: E402
    PreparedExternalResidualData,
)
from ar_raphu.diagnostics.train_utils import (  # noqa: E402
    DiagnosticTrainResult,
    evaluate_diagnostic,
    seed_everything,
    train_diagnostic_model,
)
from ar_raphu.diagnostics.truth_metrics import (  # noqa: E402
    contribution_energy_metrics,
    rank2_lag_metrics,
    surface_metrics,
    wasserstein1_discrete,
)
from ar_raphu.model import ARRAPHURank1  # noqa: E402
from ar_raphu.sequence_data import PreparedDirectForecastData  # noqa: E402
from ar_raphu.synthetic import generate_synthetic_sequence  # noqa: E402


RESULT_ROOT = PROJECT_ROOT / "results" / "v3_diagnostics"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


def configure_threads(config: dict[str, Any]) -> None:
    threads = int(config["runtime"]["cpu_threads_per_worker"])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)


def job_root(
    experiment: str, variant: str, seed: int, horizon: int
) -> Path:
    root = RESULT_ROOT / experiment
    if experiment == "D3":
        root = root / f"horizon_{horizon}"
    return root / f"seed_{seed}" / variant


def prepare(
    scenario: str,
    *,
    seed: int,
    track: str,
    horizon: int,
    config: dict[str, Any],
):
    common = config["common"]
    sequence = generate_synthetic_sequence(
        scenario,
        seed=seed,
        n_samples=common["n_samples"],
        external_variables=common["external_variables"],
    )
    data = PreparedDirectForecastData.from_sequence(
        sequence.x,
        sequence.y_observed,
        track=track,
        horizon=horizon,
        L_x=common["L_x"],
        L_y=common["L_y"],
        split_target_intervals=sequence.split_target_intervals,
    )
    x_ranges, y_range = data.scaler.input_grid_ranges(
        sequence.x, sequence.y_observed
    )
    return sequence, data, x_ranges, y_range


def build_rank1(
    *,
    track: str,
    horizon: int,
    x_ranges: list[tuple[float, float]],
    y_range: tuple[float, float],
    config: dict[str, Any],
    free_q: bool,
    active_only: bool,
    device: torch.device,
) -> ARRAPHURank1:
    common = config["common"]
    inactive = (
        tuple(
            sorted(
                set(range(common["external_variables"]))
                - set(common["active_support"])
            )
        )
        if active_only
        else ()
    )
    return ARRAPHURank1(
        track=track,
        horizon=horizon,
        external_channels=common["external_variables"],
        inactive_external_channels=inactive,
        L_x=common["L_x"] if track in {"X", "XAR"} else None,
        L_y=common["L_y"] if track in {"AR", "XAR"} else None,
        input_grid_ranges_x=x_ranges if track in {"X", "XAR"} else None,
        input_grid_range_y=y_range if track in {"AR", "XAR"} else None,
        hidden_kan=common["hidden_kan"],
        grid_size=common["grid_size"],
        response_execution_mode=common["response_execution_mode"],
        external_delay_mode=(
            "free_static_logits" if free_q else "static_gamma"
        ),
    ).to(device)


def initialize_rank1_truth_mixture(
    model: ARRAPHURank1, sequence
) -> None:
    if model.external_branch is None or model.external_branch.delay_logits is None:
        raise ValueError("Truth mixture initialization requires free X logits.")
    mixture = (
        0.6 * sequence.truth["q_primary"]
        + 0.4 * sequence.truth["q_secondary"]
    )
    with torch.no_grad():
        logits = torch.zeros_like(model.external_branch.delay_logits)
        logits[list(ACTIVE_SUPPORT)] = torch.log(
            torch.as_tensor(
                mixture[list(ACTIVE_SUPPORT)],
                device=logits.device,
                dtype=logits.dtype,
            ).clamp_min(1.0e-8)
        )
        model.external_branch.delay_logits.copy_(logits)


def build_rank2(
    *,
    horizon: int,
    x_ranges: list[tuple[float, float]],
    y_range: tuple[float, float],
    sequence,
    q_mode: str,
    include_ar: bool,
    config: dict[str, Any],
    device: torch.device,
) -> ARRAPHURank2Diagnostic:
    common = config["common"]
    return ARRAPHURank2Diagnostic(
        horizon=horizon,
        input_grid_ranges_x=x_ranges,
        q_primary=sequence.truth["q_primary"],
        q_secondary=sequence.truth["q_secondary"],
        q_mode=q_mode,
        include_ar=include_ar,
        input_grid_range_y=y_range if include_ar else None,
        hidden_kan=common["hidden_kan"],
        grid_size=common["grid_size"],
        response_execution_mode=common["response_execution_mode"],
    ).to(device)


def train_fixed(
    model,
    data,
    *,
    epochs: int,
    patience: int,
    config: dict[str, Any],
    device: torch.device,
    lag_smoothness: float,
    joint_lr: float | None = None,
) -> DiagnosticTrainResult:
    common = config["common"]
    return train_diagnostic_model(
        model,
        data,
        max_epochs=epochs,
        patience=patience,
        response_lr=common["response_learning_rate"],
        lag_lr=common["lag_learning_rate"],
        ar_lr=common["ar_learning_rate"],
        joint_lr=joint_lr,
        lag_smoothness=lag_smoothness,
        validation_interval=common["validation_interval"],
        batch_size=common["physical_chunk"],
        device=device,
        scheduler_factor=common["scheduler_factor"],
        scheduler_patience_validations=common[
            "scheduler_patience_validations"
        ],
        min_learning_rate=common["min_learning_rate"],
    )


def save_training_job(
    root: Path,
    *,
    job_config: dict[str, Any],
    summary: dict[str, Any],
    history: list[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> None:
    atomic_json(root / "config.json", job_config)
    atomic_json(root / "summary.json", summary)
    atomic_csv(root / "training_log.csv", history)
    temporary = root / "best.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, root / "best.pt")


def serializable_surface(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "arrays"
    }


def save_structure_files(
    root: Path,
    model,
    surface: dict[str, Any],
) -> None:
    atomic_json(root / "surface_metrics.json", serializable_surface(surface))
    arrays = dict(surface["arrays"])
    if isinstance(model, ARRAPHURank2Diagnostic):
        first, second = (
            value.detach().cpu().numpy() for value in model.lag_kernels()
        )
        arrays["q_component_1"] = first
        arrays["q_component_2"] = second
    else:
        arrays["q_rank1"] = (
            model.external_branch._static_q().detach().cpu().numpy()
        )
    temporary = root / "lag_kernels.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, root / "lag_kernels.npz")


def base_summary(
    validation: dict[str, Any], test: dict[str, Any], data
) -> dict[str, Any]:
    return {
        "validation_rmse_scaled": float(validation["rmse"]),
        "validation_rmse_original": float(validation["rmse"])
        * float(data.scaler.y_scale),
        "validation_r2": float(validation["r2"]),
        "test_rmse_scaled": float(test["rmse"]),
        "test_rmse_original": float(test["rmse"])
        * float(data.scaler.y_scale),
        "test_r2": float(test["r2"]),
    }


def run_d1(
    variant: str,
    seed: int,
    horizon: int,
    config: dict[str, Any],
    device: torch.device,
    root: Path,
) -> None:
    sequence, data, x_ranges, y_range = prepare(
        config["D1"]["scenario"],
        seed=seed,
        track="X",
        horizon=horizon,
        config=config,
    )
    learned = variant == "rank2_learned_q_truth_init"
    model = build_rank2(
        horizon=horizon,
        x_ranges=x_ranges,
        y_range=y_range,
        sequence=sequence,
        q_mode="free_truth_init" if learned else "oracle_fixed",
        include_ar=False,
        config=config,
        device=device,
    )
    result = train_fixed(
        model,
        data,
        epochs=(
            config["D1"]["learned_q_epochs"]
            if learned
            else config["D1"]["oracle_epochs"]
        ),
        patience=config["D1"]["patience"],
        config=config,
        device=device,
        lag_smoothness=(
            config["common"]["free_lag_smoothness"] if learned else 0.0
        ),
    )
    validation = evaluate_diagnostic(
        model,
        data,
        "validation",
        batch_size=config["common"]["physical_chunk"],
        device=device,
        return_contributions=True,
    )
    test = evaluate_diagnostic(
        model,
        data,
        "test",
        batch_size=config["common"]["physical_chunk"],
        device=device,
        return_contributions=True,
    )
    surface = surface_metrics(
        model, sequence, data, rank2_truth=True, points=201
    )
    closure = float(
        np.max(
            np.abs(
                validation["prediction"]
                - (
                    float(model.bias.detach().cpu())
                    + validation["contributions"].sum(axis=1)
                )
            )
        )
    )
    summary = {
        **base_summary(validation, test, data),
        "experiment": "D1",
        "variant": variant,
        "seed": seed,
        "horizon": horizon,
        "best_epoch": result.best_epoch,
        "mean_surface_nrmse": surface["mean_surface_nrmse"],
        "mean_surface_correlation": surface["mean_surface_correlation"],
        "contribution_closure_max_abs": closure,
    }
    if learned:
        summary.update(rank2_lag_metrics(model, sequence))
    save_training_job(
        root,
        job_config={
            "experiment": "D1",
            "variant": variant,
            "seed": seed,
            "horizon": horizon,
            "frozen_common": config["common"],
            "frozen_experiment": config["D1"],
        },
        summary=summary,
        history=result.history,
        checkpoint={"model_state": cpu_state(result.best_state)},
    )
    save_structure_files(root, model, surface)


def run_d2(
    variant: str,
    seed: int,
    horizon: int,
    config: dict[str, Any],
    device: torch.device,
    root: Path,
) -> None:
    sequence, data, x_ranges, y_range = prepare(
        config["D2"]["scenario"],
        seed=seed,
        track="X",
        horizon=horizon,
        config=config,
    )
    if variant == "rank1_free_q":
        model = build_rank1(
            track="X",
            horizon=horizon,
            x_ranges=x_ranges,
            y_range=y_range,
            config=config,
            free_q=True,
            active_only=True,
            device=device,
        )
        initialize_rank1_truth_mixture(model, sequence)
    else:
        model = build_rank2(
            horizon=horizon,
            x_ranges=x_ranges,
            y_range=y_range,
            sequence=sequence,
            q_mode="free_truth_init",
            include_ar=False,
            config=config,
            device=device,
        )
    result = train_fixed(
        model,
        data,
        epochs=config["D2"]["epochs"],
        patience=config["D2"]["patience"],
        config=config,
        device=device,
        lag_smoothness=config["common"]["free_lag_smoothness"],
    )
    validation = evaluate_diagnostic(
        model,
        data,
        "validation",
        batch_size=config["common"]["physical_chunk"],
        device=device,
        return_contributions=True,
    )
    test = evaluate_diagnostic(
        model,
        data,
        "test",
        batch_size=config["common"]["physical_chunk"],
        device=device,
    )
    surface = surface_metrics(
        model, sequence, data, rank2_truth=True, points=201
    )
    summary = {
        **base_summary(validation, test, data),
        "experiment": "D2",
        "variant": variant,
        "seed": seed,
        "horizon": horizon,
        "best_epoch": result.best_epoch,
        "mean_surface_nrmse": surface["mean_surface_nrmse"],
        "mean_surface_correlation": surface["mean_surface_correlation"],
    }
    save_training_job(
        root,
        job_config={
            "experiment": "D2",
            "variant": variant,
            "seed": seed,
            "horizon": horizon,
            "frozen_common": config["common"],
            "frozen_experiment": config["D2"],
        },
        summary=summary,
        history=result.history,
        checkpoint={"model_state": cpu_state(result.best_state)},
    )
    save_structure_files(root, model, surface)


def _ar_predictions(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    *,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    return {
        partition: np.asarray(
            evaluate_diagnostic(
                model,
                data,
                partition,
                batch_size=config["common"]["physical_chunk"],
                device=device,
            )["prediction"]
        )
        for partition in ("train", "validation", "test")
    }


def run_d3(
    variant: str,
    seed: int,
    horizon: int,
    config: dict[str, Any],
    device: torch.device,
    root: Path,
) -> None:
    sequence, base, x_ranges, y_range = prepare(
        config["D3"]["scenario"],
        seed=seed,
        track="XAR",
        horizon=horizon,
        config=config,
    )
    ar_data = PreparedDirectForecastData.from_sequence(
        sequence.x,
        sequence.y_observed,
        track="AR",
        horizon=horizon,
        L_x=config["common"]["L_x"],
        L_y=config["common"]["L_y"],
        split_target_intervals=sequence.split_target_intervals,
    )
    ar_model = build_rank1(
        track="AR",
        horizon=horizon,
        x_ranges=x_ranges,
        y_range=y_range,
        config=config,
        free_q=False,
        active_only=False,
        device=device,
    )
    ar_result = train_fixed(
        ar_model,
        ar_data,
        epochs=config["D3"]["ar_epochs"],
        patience=config["D3"]["patience"],
        config=config,
        device=device,
        lag_smoothness=0.0,
    )
    predictions = _ar_predictions(
        ar_model, ar_data, config=config, device=device
    )
    residual_data = PreparedExternalResidualData.from_ar_predictions(
        base, predictions
    )
    if variant == "residual_rank1_free_q":
        residual_model = build_rank1(
            track="X",
            horizon=horizon,
            x_ranges=x_ranges,
            y_range=y_range,
            config=config,
            free_q=True,
            active_only=True,
            device=device,
        )
        initialize_rank1_truth_mixture(residual_model, sequence)
    else:
        residual_model = build_rank2(
            horizon=horizon,
            x_ranges=x_ranges,
            y_range=y_range,
            sequence=sequence,
            q_mode="free_truth_init",
            include_ar=False,
            config=config,
            device=device,
        )
    residual_result = train_fixed(
        residual_model,
        residual_data,
        epochs=config["D3"]["residual_model_epochs"],
        patience=config["D3"]["patience"],
        config=config,
        device=device,
        lag_smoothness=config["common"]["free_lag_smoothness"],
    )
    residual_validation = evaluate_diagnostic(
        residual_model,
        residual_data,
        "validation",
        batch_size=config["common"]["physical_chunk"],
        device=device,
    )
    residual_test = evaluate_diagnostic(
        residual_model,
        residual_data,
        "test",
        batch_size=config["common"]["physical_chunk"],
        device=device,
    )
    ar_validation = evaluate_diagnostic(
        ar_model,
        ar_data,
        "validation",
        batch_size=config["common"]["physical_chunk"],
        device=device,
    )
    ar_test = evaluate_diagnostic(
        ar_model,
        ar_data,
        "test",
        batch_size=config["common"]["physical_chunk"],
        device=device,
    )
    joint_validation_prediction = (
        ar_validation["prediction"] + residual_validation["prediction"]
    )
    joint_test_prediction = ar_test["prediction"] + residual_test["prediction"]

    def rmse(target, prediction) -> float:
        return float(np.sqrt(np.mean((target - prediction) ** 2)))

    surface = surface_metrics(
        residual_model, sequence, base, rank2_truth=True, points=201
    )
    history = [
        {**row, "phase": "ar", "phase_epoch": row["epoch"]}
        for row in ar_result.history
    ] + [
        {**row, "phase": "residual", "phase_epoch": row["epoch"]}
        for row in residual_result.history
    ]
    summary = {
        "experiment": "D3",
        "variant": variant,
        "seed": seed,
        "horizon": horizon,
        "ar_best_epoch": ar_result.best_epoch,
        "residual_best_epoch": residual_result.best_epoch,
        "validation_innovation_r2": float(residual_validation["r2"]),
        "test_innovation_r2": float(residual_test["r2"]),
        "validation_ar_rmse_scaled": float(ar_validation["rmse"]),
        "test_ar_rmse_scaled": float(ar_test["rmse"]),
        "validation_ar_plus_innovation_rmse_scaled": rmse(
            ar_validation["target"], joint_validation_prediction
        ),
        "test_ar_plus_innovation_rmse_scaled": rmse(
            ar_test["target"], joint_test_prediction
        ),
        "validation_delta_x_given_ar": float(ar_validation["rmse"])
        - rmse(ar_validation["target"], joint_validation_prediction),
        "test_delta_x_given_ar": float(ar_test["rmse"])
        - rmse(ar_test["target"], joint_test_prediction),
        "mean_surface_nrmse": surface["mean_surface_nrmse"],
        "mean_surface_correlation": surface["mean_surface_correlation"],
    }
    save_training_job(
        root,
        job_config={
            "experiment": "D3",
            "variant": variant,
            "seed": seed,
            "horizon": horizon,
            "frozen_common": config["common"],
            "frozen_experiment": config["D3"],
        },
        summary=summary,
        history=history,
        checkpoint={
            "ar_model_state": cpu_state(ar_result.best_state),
            "residual_model_state": cpu_state(residual_result.best_state),
        },
    )
    save_structure_files(root, residual_model, surface)


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def _phase_history(
    result: DiagnosticTrainResult, phase: str, offset: int
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "epoch": offset + int(row["epoch"]),
            "phase": phase,
            "phase_epoch": int(row["epoch"]),
        }
        for row in result.history
    ]


def run_d4(
    variant: str,
    seed: int,
    horizon: int,
    config: dict[str, Any],
    device: torch.device,
    root: Path,
) -> None:
    sequence, data, x_ranges, y_range = prepare(
        config["D4"]["scenario"],
        seed=seed,
        track="XAR",
        horizon=horizon,
        config=config,
    )
    d4 = config["D4"]
    if variant == "simultaneous":
        model = build_rank1(
            track="XAR",
            horizon=horizon,
            x_ranges=x_ranges,
            y_range=y_range,
            config=config,
            free_q=False,
            active_only=False,
            device=device,
        )
        result = train_fixed(
            model,
            data,
            epochs=d4["simultaneous_epochs"],
            patience=d4["patience"],
            config=config,
            device=device,
            lag_smoothness=0.0,
        )
        history = _phase_history(result, "simultaneous", 0)
    elif variant == "x_first":
        x_data = PreparedDirectForecastData.from_sequence(
            sequence.x,
            sequence.y_observed,
            track="X",
            horizon=horizon,
            L_x=config["common"]["L_x"],
            L_y=config["common"]["L_y"],
            split_target_intervals=sequence.split_target_intervals,
        )
        x_model = build_rank1(
            track="X",
            horizon=horizon,
            x_ranges=x_ranges,
            y_range=y_range,
            config=config,
            free_q=False,
            active_only=False,
            device=device,
        )
        first = train_fixed(
            x_model,
            x_data,
            epochs=d4["x_first"]["x_pretrain_epochs"],
            patience=d4["patience"],
            config=config,
            device=device,
            lag_smoothness=0.0,
        )
        model = build_rank1(
            track="XAR",
            horizon=horizon,
            x_ranges=x_ranges,
            y_range=y_range,
            config=config,
            free_q=False,
            active_only=False,
            device=device,
        )
        model.external_branch.load_state_dict(
            x_model.external_branch.state_dict()
        )
        model.bias.data.copy_(x_model.bias.data)
        _set_requires_grad(model.external_branch, False)
        model.bias.requires_grad_(False)
        second = train_fixed(
            model,
            data,
            epochs=d4["x_first"]["ar_fit_epochs"],
            patience=d4["patience"],
            config=config,
            device=device,
            lag_smoothness=0.0,
        )
        _set_requires_grad(model.external_branch, True)
        model.external_branch.bias.requires_grad_(False)
        model.bias.requires_grad_(True)
        result = train_fixed(
            model,
            data,
            epochs=d4["x_first"]["joint_epochs"],
            patience=d4["patience"],
            config=config,
            device=device,
            lag_smoothness=0.0,
            joint_lr=config["common"]["joint_finetune_learning_rate"],
        )
        history = (
            _phase_history(first, "x_pretrain", 0)
            + _phase_history(
                second,
                "ar_fit",
                d4["x_first"]["x_pretrain_epochs"],
            )
            + _phase_history(
                result,
                "joint",
                d4["x_first"]["x_pretrain_epochs"]
                + d4["x_first"]["ar_fit_epochs"],
            )
        )
    else:
        ar_data = PreparedDirectForecastData.from_sequence(
            sequence.x,
            sequence.y_observed,
            track="AR",
            horizon=horizon,
            L_x=config["common"]["L_x"],
            L_y=config["common"]["L_y"],
            split_target_intervals=sequence.split_target_intervals,
        )
        ar_model = build_rank1(
            track="AR",
            horizon=horizon,
            x_ranges=x_ranges,
            y_range=y_range,
            config=config,
            free_q=False,
            active_only=False,
            device=device,
        )
        first = train_fixed(
            ar_model,
            ar_data,
            epochs=d4["ar_first"]["ar_pretrain_epochs"],
            patience=d4["patience"],
            config=config,
            device=device,
            lag_smoothness=0.0,
        )
        model = build_rank1(
            track="XAR",
            horizon=horizon,
            x_ranges=x_ranges,
            y_range=y_range,
            config=config,
            free_q=False,
            active_only=False,
            device=device,
        )
        model.ar_branch.load_state_dict(ar_model.ar_branch.state_dict())
        model.bias.data.copy_(ar_model.bias.data)
        _set_requires_grad(model.ar_branch, False)
        model.bias.requires_grad_(False)
        second = train_fixed(
            model,
            data,
            epochs=d4["ar_first"]["x_fit_epochs"],
            patience=d4["patience"],
            config=config,
            device=device,
            lag_smoothness=0.0,
        )
        _set_requires_grad(model.ar_branch, True)
        model.ar_branch.bias.requires_grad_(False)
        model.bias.requires_grad_(True)
        result = train_fixed(
            model,
            data,
            epochs=d4["ar_first"]["joint_epochs"],
            patience=d4["patience"],
            config=config,
            device=device,
            lag_smoothness=0.0,
            joint_lr=config["common"]["joint_finetune_learning_rate"],
        )
        history = (
            _phase_history(first, "ar_pretrain", 0)
            + _phase_history(
                second,
                "x_fit",
                d4["ar_first"]["ar_pretrain_epochs"],
            )
            + _phase_history(
                result,
                "joint",
                d4["ar_first"]["ar_pretrain_epochs"]
                + d4["ar_first"]["x_fit_epochs"],
            )
        )

    validation = evaluate_diagnostic(
        model,
        data,
        "validation",
        batch_size=config["common"]["physical_chunk"],
        device=device,
        return_contributions=True,
    )
    test = evaluate_diagnostic(
        model,
        data,
        "test",
        batch_size=config["common"]["physical_chunk"],
        device=device,
    )
    surface = surface_metrics(
        model, sequence, data, rank2_truth=False, points=201
    )
    energy = contribution_energy_metrics(validation["contributions"])
    learned_q = model.external_branch._static_q().detach().cpu().numpy()
    lag_w1 = float(
        np.mean(
            [
                wasserstein1_discrete(
                    learned_q[variable],
                    sequence.truth["q_primary"][variable],
                )
                for variable in ACTIVE_SUPPORT
            ]
        )
    )
    summary = {
        **base_summary(validation, test, data),
        **energy,
        "experiment": "D4",
        "variant": variant,
        "seed": seed,
        "horizon": horizon,
        "best_epoch": result.best_epoch,
        "mean_active_response_nrmse": surface["mean_surface_nrmse"],
        "mean_active_lag_w1": lag_w1,
    }
    save_training_job(
        root,
        job_config={
            "experiment": "D4",
            "variant": variant,
            "seed": seed,
            "horizon": horizon,
            "frozen_common": config["common"],
            "frozen_experiment": config["D4"],
        },
        summary=summary,
        history=history,
        checkpoint={"model_state": cpu_state(result.best_state)},
    )


def _load_d4_x_first(
    seed: int,
    *,
    config: dict[str, Any],
    device: torch.device,
):
    source = RESULT_ROOT / "D4" / f"seed_{seed}" / "x_first" / "best.pt"
    if not source.is_file():
        raise FileNotFoundError(
            f"D5 requires completed D4 x_first checkpoint: {source}"
        )
    sequence, data, x_ranges, y_range = prepare(
        config["D5"]["scenario"],
        seed=seed,
        track="XAR",
        horizon=config["common"]["primary_horizon"],
        config=config,
    )
    model = build_rank1(
        track="XAR",
        horizon=config["common"]["primary_horizon"],
        x_ranges=x_ranges,
        y_range=y_range,
        config=config,
        free_q=False,
        active_only=False,
        device=device,
    )
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return sequence, data, model


def run_d5(
    variant: str,
    seed: int,
    horizon: int,
    config: dict[str, Any],
    device: torch.device,
    root: Path,
) -> None:
    del variant, horizon
    sequence, data, model = _load_d4_x_first(
        seed, config=config, device=device
    )
    common = config["common"]
    train = evaluate_diagnostic(
        model,
        data,
        "train",
        batch_size=common["physical_chunk"],
        device=device,
        return_contributions=True,
    )
    validation = evaluate_diagnostic(
        model,
        data,
        "validation",
        batch_size=common["physical_chunk"],
        device=device,
        return_contributions=True,
    )
    train_x = train["contributions"][:, :10]
    validation_x = validation["contributions"][:, :10]
    centers = train_x.mean(axis=0)
    scales = np.sqrt(np.mean((train_x - centers) ** 2, axis=0)) + 1.0e-8
    train_design = (train_x - centers) / scales
    validation_design = (validation_x - centers) / scales
    train_adjusted = train["target"] - train["contributions"][:, 10]
    solutions = solve_gate_path(
        train_design,
        train_adjusted,
        config["D5"]["lambda_ratios"],
        max_iterations=config["D5"]["fista_max_iterations"],
        tolerance=config["D5"]["fista_tolerance"],
    )
    rows: list[dict[str, Any]] = []
    truth = set(common["active_support"])
    inactive_count = common["external_variables"] - len(truth)
    for ratio, solution in zip(
        config["D5"]["lambda_ratios"], solutions, strict=True
    ):
        support = set(
            np.flatnonzero(
                np.abs(solution.gates) > config["D5"]["support_threshold"]
            ).tolist()
        )
        true_positive = len(support & truth)
        false_positive = len(support - truth)
        prediction = (
            validation["contributions"][:, 10]
            + solution.intercept
            + validation_design @ solution.gates
        )
        rows.append(
            {
                "lambda_ratio": ratio,
                "lambda_value": solution.lambda_value,
                "support": json.dumps(sorted(support)),
                "recall": true_positive / len(truth),
                "false_positive_rate": false_positive / inactive_count,
                "precision": true_positive / max(len(support), 1),
                "validation_rmse": float(
                    np.sqrt(np.mean((validation["target"] - prediction) ** 2))
                ),
                "gate_values": json.dumps(solution.gates.tolist()),
                "intercept": solution.intercept,
                "kkt_residual": solution.kkt_residual,
                "iterations": solution.iterations,
                "converged": solution.converged,
            }
        )
    success = any(
        row["recall"] >= 0.80 and row["false_positive_rate"] <= 0.10
        for row in rows
    )
    summary = {
        "experiment": "D5",
        "variant": "gate_path",
        "seed": seed,
        "horizon": common["primary_horizon"],
        "path_contains_recoverable_support": success,
        "path_length": len(rows),
        "no_lambda_selected": True,
    }
    save_training_job(
        root,
        job_config={
            "experiment": "D5",
            "variant": "gate_path",
            "seed": seed,
            "horizon": common["primary_horizon"],
            "frozen_common": common,
            "frozen_experiment": config["D5"],
        },
        summary=summary,
        history=[
            {
                "path_index": index,
                "lambda_ratio": row["lambda_ratio"],
                "validation_rmse": row["validation_rmse"],
            }
            for index, row in enumerate(rows)
        ],
        checkpoint={
            "source_d4_x_first_model_state": cpu_state(model.state_dict()),
            "gate_path": [
                {
                    "lambda_value": solution.lambda_value,
                    "intercept": solution.intercept,
                    "gates": torch.as_tensor(solution.gates),
                }
                for solution in solutions
            ],
        },
    )
    atomic_csv(root / "gate_path.csv", rows)


def _diagnostic_contributions(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    partition: str,
    *,
    count: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    batch = next(
        data.iter_contiguous_batches(
            partition, batch_size=count, device=device
        )
    )
    with torch.no_grad():
        _, auxiliary = model.forward_contiguous(
            batch.get("x_sequence"),
            batch.get("y_sequence"),
            return_aux=True,
        )
    return (
        auxiliary["component_contribution"].detach().cpu().numpy(),
        batch["target"].detach().cpu().numpy(),
    )


def _row_gradient_norm(
    parameter: torch.nn.Parameter | None, variable: int
) -> float:
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(parameter.grad[variable].detach().norm().cpu())


def _d6_snapshot(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    *,
    epoch: int,
    phase: str,
    train_mse: float,
    validation_rmse: float,
    shrink: torch.Tensor | None,
    penalty_scale: float,
    support: list[int],
    sample_count: int,
    device: torch.device,
) -> dict[str, Any]:
    external = model.external_branch
    ar = model.ar_branch
    ar_kan_grad = gradient_norm(ar.response_branches.parameters())
    ar_gamma_grad = gradient_norm(ar.delay_prior.parameters())
    ar_total_grad = float(np.hypot(ar_kan_grad, ar_gamma_grad))
    train_contributions, train_target = _diagnostic_contributions(
        model, data, "train", count=sample_count, device=device
    )
    validation_contributions, validation_target = _diagnostic_contributions(
        model, data, "validation", count=sample_count, device=device
    )
    target_std = float(train_target.std())
    validation_target_std = float(validation_target.std())
    row: dict[str, Any] = {
        "epoch": epoch,
        "phase": phase,
        "train_mse": train_mse,
        "validation_rmse": validation_rmse,
        "ar_kan_grad_norm": ar_kan_grad,
        "ar_gamma_grad_norm": ar_gamma_grad,
        "ar_total_grad_norm": ar_total_grad,
        "penalty_scale": penalty_scale,
        "support": support,
        "x_total_contribution_rms_over_target_std": float(
            np.sqrt(np.mean(train_contributions[:, :10].sum(axis=1) ** 2))
            / max(target_std, 1.0e-12)
        ),
        "ar_contribution_rms_over_target_std": float(
            np.sqrt(np.mean(train_contributions[:, 10] ** 2))
            / max(target_std, 1.0e-12)
        ),
        "validation_x_total_contribution_rms_over_target_std": float(
            np.sqrt(
                np.mean(validation_contributions[:, :10].sum(axis=1) ** 2)
            )
            / max(validation_target_std, 1.0e-12)
        ),
        "validation_ar_contribution_rms_over_target_std": float(
            np.sqrt(np.mean(validation_contributions[:, 10] ** 2))
            / max(validation_target_std, 1.0e-12)
        ),
    }
    x_gradient_squares = 0.0
    q = external._static_q().detach()
    for variable in range(10):
        branch = external.response_branches.branches[variable]
        kan_grad = gradient_norm(branch.parameters())
        gamma_grad = float(
            np.hypot(
                _row_gradient_norm(external.delay_prior.raw_mean, variable),
                _row_gradient_norm(external.delay_prior.raw_std, variable),
            )
        )
        total_grad = float(np.hypot(kan_grad, gamma_grad))
        x_gradient_squares += total_grad**2
        contribution_ratio = float(
            np.sqrt(np.mean(train_contributions[:, variable] ** 2))
            / max(target_std, 1.0e-12)
        )
        validation_contribution_ratio = float(
            np.sqrt(
                np.mean(validation_contributions[:, variable] ** 2)
            )
            / max(validation_target_std, 1.0e-12)
        )
        row.update(
            {
                f"x_{variable}_kan_grad_norm": kan_grad,
                f"x_{variable}_gamma_grad_norm": gamma_grad,
                f"x_{variable}_total_grad_norm": total_grad,
                f"x_{variable}_gradient_ratio": total_grad
                / (ar_total_grad + 1.0e-12),
                f"x_{variable}_parameter_norm": parameter_norm(
                    branch.parameters()
                ),
                f"x_{variable}_contribution_signal_ratio": contribution_ratio,
                f"x_{variable}_validation_contribution_signal_ratio": (
                    validation_contribution_ratio
                ),
                f"x_{variable}_shrink": (
                    float(shrink[variable].detach().cpu())
                    if shrink is not None
                    else 1.0
                ),
                **{
                    f"x_{variable}_{key}": value
                    for key, value in kernel_statistics(q[variable]).items()
                },
            }
        )
    x_total_grad = float(np.sqrt(x_gradient_squares))
    row["x_total_grad_norm"] = x_total_grad
    row["x_over_ar_gradient_ratio"] = x_total_grad / (
        ar_total_grad + 1.0e-12
    )
    return row


def run_d6(
    variant: str,
    seed: int,
    horizon: int,
    config: dict[str, Any],
    device: torch.device,
    root: Path,
) -> None:
    del variant
    sequence, data, x_ranges, y_range = prepare(
        config["D6"]["scenario"],
        seed=seed,
        track="XAR",
        horizon=horizon,
        config=config,
    )
    model = build_rank1(
        track="XAR",
        horizon=horizon,
        x_ranges=x_ranges,
        y_range=y_range,
        config=config,
        free_q=False,
        active_only=False,
        device=device,
    )
    d6 = config["D6"]
    common = config["common"]
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=common["response_learning_rate"],
    )
    timeline: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_state = None
    best_epoch = 0

    def full_gradient_step() -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = data.target_count("train")
        squared = 0.0
        for batch in data.iter_contiguous_batches(
            "train",
            batch_size=common["physical_chunk"],
            device=device,
        ):
            prediction = model.forward_contiguous(
                batch["x_sequence"],
                batch["y_sequence"],
                return_aux=False,
            ).squeeze(-1)
            error = (prediction - batch["target"]).square().sum()
            (error / total).backward()
            squared += float(error.detach().cpu())
        return squared / total

    for epoch in range(1, d6["warmup_epochs"] + 1):
        train_mse = full_gradient_step()
        record = epoch % d6["log_interval"] == 0
        validation_rmse = float("nan")
        if record:
            validation_rmse = float(
                evaluate_diagnostic(
                    model,
                    data,
                    "validation",
                    batch_size=common["physical_chunk"],
                    device=device,
                )["rmse"]
            )
            if validation_rmse < best_validation:
                best_validation = validation_rmse
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            timeline.append(
                _d6_snapshot(
                    model,
                    data,
                    epoch=epoch,
                    phase="warmup",
                    train_mse=train_mse,
                    validation_rmse=validation_rmse,
                    shrink=None,
                    penalty_scale=0.0,
                    support=list(range(10)),
                    sample_count=d6["diagnostic_sample_count"],
                    device=device,
                )
            )
        optimizer.step()
    if best_state is None:
        raise RuntimeError("D6 warmup produced no validation checkpoint.")

    model.load_state_dict(best_state)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=common["response_learning_rate"],
    )
    norms = model.external_branch.response_branches.compute_branch_norms()
    lambda_group = (
        d6["pruning_scale"]
        * float(norms.median().detach().cpu())
        / common["response_learning_rate"]
    )
    best_pruning_validation = float("inf")
    best_pruning_state = copy.deepcopy(best_state)
    best_pruning_epoch = 0
    for epoch in range(1, d6["pruning_epochs"] + 1):
        train_mse = full_gradient_step()
        record = epoch % d6["log_interval"] == 0
        penalty_scale = min(1.0, epoch / d6["ramp_epochs"])
        snapshot = None
        if record:
            snapshot = _d6_snapshot(
                model,
                data,
                epoch=d6["warmup_epochs"] + epoch,
                phase="pruning",
                train_mse=train_mse,
                validation_rmse=float("nan"),
                shrink=None,
                penalty_scale=penalty_scale,
                support=torch.where(
                    (
                        model.external_branch.active_mask
                        & model.external_branch.selection_mask
                    )
                    .detach()
                    .cpu()
                )[0].tolist(),
                sample_count=d6["diagnostic_sample_count"],
                device=device,
            )
        optimizer.step()
        pre_norms, shrink = apply_group_proximal_step(
            model.external_branch.response_branches,
            optimizer.param_groups[0]["lr"],
            lambda_group * penalty_scale,
        )
        post_norms = pre_norms * shrink
        support = torch.where(
            (
                (post_norms > 1.0e-8)
                & model.external_branch.active_mask
                & model.external_branch.selection_mask
            )
            .detach()
            .cpu()
        )[0].tolist()
        for variable in range(10):
            if variable not in set(support):
                model.external_branch.prune_variable(variable, optimizer)
        if record:
            validation_rmse = float(
                evaluate_diagnostic(
                    model,
                    data,
                    "validation",
                    batch_size=common["physical_chunk"],
                    device=device,
                )["rmse"]
            )
            if validation_rmse < best_pruning_validation:
                best_pruning_validation = validation_rmse
                best_pruning_state = copy.deepcopy(model.state_dict())
                best_pruning_epoch = epoch
            assert snapshot is not None
            snapshot["validation_rmse"] = validation_rmse
            snapshot["support"] = support
            for variable in range(10):
                snapshot[f"x_{variable}_shrink"] = float(
                    shrink[variable].detach().cpu()
                )
            timeline.append(snapshot)

    starved = detect_starvation(
        timeline,
        active_support=tuple(common["active_support"]),
        **d6["starvation"],
    )
    collapse = proximal_collapse(timeline)
    summary = {
        "experiment": "D6",
        "variant": "gradient_timeline",
        "seed": seed,
        "horizon": horizon,
        "warmup_best_epoch": best_epoch,
        "pruning_best_epoch": best_pruning_epoch,
        "starved_true_variables": starved,
        "starved_true_variable_count": len(starved),
        "proximal_collapse": collapse,
    }
    csv_rows = [
        {
            **row,
            "support": json.dumps(row["support"]),
        }
        for row in timeline
    ]
    save_training_job(
        root,
        job_config={
            "experiment": "D6",
            "variant": "gradient_timeline",
            "seed": seed,
            "horizon": horizon,
            "frozen_common": common,
            "frozen_experiment": d6,
        },
        summary=summary,
        history=[
            {
                "epoch": row["epoch"],
                "phase": row["phase"],
                "train_mse": row["train_mse"],
                "validation_rmse": row["validation_rmse"],
            }
            for row in timeline
        ],
        checkpoint={"model_state": cpu_state(best_pruning_state)},
    )
    atomic_csv(root / "gradient_timeline.csv", csv_rows)


RUNNERS = {
    "D1": run_d1,
    "D2": run_d2,
    "D3": run_d3,
    "D4": run_d4,
    "D5": run_d5,
    "D6": run_d6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(RUNNERS), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--horizon", type=int, choices=(1, 5, 10), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_diagnostic_config()
    validate_job(args.experiment, args.variant, args.seed, args.horizon)
    configure_threads(config)
    seed_everything(
        args.seed,
        deterministic=config["common"]["deterministic_algorithms"],
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    root = job_root(
        args.experiment, args.variant, args.seed, args.horizon
    )
    if (root / "summary.json").is_file() and not args.force:
        print(f"SKIP_COMPLETED {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    RUNNERS[args.experiment](
        args.variant,
        args.seed,
        args.horizon,
        config,
        device,
        root,
    )
    print(f"COMPLETED {root}")


if __name__ == "__main__":
    main()
