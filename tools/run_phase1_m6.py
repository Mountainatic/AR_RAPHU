#!/usr/bin/env python3
"""Run v2 M6 free-smooth-dynamic-kernel rank-1 refits for Phase 1 E3."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from STAGE1_DUAL_SOLVER_V20_bundle.stage1.dual_metrics import prediction_metrics
from ar_raphu.m6 import initialize_m6_from_m5_gamma
from ar_raphu.model_selection import validation_one_se_select
from ar_raphu.phase1_evidence import (
    partition_predictions_and_contributions,
    response_grid_arrays,
    static_lag_kernels,
)
from ar_raphu.protocol_config import load_protocol_config
from ar_raphu.runtime_environment import require_runtime_environment
from ar_raphu.training import (
    free_lag_logit_roughness,
    refit_m6_free_lag,
    seed_everything,
)
from tools.run_phase1_scheme_a import (
    atomic_csv,
    atomic_json,
    build_model,
    configure_runtime_threads,
    cpu_state,
    prepare,
    runtime_payload,
)


EXPERIMENT_ROOT = PROJECT_ROOT / "results" / "phase1" / "E3_AR-S2_G2"
M5_ROOT = EXPERIMENT_ROOT / "Track-XAR"
ROOT = EXPERIMENT_ROOT / "M6"


def config_id(smoothness: float) -> str:
    return f"lambda={smoothness:.8g}"


def config_dir(seed: int, smoothness: float) -> Path:
    return ROOT / f"seed_{seed}" / f"lambda_{smoothness:.8g}"


def declared_weights(config: dict[str, Any]) -> list[float]:
    return [
        float(value)
        for value in config["phase1_model_selection"]["M6"][
            "second_difference_smoothness_weights"
        ]
    ]


def load_m5(
    seed: int, device: torch.device
) -> tuple[Any, list[tuple[float, float]], tuple[float, float], Any, list[int]]:
    selection_path = M5_ROOT / "validation_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("M5 validation selection must be frozen before M6.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_used") is not False:
        raise RuntimeError("M5 selection metadata does not prove validation-only use.")
    selected_scale = float(str(selection["config_id"]).split("=")[1])
    checkpoint_path = (
        M5_ROOT
        / f"seed_{seed}"
        / "forks"
        / f"s0_{selected_scale:.6f}"
        / "best.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    _, data, x_ranges, y_range = prepare(
        "AR-S2", "XAR", seed, smoke=False
    )
    model = build_model("XAR", x_ranges, y_range, device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    support = sorted(int(value) for value in checkpoint["terminal_support"])
    if not support:
        raise RuntimeError("M6 requires a non-empty frozen M5 support.")
    return data, x_ranges, y_range, model, support


def fit_job(args: argparse.Namespace, device: torch.device) -> None:
    config = load_protocol_config(require_phase1_frozen=True)
    if args.smoothness not in declared_weights(config):
        raise ValueError("M6 smoothing weight is not predeclared.")
    output_root = config_dir(args.seed, args.smoothness)
    summary_path = output_root / "summary.json"
    if summary_path.is_file() and not args.force:
        print(summary_path)
        return

    seed_everything(
        args.seed, deterministic=bool(config["training"]["deterministic_algorithms"])
    )
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data, x_ranges, y_range, m5_model, support = load_m5(args.seed, device)
    m6_model = build_model(
        "XAR",
        x_ranges,
        y_range,
        device,
        external_delay_mode="free_static_logits",
    )
    initialization = initialize_m6_from_m5_gamma(
        m5_model, m6_model, support
    )
    batch_size = int(config["training"]["batch_size"]["physical_chunk"])
    with torch.no_grad():
        m5_validation = partition_predictions_and_contributions(
            m5_model,
            data,
            "validation",
            batch_size=batch_size,
            device=device,
        )[0]
        m6_validation = partition_predictions_and_contributions(
            m6_model,
            data,
            "validation",
            batch_size=batch_size,
            device=device,
        )[0]
    prediction_error = float(np.max(np.abs(m5_validation - m6_validation)))
    if prediction_error > 2.0e-5:
        raise RuntimeError(
            f"M5→M6 initialization changed predictions by {prediction_error}."
        )

    reference = config["training"]["source_backed_v20_reference"]
    result = refit_m6_free_lag(
        m6_model,
        data,
        support,
        smoothness_weight=args.smoothness,
        epochs=int(reference["refit_epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        patience=int(reference["refit_patience"]),
        batch_size=batch_size,
        device=device,
        validation_interval=5,
    )
    m6_model.load_state_dict(result.best_state)
    best_roughness = float(free_lag_logit_roughness(m6_model).detach().cpu())
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / "best.pt.tmp"
    torch.save(
        {
            "state_dict": cpu_state(result.best_state),
            "terminal_state": cpu_state(result.terminal_state),
            "terminal_support": result.terminal_support,
            "smoothness_weight": args.smoothness,
            "initialization": "log_of_selected_M5_Gamma_kernel",
        },
        temporary,
    )
    os.replace(temporary, output_root / "best.pt")
    atomic_csv(output_root / "training_log.csv", result.history)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    atomic_json(
        summary_path,
        {
            "status": "COMPLETED",
            "model": "M6",
            "scenario": "AR-S2",
            "track": "XAR",
            "seed": args.seed,
            "config_id": config_id(args.smoothness),
            "second_difference_smoothness_weight": args.smoothness,
            "validation_rmse_scaled": result.best_validation_rmse,
            "best_epoch": result.best_epoch,
            "frozen_M5_support": support,
            "terminal_support": result.terminal_support,
            "best_lag_logit_roughness": best_roughness,
            "initialization_audit": {
                **initialization,
                "prediction_max_abs_error_scaled": prediction_error,
            },
            "objective": "mean_train_MSE_plus_lambda_times_sum_squared_lag_logit_second_differences",
            "joint_refit": "fixed_support_external_responses_external_lag_logits_AR_branch_and_bias",
            "dtype": "float32",
            "test_accessed": False,
            "runtime": runtime_payload(
                started=started, device=device, peak_memory=peak_memory
            ),
        },
    )
    print(summary_path)


def select_configuration() -> dict[str, Any]:
    config = load_protocol_config(require_phase1_frozen=True)
    weights = declared_weights(config)
    declared_ids = [config_id(weight) for weight in weights]
    lookup = {config_id(weight): weight for weight in weights}
    rows: list[dict[str, Any]] = []
    for seed in config["training"]["seeds"]["screening"]:
        for weight in weights:
            path = config_dir(seed, weight) / "summary.json"
            if not path.is_file():
                raise RuntimeError(f"Missing M6 summary: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("test_accessed") is not False:
                raise RuntimeError("M6 candidate accessed test before selection.")
            rows.append(
                {
                    "config_id": summary["config_id"],
                    "unit_id": f"seed_{seed}",
                    "validation_loss": summary["validation_rmse_scaled"],
                }
            )
    selected = validation_one_se_select(
        rows,
        declared_config_order=declared_ids,
        complexity_key=lambda item: (-lookup[item],),
    )
    selected["selection_unit"] = "independent_seed_replicate"
    selected["complexity_order"] = [
        "smoothness_weight_descending",
        "predeclared_configuration_order",
    ]
    selected["hyperparameters_selected_using"] = "validation_prediction_loss_only"
    atomic_json(ROOT / "validation_selection.json", selected)
    return selected


def aggregate(device: torch.device) -> None:
    config = load_protocol_config(require_phase1_frozen=True)
    selection_path = ROOT / "validation_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("M6 validation selection has not been frozen.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_used") is not False:
        raise RuntimeError("M6 selection is not validation-only.")
    selected_id = str(selection["selected_config_id"])
    weights = {config_id(value): value for value in declared_weights(config)}
    smoothness = weights[selected_id]
    batch_size = int(config["training"]["batch_size"]["physical_chunk"])

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    contribution_frames: list[pd.DataFrame] = []
    external_kernels: list[np.ndarray] = []
    ar_kernels: list[np.ndarray] = []
    response_payload: dict[str, np.ndarray] = {}
    for seed in config["training"]["seeds"]["screening"]:
        data, x_ranges, y_range, _, support = load_m5(seed, device)
        checkpoint = torch.load(
            config_dir(seed, smoothness) / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model = build_model(
            "XAR",
            x_ranges,
            y_range,
            device,
            external_delay_mode="free_static_logits",
        )
        model.load_state_dict(checkpoint["state_dict"])
        predicted_scaled, observed_scaled, indices, components_scaled = (
            partition_predictions_and_contributions(
                model,
                data,
                "test",
                batch_size=batch_size,
                device=device,
            )
        )
        predicted = data.scaler.inverse_y(predicted_scaled)
        observed = data.scaler.inverse_y(observed_scaled)
        metrics = prediction_metrics(observed, predicted)
        metrics.update(
            {
                "seed": seed,
                "selected_config_id": selected_id,
                "terminal_support": support,
            }
        )
        metric_rows.append(metrics)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "partition": "test",
                    "target_index": indices,
                    "observed": observed,
                    "predicted": predicted,
                    "residual": predicted - observed,
                }
            )
        )
        components = components_scaled * data.scaler.y_scale
        contribution_payload: dict[str, Any] = {
            "seed": np.full(len(indices), seed),
            "partition": np.full(len(indices), "test"),
            "target_index": indices,
            "intercept": np.full(
                len(indices),
                float(model.bias.detach().cpu()) * data.scaler.y_scale
                + data.scaler.y_mean,
            ),
        }
        for variable in range(model.external_channels):
            contribution_payload[f"x_{variable}"] = components[:, variable]
        contribution_payload["ar_y"] = components[:, -1]
        contribution_frames.append(pd.DataFrame(contribution_payload))

        external_q, ar_q = static_lag_kernels(model)
        external_kernels.append(external_q)
        ar_kernels.append(ar_q)
        arrays = response_grid_arrays(model, x_ranges, y_range, points=401)
        for name, values in arrays.items():
            response_payload[f"seed{seed}_{name}"] = values

    means = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in ("rmse", "mae", "r2")
    }
    standard_deviations = {
        key: float(np.std([row[key] for row in metric_rows], ddof=1))
        for key in ("rmse", "mae", "r2")
    }
    atomic_json(
        ROOT / "test_metrics.json",
        {
            "status": "COMPLETED",
            "model": "M6",
            "scenario": "AR-S2",
            "selected_config_id": selected_id,
            "per_seed": metric_rows,
            "mean": means,
            "sample_standard_deviation": standard_deviations,
            "hyperparameters_frozen_before_test": True,
            "rank_decisions_used_for_hyperparameter_selection": False,
        },
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        ROOT / "predictions.parquet", index=False
    )
    pd.concat(contribution_frames, ignore_index=True).to_parquet(
        ROOT / "component_contributions.parquet", index=False
    )
    np.savez_compressed(
        ROOT / "lag_kernels.npz",
        external=np.stack(external_kernels),
        ar=np.stack(ar_kernels),
    )
    np.savez_compressed(ROOT / "response_grid.npz", **response_payload)
    print(ROOT / "test_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fit", "select", "aggregate"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoothness", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    require_runtime_environment()
    configure_runtime_threads()
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.mode == "fit":
        if args.seed is None or args.smoothness is None:
            raise ValueError("fit requires --seed and --smoothness.")
        fit_job(args, device)
    elif args.mode == "select":
        selected = select_configuration()
        print(ROOT / "validation_selection.json", selected)
    else:
        aggregate(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
