#!/usr/bin/env python3
"""Resumable Phase-1 Scheme-A jobs with validation-only configuration selection."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
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

from STAGE1_DUAL_SOLVER_V20_bundle.stage1.dual_metrics import (  # noqa: E402
    prediction_metrics,
)
from STAGE1_DUAL_SOLVER_V20_bundle.stage1.experiment_utils import (  # noqa: E402
    cross_seed_one_se_select,
)
from ar_raphu.model import ARRAPHURank1  # noqa: E402
from ar_raphu.phase1_evidence import (  # noqa: E402
    partition_predictions_and_contributions,
    response_grid_arrays,
    static_lag_kernels,
    support_frequency,
)
from ar_raphu.protocol_config import load_protocol_config  # noqa: E402
from ar_raphu.runtime_environment import require_runtime_environment  # noqa: E402
from ar_raphu.sequence_data import PreparedDirectForecastData  # noqa: E402
from ar_raphu.synthetic import SCENARIOS, generate_synthetic_sequence  # noqa: E402
from ar_raphu.training import (  # noqa: E402
    evaluate_rmse,
    external_support,
    prune_external_path,
    refit_fixed_external_support,
    seed_everything,
    train_dense_warmup,
)


RESULT_ROOT = PROJECT_ROOT / "results" / "phase1"


def configure_runtime_threads() -> None:
    intraop = int(os.environ.get("AR_RAPHU_TORCH_THREADS", "1"))
    interop = int(os.environ.get("AR_RAPHU_TORCH_INTEROP_THREADS", "1"))
    if intraop < 1 or interop < 1:
        raise ValueError("PyTorch thread limits must be positive.")
    torch.set_num_threads(intraop)
    torch.set_num_interop_threads(interop)


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
        raise ValueError("Cannot write an empty training log.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_name(scenario: str) -> str:
    return {
        "AR-S0": "E1",
        "AR-S1": "E2",
        "AR-S2": "E3",
        "AR-S3": "E4",
    }.get(scenario, f"SUPPORT_{scenario}")


def track_dir(scenario: str, track: str, smoke: bool) -> Path:
    suffix = "_SMOKE" if smoke else ""
    return (
        RESULT_ROOT
        / f"{experiment_name(scenario)}_{scenario}_G2{suffix}"
        / f"Track-{track}"
    )


def prepare(
    scenario: str, track: str, seed: int, *, smoke: bool
) -> tuple[Any, PreparedDirectForecastData, list, tuple]:
    config = load_protocol_config(require_phase1_frozen=True)
    count = 512 if smoke else config["phase1_synthetic"]["sample_sizes"]["core"]
    sequence = generate_synthetic_sequence(
        scenario,
        seed=seed,
        n_samples=count,
        external_variables=10,
    )
    data = PreparedDirectForecastData.from_sequence(
        sequence.x,
        sequence.y_observed,
        track=track,
        horizon=1,
        L_x=64,
        L_y=32,
        split_target_intervals=sequence.split_target_intervals,
    )
    x_ranges, y_range = data.scaler.input_grid_ranges(
        sequence.x, sequence.y_observed
    )
    return sequence, data, x_ranges, y_range


def build_model(
    track: str,
    x_ranges: list[tuple[float, float]],
    y_range: tuple[float, float],
    device: torch.device,
) -> ARRAPHURank1:
    return ARRAPHURank1(
        track=track,
        horizon=1,
        external_channels=10,
        inactive_external_channels=(),
        L_x=64 if track in {"X", "XAR"} else None,
        L_y=32 if track in {"AR", "XAR"} else None,
        input_grid_ranges_x=x_ranges if track in {"X", "XAR"} else None,
        input_grid_range_y=y_range if track in {"AR", "XAR"} else None,
        hidden_kan=8,
        grid_size=7,
        response_execution_mode="vectorized",
    ).to(device)


def budgets(config: dict, smoke: bool) -> dict[str, int]:
    if smoke:
        return {
            "warmup_epochs": 10,
            "warmup_patience": 10,
            "prune_epochs": 12,
            "ramp_epochs": 3,
            "full_penalty_min_epochs": 3,
            "stable_epochs": 3,
            "refit_epochs": 10,
            "refit_patience": 10,
            "validation_interval": 2,
        }
    reference = config["training"]["source_backed_v20_reference"]
    return {
        "warmup_epochs": reference["warmup_epochs"],
        "warmup_patience": reference["warmup_patience"],
        "prune_epochs": reference["prune_epochs"],
        "ramp_epochs": reference["ramp_epochs"],
        "full_penalty_min_epochs": 200,
        "stable_epochs": 80,
        "refit_epochs": reference["refit_epochs"],
        "refit_patience": reference["refit_patience"],
        "validation_interval": 5,
    }


def runtime_payload(
    *,
    started: float,
    device: torch.device,
    peak_memory: int,
) -> dict[str, Any]:
    return {
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_memory_bytes": peak_memory,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }


def write_common_manifest(
    root: Path,
    sequence,
    *,
    track: str,
    seed: int,
    smoke: bool,
) -> None:
    config_path = PROJECT_ROOT / "configs" / "protocol_v2.yaml"
    atomic_json(
        root / "dataset_manifest.json",
        {
            "dataset": "fully_synthetic_AR_RAPHU",
            "scenario": sequence.scenario,
            "generator_version": sequence.truth["generator_version"],
            "seed": seed,
            "usable_target_count": sequence.usable_target_count,
            "sequence_rows": len(sequence.y_observed),
            "external_variables": sequence.x.shape[1],
            "target_intervals": sequence.split_target_intervals,
            "truth_access_policy": (
                "not_used_for_configuration_selection; post_selection_audit_only"
            ),
            "smoke_not_scientific_evidence": smoke,
        },
    )
    atomic_json(
        root / "seed.json",
        {"data_seed": seed, "model_seed": seed, "fork_seed_rule": "seed_and_scale"},
    )
    atomic_json(
        root / "model_semantics.json",
        {
            "track": track,
            "horizon": 1,
            "L_x": 64,
            "L_y": 32,
            "future_X": False,
            "external_channels": 10,
            "inactive_external_channels": [],
            "V20_static_Gamma": True,
            "KAN": {"hidden": 8, "grid": 7, "spline_order": 3},
        },
    )
    atomic_json(
        root / "config_reference.json",
        {
            "path": str(config_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(config_path),
        },
    )


def dense_job(args: argparse.Namespace, device: torch.device) -> None:
    config = load_protocol_config(require_phase1_frozen=True)
    cfg = budgets(config, args.smoke)
    sequence, data, x_ranges, y_range = prepare(
        args.scenario, args.track, args.seed, smoke=args.smoke
    )
    root = track_dir(args.scenario, args.track, args.smoke) / f"seed_{args.seed}"
    summary_path = root / "dense_summary.json"
    if summary_path.exists() and not args.force:
        print(summary_path)
        return
    seed_everything(args.seed, deterministic=True)
    model = build_model(args.track, x_ranges, y_range, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = train_dense_warmup(
        model,
        data,
        epochs=cfg["warmup_epochs"],
        learning_rate=config["training"]["learning_rate"],
        patience=cfg["warmup_patience"],
        batch_size=config["training"]["batch_size"]["physical_chunk"],
        device=device,
        validation_interval=cfg["validation_interval"],
    )
    model.load_state_dict(result.best_state)
    predicted_scaled, observed_scaled, target_indices = predictions(
        model,
        data,
        partition="test",
        batch_size=config["training"]["batch_size"]["physical_chunk"],
        device=device,
    )
    test_scaled_rmse = float(
        np.sqrt(np.mean(np.square(predicted_scaled - observed_scaled)))
    )
    predicted = data.scaler.inverse_y(predicted_scaled)
    observed = data.scaler.inverse_y(observed_scaled)
    physical_metrics = prediction_metrics(observed, predicted)
    checkpoint = {
        "state_dict": cpu_state(result.best_state),
        "seed": args.seed,
        "scenario": args.scenario,
        "track": args.track,
    }
    root.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, root / "best.pt")
    atomic_csv(root / "train_log.csv", result.history)
    write_common_manifest(
        root, sequence, track=args.track, seed=args.seed, smoke=args.smoke
    )
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    atomic_json(root / "runtime.json", runtime_payload(
        started=started, device=device, peak_memory=peak
    ))
    atomic_json(
        summary_path,
        {
            "status": "COMPLETED",
            "best_epoch": result.best_epoch,
            "validation_rmse_scaled": result.best_validation_rmse,
            "test_rmse_scaled": test_scaled_rmse,
            "test_metrics_physical": physical_metrics,
            "external_support": external_support(model),
            "test_opened_after_fixed_dense_configuration": True,
            "smoke_not_scientific_evidence": args.smoke,
        },
    )
    pd.DataFrame(
        {
            "seed": args.seed,
            "partition": "test",
            "target_index": target_indices,
            "observed": observed,
            "predicted": predicted,
            "residual": predicted - observed,
        }
    ).to_parquet(root / "predictions.parquet", index=False)
    print(summary_path)


def audit_dense_artifacts(args: argparse.Namespace, device: torch.device) -> None:
    """Recover missing dense prediction artifacts without refitting or selection."""

    config = load_protocol_config(require_phase1_frozen=True)
    sequence, data, x_ranges, y_range = prepare(
        args.scenario, args.track, args.seed, smoke=args.smoke
    )
    root = track_dir(args.scenario, args.track, args.smoke) / f"seed_{args.seed}"
    summary_path = root / "dense_summary.json"
    checkpoint_path = root / "best.pt"
    prediction_path = root / "predictions.parquet"
    if prediction_path.exists() and not args.force:
        print(prediction_path)
        return
    if not summary_path.exists() or not checkpoint_path.exists():
        raise RuntimeError("Dense fit artifacts are incomplete; cannot audit-replay.")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = build_model(args.track, x_ranges, y_range, device)
    model.load_state_dict(checkpoint["state_dict"])
    predicted_scaled, observed_scaled, target_indices = predictions(
        model,
        data,
        partition="test",
        batch_size=config["training"]["batch_size"]["physical_chunk"],
        device=device,
    )
    predicted = data.scaler.inverse_y(predicted_scaled)
    observed = data.scaler.inverse_y(observed_scaled)
    pd.DataFrame(
        {
            "seed": args.seed,
            "partition": "test",
            "target_index": target_indices,
            "observed": observed,
            "predicted": predicted,
            "residual": predicted - observed,
        }
    ).to_parquet(prediction_path, index=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replay_rmse = float(
        np.sqrt(np.mean(np.square(predicted_scaled - observed_scaled)))
    )
    if not np.isclose(
        replay_rmse,
        float(summary["test_rmse_scaled"]),
        rtol=2.0e-6,
        atol=2.0e-7,
    ):
        raise RuntimeError("Audit replay disagrees with the original test metric.")
    summary["test_metrics_physical"] = prediction_metrics(observed, predicted)
    summary["artifact_recovery"] = {
        "status": "COMPLETED",
        "reason": "initial_dense_runner_saved_metric_but_not_predictions",
        "checkpoint_refit": False,
        "configuration_selection_performed": False,
        "test_read_only_replay": True,
        "scaled_rmse_consistency_verified": True,
    }
    atomic_json(summary_path, summary)
    print(prediction_path)


def warmup_job(args: argparse.Namespace, device: torch.device) -> None:
    if args.track not in {"X", "XAR"}:
        raise ValueError("Sparse warmup requires Track-X or Track-XAR.")
    config = load_protocol_config(require_phase1_frozen=True)
    cfg = budgets(config, args.smoke)
    sequence, data, x_ranges, y_range = prepare(
        args.scenario, args.track, args.seed, smoke=args.smoke
    )
    root = track_dir(args.scenario, args.track, args.smoke) / f"seed_{args.seed}"
    checkpoint_path = root / "warmup.pt"
    if checkpoint_path.exists() and not args.force:
        print(checkpoint_path)
        return
    seed_everything(args.seed, deterministic=True)
    model = build_model(args.track, x_ranges, y_range, device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = train_dense_warmup(
        model,
        data,
        epochs=cfg["warmup_epochs"],
        learning_rate=config["training"]["learning_rate"],
        patience=cfg["warmup_patience"],
        batch_size=config["training"]["batch_size"]["physical_chunk"],
        device=device,
        validation_interval=cfg["validation_interval"],
    )
    model.load_state_dict(result.best_state)
    norms = (
        model.external_branch.response_branches.compute_branch_norms()
        .detach()
        .cpu()
    )
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": cpu_state(result.best_state),
            "seed": args.seed,
            "median_external_branch_norm": float(norms.median()),
            "external_branch_norms": norms,
            "best_validation_rmse": result.best_validation_rmse,
        },
        checkpoint_path,
    )
    atomic_csv(root / "warmup_train_log.csv", result.history)
    write_common_manifest(
        root, sequence, track=args.track, seed=args.seed, smoke=args.smoke
    )
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    atomic_json(
        root / "warmup_runtime.json",
        runtime_payload(started=started, device=device, peak_memory=peak),
    )
    print(checkpoint_path)


def fork_job(args: argparse.Namespace, device: torch.device) -> None:
    if args.scale is None:
        raise ValueError("--scale is required for a fork.")
    config = load_protocol_config(require_phase1_frozen=True)
    cfg = budgets(config, args.smoke)
    sequence, data, x_ranges, y_range = prepare(
        args.scenario, args.track, args.seed, smoke=args.smoke
    )
    seed_root = (
        track_dir(args.scenario, args.track, args.smoke) / f"seed_{args.seed}"
    )
    warmup = torch.load(
        seed_root / "warmup.pt", map_location=device, weights_only=False
    )
    fork_root = seed_root / "forks" / f"s0_{args.scale:.6f}"
    summary_path = fork_root / "summary.json"
    if summary_path.exists() and not args.force:
        print(summary_path)
        return
    fork_seed = args.seed * 10_000 + int(round(args.scale * 1_000_000))
    seed_everything(fork_seed, deterministic=True)
    model = build_model(args.track, x_ranges, y_range, device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prune = prune_external_path(
        model,
        data,
        warmup["state_dict"],
        requested_scale=args.scale,
        median_warmup_branch_norm=warmup["median_external_branch_norm"],
        epochs=cfg["prune_epochs"],
        learning_rate=config["training"]["learning_rate"],
        ramp_epochs=cfg["ramp_epochs"],
        full_penalty_min_epochs=cfg["full_penalty_min_epochs"],
        stable_epochs=cfg["stable_epochs"],
        batch_size=config["training"]["batch_size"]["physical_chunk"],
        device=device,
        validation_interval=cfg["validation_interval"],
    )
    refit_model = build_model(args.track, x_ranges, y_range, device)
    refit = refit_fixed_external_support(
        refit_model,
        data,
        prune.terminal_state,
        prune.terminal_support,
        epochs=cfg["refit_epochs"],
        learning_rate=config["training"]["learning_rate"],
        patience=cfg["refit_patience"],
        batch_size=config["training"]["batch_size"]["physical_chunk"],
        device=device,
        validation_interval=cfg["validation_interval"],
    )
    fork_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": cpu_state(refit.best_state),
            "terminal_support": prune.terminal_support,
            "seed": args.seed,
            "requested_scale": args.scale,
        },
        fork_root / "best.pt",
    )
    atomic_csv(fork_root / "prune_train_log.csv", prune.history)
    atomic_csv(fork_root / "refit_train_log.csv", refit.history)
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    atomic_json(
        fork_root / "runtime.json",
        runtime_payload(started=started, device=device, peak_memory=peak),
    )
    atomic_json(
        summary_path,
        {
            "status": "COMPLETED",
            "seed": args.seed,
            "requested_scale": args.scale,
            "config_id": f"s0={args.scale:.8g}",
            "terminal_support": prune.terminal_support,
            "active_count": len(prune.terminal_support),
            "stable": prune.stable,
            "support_stable_epoch": prune.support_stable_epoch,
            "prune_validation_rmse": prune.best_validation_rmse,
            "refit_val_rmse": refit.best_validation_rmse,
            "test_accessed": False,
            "smoke_not_scientific_evidence": args.smoke,
        },
    )
    print(summary_path)


@torch.no_grad()
def predictions(
    model: ARRAPHURank1,
    data: PreparedDirectForecastData,
    *,
    partition: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predicted, observed, indices = [], [], []
    for batch in data.iter_contiguous_batches(
        partition, batch_size=batch_size, device=device
    ):
        output = model.forward_contiguous(
            batch.get("x_sequence"),
            batch.get("y_sequence"),
            return_aux=False,
        ).squeeze(-1)
        predicted.append(output.cpu().numpy())
        observed.append(batch["target"].cpu().numpy())
        indices.append(batch["target_index"].cpu().numpy())
    return (
        np.concatenate(predicted),
        np.concatenate(observed),
        np.concatenate(indices),
    )


def select_sparse_configuration(args: argparse.Namespace) -> float:
    config = load_protocol_config(require_phase1_frozen=True)
    root = track_dir(args.scenario, args.track, args.smoke)
    seeds = (
        [0, 1]
        if args.smoke
        else config["training"]["seeds"]["screening"]
    )
    scales = (
        [0.003, 0.006]
        if args.smoke
        else config["training"]["source_backed_v20_reference"][
            "pruning_scales"
        ]
    )
    rows = []
    for seed in seeds:
        for scale in scales:
            path = (
                root
                / f"seed_{seed}"
                / "forks"
                / f"s0_{scale:.6f}"
                / "summary.json"
            )
            if not path.exists():
                raise RuntimeError(f"Missing fork summary: {path}")
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    selected = cross_seed_one_se_select(rows)
    selected_scale = float(selected["config_id"].split("=")[1])
    atomic_json(
        root / "validation_selection.json",
        {
            **selected,
            "selection_data": "validation_only",
            "truth_used": False,
            "test_used": False,
        },
    )
    return selected_scale


def aggregate(args: argparse.Namespace, device: torch.device) -> None:
    config = load_protocol_config(require_phase1_frozen=True)
    root = track_dir(args.scenario, args.track, args.smoke)
    seeds = (
        [0, 1]
        if args.smoke
        else config["training"]["seeds"]["screening"]
    )
    selected_scale = select_sparse_configuration(args)

    metric_rows = []
    prediction_frames = []
    contribution_frames = []
    external_kernels = []
    ar_kernels = []
    response_arrays: dict[str, np.ndarray] = {}
    supports: list[list[int]] = []
    for seed in seeds:
        sequence, data, x_ranges, y_range = prepare(
            args.scenario, args.track, seed, smoke=args.smoke
        )
        checkpoint_path = (
            root
            / f"seed_{seed}"
            / "forks"
            / f"s0_{selected_scale:.6f}"
            / "best.pt"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model = build_model(args.track, x_ranges, y_range, device)
        model.load_state_dict(checkpoint["state_dict"])
        (
            predicted_scaled,
            observed_scaled,
            target_indices,
            component_scaled,
        ) = partition_predictions_and_contributions(
            model,
            data,
            "test",
            batch_size=config["training"]["batch_size"]["physical_chunk"],
            device=device,
        )
        predicted = data.scaler.inverse_y(predicted_scaled)
        observed = data.scaler.inverse_y(observed_scaled)
        metrics = prediction_metrics(observed, predicted)
        metrics.update(
            {
                "seed": seed,
                "selected_scale": selected_scale,
                "terminal_support": checkpoint["terminal_support"],
            }
        )
        metric_rows.append(metrics)
        supports.append(list(checkpoint["terminal_support"]))
        external_q, ar_q = static_lag_kernels(model)
        if external_q is not None:
            external_kernels.append(external_q)
        if ar_q is not None:
            ar_kernels.append(ar_q)
        arrays = response_grid_arrays(
            model, x_ranges, y_range, points=401
        )
        for name, values in arrays.items():
            if name.startswith("x_grid_v"):
                variable = int(name.split("v")[-1])
                response_arrays[f"seed{seed}_{name}_scaled"] = values
                response_arrays[f"seed{seed}_{name}_original"] = (
                    values * data.scaler.x_scale[variable]
                    + data.scaler.x_mean[variable]
                )
            elif name == "y_grid":
                response_arrays[f"seed{seed}_y_grid_scaled"] = values
                response_arrays[f"seed{seed}_y_grid_original"] = (
                    values * data.scaler.y_scale + data.scaler.y_mean
                )
            else:
                response_arrays[f"seed{seed}_{name}_physical"] = (
                    values * data.scaler.y_scale
                )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "partition": "test",
                    "target_index": target_indices,
                    "observed": observed,
                    "predicted": predicted,
                    "residual": predicted - observed,
                }
            )
        )
        component_physical = component_scaled * data.scaler.y_scale
        contribution_payload: dict[str, Any] = {
            "seed": np.full(len(target_indices), seed),
            "partition": np.full(len(target_indices), "test"),
            "target_index": target_indices,
            "intercept": np.full(
                len(target_indices),
                float(model.bias.detach().cpu()) * data.scaler.y_scale
                + data.scaler.y_mean,
            ),
        }
        for variable in range(model.external_channels):
            contribution_payload[f"x_{variable}"] = component_physical[
                :, variable
            ]
        contribution_payload["ar_y"] = component_physical[:, -1]
        contribution_frames.append(pd.DataFrame(contribution_payload))
    atomic_json(
        root / "test_metrics.json",
        {
            "status": "COMPLETED",
            "selected_scale": selected_scale,
            "per_seed": metric_rows,
            "truth_audit_after_selection": {
                "true_support_by_scenario": (
                    [] if args.scenario == "AR-S0" else [0, 1, 2]
                )
            },
        },
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        root / "predictions.parquet", index=False
    )
    pd.concat(contribution_frames, ignore_index=True).to_parquet(
        root / "contributions.parquet", index=False
    )
    if external_kernels:
        np.save(root / "lag_kernel.npy", np.stack(external_kernels))
    if ar_kernels:
        np.save(root / "ar_lag_kernel.npy", np.stack(ar_kernels))
    np.savez_compressed(root / "response_grid.npz", **response_arrays)
    np.savez_compressed(
        root / "surface_grid.npz",
        status=np.asarray(["NOT_APPLICABLE"]),
        reason=np.asarray(["Scheme_A_rank1_has_no_B_surface"]),
    )
    atomic_json(
        root / "support.json",
        {
            "status": "COMPLETED",
            "selected_scale": selected_scale,
            "per_seed": [
                {"seed": seed, "support": support}
                for seed, support in zip(seeds, supports, strict=True)
            ],
            "frequency": support_frequency(
                supports, variables=model.external_channels
            ),
        },
    )
    atomic_json(
        root / "singular_values.json",
        {
            "status": "NOT_APPLICABLE",
            "reason": "Scheme A is the anchored rank-1 model; rank SVD is M8.",
        },
    )
    atomic_csv(
        root / "conditional_delay.csv",
        [
            {
                "status": "NOT_APPLICABLE",
                "reason": "Static rank-1 kernels have no amplitude-conditioned delay.",
            }
        ],
    )
    print(root / "test_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "dense",
            "audit-dense",
            "warmup",
            "fork",
            "select",
            "aggregate",
        ],
    )
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--track", choices=["X", "AR", "XAR"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    require_runtime_environment()
    configure_runtime_threads()
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.mode == "dense":
        dense_job(args, device)
    elif args.mode == "audit-dense":
        audit_dense_artifacts(args, device)
    elif args.mode == "warmup":
        warmup_job(args, device)
    elif args.mode == "fork":
        fork_job(args, device)
    elif args.mode == "select":
        selected = select_sparse_configuration(args)
        print(
            track_dir(args.scenario, args.track, args.smoke)
            / "validation_selection.json",
            selected,
        )
    else:
        aggregate(args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
