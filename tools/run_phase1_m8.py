#!/usr/bin/env python3
"""Run v2 M8 A-anchored orthogonal external residual surfaces."""

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
from STAGE1_DUAL_SOLVER_V20_bundle.stage1.variational_v20 import (
    SplineBasisBankV20,
)
from ar_raphu.model_selection import validation_one_se_select
from ar_raphu.orthogonal_surface import (
    orthogonal_lag_bases,
    solve_surface_ridge,
    surface_design,
    surface_penalty,
)
from ar_raphu.protocol_config import load_protocol_config
from ar_raphu.rank_audit import (
    discrete_lag_gram,
    empirical_amplitude_gram,
    gram_whitened_rank_audit,
)
from ar_raphu.runtime_environment import require_runtime_environment
from tools.run_phase1_m7 import (
    ROOT as M7_ROOT,
    basis_for,
    config_dir as m7_config_dir,
    load_anchor,
    partition_inputs,
    predict_scaled as predict_m7_scaled,
)
from tools.run_phase1_scheme_a import configure_runtime_threads


ROOT = PROJECT_ROOT / "results" / "phase1" / "E2_AR-S1_G2" / "M8"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def config_id(lag_grid: int, smoothness: float) -> str:
    return f"gtau={lag_grid};lambda={smoothness:.8g}"


def config_dir(seed: int, lag_grid: int, smoothness: float) -> Path:
    return (
        ROOT
        / f"seed_{seed}"
        / f"lag_grid_{lag_grid}"
        / f"lambda_{smoothness:.8g}"
    )


def declared_configs(config: dict[str, Any]) -> list[tuple[int, float]]:
    m8 = config["phase1_model_selection"]["M8"]
    return [
        (int(grid), float(smoothness))
        for grid in m8["lag_spline_grid_sizes"]
        for smoothness in m8["residual_smoothness_weights"]
    ]


def selected_m7() -> tuple[int, float]:
    selection_path = M7_ROOT / "validation_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("M7 validation selection must be frozen before M8.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_used") is not False:
        raise RuntimeError("M7 selection metadata is not validation-only.")
    parts = dict(
        item.split("=", 1)
        for item in selection["selected_config_id"].split(";")
    )
    return int(parts["gx"]), float(parts["lambda"])


def load_m7(seed: int, device: torch.device):
    grid_size, smoothness = selected_m7()
    data, x_ranges, anchor, support, _ = load_anchor(seed, device)
    checkpoint = torch.load(
        m7_config_dir(seed, grid_size, smoothness) / "fit.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint["support"] != support:
        raise RuntimeError("M7 checkpoint support differs from Scheme-A anchor.")
    bank = SplineBasisBankV20(
        [x_ranges[variable] for variable in support],
        grid_size=grid_size,
        spline_order=3,
        dtype=torch.float64,
    ).to(device)
    return data, x_ranges, anchor, support, checkpoint, bank


def partition_m7(
    data,
    anchor,
    support: list[int],
    checkpoint: dict[str, Any],
    bank: SplineBasisBankV20,
    partition: str,
    *,
    device: torch.device,
    batch_size: int,
):
    windows, target, ar_contribution, indices = partition_inputs(
        data,
        anchor,
        support,
        partition,
        device=device,
        batch_size=batch_size,
    )
    amplitude = basis_for(bank, windows)
    prediction = predict_m7_scaled(
        amplitude,
        checkpoint["q"].to(device),
        checkpoint["coefficients"].to(device),
        checkpoint["centered_bias"].to(device),
        checkpoint["response_means"].to(device),
        ar_contribution,
    )
    return amplitude, target, prediction, indices


def fit_job(args: argparse.Namespace, device: torch.device) -> None:
    config = load_protocol_config(require_phase1_frozen=True)
    candidate = (args.lag_grid, args.smoothness)
    if candidate not in set(declared_configs(config)):
        raise ValueError("M8 configuration is not predeclared.")
    output_root = config_dir(args.seed, *candidate)
    summary_path = output_root / "summary.json"
    if summary_path.is_file() and not args.force:
        print(summary_path)
        return

    started = time.perf_counter()
    data, _, anchor, support, checkpoint, bank = load_m7(args.seed, device)
    batch_size = int(config["training"]["batch_size"]["physical_chunk"])
    train_basis, train_target, train_m7, _ = partition_m7(
        data,
        anchor,
        support,
        checkpoint,
        bank,
        "train",
        device=device,
        batch_size=batch_size,
    )
    val_basis, val_target, val_m7, _ = partition_m7(
        data,
        anchor,
        support,
        checkpoint,
        bank,
        "validation",
        device=device,
        batch_size=batch_size,
    )
    lag_basis = orthogonal_lag_bases(
        checkpoint["q"].to(device), args.lag_grid
    )
    train_design = surface_design(train_basis, lag_basis)
    val_design = surface_design(val_basis, lag_basis)
    penalty = surface_penalty(
        len(support),
        lag_basis.shape[-1],
        train_basis.shape[-1],
        device=device,
        dtype=torch.float64,
    )
    result = solve_surface_ridge(
        train_design,
        train_target - train_m7,
        penalty,
        smoothness=args.smoothness,
    )
    validation_prediction = val_m7 + val_design @ result.coefficients
    validation_rmse = float(
        (validation_prediction - val_target).square().mean().sqrt().cpu()
    )
    coefficients = result.coefficients.reshape(
        len(support), lag_basis.shape[-1], train_basis.shape[-1]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / "fit.pt.tmp"
    torch.save(
        {
            "coefficients": coefficients.detach().cpu(),
            "lag_basis": lag_basis.detach().cpu(),
            "support": support,
            "lag_grid": args.lag_grid,
            "smoothness": args.smoothness,
            "m7_grid_size": bank.grid_size,
            "m7_checkpoint": str(
                m7_config_dir(args.seed, *selected_m7()) / "fit.pt"
            ),
            "anchor_constraint": "q_transpose_W_C_D_equals_zero",
        },
        temporary,
    )
    os.replace(temporary, output_root / "fit.pt")
    orthogonality_error = float(
        torch.einsum(
            "nl,nlm->nm", checkpoint["q"].to(device), lag_basis
        )
        .abs()
        .max()
        .cpu()
    )
    atomic_json(
        summary_path,
        {
            "status": "COMPLETED",
            "model": "M8",
            "seed": args.seed,
            "config_id": config_id(*candidate),
            "lag_grid": args.lag_grid,
            "smoothing_weight": args.smoothness,
            "validation_rmse_scaled": validation_rmse,
            "support": support,
            "kkt_residual": result.kkt_residual,
            "objective": result.objective,
            "numerical_relative_jitter": 1.0e-10,
            "numerical_jitter": result.numerical_jitter,
            "anchor_orthogonality_max_abs": orthogonality_error,
            "dtype": "float64",
            "test_accessed": False,
            "rank_audit_accessed": False,
            "wall_seconds": time.perf_counter() - started,
        },
    )
    print(summary_path)


def candidate_rows(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, float]]]:
    lookup = {
        config_id(grid, smoothness): (grid, smoothness)
        for grid, smoothness in declared_configs(config)
    }
    rows = []
    for seed in config["training"]["seeds"]["screening"]:
        for identifier, (grid, smoothness) in lookup.items():
            path = config_dir(seed, grid, smoothness) / "summary.json"
            if not path.is_file():
                raise RuntimeError(f"Missing M8 candidate summary: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if (
                summary.get("test_accessed") is not False
                or summary.get("rank_audit_accessed") is not False
            ):
                raise RuntimeError("M8 candidate used a forbidden selection input.")
            rows.append(
                {
                    "config_id": identifier,
                    "unit_id": f"seed_{seed}",
                    "validation_loss": summary["validation_rmse_scaled"],
                }
            )
    return rows, lookup


def select_configuration() -> dict[str, Any]:
    config = load_protocol_config(require_phase1_frozen=True)
    m8 = config["phase1_model_selection"]["M8"]
    rows, lookup = candidate_rows(config)
    grids = [int(value) for value in m8["lag_spline_grid_sizes"]]
    weights = [float(value) for value in m8["residual_smoothness_weights"]]
    pilot = float(m8["pilot_smoothing_weight"])
    per_weight: dict[str, dict[str, Any]] = {}
    for weight in weights:
        identifiers = [config_id(grid, weight) for grid in grids]
        selection = validation_one_se_select(
            [row for row in rows if lookup[row["config_id"]][1] == weight],
            declared_config_order=identifiers,
            complexity_key=lambda item: (lookup[item][0],),
        )
        per_weight[f"{weight:.8g}"] = selection

    pilot_selection = per_weight[f"{pilot:.8g}"]
    pilot_grid = lookup[pilot_selection["selected_config_id"]][0]
    membership_count = sum(
        config_id(pilot_grid, weight)
        in per_weight[f"{weight:.8g}"]["eligible_config_ids"]
        for weight in weights
    )
    required = int(
        m8["lag_grid_stability_rule"]["minimum_one_se_memberships"]
    )
    unstable = membership_count < required
    if unstable:
        identifiers = [
            config_id(grid, weight) for grid, weight in declared_configs(config)
        ]
        final = validation_one_se_select(
            rows,
            declared_config_order=identifiers,
            complexity_key=lambda item: (
                lookup[item][0],
                -lookup[item][1],
            ),
        )
        branch = "full_cartesian_fallback"
    else:
        identifiers = [config_id(pilot_grid, weight) for weight in weights]
        final = validation_one_se_select(
            [row for row in rows if lookup[row["config_id"]][0] == pilot_grid],
            declared_config_order=identifiers,
            complexity_key=lambda item: (-lookup[item][1],),
        )
        branch = "stable_sequential"

    final.update(
        {
            "selection_unit": "independent_seed_replicate",
            "pilot_smoothing_weight": pilot,
            "pilot_selected_lag_grid": pilot_grid,
            "pilot_grid_one_se_membership_count": membership_count,
            "required_memberships": required,
            "M8_GRID_SMOOTHING_INTERACTION": unstable,
            "selection_branch": branch,
            "per_weight_lag_grid_one_se": per_weight,
            "rank_inputs_used_for_selection": False,
        }
    )
    atomic_json(ROOT / "validation_selection.json", final)
    return final


def aggregate(device: torch.device) -> None:
    config = load_protocol_config(require_phase1_frozen=True)
    selection_path = ROOT / "validation_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("M8 validation selection has not been frozen.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = dict(
        item.split("=", 1)
        for item in selection["selected_config_id"].split(";")
    )
    lag_grid, smoothness = int(selected["gtau"]), float(selected["lambda"])
    batch_size = int(config["training"]["batch_size"]["physical_chunk"])
    metrics = []
    predictions = []
    rank_audits: list[dict[str, Any]] = []
    for seed in config["training"]["seeds"]["screening"]:
        data, _, anchor, support, m7_checkpoint, bank = load_m7(seed, device)
        test_basis, test_target, test_m7, indices = partition_m7(
            data,
            anchor,
            support,
            m7_checkpoint,
            bank,
            "test",
            device=device,
            batch_size=batch_size,
        )
        checkpoint = torch.load(
            config_dir(seed, lag_grid, smoothness) / "fit.pt",
            map_location=device,
            weights_only=False,
        )
        residual = (
            surface_design(test_basis, checkpoint["lag_basis"].to(device))
            @ checkpoint["coefficients"].to(device).reshape(-1)
        )
        prediction_scaled = test_m7 + residual
        observed = data.scaler.inverse_y(test_target.cpu().numpy())
        predicted = data.scaler.inverse_y(prediction_scaled.cpu().numpy())
        row = prediction_metrics(observed, predicted)
        row.update(
            {
                "seed": seed,
                "selected_config_id": selection["selected_config_id"],
                "support": support,
            }
        )
        metrics.append(row)
        predictions.append(
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

        train_x = torch.as_tensor(
            data.x_scaled[: data.scaler.fit_stop],
            device=device,
            dtype=torch.float64,
        )
        q = m7_checkpoint["q"].cpu().numpy()
        m7_coefficients = m7_checkpoint["coefficients"].cpu().numpy()
        lag_residual_basis = checkpoint["lag_basis"].cpu().numpy()
        residual_coefficients = checkpoint["coefficients"].cpu().numpy()
        for local_index, variable in enumerate(support):
            lag_basis = np.column_stack(
                [q[local_index], lag_residual_basis[local_index]]
            )
            coefficients = np.vstack(
                [
                    m7_coefficients[local_index][None, :],
                    residual_coefficients[local_index],
                ]
            )
            amplitude_basis = (
                bank.evaluate_grid(local_index, train_x[:, variable])
                .cpu()
                .numpy()
            )
            audit = gram_whitened_rank_audit(
                coefficients,
                discrete_lag_gram(lag_basis),
                empirical_amplitude_gram(amplitude_basis),
            )
            rank_audits.append(
                {
                    "seed": seed,
                    "variable": variable,
                    "singular_values": audit.singular_values.tolist(),
                    "first_singular_energy": audit.first_singular_energy,
                    "nonseparability": audit.nonseparability,
                }
            )
    atomic_json(
        ROOT / "test_metrics.json",
        {
            "status": "COMPLETED",
            "model": "M8",
            "selected_config_id": selection["selected_config_id"],
            "selection_branch": selection["selection_branch"],
            "per_seed": metrics,
            "hyperparameters_frozen_before_test": True,
        },
    )
    atomic_json(
        ROOT / "rank_audit.json",
        {
            "status": "COMPLETED",
            "model": "M8",
            "selected_config_id": selection["selected_config_id"],
            "selection_used_rank_inputs": False,
            "audit": "Gram_whitened_SVD",
            "bootstrap_status": "NOT_YET_RUN",
            "per_seed_variable": rank_audits,
        },
    )
    pd.concat(predictions, ignore_index=True).to_parquet(
        ROOT / "predictions.parquet", index=False
    )
    print(ROOT / "test_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fit", "select", "aggregate"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lag-grid", type=int)
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
        if args.seed is None or args.lag_grid is None or args.smoothness is None:
            raise ValueError("M8 fit requires seed, lag-grid, and smoothness.")
        fit_job(args, device)
    elif args.mode == "select":
        selected = select_configuration()
        print(ROOT / "validation_selection.json", selected["selected_config_id"])
    else:
        aggregate(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
