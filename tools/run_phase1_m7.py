#!/usr/bin/env python3
"""Run the v2 M7 Scheme-A-anchored convex spline refit for Phase 1."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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
    solve_fixed_q_fista_v20,
)
from ar_raphu.model_selection import validation_one_se_select
from ar_raphu.phase1_evidence import (
    partition_predictions_and_contributions,
    static_lag_kernels,
)
from ar_raphu.protocol_config import load_protocol_config
from ar_raphu.runtime_environment import require_runtime_environment
from tools.run_phase1_scheme_a import (
    build_model,
    configure_runtime_threads,
    prepare,
)


ROOT = PROJECT_ROOT / "results" / "phase1" / "E2_AR-S1_G2" / "M7"
ANCHOR_ROOT = (
    PROJECT_ROOT / "results" / "phase1" / "E2_AR-S1_G2" / "Track-XAR"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def config_id(grid_size: int, smoothness: float) -> str:
    return f"gx={grid_size};lambda={smoothness:.8g}"


def config_dir(seed: int, grid_size: int, smoothness: float) -> Path:
    return (
        ROOT
        / f"seed_{seed}"
        / f"grid_{grid_size}"
        / f"lambda_{smoothness:.8g}"
    )


def declared_configs(config: dict[str, Any]) -> list[tuple[int, float]]:
    m7 = config["phase1_model_selection"]["M7"]
    return [
        (int(grid), float(smoothness))
        for grid in m7["amplitude_spline_grid_sizes"]
        for smoothness in m7["smoothness_weights"]
    ]


def load_anchor(seed: int, device: torch.device):
    selection_path = ANCHOR_ROOT / "validation_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("M5 validation selection must be frozen before M7.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_used") is not False:
        raise RuntimeError("M5 selection metadata does not prove validation-only use.")
    selected_scale = float(str(selection["config_id"]).split("=")[1])
    checkpoint_path = (
        ANCHOR_ROOT
        / f"seed_{seed}"
        / "forks"
        / f"s0_{selected_scale:.6f}"
        / "best.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    _, data, x_ranges, y_range = prepare(
        "AR-S1", "XAR", seed, smoke=False
    )
    model = build_model("XAR", x_ranges, y_range, device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    support = [int(value) for value in checkpoint["terminal_support"]]
    if not support:
        raise RuntimeError("M7 requires a non-empty frozen M5 external support.")
    external_q, _ = static_lag_kernels(model)
    if external_q is None:
        raise RuntimeError("M5 checkpoint has no external lag kernel.")
    return data, x_ranges, model, support, external_q[support]


def partition_inputs(
    data,
    model,
    support: list[int],
    partition: str,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    windows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    indices: list[np.ndarray] = []
    for batch in data.iter_batches(
        partition, batch_size=batch_size, device=device
    ):
        windows.append(batch["x_window"][:, support])
        targets.append(batch["target"])
        indices.append(batch["target_index"].cpu().numpy())
    _, _, evidence_indices, components = (
        partition_predictions_and_contributions(
            model,
            data,
            partition,
            batch_size=batch_size,
            device=device,
        )
    )
    joined_indices = np.concatenate(indices)
    if not np.array_equal(joined_indices, evidence_indices):
        raise RuntimeError("M7 anchor contribution indices are misaligned.")
    ar_contribution = torch.as_tensor(
        components[:, -1], device=device, dtype=torch.float64
    )
    return (
        torch.cat(windows).to(torch.float64),
        torch.cat(targets).to(torch.float64),
        ar_contribution,
        joined_indices,
    )


def basis_for(
    bank: SplineBasisBankV20, windows: torch.Tensor
) -> torch.Tensor:
    return bank(torch.flip(windows, dims=(-1,)))


def centered_parameters(
    bank: SplineBasisBankV20,
    coefficients: torch.Tensor,
    bias: torch.Tensor,
    data,
    support: list[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    means: list[torch.Tensor] = []
    train_x = torch.as_tensor(
        data.x_scaled[: data.scaler.fit_stop],
        device=device,
        dtype=torch.float64,
    )
    for local_index, variable in enumerate(support):
        raw_basis = bank.evaluate_grid(local_index, train_x[:, variable])
        means.append(raw_basis.mean(0) @ coefficients[local_index])
    response_means = torch.stack(means)
    return bias + response_means.sum(), response_means


def predict_scaled(
    basis: torch.Tensor,
    q: torch.Tensor,
    coefficients: torch.Tensor,
    centered_bias: torch.Tensor,
    response_means: torch.Tensor,
    ar_contribution: torch.Tensor,
) -> torch.Tensor:
    raw = torch.einsum("bnlm,nm,nl->b", basis, coefficients, q)
    external = raw - response_means.sum()
    return centered_bias + external + ar_contribution


def solve_unpenalized_rank_robust(
    basis: torch.Tensor,
    q: torch.Tensor,
    target: torch.Tensor,
) -> SimpleNamespace:
    """Solve the lambda=0 candidate exactly with rank-revealing FP64 SVD."""

    phi = torch.einsum("bnlm,nl->bnm", basis, q)
    sample_count, variable_count, basis_count = phi.shape
    design = phi.reshape(sample_count, -1).cpu()
    response = target.reshape(-1).to(torch.float64).cpu()
    design_mean = design.mean(0, keepdim=True)
    response_mean = response.mean()
    centered_design = design - design_mean
    centered_response = response - response_mean
    solution = torch.linalg.lstsq(
        centered_design, centered_response, driver="gelsd"
    ).solution
    residual = centered_design @ solution - centered_response
    gradient = centered_design.T @ residual / sample_count
    coefficients = solution.reshape(variable_count, basis_count).to(basis.device)
    bias = (
        response_mean - design_mean.reshape(-1).dot(solution)
    ).to(basis.device)
    return SimpleNamespace(
        coefficients=coefficients,
        bias=bias,
        iterations=1,
        converged=True,
        kkt_residual=float(gradient.norm()),
        objective=float(0.5 * residual.square().mean()),
        history=[{"relative_step": 0.0}],
        solver="fp64_rank_revealing_lstsq_gelsd",
    )


def fit_job(args: argparse.Namespace, device: torch.device) -> None:
    config = load_protocol_config(require_phase1_frozen=True)
    allowed = set(declared_configs(config))
    candidate = (args.grid_size, args.smoothness)
    if candidate not in allowed:
        raise ValueError("M7 configuration is not predeclared.")
    output_root = config_dir(args.seed, *candidate)
    summary_path = output_root / "summary.json"
    if summary_path.is_file() and not args.force:
        print(summary_path)
        return

    started = time.perf_counter()
    data, x_ranges, anchor, support, q_array = load_anchor(args.seed, device)
    batch_size = int(config["training"]["batch_size"]["physical_chunk"])
    train_windows, train_target, train_ar, _ = partition_inputs(
        data, anchor, support, "train", device=device, batch_size=batch_size
    )
    val_windows, val_target, val_ar, _ = partition_inputs(
        data,
        anchor,
        support,
        "validation",
        device=device,
        batch_size=batch_size,
    )
    bank = SplineBasisBankV20(
        [x_ranges[variable] for variable in support],
        grid_size=args.grid_size,
        spline_order=3,
        dtype=torch.float64,
    ).to(device)
    train_basis = basis_for(bank, train_windows)
    val_basis = basis_for(bank, val_windows)
    q = torch.as_tensor(q_array, device=device, dtype=torch.float64)
    if args.smoothness == 0.0:
        result = solve_unpenalized_rank_robust(
            train_basis, q, train_target - train_ar
        )
    else:
        result = solve_fixed_q_fista_v20(
            train_basis,
            q,
            train_target - train_ar,
            lambda_group=0.0,
            lambda_smooth=args.smoothness,
            roughness_matrix=bank.roughness,
            max_iter=100000,
            tolerance=1.0e-7,
            kkt_tolerance=1.0e-5,
        )
        result.solver = "monotone_restarted_FISTA_v20"
    centered_bias, response_means = centered_parameters(
        bank,
        result.coefficients,
        result.bias,
        data,
        support,
        device=device,
    )
    validation_prediction = predict_scaled(
        val_basis,
        q,
        result.coefficients,
        centered_bias,
        response_means,
        val_ar,
    )
    validation_rmse = float(
        (validation_prediction - val_target).square().mean().sqrt().cpu()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / "fit.pt.tmp"
    torch.save(
        {
            "coefficients": result.coefficients.detach().cpu(),
            "centered_bias": centered_bias.detach().cpu(),
            "response_means": response_means.detach().cpu(),
            "q": q.detach().cpu(),
            "support": support,
            "grids": bank.grids.detach().cpu(),
            "roughness": bank.roughness.detach().cpu(),
            "grid_size": args.grid_size,
            "smoothness": args.smoothness,
            "anchor": "selected_M5_XAR",
        },
        temporary,
    )
    os.replace(temporary, output_root / "fit.pt")
    atomic_json(
        summary_path,
        {
            "status": "COMPLETED",
            "model": "M7",
            "seed": args.seed,
            "config_id": config_id(*candidate),
            "amplitude_grid": args.grid_size,
            "smoothing_weight": args.smoothness,
            "validation_rmse_scaled": validation_rmse,
            "support": support,
            "fista_iterations": result.iterations,
            "fista_converged": result.converged,
            "convex_solver": result.solver,
            "kkt_residual": result.kkt_residual,
            "last_recorded_relative_step": result.history[-1]["relative_step"],
            "relative_step_tolerance": 1.0e-7,
            "kkt_tolerance": 1.0e-5,
            "objective": result.objective,
            "dtype": "float64",
            "test_accessed": False,
            "wall_seconds": time.perf_counter() - started,
        },
    )
    print(summary_path)


def select_configuration() -> dict[str, Any]:
    config = load_protocol_config(require_phase1_frozen=True)
    candidates = declared_configs(config)
    declared_ids = [config_id(*candidate) for candidate in candidates]
    lookup = {
        config_id(grid, smoothness): (grid, smoothness)
        for grid, smoothness in candidates
    }
    rows: list[dict[str, Any]] = []
    for seed in config["training"]["seeds"]["screening"]:
        for grid, smoothness in candidates:
            path = config_dir(seed, grid, smoothness) / "summary.json"
            if not path.is_file():
                raise RuntimeError(f"Missing M7 summary: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("test_accessed") is not False:
                raise RuntimeError("M7 candidate accessed test before selection.")
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
        complexity_key=lambda item: (
            lookup[item][0],
            -lookup[item][1],
        ),
    )
    selected["selection_unit"] = "independent_seed_replicate"
    selected["complexity_order"] = [
        "amplitude_grid_ascending",
        "smoothing_weight_descending",
        "predeclared_configuration_order",
    ]
    atomic_json(ROOT / "validation_selection.json", selected)
    return selected


def aggregate(device: torch.device) -> None:
    config = load_protocol_config(require_phase1_frozen=True)
    selection_path = ROOT / "validation_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("M7 validation selection has not been frozen.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_id = selection["selected_config_id"]
    candidates = {
        config_id(grid, smoothness): (grid, smoothness)
        for grid, smoothness in declared_configs(config)
    }
    grid_size, smoothness = candidates[selected_id]
    batch_size = int(config["training"]["batch_size"]["physical_chunk"])
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    response_payload: dict[str, np.ndarray] = {}
    for seed in config["training"]["seeds"]["screening"]:
        data, x_ranges, anchor, support, _ = load_anchor(seed, device)
        test_windows, test_target, test_ar, indices = partition_inputs(
            data,
            anchor,
            support,
            "test",
            device=device,
            batch_size=batch_size,
        )
        checkpoint = torch.load(
            config_dir(seed, grid_size, smoothness) / "fit.pt",
            map_location=device,
            weights_only=False,
        )
        bank = SplineBasisBankV20(
            [x_ranges[variable] for variable in support],
            grid_size=grid_size,
            spline_order=3,
            dtype=torch.float64,
        ).to(device)
        prediction_scaled = predict_scaled(
            basis_for(bank, test_windows),
            checkpoint["q"].to(device),
            checkpoint["coefficients"].to(device),
            checkpoint["centered_bias"].to(device),
            checkpoint["response_means"].to(device),
            test_ar,
        )
        predicted = data.scaler.inverse_y(prediction_scaled.cpu().numpy())
        observed = data.scaler.inverse_y(test_target.cpu().numpy())
        row = prediction_metrics(observed, predicted)
        row.update(
            {
                "seed": seed,
                "selected_config_id": selected_id,
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
        for local_index, variable in enumerate(support):
            lower, upper = x_ranges[variable]
            grid = torch.linspace(
                lower, upper, 401, device=device, dtype=torch.float64
            )
            values = (
                bank.evaluate_grid(local_index, grid)
                @ checkpoint["coefficients"][local_index].to(device)
                - checkpoint["response_means"][local_index].to(device)
            )
            response_payload[f"seed{seed}_x_grid_v{variable}_scaled"] = (
                grid.cpu().numpy()
            )
            response_payload[f"seed{seed}_x_response_v{variable}_scaled"] = (
                values.cpu().numpy()
            )
    atomic_json(
        ROOT / "test_metrics.json",
        {
            "status": "COMPLETED",
            "model": "M7",
            "selected_config_id": selected_id,
            "per_seed": metrics,
            "hyperparameters_frozen_before_test": True,
        },
    )
    pd.concat(predictions, ignore_index=True).to_parquet(
        ROOT / "predictions.parquet", index=False
    )
    np.savez_compressed(ROOT / "response_grid.npz", **response_payload)
    print(ROOT / "test_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fit", "select", "aggregate"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--grid-size", type=int)
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
        if args.seed is None or args.grid_size is None or args.smoothness is None:
            raise ValueError("M7 fit requires seed, grid-size, and smoothness.")
        fit_job(args, device)
    elif args.mode == "select":
        selected = select_configuration()
        print(ROOT / "validation_selection.json", selected["selected_config_id"])
    else:
        aggregate(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
