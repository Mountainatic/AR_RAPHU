#!/usr/bin/env python3
"""Formal residual moving-block bootstrap for the frozen E2 M8 rank audit."""

from __future__ import annotations

import argparse
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

from ar_raphu.orthogonal_surface import surface_design, surface_penalty
from ar_raphu.protocol_config import load_protocol_config
from ar_raphu.rank_audit import (
    discrete_lag_gram,
    empirical_amplitude_gram,
    gram_whitened_rank_audit,
)
from ar_raphu.runtime_environment import require_runtime_environment
from ar_raphu.statistics import (
    benjamini_hochberg,
    moving_block_indices,
    residual_acf_block_length,
)
from tools.run_phase1_m8 import (
    ROOT,
    config_dir,
    load_m7,
    partition_m7,
)
from tools.run_phase1_scheme_a import configure_runtime_threads


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def frozen_m8_config() -> tuple[int, float, str]:
    path = ROOT / "validation_selection.json"
    if not path.is_file():
        raise RuntimeError("M8 validation selection must be frozen first.")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("rank_inputs_used_for_selection") is not False:
        raise RuntimeError("M8 hyperparameters were not frozen independently of rank.")
    parts = dict(
        item.split("=", 1)
        for item in selection["selected_config_id"].split(";")
    )
    return (
        int(parts["gtau"]),
        float(parts["lambda"]),
        selection["selected_config_id"],
    )


def rank_statistic(
    *,
    q: np.ndarray,
    m7_coefficients: np.ndarray,
    lag_residual_basis: np.ndarray,
    residual_coefficients: np.ndarray,
    amplitude_grams: list[np.ndarray],
) -> np.ndarray:
    statistics = []
    for variable in range(len(q)):
        lag_basis = np.column_stack(
            [q[variable], lag_residual_basis[variable]]
        )
        coefficients = np.vstack(
            [
                m7_coefficients[variable][None, :],
                residual_coefficients[variable],
            ]
        )
        audit = gram_whitened_rank_audit(
            coefficients,
            discrete_lag_gram(lag_basis),
            amplitude_grams[variable],
        )
        statistics.append(audit.nonseparability)
    return np.asarray(statistics, dtype=np.float64)


def seed_job(seed: int, device: torch.device, *, force: bool) -> Path:
    output = ROOT / "bootstrap" / f"seed_{seed}.json"
    if output.is_file() and not force:
        return output
    config = load_protocol_config(require_phase1_frozen=True)
    lag_grid, smoothness, selected_id = frozen_m8_config()
    replicates = int(config["statistics"]["bootstrap_replicates"]["formal"])
    data, _, anchor, support, m7_checkpoint, bank = load_m7(seed, device)
    if not support:
        raise RuntimeError("E2 M8 bootstrap requires nonempty frozen support.")
    batch_size = int(config["training"]["batch_size"]["physical_chunk"])
    train_basis, train_target, train_m7, _ = partition_m7(
        data,
        anchor,
        support,
        m7_checkpoint,
        bank,
        "train",
        device=device,
        batch_size=batch_size,
    )
    checkpoint = torch.load(
        config_dir(seed, lag_grid, smoothness) / "fit.pt",
        map_location=device,
        weights_only=False,
    )
    lag_basis = checkpoint["lag_basis"].to(device=device, dtype=torch.float64)
    design = surface_design(train_basis, lag_basis).to(torch.float64)
    penalty = surface_penalty(
        len(support),
        lag_basis.shape[-1],
        train_basis.shape[-1],
        device=device,
        dtype=torch.float64,
    )
    gram = design.T @ design / design.shape[0]
    system = gram + smoothness * penalty
    scale = float(system.diagonal().abs().mean().detach().cpu())
    numerical_jitter = 1.0e-10 * max(scale, 1.0)
    system = system + numerical_jitter * torch.eye(
        system.shape[0], device=device, dtype=torch.float64
    )

    null_residual = (
        train_target - train_m7
    ).detach().cpu().numpy().astype(np.float64)
    null_residual -= null_residual.mean()
    block_length = residual_acf_block_length(null_residual)
    bootstrap_seed = 880000 + int(seed)
    indices = moving_block_indices(
        len(null_residual),
        block_length,
        replicates=replicates,
        rng=np.random.default_rng(bootstrap_seed),
    )
    sampled = torch.as_tensor(
        null_residual[indices], device=device, dtype=torch.float64
    )
    right = sampled @ design / design.shape[0]
    bootstrap_coefficients = torch.linalg.solve(system, right.T).T
    bootstrap_coefficients = (
        bootstrap_coefficients.reshape(
            replicates,
            len(support),
            lag_basis.shape[-1],
            train_basis.shape[-1],
        )
        .detach()
        .cpu()
        .numpy()
    )

    train_x = torch.as_tensor(
        data.x_scaled[: data.scaler.fit_stop],
        device=device,
        dtype=torch.float64,
    )
    amplitude_grams = [
        empirical_amplitude_gram(
            bank.evaluate_grid(local_index, train_x[:, variable])
            .detach()
            .cpu()
            .numpy()
        )
        for local_index, variable in enumerate(support)
    ]
    q = m7_checkpoint["q"].detach().cpu().numpy()
    m7_coefficients = (
        m7_checkpoint["coefficients"].detach().cpu().numpy()
    )
    lag_residual_basis = lag_basis.detach().cpu().numpy()
    observed_coefficients = (
        checkpoint["coefficients"].detach().cpu().numpy()
    )
    observed = rank_statistic(
        q=q,
        m7_coefficients=m7_coefficients,
        lag_residual_basis=lag_residual_basis,
        residual_coefficients=observed_coefficients,
        amplitude_grams=amplitude_grams,
    )
    null = np.stack(
        [
            rank_statistic(
                q=q,
                m7_coefficients=m7_coefficients,
                lag_residual_basis=lag_residual_basis,
                residual_coefficients=bootstrap_coefficients[index],
                amplitude_grams=amplitude_grams,
            )
            for index in range(replicates)
        ]
    )
    global_observed = float(observed.max())
    global_null = null.max(axis=1)
    global_p = float(
        (1 + np.count_nonzero(global_null >= global_observed))
        / (replicates + 1)
    )
    variable_p = (
        1 + np.count_nonzero(null >= observed[None, :], axis=0)
    ) / (replicates + 1)
    reject, adjusted = benjamini_hochberg(
        variable_p,
        q=float(config["statistics"]["multiple_testing_correction"]["FDR_q"]),
    )
    alpha = float(config["statistics"]["significance_alpha"])
    global_reject = global_p < alpha
    variable_rows = [
        {
            "variable": int(variable),
            "observed_nonseparability": float(observed[index]),
            "raw_p_value": float(variable_p[index]),
            "BH_adjusted_q_value": float(adjusted[index]),
            "BH_reject_before_global_gate": bool(reject[index]),
            "rank2_reject": bool(global_reject and reject[index]),
        }
        for index, variable in enumerate(support)
    ]
    atomic_json(
        output,
        {
            "status": "COMPLETED",
            "model": "M8",
            "scenario": "AR-S1",
            "truth_rank": 1,
            "seed": seed,
            "selected_config_id": selected_id,
            "null_model": "frozen_M7_rank1",
            "bootstrap": "residual_moving_block_refit_M8",
            "bootstrap_seed": bootstrap_seed,
            "replicates": replicates,
            "block_length": block_length,
            "block_length_rule": config["statistics"]["block_length_rule"],
            "global_statistic": "max_variable_Gram_whitened_nonseparability",
            "global_observed": global_observed,
            "global_p_value": global_p,
            "global_reject": global_reject,
            "alpha": alpha,
            "variable_tests_gated_by_global": True,
            "multiple_testing": config["statistics"][
                "multiple_testing_correction"
            ],
            "variables": variable_rows,
            "test_partition_accessed": False,
            "dtype": "float64",
            "numerical_jitter": numerical_jitter,
        },
    )
    return output


def aggregate() -> Path:
    config = load_protocol_config(require_phase1_frozen=True)
    seeds = [int(value) for value in config["training"]["seeds"]["screening"]]
    records = []
    for seed in seeds:
        path = ROOT / "bootstrap" / f"seed_{seed}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing bootstrap result: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("test_partition_accessed") is not False:
            raise RuntimeError("Bootstrap rank audit accessed test data.")
        records.append(record)
    variable_tests = [
        row
        for record in records
        for row in record["variables"]
    ]
    rank2_rejections = sum(row["rank2_reject"] for row in variable_tests)
    global_rejections = sum(record["global_reject"] for record in records)
    output = ROOT / "bootstrap_rank_audit.json"
    atomic_json(
        output,
        {
            "status": "COMPLETED",
            "model": "M8",
            "scenario": "AR-S1",
            "truth_rank": 1,
            "replicates_per_seed": int(
                config["statistics"]["bootstrap_replicates"]["formal"]
            ),
            "seed_count": len(seeds),
            "global_rejection_count": global_rejections,
            "global_false_positive_rate": global_rejections / len(seeds),
            "variable_rejection_count": rank2_rejections,
            "variable_false_positive_rate": (
                rank2_rejections / len(variable_tests)
                if variable_tests
                else 0.0
            ),
            "rank1_false_positive_threshold": float(
                config["statistics"]["rank1_false_positive_threshold"]
            ),
            "rank1_false_positive_gate_passed": (
                global_rejections / len(seeds)
                <= float(config["statistics"]["rank1_false_positive_threshold"])
            ),
            "test_partition_accessed": False,
            "per_seed": records,
        },
    )
    rank_path = ROOT / "rank_audit.json"
    rank_payload = json.loads(rank_path.read_text(encoding="utf-8"))
    rank_payload.update(
        {
            "bootstrap_status": "COMPLETED",
            "bootstrap_result": str(output.relative_to(PROJECT_ROOT)),
            "bootstrap_test_partition_accessed": False,
            "rank1_false_positive_gate_passed": (
                global_rejections / len(seeds)
                <= float(config["statistics"]["rank1_false_positive_threshold"])
            ),
        }
    )
    atomic_json(rank_path, rank_payload)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["seed", "aggregate"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    require_runtime_environment()
    configure_runtime_threads()
    args = parse_args()
    if args.mode == "aggregate":
        print(aggregate())
        return 0
    if args.seed is None:
        raise ValueError("seed mode requires --seed.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    print(seed_job(args.seed, device, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
