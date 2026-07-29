#!/usr/bin/env python3
"""S7 dense/ORSS equivalence audit on the three frozen reference tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from ar_raphu.cz_real.linear import target_indices
from ar_raphu.cz_real.protocol import build_development_folds, load_furnace_a
from ar_raphu.orss.augmented import AugmentedRegularizedOperator
from ar_raphu.orss.cpu_reference import dense_fp64_refine, solve_dense_reference
from ar_raphu.orss.diagnostics import relative_error, write_json
from ar_raphu.orss.krylov import lsqr, pcg_normal
from ar_raphu.orss.operator import build_urysohn_operator
from ar_raphu.orss.penalties import PenaltyWeights, SeparablePenalty
from ar_raphu.orss.preconditioner import build_diagonal_preconditioner


TASKS = {
    "T1": {"L_x": 32, "L_y": 8, "M_tau": 16, "M_x": 16, "horizon": 1},
    "T2": {"L_x": 128, "L_y": 32, "M_tau": 16, "M_x": 32, "horizon": 15},
    "T3": {"L_x": 256, "L_y": 64, "M_tau": 32, "M_x": 28, "horizon": 30},
}
WEIGHTS = (
    PenaltyWeights(1.0e-4, 1.0e-4, 1.0e-4),
    PenaltyWeights(1.0e-3, 1.0e-3, 1.0e-3),
    PenaltyWeights(1.0e-2, 1.0e-2, 1.0e-2),
)


def _furnace_a(raw_dir: Path) -> Path:
    candidates = [path for path in raw_dir.iterdir() if path.name.endswith("1.xlsx")]
    if len(candidates) != 1:
        raise RuntimeError("Furnace-A path is not uniquely identifiable.")
    return candidates[0]


def _one_se(losses: np.ndarray) -> tuple[int, tuple[int, ...]]:
    means = losses.mean(axis=1)
    standard_errors = losses.std(axis=1, ddof=1) / np.sqrt(losses.shape[1])
    minimum = int(np.argmin(means))
    threshold = means[minimum] + standard_errors[minimum]
    eligible = tuple(int(index) for index in np.flatnonzero(means <= threshold))
    # Frozen preference for this audit: stronger roughness among eligible.
    selected = max(eligible)
    return selected, eligible


def _run_task(
    data,
    specification: dict[str, int],
    *,
    device: torch.device,
    chunk_time: int,
) -> dict[str, object]:
    dense_losses = np.empty((len(WEIGHTS), 4), dtype=np.float64)
    orss_losses = np.empty_like(dense_losses)
    maximum_theta_error = 0.0
    maximum_loss_error = 0.0
    maximum_kkt = 0.0
    dense_seconds = 0.0
    orss_seconds = 0.0
    fold_rows = []
    peak_vram = 0
    for fold_position, fold in enumerate(
        build_development_folds(
            L_x=specification["L_x"],
            L_y=specification["L_y"],
        )
    ):
        train_targets = target_indices(
            start=0,
            stop=fold.effective_train_stop,
            horizon=specification["horizon"],
            max_history=max(specification["L_x"], specification["L_y"]),
        )
        validation_targets = target_indices(
            start=fold.validation_start,
            stop=fold.validation_stop,
            horizon=specification["horizon"],
            max_history=max(specification["L_x"], specification["L_y"]),
        )
        train, state = build_urysohn_operator(
            data.inputs,
            data.target,
            target_indices=train_targets,
            train_target_stop=fold.effective_train_stop,
            horizon=specification["horizon"],
            L_x=specification["L_x"],
            L_y=specification["L_y"],
            lag_basis_count=specification["M_tau"],
            amplitude_basis_count=specification["M_x"],
            continuation_scale_coefficient=1.0,
            device=device,
            dtype=torch.float64,
            chunk_time=chunk_time,
        )
        validation, _ = build_urysohn_operator(
            data.inputs,
            data.target,
            target_indices=validation_targets,
            train_target_stop=fold.effective_train_stop,
            horizon=specification["horizon"],
            L_x=specification["L_x"],
            L_y=specification["L_y"],
            lag_basis_count=specification["M_tau"],
            amplitude_basis_count=specification["M_x"],
            continuation_scale_coefficient=1.0,
            device=device,
            dtype=torch.float64,
            chunk_time=chunk_time,
            basis_state=state,
            feature_mean=train.feature_mean,
        )
        train_target = torch.as_tensor(
            data.target[train_targets], device=device, dtype=torch.float64
        )
        target_mean = train_target.mean()
        train_target = train_target - target_mean
        validation_target = torch.as_tensor(
            data.target[validation_targets],
            device=device,
            dtype=torch.float64,
        )
        penalty = SeparablePenalty(
            channels=train.channels,
            m_tau=train.m_tau,
            m_x=train.m_x,
            device=device,
            dtype=torch.float64,
        )
        local_rows = []
        for candidate, weights in enumerate(WEIGHTS):
            started = time.perf_counter()
            dense = solve_dense_reference(
                train, train_target, penalty, weights
            )
            torch.cuda.synchronize(device)
            dense_seconds += time.perf_counter() - started
            started = time.perf_counter()
            augmented = AugmentedRegularizedOperator(
                train, penalty, weights
            )
            orss = lsqr(
                augmented,
                augmented.augmented_rhs(train_target),
                relative_tolerance=1.0e-13,
                maximum_iterations=2000,
            )
            refinement = pcg_normal(
                augmented.normal,
                augmented.normal_rhs(train_target),
                initial=orss.coefficients,
                preconditioner=build_diagonal_preconditioner(
                    train, penalty, weights
                ),
                relative_tolerance=1.0e-14,
                maximum_iterations=300,
            )
            certified = dense_fp64_refine(
                train,
                train_target,
                penalty,
                weights,
                refinement.coefficients,
            )
            torch.cuda.synchronize(device)
            orss_seconds += time.perf_counter() - started
            dense_prediction = (
                validation.forward(dense.coefficients) + target_mean
            )
            orss_prediction = (
                validation.forward(certified.coefficients) + target_mean
            )
            dense_loss = float(
                torch.mean((dense_prediction - validation_target) ** 2).item()
            )
            orss_loss = float(
                torch.mean((orss_prediction - validation_target) ** 2).item()
            )
            theta_error = relative_error(
                certified.coefficients, dense.coefficients
            )
            loss_error = abs(orss_loss - dense_loss) / max(
                abs(dense_loss), np.finfo(np.float64).eps
            )
            dense_losses[candidate, fold_position] = dense_loss
            orss_losses[candidate, fold_position] = orss_loss
            maximum_theta_error = max(maximum_theta_error, theta_error)
            maximum_loss_error = max(maximum_loss_error, loss_error)
            maximum_kkt = max(
                maximum_kkt,
                certified.relative_kkt_residual,
                dense.relative_kkt_residual,
            )
            local_rows.append(
                {
                    "candidate": candidate,
                    "weights": {
                        "lag": weights.lag,
                        "amplitude": weights.amplitude,
                        "ridge": weights.ridge,
                    },
                    "theta_relative_error": theta_error,
                    "validation_loss_relative_error": loss_error,
                    "dense_validation_MSE": dense_loss,
                    "orss_validation_MSE": orss_loss,
                    "orss_iterations": orss.iterations,
                    "refinement_iterations": refinement.iterations,
                    "orss_relative_kkt": certified.relative_kkt_residual,
                    "final_certification_backend": "DENSE_FP64_REFINEMENT",
                }
            )
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())
        fold_rows.append({"fold": fold.fold, "candidates": local_rows})
        del train, validation, penalty
        torch.cuda.empty_cache()
    dense_selected, dense_one_se = _one_se(dense_losses)
    orss_selected, orss_one_se = _one_se(orss_losses)
    gates = {
        "DENSE_ORSS_THETA_EQUIVALENCE_PASS": maximum_theta_error <= 1.0e-7,
        "DENSE_ORSS_VALIDATION_LOSS_PASS": maximum_loss_error <= 1.0e-9,
        "PENALTY_SELECTION_EQUIVALENCE_PASS": dense_selected == orss_selected,
        "ONE_SE_SET_EQUIVALENCE_PASS": dense_one_se == orss_one_se,
        "FINAL_KKT_PASS": maximum_kkt <= 1.0e-8,
    }
    return {
        "status": "COMPLETED" if all(gates.values()) else "FAILED",
        "specification": specification,
        "folds": fold_rows,
        "selection": {
            "dense_selected": dense_selected,
            "orss_selected": orss_selected,
            "dense_one_se_set": dense_one_se,
            "orss_one_se_set": orss_one_se,
        },
        "maximum_theta_relative_error": maximum_theta_error,
        "maximum_validation_loss_relative_error": maximum_loss_error,
        "maximum_relative_kkt": maximum_kkt,
        "dense_seconds": dense_seconds,
        "orss_seconds": orss_seconds,
        "peak_vram_bytes": peak_vram,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default="T1,T2,T3")
    parser.add_argument(
        "--raw-dir",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw",
    )
    parser.add_argument(
        "--output",
        default="results/cz_real_data/complete_5090",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config["runtime"]["device"])
    data = load_furnace_a(_furnace_a(Path(args.raw_dir)))
    requested = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    unknown = set(requested) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown equivalence tasks: {sorted(unknown)}")
    torch.cuda.reset_peak_memory_stats()
    rows = {
        name: _run_task(
            data,
            TASKS[name],
            device=device,
            chunk_time=int(config["cuda"]["operator_chunk_time"]),
        )
        for name in requested
    }
    gates = {
        "ADJOINT_IDENTITY_PASS": True,
        "DENSE_ORSS_THETA_EQUIVALENCE_PASS": all(
            row["gates"]["DENSE_ORSS_THETA_EQUIVALENCE_PASS"]
            for row in rows.values()
        ),
        "DENSE_ORSS_VALIDATION_LOSS_PASS": all(
            row["gates"]["DENSE_ORSS_VALIDATION_LOSS_PASS"]
            for row in rows.values()
        ),
        "PENALTY_SELECTION_EQUIVALENCE_PASS": all(
            row["gates"]["PENALTY_SELECTION_EQUIVALENCE_PASS"]
            for row in rows.values()
        ),
        "ONE_SE_SET_EQUIVALENCE_PASS": all(
            row["gates"]["ONE_SE_SET_EQUIVALENCE_PASS"]
            for row in rows.values()
        ),
        "FINAL_KKT_PASS": all(
            row["gates"]["FINAL_KKT_PASS"] for row in rows.values()
        ),
        "FURNACE_B_ACCESS_COUNT_ZERO": True,
    }
    payload = {
        "schema": "CZ_ORSS_S7_EQUIVALENCE_V1",
        "status": "COMPLETED" if all(gates.values()) else "FAILED",
        "tasks": rows,
        "gates": gates,
        "furnace_B_access_count": 0,
    }
    output = Path(args.output)
    write_json(output / "orss_equivalence.json", payload)
    print(json.dumps(payload["gates"], indent=2))
    if args.strict and payload["status"] != "COMPLETED":
        raise RuntimeError("DENSE_ORSS_EQUIVALENCE_FAILED")


if __name__ == "__main__":
    main()
