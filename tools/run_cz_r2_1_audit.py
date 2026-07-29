#!/usr/bin/env python3
"""S8 ORSS correctness audit, preserving the frozen CZ R2.1 gates."""

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
from ar_raphu.orss.diagnostics import write_json
from ar_raphu.orss.krylov import lsqr
from ar_raphu.orss.operator import UrysohnLinearOperator, build_urysohn_operator
from ar_raphu.orss.penalties import PenaltyWeights, SeparablePenalty


def _furnace_a(raw_dir: Path) -> Path:
    candidates = [path for path in raw_dir.iterdir() if path.name.endswith("1.xlsx")]
    if len(candidates) != 1:
        raise RuntimeError("Furnace-A path is not uniquely identifiable.")
    return candidates[0]


def _subset_operator(
    operator: UrysohnLinearOperator, indices: slice
) -> UrysohnLinearOperator:
    return UrysohnLinearOperator(
        operator.branches[indices],
        feature_mean=operator.feature_mean[indices],
        chunk_time=operator.chunk_time,
    )


def _fit_zero(
    operator: UrysohnLinearOperator, centered_target: torch.Tensor
) -> dict[str, float | int | bool]:
    penalty = SeparablePenalty(
        channels=operator.channels,
        m_tau=operator.m_tau,
        m_x=operator.m_x,
        device=operator.device,
        dtype=operator.dtype,
    )
    augmented = AugmentedRegularizedOperator(
        operator, penalty, PenaltyWeights(0.0, 0.0, 0.0)
    )
    result = lsqr(
        augmented,
        augmented.augmented_rhs(centered_target),
        relative_tolerance=1.0e-10,
        maximum_iterations=2500,
    )
    prediction = operator.forward(result.coefficients)
    return {
        "training_MSE_mm2": float(
            torch.mean((prediction - centered_target) ** 2).item()
        ),
        "relative_kkt_residual": result.relative_kkt_residual,
        "iterations": result.iterations,
        "converged": result.relative_kkt_residual <= 1.0e-8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--solver", choices=("orss",), default="orss")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--raw-dir",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw",
    )
    parser.add_argument(
        "--output",
        default="results/cz_real_data/complete_5090/R2_1",
    )
    parser.add_argument("--furnace-a-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    if not args.furnace_a_only:
        raise RuntimeError("FURNACE_B_ACCESSED_BEFORE_FREEZE")
    device = torch.device(config["runtime"]["device"])
    data = load_furnace_a(_furnace_a(Path(args.raw_dir)))
    legacy_path = Path("results/cz_real_v1/R2_1/R2_1_AUDIT_RESULTS.json")
    if not legacy_path.exists():
        raise RuntimeError("Frozen R2.1 alignment audit is missing.")
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    records = []
    for fold in build_development_folds(L_x=32, L_y=32):
        targets = target_indices(
            start=0,
            stop=fold.effective_train_stop,
            horizon=1,
            max_history=32,
        )
        operator, _ = build_urysohn_operator(
            data.inputs,
            data.target,
            target_indices=targets,
            train_target_stop=fold.effective_train_stop,
            horizon=1,
            L_x=32,
            L_y=32,
            lag_basis_count=16,
            amplitude_basis_count=32,
            continuation_scale_coefficient=1.0,
            device=device,
            dtype=torch.float64,
            chunk_time=int(config["cuda"]["operator_chunk_time"]),
        )
        target = torch.as_tensor(
            data.target[targets], device=device, dtype=torch.float64
        )
        target = target - target.mean()
        ar = _fit_zero(_subset_operator(operator, slice(-1, None)), target)
        x_only = _fit_zero(_subset_operator(operator, slice(0, -1)), target)
        xar = _fit_zero(operator, target)
        nestedness_margin = float(ar["training_MSE_mm2"]) - float(
            xar["training_MSE_mm2"]
        )
        records.append(
            {
                "fold": fold.fold,
                "AR": ar,
                "X_only": x_only,
                "XAR": xar,
                "AR_train_MSE_minus_XAR_train_MSE": nestedness_margin,
                "XAR_TRAINING_NESTEDNESS_PASS": nestedness_margin >= -1.0e-10,
            }
        )
        del operator
        torch.cuda.empty_cache()
    legacy_gates = legacy["gates"]
    gates = {
        "XAR_TRAINING_NESTEDNESS_PASS": all(
            bool(row["XAR_TRAINING_NESTEDNESS_PASS"]) for row in records
        ),
        "TARGET_ALIGNMENT_PASS": bool(legacy_gates["TARGET_ALIGNMENT_PASS"]),
        "NO_FUTURE_X_PASS": bool(legacy_gates["NO_FUTURE_X_PASS"]),
        "PURGE_PASS": bool(legacy_gates["PURGE_PASS"]),
        "CONTINUATION_TRAIN_IDENTITY_PASS": bool(
            legacy_gates["CONTINUATION_IN_SUPPORT_IDENTITY_PASS"]
        ),
        "FURNACE_B_ACCESS_COUNT_ZERO": True,
        "FINAL_KKT_PASS": all(
            max(
                float(row[model]["relative_kkt_residual"])
                for model in ("AR", "X_only", "XAR")
            )
            <= 1.0e-8
            for row in records
        ),
    }
    payload = {
        "schema": "CZ_R2_1_ORSS_AUDIT_V1",
        "status": "COMPLETED" if all(gates.values()) else "FAILED",
        "solver": "ORSS_ZERO_ENDPOINT_LSQR_MINIMUM_NORM",
        "folds": records,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - started,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
    }
    output = Path(args.output)
    write_json(output / "R2_1_ORSS_AUDIT_RESULTS.json", payload)
    write_json(
        output / "R2_1_STATUS.json",
        {
            "stage": "R2.1_NESTEDNESS_AND_ALIGNMENT_AUDIT",
            "status": payload["status"],
            "gates": gates,
            "next_stage": (
                "R3A_NATIVE_HISTORY_SELECTION"
                if payload["status"] == "COMPLETED"
                else "HARD_STOP"
            ),
        },
    )
    print(json.dumps(gates, indent=2))
    if args.strict and payload["status"] != "COMPLETED":
        raise RuntimeError("R2_1_PASS_FAILED")


if __name__ == "__main__":
    main()

