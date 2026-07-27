#!/usr/bin/env python3
"""Non-scientific PWH CPU-FP64 full-kernel integration smoke."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ar_raphu.datasets.loaders import load_pwh
from ar_raphu.spectral.design import build_spectral_design
from ar_raphu.spectral.penalties import tensor_penalty
from ar_raphu.spectral.solver import solve_full_kernel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/public_benchmarks/pb1/smoke/pwh_full_kernel.json"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    dataset = load_pwh(args.raw_root, include_test=False)
    first_sequence = np.unique(dataset.sequence_id)[0]
    sequence_rows = np.flatnonzero(dataset.sequence_id == first_sequence)
    x = np.asarray(dataset.x[sequence_rows], dtype=np.float64)
    y = np.asarray(dataset.y[sequence_rows, 0], dtype=np.float64)
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0)
    y_mean, y_scale = float(y.mean()), float(y.std())
    x = (x - x_mean) / x_scale
    y = (y - y_mean) / y_scale

    horizon = 1
    history = 16
    target_stop = min(len(x), 4096)
    target_indices = np.arange(history - 1 + horizon, target_stop)
    design = build_spectral_design(
        x,
        target_indices=target_indices,
        train_target_stop=target_stop,
        horizon=horizon,
        L_x=history,
        lag_basis_count=16,
        amplitude_basis_count=16,
    )
    penalty = tensor_penalty(
        design.lag_gram,
        design.amplitude_grams,
        lag_smoothness=1.0e-3,
        amplitude_smoothness=1.0e-3,
        ridge_weight=1.0e-4,
    )
    fit = solve_full_kernel(
        design.matrix,
        y[target_indices],
        penalty,
        fit_intercept=True,
    )
    residual = fit.predictions - y[target_indices]
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "stage": "SMOKE",
        "scientific_evidence": False,
        "dataset": "pwh",
        "official_test_accessed": False,
        "track": "X",
        "horizon": horizon,
        "history": history,
        "record": str(first_sequence),
        "n_samples": int(len(target_indices)),
        "design_shape": list(design.matrix.shape),
        "dtype": str(design.matrix.dtype),
        "device": "cpu",
        "relative_kkt_residual": fit.relative_kkt_residual,
        "condition_number": fit.condition_number,
        "train_rmse_standardized": float(np.sqrt(np.mean(residual * residual))),
        "wall_seconds": time.perf_counter() - started,
        "status": (
            "COMPLETED"
            if fit.relative_kkt_residual <= 1.0e-8
            else "FAILED"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
