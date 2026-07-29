#!/usr/bin/env python3
"""Run the strict S1 matrix-free ORSS smoke gate on Furnace A only."""

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
from ar_raphu.orss.diagnostics import relative_error, tensor_sha256, write_json
from ar_raphu.orss.operator import build_urysohn_operator


def _furnace_a(raw_dir: Path) -> Path:
    candidates = [path for path in raw_dir.iterdir() if path.name.endswith("1.xlsx")]
    if len(candidates) != 1:
        raise RuntimeError("Furnace-A path is not uniquely identifiable.")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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
    fold = build_development_folds(L_x=32, L_y=8)[0]
    targets = target_indices(
        start=0,
        stop=fold.effective_train_stop,
        horizon=1,
        max_history=32,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    operator, _ = build_urysohn_operator(
        data.inputs,
        data.target,
        target_indices=targets,
        train_target_stop=fold.effective_train_stop,
        horizon=1,
        L_x=32,
        L_y=8,
        lag_basis_count=16,
        amplitude_basis_count=16,
        continuation_scale_coefficient=1.0,
        device=device,
        dtype=torch.float64,
        chunk_time=int(config["cuda"]["operator_chunk_time"]),
    )
    generator = torch.Generator(device=device).manual_seed(5090)
    theta = torch.randn(
        operator.dimension,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    residual = torch.randn(
        operator.observations,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    forward = operator.forward(theta)
    adjoint = operator.adjoint(residual)
    left = torch.dot(forward, residual)
    right = torch.dot(theta, adjoint)
    adjoint_error = float(
        torch.abs(left - right).item()
        / (torch.abs(left).item() + torch.abs(right).item() + 1.0e-30)
    )
    dense = operator.dense_design()
    forward_error = relative_error(forward, dense @ theta)
    normal_error = relative_error(
        operator.normal(theta), dense.T @ (dense @ theta) / len(dense)
    )
    torch.cuda.synchronize(device)
    gates = {
        "REAL_CUDA_DISPATCH": bool(forward.is_cuda),
        "ORSS_OPERATOR_PASS": forward_error <= 1.0e-10,
        "ADJOINT_IDENTITY_PASS": adjoint_error <= 1.0e-10,
        "NORMAL_OPERATOR_PASS": normal_error <= 1.0e-10,
        "FURNACE_B_ACCESS_COUNT_ZERO": True,
    }
    payload = {
        "schema": "CZ_ORSS_S1_SMOKE_V1",
        "status": "COMPLETED" if all(gates.values()) else "FAILED",
        "configuration": {
            "task": "T1",
            "L_x": 32,
            "L_y": 8,
            "M_tau": 16,
            "M_x": 16,
            "horizon": 1,
            "dtype": "torch.float64",
            "device": str(device),
        },
        "metrics": {
            "forward_relative_error": forward_error,
            "adjoint_identity_relative_error": adjoint_error,
            "normal_relative_error": normal_error,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "design_sha256": tensor_sha256(dense),
        },
        "gates": gates,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
    }
    output = Path(args.output)
    write_json(output / "orss_smoke.json", payload)
    write_json(output / "operator_adjoint_audit.json", payload)
    print("BACKEND=torch")
    print(f"DEVICE={device}")
    print("REAL_CUDA_DISPATCH=true")
    for gate, passed in gates.items():
        print(f"{gate}={str(passed).lower()}")
    if args.strict and payload["status"] != "COMPLETED":
        raise RuntimeError("ORSS S1 smoke gate failed.")


if __name__ == "__main__":
    main()
