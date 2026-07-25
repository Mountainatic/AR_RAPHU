#!/usr/bin/env python
"""Benchmark legacy window KAN, vectorized window KAN, and sequence convolution.

The script uses a v17 selected checkpoint when present.  It reports numerical
and gradient equivalence before timing so a speedup is never accepted at the
cost of changing the model.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from stage1.model import Stage1TargetDelayKAN
from stage1.sequence_ops import ContiguousWindowBatch


def input_ranges(x: np.ndarray, indices) -> list[tuple[float, float]]:
    values = x[np.asarray(indices)].transpose(1, 0, 2).reshape(x.shape[1], -1)
    lo, hi = values.min(1), values.max(1)
    pad = np.maximum((hi - lo) * .05, .05)
    return [(float(a - p), float(b + p)) for a, b, p in zip(lo, hi, pad)]


def build_model(ranges, device):
    return Stage1TargetDelayKAN(
        10, 32, hidden_kan=8, kan_grid_size=7, epsilon=0.,
        active_mask=torch.ones(10, dtype=torch.bool),
        input_grid_ranges=ranges,
        response_execution_mode="vectorized",
    ).to(device)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn, device, *, warmup=2, repeats=10):
    for _ in range(warmup):
        fn()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - started) / repeats


def gradient_map(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--windows", type=int, default=None,
                        help="default: full train on CUDA, 128 on CPU")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "results_stage1/STAGE1_OPERATOR_ACCEL_V19/hotspot_benchmark.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    data = np.load(ROOT / "results_stage1/KAN_O1_v12/data_snapshot.npz")
    split = json.loads((ROOT / "results_stage1/KAN_O1_v12/split_manifest.json").read_text())
    train = np.asarray(split["train"])
    count = args.windows or (len(train) if device.type == "cuda" else min(128, len(train)))
    selected = train[:count]
    ranges = input_ranges(data["X"], train)
    model = build_model(ranges, device)
    checkpoint_path = ROOT / "results_stage1/M2_JOINT_RECOVERY_v17/M2_clean/checkpoints/seed_0/best.pt"
    checkpoint_loaded = checkpoint_path.exists()
    if checkpoint_loaded:
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
    model.eval()
    windows = torch.as_tensor(data["X"][selected], dtype=torch.float32, device=device)
    target = torch.as_tensor(data["y_clean"][selected], dtype=torch.float32, device=device)
    cached = ContiguousWindowBatch.from_windows(windows, target)

    with torch.no_grad():
        model.response_branches.execution_mode = "legacy"
        legacy_prediction = model(windows, return_aux=False)
        model.response_branches.execution_mode = "vectorized"
        vector_prediction = model(windows, return_aux=False)
        sequence_prediction = model.forward_contiguous(cached, return_aux=False)

    # Gradient equivalence on a smaller prefix keeps the audit cheap on CPU.
    gradient_count = min(32, count)
    gradient_windows = windows[:gradient_count]
    gradient_target = target[:gradient_count]
    gradient_cached = ContiguousWindowBatch.from_windows(gradient_windows, gradient_target)
    legacy_model = copy.deepcopy(model)
    sequence_model = copy.deepcopy(model)
    legacy_model.response_branches.execution_mode = "legacy"
    legacy_loss = (legacy_model(gradient_windows, return_aux=False).squeeze(-1) - gradient_target).square().mean()
    sequence_loss = (sequence_model.forward_contiguous(gradient_cached, return_aux=False).squeeze(-1) - gradient_target).square().mean()
    legacy_loss.backward(); sequence_loss.backward()
    gradient_errors = []
    for name in gradient_map(legacy_model):
        left = gradient_map(legacy_model)[name]
        right = gradient_map(sequence_model)[name]
        if left is None or right is None:
            continue
        gradient_errors.append(float((left - right).abs().max()))

    model.eval()
    repeats = args.repeats
    with torch.no_grad():
        model.response_branches.execution_mode = "legacy"
        legacy_seconds = benchmark(lambda: model(windows, return_aux=False), device,
                                   repeats=max(1, repeats // 3))
        model.response_branches.execution_mode = "vectorized"
        vector_seconds = benchmark(lambda: model(windows, return_aux=False), device,
                                   repeats=repeats)
        sequence_seconds = benchmark(
            lambda: model.forward_contiguous(cached, return_aux=False), device,
            repeats=repeats)

    report = {
        "device": str(device),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "checkpoint_loaded": checkpoint_loaded,
        "windows": count,
        "variables": int(windows.shape[1]),
        "lag": int(windows.shape[2]),
        "legacy_response_evaluations": int(count * windows.shape[1] * windows.shape[2]),
        "sequence_response_evaluations": int((count + windows.shape[2] - 1) * windows.shape[1]),
        "theoretical_response_reuse_factor": float(
            count * windows.shape[2] / (count + windows.shape[2] - 1)),
        "max_abs_prediction_error_vectorized_vs_legacy": float((vector_prediction - legacy_prediction).abs().max().cpu()),
        "max_abs_prediction_error_sequence_vs_legacy": float((sequence_prediction - legacy_prediction).abs().max().cpu()),
        "max_abs_gradient_error_sequence_vs_legacy": max(gradient_errors, default=0.0),
        "legacy_forward_seconds": legacy_seconds,
        "vectorized_window_forward_seconds": vector_seconds,
        "sequence_forward_seconds": sequence_seconds,
        "speedup_sequence_vs_legacy": legacy_seconds / sequence_seconds,
        "speedup_sequence_vs_vectorized_window": vector_seconds / sequence_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
