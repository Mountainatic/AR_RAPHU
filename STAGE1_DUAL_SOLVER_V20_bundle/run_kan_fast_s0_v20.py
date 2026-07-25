#!/usr/bin/env python
"""Formal V20 fast-KAN S0 line.

Protocol
--------
1. One exact sequence-first dense warmup per seed.
2. Every regularization scale independently forks from the same warmup state.
3. Every fork performs authoritative terminal pruning and fixed-support refit.
4. The final scale is selected across seeds using validation RMSE only.
5. Truth metrics and test metrics are reported after selection.

The script supports job-level execution so several tiny models can run
concurrently on one GPU to raise utilization.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from stage1.dual_metrics import (contribution_metric_rows, delay_metric_rows,
                                 function_metric_rows, prediction_metrics)
from stage1.experiment_utils import (cross_seed_one_se_select,
                                     input_ranges_from_train, read_json,
                                     save_json, seed_everything,
                                     support_truth_metrics, torch_environment,
                                     write_csv)
from stage1.independent_path import run_independent_fork
from stage1.model import Stage1TargetDelayKAN
from stage1.protocol import train_warmup
from stage1.sequence_ops import ContiguousWindowBatch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results_stage1/KAN_O1_v12/data_snapshot.npz"
SPLIT = ROOT / "results_stage1/KAN_O1_v12/split_manifest.json"
OUT = ROOT / "results_stage1/STAGE1_DUAL_SOLVER_V20/kan_fast"


def build_model(ranges, device):
    return Stage1TargetDelayKAN(
        10, 32, hidden_kan=8, kan_grid_size=7, epsilon=0.0,
        active_mask=torch.ones(10, dtype=torch.bool, device=device),
        input_grid_ranges=ranges, response_execution_mode="auto",
    ).to(device)


def load_context(device):
    z = np.load(DATA)
    x = z["X"].astype(np.float32)
    y_clean = z["y_clean"].astype(np.float32)
    y_observed = z["Y_observed"].reshape(-1).astype(np.float32)
    true_q = z["true_h"].astype(np.float64)
    split = read_json(SPLIT)
    train, val, test = [np.asarray(split[key], dtype=int) for key in ("train", "val", "test")]
    ranges = input_ranges_from_train(x, train)

    def batch(indices, target):
        windows = torch.as_tensor(x[indices], device=device)
        y = torch.as_tensor(target[indices], device=device)
        return ContiguousWindowBatch.from_windows(windows, y)

    return {
        "x": x, "y_clean": y_clean, "y_observed": y_observed, "true_q": true_q,
        "train_indices": train, "val_indices": val, "test_indices": test,
        "ranges": ranges,
        "train": batch(train, y_clean), "val": batch(val, y_clean), "test": batch(test, y_clean),
        "test_observed": torch.as_tensor(y_observed[test], device=device),
    }


def budgets(smoke: bool):
    if smoke:
        return dict(warmup_epochs=30, warmup_patience=30, prune_epochs=60,
                    ramp_epochs=10, full_penalty_min_epochs=10, stable_epochs=10,
                    refit_epochs=40, refit_patience=40,
                    validation_interval=5, diagnostic_interval=20)
    return dict(warmup_epochs=3000, warmup_patience=300, prune_epochs=1800,
                ramp_epochs=300, full_penalty_min_epochs=200, stable_epochs=80,
                refit_epochs=5000, refit_patience=500,
                validation_interval=5, diagnostic_interval=20)


def warmup_path(seed: int):
    return OUT / "warmup" / f"seed_{seed}" / "warmup.pt"


def fork_dir(seed: int, scale: float):
    return OUT / "forks" / f"seed_{seed}" / f"s0_{scale:.6f}"


def run_warmup(seed: int, device, smoke: bool):
    ctx, cfg = load_context(device), budgets(smoke)
    seed_everything(seed)
    model = build_model(ctx["ranges"], device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = train_warmup(
        model, ctx["train"], None, ctx["val"], None,
        epochs=cfg["warmup_epochs"], lr=0.003, patience=cfg["warmup_patience"],
        validation_interval=cfg["validation_interval"],
        diagnostic_interval=cfg["diagnostic_interval"],
        history_interval=cfg["validation_interval"],
    )
    elapsed = time.perf_counter() - started
    branch_norms = model.response_branches.compute_branch_norms().detach().cpu()
    path = warmup_path(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": result.best_state,
        "seed": seed,
        "median_branch_norm": float(branch_norms.median()),
        "branch_norms": branch_norms,
        "best_val_rmse": result.best_val_rmse,
        "best_epoch": result.best_epoch,
        "history": result.history,
        "config": cfg,
        "runtime_seconds": elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    torch.save(payload, path)
    save_json(path.parent / "summary.json", {
        "seed": seed, "best_val_rmse": result.best_val_rmse,
        "best_epoch": result.best_epoch, "median_branch_norm": float(branch_norms.median()),
        "runtime_seconds": elapsed, "peak_memory_bytes": payload["peak_memory_bytes"],
    })
    return path


def run_fork(seed: int, scale: float, device, smoke: bool):
    ctx, cfg = load_context(device), budgets(smoke)
    payload = torch.load(warmup_path(seed), map_location=device, weights_only=False)
    seed_everything(seed * 10000 + int(round(scale * 1_000_000)))
    model = build_model(ctx["ranges"], device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = run_independent_fork(
        model, payload["state_dict"], seed=seed, requested_scale=scale,
        median_warmup_branch_norm=payload["median_branch_norm"],
        x_train=ctx["train"], y_train=None, x_val=ctx["val"], y_val=None,
        lr=0.003, prune_epochs=cfg["prune_epochs"], ramp_epochs=cfg["ramp_epochs"],
        full_penalty_min_epochs=cfg["full_penalty_min_epochs"],
        stable_epochs=cfg["stable_epochs"], refit_epochs=cfg["refit_epochs"],
        refit_patience=cfg["refit_patience"],
        validation_interval=cfg["validation_interval"],
        diagnostic_interval=cfg["diagnostic_interval"],
        history_interval=cfg["validation_interval"],
    )
    model.load_state_dict(result.refit_state)
    model.eval()
    with torch.no_grad():
        pred_test = model.forward_contiguous(ctx["test"], return_aux=False).squeeze(-1)
        pred_train = model.forward_contiguous(ctx["train"], return_aux=False).squeeze(-1)
        learned_q = model.delay_prior().detach().cpu()
    elapsed = time.perf_counter() - started
    out = fork_dir(seed, scale)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": result.refit_state,
        "selection_mask": model.selection_mask.detach().cpu(),
        "learned_q": learned_q,
        "metadata": {
            "run_id": "STAGE1_DUAL_SOLVER_V20_KAN_FAST",
            "seed": seed, "requested_scale": scale,
            "lambda_group": result.lambda_group,
            "shared_warmup_independent_fork": True,
            "validation_only_configuration_selection": True,
            "sequence_fast_path": True,
        },
    }
    torch.save(checkpoint, out / "best.pt")
    row = result.summary()
    row.update({
        "train_rmse": float((pred_train - ctx["train"].target).square().mean().sqrt().cpu()),
        "test_clean_rmse": float((pred_test - ctx["test"].target).square().mean().sqrt().cpu()),
        "test_observed_rmse": float((pred_test - ctx["test_observed"]).square().mean().sqrt().cpu()),
        "runtime_seconds": elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "prune_epochs_actual": result.prune_history[-1]["epoch"] if result.prune_history else 0,
        "refit_epochs_actual": result.refit_history[-1]["epoch"] if result.refit_history else 0,
    })
    save_json(out / "summary.json", row)
    save_json(out / "prune_history.json", result.prune_history)
    save_json(out / "refit_history.json", result.refit_history)
    np.savez_compressed(out / "predictions.npz",
                        y_test_pred=pred_test.detach().cpu().numpy(),
                        y_test_clean=ctx["test"].target.detach().cpu().numpy(),
                        y_test_observed=ctx["test_observed"].detach().cpu().numpy(),
                        learned_q=learned_q.numpy())
    return row


def aggregate(seeds, scales, device):
    rows = []
    for seed in seeds:
        for scale in scales:
            path = fork_dir(seed, scale) / "summary.json"
            if path.exists():
                rows.append(read_json(path))
    expected = len(seeds) * len(scales)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} fork summaries, found {len(rows)}")
    selected = cross_seed_one_se_select(rows)
    selected_scale = float(selected["config_id"].split("=")[1])
    selected_rows = [r for r in rows if abs(float(r["requested_scale"]) - selected_scale) < 1e-12]
    ctx = load_context(device)
    support_rows, delay_rows, function_rows, contribution_rows = [], [], [], []
    predictions, learned_qs = [], []
    function_arrays, contribution_arrays = {}, {}

    for row in selected_rows:
        seed = int(row["seed"])
        checkpoint = torch.load(fork_dir(seed, selected_scale) / "best.pt",
                                map_location=device, weights_only=False)
        model = build_model(ctx["ranges"], device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            pred = model.forward_contiguous(ctx["test"], return_aux=False).squeeze(-1)
            q = model.delay_prior().detach().cpu().numpy()
        predictions.append(pred.cpu().numpy())
        learned_qs.append(q)
        support = list(map(int, row["terminal_support"]))
        support_rows.append({"seed": seed, "terminal_support": support,
                             "active_count": len(support), **support_truth_metrics(support),
                             "test_clean_rmse": row["test_clean_rmse"]})
        delay_rows.extend(delay_metric_rows(q, ctx["true_q"], seed=seed))
        fr, fa, lc, tc = function_metric_rows(
            model, ctx["train"].raw_sequence, ctx["ranges"], seed=seed, solver="kan")
        cr, ca = contribution_metric_rows(
            model, ctx["test"].windows, ctx["true_q"], seed=seed, solver="kan",
            learned_centers=lc, truth_centers=tc)
        function_rows.extend(fr); contribution_rows.extend(cr)
        function_arrays.update({f"seed{seed}_{k}": v for k, v in fa.items()})
        contribution_arrays.update({f"seed{seed}_{k}": v for k, v in ca.items()})

    out = OUT / "selected"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "all_fork_metrics.csv", rows)
    write_csv(out / "support_metrics.csv", support_rows)
    write_csv(out / "delay_metrics.csv", delay_rows)
    write_csv(out / "function_metrics.csv", function_rows)
    write_csv(out / "contribution_metrics.csv", contribution_rows)
    np.savez_compressed(out / "predictions.npz",
                        y_test_pred=np.stack(predictions),
                        y_test_clean=ctx["test"].target.cpu().numpy(),
                        y_test_observed=ctx["test_observed"].cpu().numpy())
    np.savez_compressed(out / "delay_distributions.npz",
                        learned_q=np.stack(learned_qs), true_q=ctx["true_q"])
    np.savez_compressed(out / "function_grids.npz", **function_arrays)
    np.savez_compressed(out / "contributions.npz", **contribution_arrays)
    save_json(out / "selected_config.json", selected)
    warmup_runtimes = [read_json(warmup_path(int(r["seed"])).parent / "summary.json")["runtime_seconds"]
                       for r in selected_rows]
    total_runtimes = [float(w) + float(r["runtime_seconds"])
                      for w, r in zip(warmup_runtimes, selected_rows)]
    summary = {
        "solver": "operator_accelerated_KAN",
        "selected_scale": selected_scale,
        "cross_seed_selection": selected,
        "mean_test_clean_rmse": float(np.mean([r["test_clean_rmse"] for r in selected_rows])),
        "std_test_clean_rmse": float(np.std([r["test_clean_rmse"] for r in selected_rows], ddof=1)) if len(selected_rows) > 1 else 0.0,
        "mean_active_count": float(np.mean([len(r["terminal_support"]) for r in selected_rows])),
        "exact_support_frequency": int(sum(r["exact_support"] for r in support_rows)),
        "mean_f1": float(np.mean([r["f1"] for r in support_rows])),
        "mean_runtime_warmup_seconds": float(np.mean(warmup_runtimes)),
        "mean_runtime_fork_seconds": float(np.mean([r["runtime_seconds"] for r in selected_rows])),
        "mean_runtime_seconds": float(np.mean(total_runtimes)),
        "total_warmup_compute_seconds": float(sum(warmup_runtimes)),
        "total_fork_search_compute_seconds": float(sum(float(r["runtime_seconds"]) for r in rows)),
        "total_search_compute_seconds": float(sum(warmup_runtimes) + sum(float(r["runtime_seconds"]) for r in rows)),
        "peak_memory_bytes_selected_jobs": int(max(int(r.get("peak_memory_bytes", 0)) for r in selected_rows)),
        "runtime_definition": "one shared warmup plus the selected independent prune/refit fork per seed; search totals reported separately",
        "environment": torch_environment(),
    }
    save_json(out / "summary.json", summary)
    return summary


def make_manifest(seeds, scales, smoke, device):
    manifests = OUT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    common = ["--device", str(device)] + (["--smoke"] if smoke else [])
    warmup_jobs = [{
        "job_id": f"kan_warmup_seed_{seed}",
        "command": ["python", "run_kan_fast_s0_v20.py", "--mode", "warmup", "--seed", str(seed), *common],
    } for seed in seeds]
    fork_jobs = [{
        "job_id": f"kan_fork_seed_{seed}_s0_{scale:.6f}",
        "command": ["python", "run_kan_fast_s0_v20.py", "--mode", "fork", "--seed", str(seed),
                    "--scale", str(scale), *common],
    } for seed in seeds for scale in scales]
    save_json(manifests / "warmup_jobs.json", {"jobs": warmup_jobs})
    save_json(manifests / "fork_jobs.json", {"jobs": fork_jobs})
    return manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["warmup", "fork", "aggregate", "all", "manifest"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.006)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--scales", default="0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.012")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    seeds = [int(v) for v in args.seeds.split(",") if v]
    scales = [float(v) for v in args.scales.split(",") if v]
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "warmup":
        print(run_warmup(args.seed, device, args.smoke))
    elif args.mode == "fork":
        print(json.dumps(run_fork(args.seed, args.scale, device, args.smoke), indent=2))
    elif args.mode == "aggregate":
        print(json.dumps(aggregate(seeds, scales, device), indent=2))
    elif args.mode == "manifest":
        print(make_manifest(seeds, scales, args.smoke, args.device))
    else:
        for seed in seeds:
            run_warmup(seed, device, args.smoke)
        for seed in seeds:
            for scale in scales:
                run_fork(seed, scale, device, args.smoke)
        print(json.dumps(aggregate(seeds, scales, device), indent=2))


if __name__ == "__main__":
    main()
