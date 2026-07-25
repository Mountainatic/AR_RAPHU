#!/usr/bin/env python
"""Formal V20 variational distributed-lag spline line.

The runner has a truth-blind two-stage search:

* screen: seed 0 evaluates the full spline/group/smoothness grid;
* formal: validation-selected candidates are run on five seeds;
* aggregate: a cross-seed one-standard-error rule fixes one configuration;
* noisy: the selected clean configuration is trained from scratch on observed
  labels without another structural grid search.

Each job is independent and can be dispatched concurrently to the same GPU.
"""
from __future__ import annotations

import argparse
import json
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
from stage1.variational_v20 import (VariationalDistributedLagSplineV20,
                                    fit_variational_v20,
                                    initialize_delay_from_seed)
from stage1.sequence_ops import windows_to_raw_sequence

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results_stage1/KAN_O1_v12/data_snapshot.npz"
SPLIT = ROOT / "results_stage1/KAN_O1_v12/split_manifest.json"
OUT = ROOT / "results_stage1/STAGE1_DUAL_SOLVER_V20/variational"


def load_context(device, target_kind="clean"):
    z = np.load(DATA)
    x = z["X"].astype(np.float32)
    y_clean = z["y_clean"].astype(np.float32)
    y_observed = z["Y_observed"].reshape(-1).astype(np.float32)
    true_q = z["true_h"].astype(np.float64)
    split = read_json(SPLIT)
    train, val, test = [np.asarray(split[key], dtype=int) for key in ("train", "val", "test")]
    target = y_clean if target_kind == "clean" else y_observed
    ranges = input_ranges_from_train(x, train)
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(target, device=device)
    return {
        "x": x, "windows": xt, "target": yt, "y_clean": y_clean,
        "y_observed": y_observed, "true_q": true_q,
        "train": train, "val": val, "test": test, "ranges": ranges,
    }


def config_id(grid_size, lambda_group, lambda_smooth, delay_solver="adam"):
    return f"{delay_solver}_g{int(grid_size)}_lg{lambda_group:.3e}_ls{lambda_smooth:.3e}"


def parse_config_id(value: str):
    pieces = value.split("_")
    return {
        "delay_solver": pieces[0],
        "grid_size": int(pieces[1][1:]),
        "lambda_group": float(pieces[2][2:]),
        "lambda_smooth": float(pieces[3][2:]),
    }


def job_dir(stage: str, target_kind: str, seed: int, cid: str):
    return OUT / stage / target_kind / f"seed_{seed}" / cid


def fit_budgets(smoke: bool):
    if smoke:
        return dict(outer_iterations=2, outer_patience=2, delay_steps=5,
                    fista_max_iter=80, fista_tolerance=1e-4, fista_kkt_tolerance=1e-2)
    return dict(outer_iterations=20, outer_patience=6, delay_steps=100,
                fista_max_iter=2500, fista_tolerance=1e-7, fista_kkt_tolerance=1e-5)


def run_job(seed: int, grid_size: int, lambda_group: float, lambda_smooth: float,
            device, *, stage: str, target_kind: str, smoke: bool, delay_solver: str = "adam"):
    ctx = load_context(device, target_kind)
    seed_everything(seed)
    model = VariationalDistributedLagSplineV20(
        10, 32, ctx["ranges"], grid_size=grid_size, spline_order=3).to(device)
    initialize_delay_from_seed(model, seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    basis_started = time.perf_counter()
    raw_train = windows_to_raw_sequence(ctx["windows"][ctx["train"]])
    raw_val = windows_to_raw_sequence(ctx["windows"][ctx["val"]])
    raw_test = windows_to_raw_sequence(ctx["windows"][ctx["test"]])
    seq_train = model.basis_bank.sequence_basis(raw_train)
    seq_val = model.basis_bank.sequence_basis(raw_val)
    seq_test = model.basis_bank.sequence_basis(raw_test)
    basis_train = model.basis_bank.lag_basis_from_sequence(seq_train, 32).detach()
    basis_val = model.basis_bank.lag_basis_from_sequence(seq_val, 32).detach()
    basis_test = model.basis_bank.lag_basis_from_sequence(seq_test, 32).detach()
    basis_seconds = time.perf_counter() - basis_started
    fit = fit_variational_v20(
        model, basis_train, ctx["target"][ctx["train"]],
        basis_val, ctx["target"][ctx["val"]],
        lambda_group=lambda_group, lambda_smooth=lambda_smooth,
        delay_lr=0.003, delay_solver=delay_solver, **fit_budgets(smoke))
    model.load_state_dict(fit.best_state)
    model.eval()
    with torch.no_grad():
        pred_train = model.predict_from_basis(basis_train)
        pred_val = model.predict_from_basis(basis_val)
        pred_test = model.predict_from_basis(basis_test)
        learned_q = model.delay_prior().detach().cpu().numpy()
    runtime = time.perf_counter() - started
    cid = config_id(grid_size, lambda_group, lambda_smooth, delay_solver)
    out = job_dir(stage, target_kind, seed, cid)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "selection_mask": model.selection_mask.detach().cpu(),
        "learned_q": torch.as_tensor(learned_q),
        "metadata": {
            "run_id": "STAGE1_DUAL_SOLVER_V20_VARIATIONAL",
            "seed": seed, "config_id": cid, "target_kind": target_kind,
            "delay_solver": delay_solver,
            "validation_only_configuration_selection": True,
            "solver": "monotone_FISTA_plus_low_dimensional_Gamma_block",
        },
    }
    torch.save(checkpoint, out / "best.pt")
    support = torch.where(model.selection_mask)[0].cpu().tolist()
    train_metrics = prediction_metrics(ctx["target"][ctx["train"]].cpu().numpy(), pred_train.cpu().numpy())
    val_metrics = prediction_metrics(ctx["target"][ctx["val"]].cpu().numpy(), pred_val.cpu().numpy())
    test_clean_metrics = prediction_metrics(ctx["y_clean"][ctx["test"]], pred_test.cpu().numpy())
    test_observed_metrics = prediction_metrics(ctx["y_observed"][ctx["test"]], pred_test.cpu().numpy())
    last_outer = fit.outer_history[-1]
    row = {
        "seed": seed, "config_id": cid, "grid_size": grid_size,
        "delay_solver": delay_solver,
        "lambda_group": lambda_group, "lambda_smooth": lambda_smooth,
        "target_kind": target_kind,
        "terminal_support": support, "active_count": len(support),
        "refit_val_rmse": fit.best_val_rmse,
        "train_rmse": train_metrics["rmse"],
        "test_clean_rmse": test_clean_metrics["rmse"],
        "test_observed_rmse": test_observed_metrics["rmse"],
        "best_outer_iteration": fit.best_outer_iteration,
        "outer_iterations_actual": len(fit.outer_history),
        "fista_iterations_last": int(last_outer["fista_iterations"]),
        "fista_converged_last": bool(last_outer["fista_converged"]),
        "kkt_residual_last": float(last_outer["kkt_residual"]),
        "converged": bool(fit.converged),
        "stable": bool(fit.converged),
        "basis_seconds": basis_seconds,
        "runtime_seconds": runtime,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    save_json(out / "summary.json", row)
    write_csv(out / "outer_history.csv", fit.outer_history)
    write_csv(out / "fista_history.csv", fit.fista_history)
    np.savez_compressed(out / "predictions.npz",
                        y_train_pred=pred_train.cpu().numpy(),
                        y_val_pred=pred_val.cpu().numpy(),
                        y_test_pred=pred_test.cpu().numpy(),
                        y_test_clean=ctx["y_clean"][ctx["test"]],
                        y_test_observed=ctx["y_observed"][ctx["test"]],
                        learned_q=learned_q,
                        coefficients=model.coefficients.cpu().numpy(),
                        knots=model.basis_bank.grids.cpu().numpy(),
                        roughness_matrix=model.basis_bank.roughness.cpu().numpy())
    return row


def search_space(smoke: bool):
    if smoke:
        grids, groups, smooths = [8], [1e-3, 1e-2], [0.0]
    else:
        grids = [8, 12, 16]
        groups = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        smooths = [0.0, 1e-4, 1e-3]
    configs = [{"delay_solver": "adam", "grid_size": g, "lambda_group": lg,
                "lambda_smooth": ls, "config_id": config_id(g, lg, ls, "adam")}
               for g in grids for lg in groups for ls in smooths]
    if not smoke:
        # Focused variable-projection-style L-BFGS probes, avoiding a doubled
        # Cartesian grid while still testing a second outer solver.
        for lg in [1e-3, 3e-3, 1e-2]:
            for ls in [0.0, 1e-3]:
                configs.append({"delay_solver": "lbfgs", "grid_size": 12,
                                "lambda_group": lg, "lambda_smooth": ls,
                                "config_id": config_id(12, lg, ls, "lbfgs")})
    return configs


def collect_rows(stage, target_kind):
    rows = []
    root = OUT / stage / target_kind
    if root.exists():
        for path in sorted(root.glob("seed_*/*/summary.json")):
            rows.append(read_json(path))
    return rows


def select_screen_candidates(smoke=False, top_k=8):
    rows = collect_rows("screen", "clean")
    if not rows:
        raise RuntimeError("screen results are missing")
    rows.sort(key=lambda r: (r["refit_val_rmse"], r["active_count"], r["kkt_residual_last"]))
    best = rows[0]["refit_val_rmse"]
    candidates = {r["config_id"] for r in rows[:top_k]}
    candidates |= {r["config_id"] for r in rows
                   if r["refit_val_rmse"] <= 1.5 * best and r["active_count"] <= 6}
    # Preserve at least the best configuration for every observed active count.
    by_count = {}
    for row in rows:
        by_count.setdefault(row["active_count"], row["config_id"])
    candidates |= set(by_count.values())
    selected = sorted(candidates)
    save_json(OUT / "screen_selection.json", {
        "selection_uses_truth": False,
        "minimum_seed0_validation_rmse": best,
        "candidate_config_ids": selected,
        "screen_rows": rows,
    })
    return selected


def aggregate_formal(device, seeds, target_kind="clean"):
    rows = collect_rows("formal", target_kind)
    if not rows:
        raise RuntimeError("formal results are missing")
    selected = cross_seed_one_se_select(rows)
    cid = selected["config_id"]
    chosen = [r for r in rows if r["config_id"] == cid]
    if len(chosen) != len(seeds):
        raise RuntimeError(f"selected config has {len(chosen)} seeds, expected {len(seeds)}")
    cfg = parse_config_id(cid)
    ctx = load_context(device, target_kind)
    support_rows, delay_rows, function_rows, contribution_rows = [], [], [], []
    predictions, qs, coefficients = [], [], []
    function_arrays, contribution_arrays = {}, {}
    for row in chosen:
        seed = int(row["seed"])
        checkpoint = torch.load(job_dir("formal", target_kind, seed, cid) / "best.pt",
                                map_location=device, weights_only=False)
        model = VariationalDistributedLagSplineV20(
            10, 32, ctx["ranges"], grid_size=cfg["grid_size"], spline_order=3).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            basis_test = model.basis(ctx["windows"][ctx["test"]])
            pred = model.predict_from_basis(basis_test)
            q = model.delay_prior().cpu().numpy()
        support = torch.where(model.selection_mask)[0].cpu().tolist()
        truth = support_truth_metrics(support)
        support_rows.append({"seed": seed, "terminal_support": support,
                             "active_count": len(support), **truth,
                             "test_clean_rmse": row["test_clean_rmse"]})
        delay_rows.extend(delay_metric_rows(q, ctx["true_q"], seed=seed))
        raw_train = torch.as_tensor(ctx["x"][ctx["train"][0], :, :], device=device)  # placeholder overwritten below
        # Reconstruct unique train sequence from contiguous windows.
        raw_train = windows_to_raw_sequence(ctx["windows"][ctx["train"]])
        fr, fa, lc, tc = function_metric_rows(
            model, raw_train, ctx["ranges"], seed=seed, solver="variational")
        cr, ca = contribution_metric_rows(
            model, ctx["windows"][ctx["test"]], ctx["true_q"], seed=seed,
            solver="variational", learned_centers=lc, truth_centers=tc)
        function_rows.extend(fr); contribution_rows.extend(cr)
        function_arrays.update({f"seed{seed}_{k}": v for k, v in fa.items()})
        contribution_arrays.update({f"seed{seed}_{k}": v for k, v in ca.items()})
        predictions.append(pred.cpu().numpy()); qs.append(q)
        coefficients.append(model.coefficients.cpu().numpy())

    out = OUT / "selected" / target_kind
    out.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / f"formal_{target_kind}_all_config_metrics.csv", rows)
    write_csv(out / "seed_metrics.csv", chosen)
    write_csv(out / "support_metrics.csv", support_rows)
    write_csv(out / "delay_metrics.csv", delay_rows)
    write_csv(out / "function_metrics.csv", function_rows)
    write_csv(out / "contribution_metrics.csv", contribution_rows)
    np.savez_compressed(out / "predictions.npz", y_test_pred=np.stack(predictions),
                        y_test_clean=ctx["y_clean"][ctx["test"]],
                        y_test_observed=ctx["y_observed"][ctx["test"]])
    np.savez_compressed(out / "delay_distributions.npz", learned_q=np.stack(qs), true_q=ctx["true_q"])
    np.savez_compressed(out / "spline_coefficients.npz", coefficients=np.stack(coefficients),
                        knots=model.basis_bank.grids.cpu().numpy(),
                        roughness_matrix=model.basis_bank.roughness.cpu().numpy())
    np.savez_compressed(out / "function_grids.npz", **function_arrays)
    np.savez_compressed(out / "contributions.npz", **contribution_arrays)
    save_json(out / "selected_config.json", {**selected, **cfg})
    summary = {
        "solver": "variational_distributed_lag_spline",
        "target_kind": target_kind,
        "selected_config": {**selected, **cfg},
        "mean_test_clean_rmse": float(np.mean([r["test_clean_rmse"] for r in chosen])),
        "std_test_clean_rmse": float(np.std([r["test_clean_rmse"] for r in chosen], ddof=1)) if len(chosen) > 1 else 0.0,
        "mean_test_observed_rmse": float(np.mean([r["test_observed_rmse"] for r in chosen])),
        "mean_f1": float(np.mean([r["f1"] for r in support_rows])),
        "exact_support_frequency": int(sum(r["exact_support"] for r in support_rows)),
        "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in chosen])),
        "total_formal_search_compute_seconds": float(sum(float(r["runtime_seconds"]) for r in rows)),
        "selected_config_compute_seconds": float(sum(float(r["runtime_seconds"]) for r in chosen)),
        "peak_memory_bytes_selected_jobs": int(max(int(r.get("peak_memory_bytes", 0)) for r in chosen)),
        "all_fista_converged": bool(all(r["converged"] for r in chosen)),
        "environment": torch_environment(),
    }
    save_json(out / "summary.json", summary)
    return summary


def make_manifests(seeds, smoke, device):
    manifests = OUT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    common = ["--device", str(device)] + (["--smoke"] if smoke else [])
    screen_jobs = []
    for cfg in search_space(smoke):
        screen_jobs.append({
            "job_id": f"var_screen_seed0_{cfg['config_id']}",
            "command": ["python", "run_variational_stage1_s0_v20.py", "--mode", "job",
                        "--stage", "screen", "--target", "clean", "--seed", "0",
                        "--grid-size", str(cfg["grid_size"]),
                        "--lambda-group", str(cfg["lambda_group"]),
                        "--lambda-smooth", str(cfg["lambda_smooth"]),
                        "--delay-solver", cfg["delay_solver"], *common],
        })
    save_json(manifests / "variational_screen_jobs.json", {"jobs": screen_jobs})
    return manifests


def make_formal_manifest(seeds, smoke, device):
    candidates = select_screen_candidates(smoke)
    manifests = OUT / "manifests"
    common = ["--device", str(device)] + (["--smoke"] if smoke else [])
    jobs = []
    for seed in seeds:
        for cid in candidates:
            cfg = parse_config_id(cid)
            jobs.append({
                "job_id": f"var_formal_seed{seed}_{cid}",
                "command": ["python", "run_variational_stage1_s0_v20.py", "--mode", "job",
                            "--stage", "formal", "--target", "clean", "--seed", str(seed),
                            "--grid-size", str(cfg["grid_size"]),
                            "--lambda-group", str(cfg["lambda_group"]),
                            "--lambda-smooth", str(cfg["lambda_smooth"]),
                        "--delay-solver", cfg["delay_solver"], *common],
            })
    save_json(manifests / "variational_formal_jobs.json", {"jobs": jobs})
    return manifests / "variational_formal_jobs.json"


def make_noisy_manifest(seeds, smoke, device):
    selected = read_json(OUT / "selected" / "clean" / "selected_config.json")
    cfg = {k: selected[k] for k in ("grid_size", "lambda_group", "lambda_smooth", "delay_solver")}
    cid = config_id(**cfg)
    common = ["--device", str(device)] + (["--smoke"] if smoke else [])
    jobs = [{
        "job_id": f"var_noisy_seed{seed}_{cid}",
        "command": ["python", "run_variational_stage1_s0_v20.py", "--mode", "job",
                    "--stage", "formal", "--target", "observed", "--seed", str(seed),
                    "--grid-size", str(cfg["grid_size"]),
                    "--lambda-group", str(cfg["lambda_group"]),
                    "--lambda-smooth", str(cfg["lambda_smooth"]),
                    "--delay-solver", cfg["delay_solver"], *common],
    } for seed in seeds]
    path = OUT / "manifests" / "variational_noisy_jobs.json"
    save_json(path, {"jobs": jobs})
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["job", "manifest", "select-screen", "aggregate", "make-noisy", "all"], default="all")
    p.add_argument("--stage", choices=["screen", "formal"], default="screen")
    p.add_argument("--target", choices=["clean", "observed"], default="clean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--lambda-group", type=float, default=1e-4)
    p.add_argument("--lambda-smooth", type=float, default=1e-4)
    p.add_argument("--delay-solver", choices=["adam", "lbfgs"], default="adam")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device)
    seeds = [int(v) for v in args.seeds.split(",") if v]
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "job":
        print(json.dumps(run_job(args.seed, args.grid_size, args.lambda_group,
                                 args.lambda_smooth, device, stage=args.stage,
                                 target_kind=args.target, smoke=args.smoke,
                                 delay_solver=args.delay_solver), indent=2))
    elif args.mode == "manifest":
        print(make_manifests(seeds, args.smoke, args.device))
    elif args.mode == "select-screen":
        print(make_formal_manifest(seeds, args.smoke, args.device))
    elif args.mode == "aggregate":
        print(json.dumps(aggregate_formal(device, seeds, args.target), indent=2))
    elif args.mode == "make-noisy":
        print(make_noisy_manifest(seeds, args.smoke, args.device))
    else:
        # Sequential fallback for CPU/smoke. Full GPU runs should use manifests.
        for cfg in search_space(args.smoke):
            run_job(0, cfg["grid_size"], cfg["lambda_group"], cfg["lambda_smooth"],
                    device, stage="screen", target_kind="clean", smoke=args.smoke,
                    delay_solver=cfg["delay_solver"])
        candidates = select_screen_candidates(args.smoke)
        for seed in seeds:
            for cid in candidates:
                cfg = parse_config_id(cid)
                run_job(seed, cfg["grid_size"], cfg["lambda_group"], cfg["lambda_smooth"],
                        device, stage="formal", target_kind="clean", smoke=args.smoke,
                        delay_solver=cfg["delay_solver"])
        print(json.dumps(aggregate_formal(device, seeds, "clean"), indent=2))
        selected = read_json(OUT / "selected" / "clean" / "selected_config.json")
        cfg = {k: selected[k] for k in ("grid_size", "lambda_group", "lambda_smooth", "delay_solver")}
        for seed in seeds:
            run_job(seed, cfg["grid_size"], cfg["lambda_group"], cfg["lambda_smooth"],
                    device, stage="formal", target_kind="observed", smoke=args.smoke,
                    delay_solver=cfg["delay_solver"])
        print(json.dumps(aggregate_formal(device, seeds, "observed"), indent=2))


if __name__ == "__main__":
    main()
