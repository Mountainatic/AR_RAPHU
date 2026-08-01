from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from prism_cpu.config import TASKS
from prism_cpu.features import attach_series, build_task_data
from prism_cpu.io import load_dataset, sha256_file
from prism_cpu.metrics import metric_row, paired_block_bootstrap
from prism_cpu.models import Fitted, fit_dynamic_models, fit_prism_models, fit_simple_models


def _save_predictions(path: Path, data, fitted: list[Fitted]) -> None:
    rows = []
    for m in fitted:
        for split, mask, pred in (("train", data.train, m.train_pred), ("validation", data.validation, m.val_pred), ("test", data.test, m.test_pred)):
            if len(pred) != int(mask.sum()):
                raise ValueError(f"prediction/mask mismatch for {m.name} {split}")
            idx = np.flatnonzero(mask)
            rows.append(pd.DataFrame({"sample_id": data.origins[idx], "dataset": data.dataset, "task": data.task_id, "split": split, "model": m.name, "y_true": data.y[mask], "y_pred": pred, "information_set": m.information_set, "profile_id": m.profile_id, "seed": 0, "dtype": "float64", "parameter_count": m.parameter_count}))
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def _metric_rows(data, fitted: list[Fitted], bootstrap_reps: int) -> list[dict]:
    base = next((m for m in fitted if m.name == "PERSISTENCE"), None)
    if base is None:
        raise RuntimeError("persistence baseline missing")
    rows = []
    for m in fitted:
        for split, mask, pred, bpred in (("train", data.train, m.train_pred, base.train_pred), ("validation", data.validation, m.val_pred, base.val_pred), ("test", data.test, m.test_pred, base.test_pred)):
            y = data.y[mask]
            met = metric_row(y, pred, baseline=bpred)
            row = {"task": data.task_id, "dataset": data.dataset, "model": m.name, "split": split, "information_set": m.information_set, "profile_id": m.profile_id, "parameter_count": m.parameter_count, **met}
            if split == "test" and m.name not in {"MEAN", "PERSISTENCE"}:
                row["bootstrap"] = paired_block_bootstrap(y, pred, bpred, reps=bootstrap_reps, block=max(8, data.horizon_steps), seed=0)
            rows.append(row)
    return rows


def _coefficients(model: object, path: Path, model_name: str) -> None:
    if model is None:
        return
    if isinstance(model, tuple):
        for i, m in enumerate(model):
            _coefficients(m, path / f"{model_name}_{i}", model_name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "class": type(model).__name__}
    if hasattr(model, "named_steps"):
        payload["steps"] = list(model.named_steps)
        est = list(model.named_steps.values())[-1]
        if hasattr(est, "coef_"):
            payload["coef_shape"] = list(np.asarray(est.coef_).shape)
            payload["coef_norm"] = float(np.linalg.norm(est.coef_))
        if hasattr(est, "alpha"):
            payload["alpha"] = float(est.alpha)
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-cap", type=int, default=50_000)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--task", action="append", default=[])
    args = ap.parse_args()
    out = args.output
    for d in ("PREDICTIONS", "KERNELS", "AR_PROFILES", "NUMERICAL_CERTIFICATES", "BOOTSTRAP", "RESOURCE_USAGE"):
        (out / d).mkdir(parents=True, exist_ok=True)
    requested = set(args.task)
    metric_rows = []
    task_status = {}
    started = time.time()
    dataset_cache = {}
    for spec in TASKS:
        if requested and spec.task_id not in requested:
            continue
        t0 = time.time()
        try:
            if spec.dataset not in dataset_cache:
                dataset_cache[spec.dataset] = load_dataset(spec.dataset, args.raw_root)
            ds = dataset_cache[spec.dataset]
            data = attach_series(build_task_data(ds, spec, sample_cap=args.sample_cap), ds)
            fitted = fit_simple_models(data) + fit_dynamic_models(data)
            prism, aux = fit_prism_models(data)
            fitted += prism
            task_metrics = _metric_rows(data, fitted, args.bootstrap)
            metric_rows.extend(task_metrics)
            pred_path = out / "PREDICTIONS" / f"{spec.task_id}.csv"
            _save_predictions(pred_path, data, fitted)
            for m in fitted:
                _coefficients(m.model, out / "KERNELS" / spec.task_id / m.name, m.name)
            np.savez_compressed(out / "AR_PROFILES" / f"{spec.task_id}.npz", origins=data.origins, split=data.split, horizon_steps=data.horizon_steps)
            metadata = {k: v for k, v in data.metadata.items() if not isinstance(v, np.ndarray)}
            task_status[spec.task_id] = {"status": "COMPLETED", "n_samples": int(len(data.y)), "counts": {s: int(np.sum(data.split == s)) for s in ("train", "validation", "test")}, "seconds": time.time() - t0, "horizon_steps": data.horizon_steps, "cadence_seconds": data.cadence_seconds, "metadata": metadata}
            print(f"COMPLETED {spec.task_id} n={len(data.y)} seconds={time.time()-t0:.1f}", flush=True)
        except Exception as exc:
            task_status[spec.task_id] = {"status": "BLOCKED_BY_MISSING_DATA" if isinstance(exc, FileNotFoundError) else "FAILED", "error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - t0}
            print(f"{task_status[spec.task_id]['status']} {spec.task_id}: {exc}", flush=True)
    if metric_rows:
        flat = []
        for row in metric_rows:
            r = {k: v for k, v in row.items() if k != "bootstrap"}
            if "bootstrap" in row:
                r.update({f"bootstrap_{k}": v for k, v in row["bootstrap"].items()})
            flat.append(r)
        pd.DataFrame(flat).to_csv(out / "FINAL_BASELINE_TABLE.csv", index=False)
    (out / "TASK_STATUS.json").write_text(json.dumps(task_status, indent=2, ensure_ascii=False), encoding="utf-8")
    resource = {"hostname": platform.node(), "python": platform.python_version(), "platform": platform.platform(), "pid": os.getpid(), "omp": os.environ.get("OMP_NUM_THREADS"), "mkl": os.environ.get("MKL_NUM_THREADS"), "openblas": os.environ.get("OPENBLAS_NUM_THREADS"), "n_jobs": os.environ.get("PRISM_N_JOBS", "1"), "seconds": time.time() - started, "dtype": "float64"}
    (out / "RESOURCE_USAGE" / "cpu_runtime.json").write_text(json.dumps(resource, indent=2), encoding="utf-8")
    completed = [k for k, v in task_status.items() if v["status"] == "COMPLETED"]
    blocked = [k for k, v in task_status.items() if v["status"].startswith("BLOCKED")]
    failed = [k for k, v in task_status.items() if v["status"] == "FAILED"]
    report = ["# PRISM CPU benchmark report", "", "Status: `COMPLETED`" if not failed else "Status: `FAILED`", "", f"Completed tasks: {', '.join(completed) or 'none'}", f"Blocked tasks: {', '.join(blocked) or 'none'}", f"Failed tasks: {', '.join(failed) or 'none'}", "", "All physical operators, metrics and saved predictions use float64. Test metrics are read only after model selection on validation.", "", "This run excludes raw source files from the results directory."]
    (out / "CPU_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision = {"status": "COMPLETED" if not failed else "FAILED", "completed_tasks": completed, "blocked_tasks": blocked, "failed_tasks": failed, "sample_cap": args.sample_cap, "bootstrap_reps": args.bootstrap, "protocol": "PRISM_INDUSTRIAL_BENCHMARK_V1"}
    (out / "CPU_FINAL_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
