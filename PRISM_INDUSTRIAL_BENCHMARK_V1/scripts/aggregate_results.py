from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from prism_cpu.metrics import paired_block_bootstrap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--shared", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()
    pred_dir = args.results / "PREDICTIONS"
    rows = []
    for path in sorted(pred_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        for (task, split, model), g in df.groupby(["task", "split", "model"], sort=False):
            base = df[(df.task == task) & (df.split == split) & (df.model == "PERSISTENCE")]
            if len(base) != len(g):
                continue
            y = g.y_true.to_numpy(float)
            p = g.y_pred.to_numpy(float)
            b = base.y_pred.to_numpy(float)
            mse = float(np.mean((y - p) ** 2))
            bmse = float(np.mean((y - b) ** 2))
            r = {"task": task, "dataset": str(g.dataset.iloc[0]), "model": model, "split": split, "information_set": str(g.information_set.iloc[0]), "profile_id": str(g.profile_id.iloc[0]), "parameter_count": int(g.parameter_count.iloc[0]), "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y-p))), "n": int(len(g)), "relative_improvement_vs_baseline": float((bmse-mse)/(bmse+1e-15))}
            if split == "test" and model not in {"MEAN", "PERSISTENCE"}:
                r.update({f"bootstrap_{k}": v for k, v in paired_block_bootstrap(y, p, b, reps=args.bootstrap, block=32, seed=0).items()})
            rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(args.results / "FINAL_BASELINE_TABLE.csv", index=False)
    categories = {
        "SIMPLE_BASELINES.csv": table[table.model.isin(["MEAN", "PERSISTENCE"])],
        "CLASSICAL_SOFT_SENSOR.csv": table[table.model.isin(["RIDGE_X", "PLS_X", "HISTGB_X"])],
        "SYSTEM_IDENTIFICATION.csv": table[table.model.isin(["AR", "ARX"])],
        "PRISM_PROFILE_AUDIT.csv": table[table.model.str.startswith("PRISM_")],
        "PRISM_MODELS.csv": table[table.model.str.startswith("PRISM_")],
    }
    for name, sub in categories.items():
        sub.to_csv(args.results / name, index=False)
    summary = table[(table.split == "test") & table.model.str.startswith("PRISM_")].copy()
    summary.to_csv(args.results / "CHANNEL_TIMESCALE_SUMMARY.csv", index=False)
    (args.results / "RESULT_STATUS.json").write_text(json.dumps({"status": "BLOCKED_BY_MISSING_DATA", "completed_task_files": sorted(p.stem for p in pred_dir.glob("*.csv")), "blocked_tasks": ["SRU_H2S", "SRU_SO2"], "sample_cap": "50000 screening", "bootstrap_replicates": args.bootstrap}, indent=2), encoding="utf-8")
    lines = ["# PRISM CPU benchmark report", "", "Status: `BLOCKED_BY_MISSING_DATA` (SRU process table unavailable)", "", f"Completed prediction files: {', '.join(sorted(p.stem for p in pred_dir.glob('*.csv')))}", "Blocked tasks: `SRU_H2S`, `SRU_SO2`", "", "This is a CPU screening run with a deterministic 50,000-sample cap per task and 500 paired moving-block bootstrap replicates. It is not a full-data final claim.", "", "Selection uses validation loss only; test predictions are read after each task's fitting and are not used for hyperparameter selection.", "", "Raw source files remain outside this results directory."]
    (args.results / "CPU_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.results / "CPU_FINAL_DECISION.json").write_text(json.dumps({"status": "BLOCKED_BY_MISSING_DATA", "completed_tasks": sorted(p.stem for p in pred_dir.glob("*.csv")), "blocked_tasks": ["SRU_H2S", "SRU_SO2"], "protocol": "PRISM_INDUSTRIAL_BENCHMARK_V1", "screening_sample_cap": 50000, "bootstrap_replicates": args.bootstrap}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

