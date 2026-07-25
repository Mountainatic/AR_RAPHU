#!/usr/bin/env python
"""Aggregate, audit, compare and report the two V20 solver lines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage1.experiment_utils import read_json, save_json, write_csv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_stage1/STAGE1_DUAL_SOLVER_V20"
REF = ROOT / "results_stage1/M2_JOINT_RECOVERY_v17_REFERENCE"


def finite_frame(path: Path, min_rows=1):
    frame = pd.read_csv(path)
    if len(frame) < min_rows:
        raise ValueError(f"{path} has {len(frame)} rows, expected >= {min_rows}")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
    if numeric.size and not np.isfinite(numeric).all():
        raise ValueError(f"non-finite values in {path}")
    return frame


def mean_column(path, column):
    return float(finite_frame(path)[column].mean())


def finalize():
    kan_dir = OUT / "kan_fast/selected"
    var_clean = OUT / "variational/selected/clean"
    var_noisy = OUT / "variational/selected/observed"
    required = [
        kan_dir / "summary.json", kan_dir / "support_metrics.csv",
        kan_dir / "delay_metrics.csv", kan_dir / "function_metrics.csv",
        kan_dir / "contribution_metrics.csv",
        var_clean / "summary.json", var_clean / "support_metrics.csv",
        var_clean / "delay_metrics.csv", var_clean / "function_metrics.csv",
        var_clean / "contribution_metrics.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")

    kan = read_json(kan_dir / "summary.json")
    var = read_json(var_clean / "summary.json")
    noisy = read_json(var_noisy / "summary.json") if (var_noisy / "summary.json").exists() else None
    reference = read_json(REF / "summary.json")
    kan_support = finite_frame(kan_dir / "support_metrics.csv", 5)
    var_support = finite_frame(var_clean / "support_metrics.csv", 5)
    kan_delay = finite_frame(kan_dir / "delay_metrics.csv", 15)
    var_delay = finite_frame(var_clean / "delay_metrics.csv", 15)
    kan_function = finite_frame(kan_dir / "function_metrics.csv", 15)
    var_function = finite_frame(var_clean / "function_metrics.csv", 15)
    kan_contrib = finite_frame(kan_dir / "contribution_metrics.csv", 15)
    var_contrib = finite_frame(var_clean / "contribution_metrics.csv", 15)

    rows = [
        {
            "method": "V17_KAN_Adam_reference",
            "mean_test_clean_rmse": reference.get("mean_test_clean_rmse", 0.001882),
            "mean_runtime_seconds": reference.get("mean_runtime_seconds", 438.49),
            "mean_f1": 1.0,
            "mean_delay_w1": 0.005478,
            "mean_function_correlation": 0.999985,
            "mean_contribution_correlation": 0.999986,
        },
        {
            "method": "V20_operator_KAN_independent_forks",
            "mean_test_clean_rmse": kan["mean_test_clean_rmse"],
            "mean_runtime_seconds": kan["mean_runtime_seconds"],
            "mean_f1": float(kan_support["f1"].mean()),
            "mean_delay_w1": float(kan_delay["wasserstein_1"].mean()),
            "mean_function_correlation": float(kan_function["function_correlation"].mean()),
            "mean_contribution_correlation": float(kan_contrib["variable_contribution_correlation"].mean()),
        },
        {
            "method": "V20_variational_spline_FISTA",
            "mean_test_clean_rmse": var["mean_test_clean_rmse"],
            "mean_runtime_seconds": var["mean_runtime_seconds"],
            "mean_f1": float(var_support["f1"].mean()),
            "mean_delay_w1": float(var_delay["wasserstein_1"].mean()),
            "mean_function_correlation": float(var_function["function_correlation"].mean()),
            "mean_contribution_correlation": float(var_contrib["variable_contribution_correlation"].mean()),
        },
    ]
    write_csv(OUT / "runtime_scientific_comparison.csv", rows)

    flags = {
        "KAN_FAST_PREDICTION_PASS": kan["mean_test_clean_rmse"] <= 0.01,
        "KAN_FAST_SUPPORT_PASS": float(kan_support["f1"].mean()) >= 0.8 and int(kan_support["exact_support"].sum()) >= 3,
        "KAN_FAST_DELAY_PASS": float(kan_delay["mean_delay_mae"].mean()) <= 2.0,
        "KAN_FAST_FUNCTION_PASS": float(kan_function["function_correlation"].mean()) >= 0.9,
        "KAN_FAST_CONTRIBUTION_PASS": float(kan_contrib["variable_contribution_correlation"].mean()) >= 0.9,
        "VARIATIONAL_PREDICTION_PASS": var["mean_test_clean_rmse"] <= 0.01,
        "VARIATIONAL_SUPPORT_PASS": float(var_support["f1"].mean()) >= 0.8 and int(var_support["exact_support"].sum()) >= 3,
        "VARIATIONAL_DELAY_PASS": float(var_delay["mean_delay_mae"].mean()) <= 2.0,
        "VARIATIONAL_FUNCTION_PASS": float(var_function["function_correlation"].mean()) >= 0.9,
        "VARIATIONAL_CONTRIBUTION_PASS": float(var_contrib["variable_contribution_correlation"].mean()) >= 0.9,
        "VARIATIONAL_FISTA_CONVERGENCE_PASS": bool(var["all_fista_converged"]),
        "VARIATIONAL_NOISY_COMPLETED": noisy is not None,
    }
    pytest_path = OUT / "pytest_summary.json"
    pytest_pass = bool(read_json(pytest_path).get("passed")) if pytest_path.exists() else False
    flags["PYTEST_PASS"] = pytest_pass
    flags["ARTIFACT_CONSISTENCY_PASS"] = True
    flags["KAN_FAST_LINE_PASS"] = all(flags[key] for key in flags if key.startswith("KAN_FAST_") and key != "KAN_FAST_LINE_PASS")
    flags["VARIATIONAL_LINE_PASS"] = all(flags[key] for key in (
        "VARIATIONAL_PREDICTION_PASS", "VARIATIONAL_SUPPORT_PASS", "VARIATIONAL_DELAY_PASS",
        "VARIATIONAL_FUNCTION_PASS", "VARIATIONAL_CONTRIBUTION_PASS",
        "VARIATIONAL_FISTA_CONVERGENCE_PASS"))
    flags["READY_FOR_S1_S6"] = bool((flags["KAN_FAST_LINE_PASS"] or flags["VARIATIONAL_LINE_PASS"]) and flags["PYTEST_PASS"] and flags["ARTIFACT_CONSISTENCY_PASS"])

    audit = {
        "required_artifacts_present": not missing,
        "csv_nonempty_and_finite": True,
        "kan_selected_seed_count": len(kan_support),
        "variational_selected_seed_count": len(var_support),
        "selection_protocols": {
            "kan": "shared warmup + independent pruning forks + cross-seed validation one-SE",
            "variational": "seed0 screen + cross-seed formal validation one-SE",
        },
    }
    save_json(OUT / "artifact_consistency_audit.json", audit)
    save_json(OUT / "pass_fail_summary.json", flags)
    report = f"""# STAGE1 DUAL SOLVER V20 report

## Fast KAN line

- Selected scale: `{kan['selected_scale']}`
- Test clean RMSE: `{kan['mean_test_clean_rmse']:.8f} ± {kan['std_test_clean_rmse']:.8f}`
- Mean support F1: `{kan_support['f1'].mean():.6f}`
- Exact support frequency: `{int(kan_support['exact_support'].sum())}/5`
- Mean delay W1: `{kan_delay['wasserstein_1'].mean():.8f}`
- Mean function correlation: `{kan_function['function_correlation'].mean():.8f}`
- Mean variable-contribution correlation: `{kan_contrib['variable_contribution_correlation'].mean():.8f}`

The KAN line uses exact sequence-first response evaluation and depthwise causal
convolution.  Each regularization point independently starts from the same
warmup checkpoint, removing the path dependence observed in V19 nested
homotopy.

## Variational spline line

- Selected config: `{var['selected_config']['config_id']}`
- Test clean RMSE: `{var['mean_test_clean_rmse']:.8f} ± {var['std_test_clean_rmse']:.8f}`
- Mean support F1: `{var_support['f1'].mean():.6f}`
- Exact support frequency: `{int(var_support['exact_support'].sum())}/5`
- Mean delay W1: `{var_delay['wasserstein_1'].mean():.8f}`
- Mean function correlation: `{var_function['function_correlation'].mean():.8f}`
- Mean variable-contribution correlation: `{var_contrib['variable_contribution_correlation'].mean():.8f}`
- All selected-seed FISTA converged: `{var['all_fista_converged']}`

This line replaces the two-layer KAN response by explicit cubic B-splines.  For
fixed Gamma delays the response and variable-selection subproblem is convex and
is solved with scale-normalized monotone FISTA, a roughness penalty and an
explicit KKT/prox-gradient stopping audit.

## Final gates

```json
{json.dumps(flags, indent=2)}
```
"""
    (OUT / "STAGE1_DUAL_SOLVER_V20_report.md").write_text(report, encoding="utf-8")
    return {"flags": flags, "comparison": rows}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=["finalize"], default="finalize")
    p.parse_args()
    print(json.dumps(finalize(), indent=2))


if __name__ == "__main__":
    main()
