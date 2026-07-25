#!/usr/bin/env python3
"""Post-selection truth and implementation audit for Phase-1 E3 M6."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.protocol_config import load_protocol_config
from ar_raphu.runtime_environment import require_runtime_environment
from ar_raphu.synthetic import generate_synthetic_sequence


EXPERIMENT_ROOT = PROJECT_ROOT / "results" / "phase1" / "E3_AR-S2_G2"
M5_ROOT = EXPERIMENT_ROOT / "Track-XAR"
M6_ROOT = EXPERIMENT_ROOT / "M6"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    require_runtime_environment()
    config = load_protocol_config(require_phase1_frozen=True)
    selection = read_json(M6_ROOT / "validation_selection.json")
    test = read_json(M6_ROOT / "test_metrics.json")
    if selection.get("test_used") is not False:
        raise RuntimeError("M6 selection was not validation-only.")
    if not test.get("hyperparameters_frozen_before_test"):
        raise RuntimeError("M6 test metadata does not prove post-freeze access.")

    seeds = [int(seed) for seed in config["training"]["seeds"]["screening"]]
    weights = [
        float(value)
        for value in config["phase1_model_selection"]["M6"][
            "second_difference_smoothness_weights"
        ]
    ]
    summaries = [
        read_json(
            M6_ROOT
            / f"seed_{seed}"
            / f"lambda_{weight:.8g}"
            / "summary.json"
        )
        for seed in seeds
        for weight in weights
    ]
    if any(summary.get("test_accessed") is not False for summary in summaries):
        raise RuntimeError("At least one M6 candidate accessed test.")
    maximum_prediction_error = max(
        summary["initialization_audit"]["prediction_max_abs_error_scaled"]
        for summary in summaries
    )
    maximum_kernel_error = max(
        summary["initialization_audit"]["kernel_max_abs_error"]
        for summary in summaries
    )
    support_preserved = all(
        summary["frozen_M5_support"] == summary["terminal_support"]
        for summary in summaries
    )

    m5_test = read_json(M5_ROOT / "test_metrics.json")
    m5_by_seed = {int(row["seed"]): row for row in m5_test["per_seed"]}
    m6_by_seed = {int(row["seed"]): row for row in test["per_seed"]}
    paired = []
    for seed in seeds:
        m5_rmse = float(m5_by_seed[seed]["rmse"])
        m6_rmse = float(m6_by_seed[seed]["rmse"])
        paired.append(
            {
                "seed": seed,
                "m5_rmse": m5_rmse,
                "m6_rmse": m6_rmse,
                "m5_minus_m6_rmse": m5_rmse - m6_rmse,
                "relative_rmse_reduction": (m5_rmse - m6_rmse) / m5_rmse,
            }
        )
    differences = np.asarray(
        [row["m5_minus_m6_rmse"] for row in paired], dtype=np.float64
    )

    m5_kernels = np.load(M5_ROOT / "lag_kernel.npy")
    m6_kernels = np.load(M6_ROOT / "lag_kernels.npz")["external"]
    truth = generate_synthetic_sequence(
        "AR-S2", seed=0, n_samples=10000, external_variables=10
    ).truth
    true_q = np.asarray(truth["q_primary"], dtype=np.float64)
    active = [int(value) for value in truth["active_support"]]
    kernel_rows = []
    for seed_index, seed in enumerate(seeds):
        support = set(int(value) for value in m5_by_seed[seed]["terminal_support"])
        for variable in active:
            if variable not in support:
                kernel_rows.append(
                    {
                        "seed": seed,
                        "variable": variable,
                        "status": "NOT_APPLICABLE_NOT_SELECTED_BY_M5",
                    }
                )
                continue
            truth_row = true_q[variable]
            m5_row = m5_kernels[seed_index, variable]
            m6_row = m6_kernels[seed_index, variable]
            lags = np.arange(len(truth_row), dtype=np.float64)
            kernel_rows.append(
                {
                    "seed": seed,
                    "variable": variable,
                    "status": "COMPLETED",
                    "m5_l1_error": float(np.abs(m5_row - truth_row).sum()),
                    "m6_l1_error": float(np.abs(m6_row - truth_row).sum()),
                    "m5_mean_lag_error": float(
                        abs(np.dot(m5_row - truth_row, lags))
                    ),
                    "m6_mean_lag_error": float(
                        abs(np.dot(m6_row - truth_row, lags))
                    ),
                }
            )
    completed = [row for row in kernel_rows if row["status"] == "COMPLETED"]
    m5_l1 = np.asarray([row["m5_l1_error"] for row in completed])
    m6_l1 = np.asarray([row["m6_l1_error"] for row in completed])

    atomic_json(
        M6_ROOT / "post_selection_truth_audit.json",
        {
            "status": "COMPLETED",
            "model": "M6",
            "scenario": "AR-S2",
            "selection_was_validation_only": True,
            "truth_access_stage": "after_hyperparameter_selection_and_test_aggregation",
            "implementation_audit": {
                "candidate_count": len(summaries),
                "all_candidates_test_accessed_false": True,
                "support_preserved_for_all_candidates": support_preserved,
                "maximum_initial_prediction_abs_error_scaled": maximum_prediction_error,
                "maximum_initial_kernel_abs_error": maximum_kernel_error,
            },
            "prediction_comparison": {
                "per_seed": paired,
                "seed_count": len(paired),
                "m6_better_seed_count": int((differences > 0).sum()),
                "m6_worse_seed_count": int((differences < 0).sum()),
                "mean_m5_minus_m6_rmse": float(differences.mean()),
                "standard_error_m5_minus_m6_rmse": float(
                    differences.std(ddof=1) / np.sqrt(len(differences))
                ),
                "mean_relative_rmse_reduction": float(
                    np.mean([row["relative_rmse_reduction"] for row in paired])
                ),
            },
            "lag_kernel_truth_comparison": {
                "scope": "true_active_variables_selected_by_frozen_M5_support",
                "completed_seed_variable_pairs": len(completed),
                "m6_lower_l1_error_pair_count": int((m6_l1 < m5_l1).sum()),
                "mean_m5_l1_error": float(m5_l1.mean()),
                "mean_m6_l1_error": float(m6_l1.mean()),
                "mean_m5_minus_m6_l1_error": float((m5_l1 - m6_l1).mean()),
                "per_seed_variable": kernel_rows,
            },
        },
    )
    print(M6_ROOT / "post_selection_truth_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
