"""Run and aggregate the 10-seed native TIM E2 screening profile."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .synthetic_c1 import REGIMES


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_one(
    project: Path,
    output_root: Path,
    regime: str,
    seed: int,
    n_samples: int,
    noise_level: str,
) -> dict[str, Any]:
    run_root = output_root / regime / f"seed_{seed:02d}"
    log = output_root / "logs" / f"{regime}__seed_{seed:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "experiments.tim_validation.e2_stagewise.native_runner",
        "all",
        "--project",
        str(project),
        "--run-root",
        str(run_root),
        "--regime",
        regime,
        "--seed",
        str(seed),
        "--n-samples",
        str(n_samples),
        "--noise-level",
        noise_level,
    ]
    environment = dict(os.environ)
    environment.pop("PRISM_FORMAL_INFERENCE_ONLY", None)
    environment.update(
        {
            "PRISM_V211_K_INNER_WORKERS": "2",
            "PRISM_V211_W_INNER_WORKERS": "2",
            "PRISM_V211_A_INNER_WORKERS": "2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=project.parent,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "regime": regime,
        "seed": seed,
        "returncode": completed.returncode,
        "run_root": str(run_root),
        "log": str(log),
    }


def summarize_run(run_root: Path) -> dict[str, Any]:
    truth = json.loads((run_root / "shared" / "GENERATOR_TRUTH.json").read_text(encoding="utf-8"))
    structure = json.loads((run_root / "STRUCTURE_SIGNATURE.json").read_text(encoding="utf-8"))
    inference = json.loads(
        (run_root / "inference" / "INFERENCE_COMPLETE.json").read_text(encoding="utf-8")
    )
    development = json.loads(
        (run_root / "DEVELOPMENT_COMPLETE.json").read_text(encoding="utf-8")
    )
    true_channels = set(int(value) for value in truth["active_channels"])
    selected_channels = set(int(value[1:]) for value in structure["admitted_channels"])
    tp = len(true_channels & selected_channels)
    fp = len(selected_channels - true_channels)
    fn = len(true_channels - selected_channels)
    precision = tp / len(selected_channels) if selected_channels else 0.0
    recall = tp / len(true_channels) if true_channels else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    true_stage = truth["stage_truth"]
    selected_stage = {
        "K": bool(structure["K_admitted"]),
        "C": bool(structure["C_admitted"]),
        "W": bool(structure["DeltaW_admitted"]),
        "A": bool(structure["A_admitted"]),
    }
    selected_scales = structure["channel_scale_class"]
    correct_scales = sum(
        selected_scales.get(f"x{channel}") == truth["scale_by_channel"].get(str(channel))
        for channel in true_channels
    )
    metrics = {
        str(record.get("model")): record
        for record in inference["records"]
        if record.get("status") == "PASS"
    }
    full = metrics.get("PRISM_V2_1_1_PHYSICS_FIRST", {})
    return {
        "regime": truth["regime"],
        "seed": truth["seed"],
        "true_channels": json.dumps(sorted(true_channels)),
        "admitted_channels": json.dumps(sorted(selected_channels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_admission_rate": fp / max(1, 8 - len(true_channels)),
        "false_rejection_rate": fn / max(1, len(true_channels)),
        "decoy_admitted": 4 in selected_channels,
        "scale_accuracy": correct_scales / max(1, len(true_channels)),
        "true_stage_vector": json.dumps(true_stage, sort_keys=True),
        "observed_stage_vector": json.dumps(selected_stage, sort_keys=True),
        "stage_exact_match": true_stage == selected_stage,
        "C_correct": true_stage["C"] == selected_stage["C"],
        "W_correct": true_stage["W"] == selected_stage["W"],
        "A_correct": true_stage["A"] == selected_stage["A"],
        "test_rmse_full": full.get("rmse"),
        "test_r2_full": full.get("r2_delta"),
        "checkpoint_reload_max_error": inference["reload_audit"][
            "maximum_absolute_prediction_error"
        ],
        "k_failed_channel_count": sum(
            status != "PASS" for status in development["stage_statuses"]["K"]
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for regime in REGIMES:
        subset = [row for row in rows if row["regime"] == regime]
        result.append(
            {
                "regime": regime,
                "seed_count": len(subset),
                "mean_precision": float(np.mean([row["precision"] for row in subset])),
                "mean_recall": float(np.mean([row["recall"] for row in subset])),
                "mean_f1": float(np.mean([row["f1"] for row in subset])),
                "mean_false_admission_rate": float(
                    np.mean([row["false_admission_rate"] for row in subset])
                ),
                "mean_false_rejection_rate": float(
                    np.mean([row["false_rejection_rate"] for row in subset])
                ),
                "decoy_admission_rate": float(
                    np.mean([row["decoy_admitted"] for row in subset])
                ),
                "mean_scale_accuracy": float(
                    np.mean([row["scale_accuracy"] for row in subset])
                ),
                "stage_exact_match_rate": float(
                    np.mean([row["stage_exact_match"] for row in subset])
                ),
                "C_accuracy": float(np.mean([row["C_correct"] for row in subset])),
                "W_accuracy": float(np.mean([row["W_correct"] for row in subset])),
                "A_accuracy": float(np.mean([row["A_correct"] for row in subset])),
                "mean_test_rmse_full": float(
                    np.mean([row["test_rmse_full"] for row in subset])
                ),
                "mean_k_failed_channel_count": float(
                    np.mean([row["k_failed_channel_count"] for row in subset])
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--noise-level", default="very_low")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(regime, seed) for regime in REGIMES for seed in range(10)]
    if not args.aggregate_only:
        outcomes: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_one,
                    project,
                    output_root,
                    regime,
                    seed,
                    args.n_samples,
                    args.noise_level,
                ): (regime, seed)
                for regime, seed in jobs
            }
            for future in as_completed(futures):
                outcomes.append(future.result())
                _write_json(output_root / "RUN_OUTCOMES.json", sorted(outcomes, key=lambda x: (x["regime"], x["seed"])))
        failures = [item for item in outcomes if item["returncode"] != 0]
        if failures:
            _write_json(
                output_root / "SCREENING_MANIFEST.json",
                {"status": "FAILED_RETAINED", "failures": failures, "jobs": len(jobs)},
            )
            return 1
    rows = [summarize_run(output_root / regime / f"seed_{seed:02d}") for regime, seed in jobs]
    aggregates = aggregate(rows)
    _write_csv(output_root / "per_seed.csv", rows)
    _write_csv(output_root / "channel_recovery.csv", rows)
    _write_csv(output_root / "scale_recovery.csv", rows)
    _write_csv(output_root / "stage_recovery.csv", rows)
    _write_csv(output_root / "aggregate.csv", aggregates)
    _write_json(
        output_root / "SCREENING_MANIFEST.json",
        {
            "status": "COMPLETED",
            "profile": "screening",
            "implementation": "native_prism_v2.1.1",
            "regimes": list(REGIMES),
            "seeds": list(range(10)),
            "jobs": len(jobs),
            "n_samples": args.n_samples,
            "noise_level": args.noise_level,
            "test_access_after_per_run_freeze_only": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
