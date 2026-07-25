#!/usr/bin/env python3
"""Run fixed B0/B1 Phase-1 baselines without using validation or test to fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from STAGE1_DUAL_SOLVER_V20_bundle.stage1.dual_metrics import prediction_metrics
from ar_raphu.baselines import (
    LinearDirectForecaster,
    persistence_predict,
    target_indices,
)
from ar_raphu.protocol_config import load_protocol_config
from ar_raphu.runtime_environment import require_runtime_environment
from ar_raphu.sequence_data import PreparedDirectForecastData
from ar_raphu.synthetic import SCENARIOS, generate_synthetic_sequence


RESULT_ROOT = PROJECT_ROOT / "results" / "phase1"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_name(scenario: str) -> str:
    return {
        "AR-S0": "E1",
        "AR-S1": "E2",
        "AR-S2": "E3",
        "AR-S3": "E4",
    }.get(scenario, f"SUPPORT_{scenario}")


def run(args: argparse.Namespace) -> Path:
    config = load_protocol_config(require_phase1_frozen=True)
    if args.model not in {"B0", "B1"}:
        raise ValueError("This runner implements fixed B0/B1 only.")
    count = (
        512
        if args.smoke
        else config["phase1_synthetic"]["sample_sizes"]["core"]
    )
    sequence = generate_synthetic_sequence(
        args.scenario,
        seed=args.seed,
        n_samples=count,
        external_variables=10,
    )
    data = PreparedDirectForecastData.from_sequence(
        sequence.x,
        sequence.y_observed,
        track="AR",
        horizon=1,
        L_x=64,
        L_y=32,
        split_target_intervals=sequence.split_target_intervals,
    )
    root = (
        RESULT_ROOT
        / f"{experiment_name(args.scenario)}_{args.scenario}_G2"
        / args.model
        / f"seed_{args.seed}"
    )
    if args.smoke:
        root = (
            RESULT_ROOT
            / f"{experiment_name(args.scenario)}_{args.scenario}_G2_SMOKE"
            / args.model
            / f"seed_{args.seed}"
        )
    summary_path = root / "test_metrics.json"
    if summary_path.exists() and not args.force:
        return summary_path

    started = time.perf_counter()
    if args.model == "B0":
        predicted_scaled = persistence_predict(data, "test")
        indices = target_indices(data, "test")
        fit = {
            "fit": "none",
            "definition": "y_hat[t+h|t]=y[t]",
            "train_target_interval": sequence.split_target_intervals["train"],
        }
    else:
        model = LinearDirectForecaster.fit(data, kind="AR")
        predicted_scaled, indices = model.predict(data, "test")
        fit = {
            "fit": "unregularized_float64_least_squares",
            "train_target_interval": sequence.split_target_intervals["train"],
            "design_rank": model.rank,
            "coefficient_count_including_intercept": len(model.coefficients) + 1,
            "singular_values": model.singular_values.tolist(),
            "coefficients": model.coefficients.tolist(),
            "intercept": model.intercept,
        }
    observed_scaled = data.y_scaled[indices].astype(np.float64)
    predicted = data.scaler.inverse_y(predicted_scaled)
    observed = data.scaler.inverse_y(observed_scaled)
    metrics = prediction_metrics(observed, predicted)
    target_std = float(np.std(observed, ddof=0))
    metrics["target_std_normalized_rmse"] = (
        metrics["rmse"] / target_std if target_std > 0 else None
    )
    metrics.update(
        {
            "status": "COMPLETED",
            "model": args.model,
            "scenario": args.scenario,
            "seed": args.seed,
            "horizon": 1,
            "future_X": False,
            "test_opened_after_model_definition_frozen": True,
            "smoke_not_scientific_evidence": args.smoke,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "fit.json", fit)
    atomic_json(summary_path, metrics)
    config_path = PROJECT_ROOT / "configs" / "protocol_v2.yaml"
    atomic_json(
        root / "config_reference.json",
        {
            "path": "configs/protocol_v2.yaml",
            "sha256": sha256(config_path),
        },
    )
    atomic_json(
        root / "runtime.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "device": "cpu",
            "dtype": "float64",
            "python": platform.python_version(),
            "cpu_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        },
    )
    pd.DataFrame(
        {
            "seed": args.seed,
            "partition": "test",
            "target_index": indices,
            "observed": observed,
            "predicted": predicted,
            "residual": predicted - observed,
        }
    ).to_parquet(root / "predictions.parquet", index=False)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--model", choices=["B0", "B1"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    require_runtime_environment()
    path = run(parse_args())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
