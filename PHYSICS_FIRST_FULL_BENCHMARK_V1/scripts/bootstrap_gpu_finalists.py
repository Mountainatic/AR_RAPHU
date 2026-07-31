#!/usr/bin/env python3
"""Paired moving-block bootstrap over saved finalist predictions."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gpu_common import atomic_json, write_csv


def _draw_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    chunks = []
    while sum(len(chunk) for chunk in chunks) < n:
        start = int(rng.integers(0, n))
        chunks.append((start + np.arange(block)) % n)
    return np.concatenate(chunks)[:n]


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "upper_95": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    results = Path(args.results).expanduser().resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    replicates = int(config["ablation_protocol"]["bootstrap_replicates"])
    block = int(config["ablation_protocol"]["bootstrap_block_rows"])
    model_ids = [item["id"] for item in config["models"]]
    payloads: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for model_id in model_ids:
        for path in sorted(
            (results / "tasks" / "finalists").glob(f"*/{model_id}/seed_*/predictions.npz")
        ):
            direction = path.parents[2].name
            seed = int(path.parent.name.split("_")[-1])
            with np.load(path, allow_pickle=False) as stored:
                mask = stored["evaluation_mask"].astype(bool)
                ids = stored["sample_id"][mask]
                target = stored["y_true"][mask].astype(np.float64)
                prediction = stored["y_pred"][mask].astype(np.float64)
            payloads[(model_id, seed, direction)] = (
                ids,
                np.column_stack((target, prediction)),
            )
    rng = np.random.default_rng(20260731)
    model_rows = []
    for model_id in model_ids:
        keys = sorted(key for key in payloads if key[0] == model_id)
        draws = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            gain_sum = 0.0
            baseline_sum = 0.0
            for key in keys:
                _, values = payloads[key]
                target, prediction = values[:, 0], values[:, 1]
                index = _draw_indices(len(target), block, rng)
                baseline = target[index] ** 2
                model_loss = (target[index] - prediction[index]) ** 2
                gain_sum += float(np.sum(baseline - model_loss))
                baseline_sum += float(np.sum(baseline))
            draws[replicate] = gain_sum / max(
                baseline_sum, np.finfo(np.float64).eps
            )
        model_rows.append(
            {
                "kind": "MODEL_VS_PERSISTENCE",
                "model_a": model_id,
                "model_b": "Persistence",
                "replicates": replicates,
                "block_rows": block,
                **_summary(draws),
            }
        )
    pair_rows = []
    for model_a, model_b in itertools.combinations(model_ids, 2):
        matched = sorted(
            set((seed, direction) for mid, seed, direction in payloads if mid == model_a)
            & set((seed, direction) for mid, seed, direction in payloads if mid == model_b)
        )
        draws = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            difference_sum = 0.0
            rows = 0
            for seed, direction in matched:
                ids_a, values_a = payloads[(model_a, seed, direction)]
                ids_b, values_b = payloads[(model_b, seed, direction)]
                if not np.array_equal(ids_a, ids_b):
                    raise RuntimeError(
                        f"FINALIST_SAMPLE_ID_MISMATCH:{model_a}:{model_b}:{seed}:{direction}"
                    )
                target = values_a[:, 0]
                if not np.array_equal(target, values_b[:, 0]):
                    raise RuntimeError("FINALIST_TARGET_MISMATCH")
                index = _draw_indices(len(target), block, rng)
                loss_a = (target[index] - values_a[index, 1]) ** 2
                loss_b = (target[index] - values_b[index, 1]) ** 2
                difference_sum += float(np.sum(loss_a - loss_b))
                rows += len(index)
            draws[replicate] = difference_sum / max(1, rows)
        pair_rows.append(
            {
                "kind": "PAIRED_MSE_A_MINUS_B",
                "model_a": model_a,
                "model_b": model_b,
                "replicates": replicates,
                "block_rows": block,
                **_summary(draws),
            }
        )
    all_rows = model_rows + pair_rows
    write_csv(results / "ABLATIONS" / "FINALIST_BOOTSTRAP.csv", all_rows)
    report = {
        "schema": "GPU_FINALIST_BLOCK_BOOTSTRAP_V1",
        "status": "PASS",
        "replicates": replicates,
        "block_rows": block,
        "models": model_rows,
        "pairs": pair_rows,
    }
    atomic_json(results / "ABLATIONS" / "FINALIST_BOOTSTRAP.json", report)
    print("GPU_FINALIST_BOOTSTRAP=" + json.dumps({"status": "PASS", "rows": len(all_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
