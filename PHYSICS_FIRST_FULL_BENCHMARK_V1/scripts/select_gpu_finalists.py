#!/usr/bin/env python3
"""Select six FP32 finalists from validation traces only.

The quota preserves both formal leaderboards and the physics residual ablation:
three input-only, two direct dynamic, and one K-residual model.  Test metrics and
predictions are never opened by this selector.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-seeds", type=int, default=5)
    args = parser.parse_args()
    results = Path(args.results).expanduser().resolve()
    source_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    specs = {
        item["id"]: item
        for item in source_config["models"]
        if item["stage"] in {"core", "frontier"}
        and item.get("label") != "NONCAUSAL_CONTROL_EXCLUDED_FROM_ONLINE_BOARD"
    }
    traces: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for model_id, spec in specs.items():
        for stage in ("core", "frontier"):
            model_root = results / "tasks" / stage
            for direction_root in sorted(model_root.glob(f"*/{model_id}")):
                direction = direction_root.parent.name
                for seed_root in sorted(direction_root.glob("seed_*")):
                    losses = []
                    for fold_path in sorted((seed_root / "selection").glob("fold_*.json")):
                        record = json.loads(fold_path.read_text(encoding="utf-8"))
                        losses.append(float(record["best_validation_loss"]))
                    if losses:
                        traces[model_id][(direction, seed_root.name)].append(
                            float(statistics.mean(losses))
                        )
    rows = []
    for model_id, by_run in traces.items():
        directions = {key[0] for key in by_run}
        seeds = {key[1] for key in by_run}
        if len(directions) < 2 or len(seeds) < args.minimum_seeds:
            continue
        values = [statistics.mean(run_values) for run_values in by_run.values()]
        rows.append(
            {
                "model_id": model_id,
                "mode": specs[model_id]["mode"],
                "validation_mse_scaled_median": float(statistics.median(values)),
                "directions": len(directions),
                "seeds": len(seeds),
            }
        )
    quotas = {"input": 3, "dynamic": 2, "residual": 1}
    selected = []
    for mode, quota in quotas.items():
        candidates = sorted(
            (row for row in rows if row["mode"] == mode),
            key=lambda row: (row["validation_mse_scaled_median"], row["model_id"]),
        )
        selected.extend(candidates[:quota])
    if len(selected) != 6:
        raise RuntimeError(f"INSUFFICIENT_FINALISTS:{len(selected)}:{rows}")
    finalist_specs = []
    for row in selected:
        spec = dict(specs[row["model_id"]])
        spec["source_model_id"] = spec["id"]
        spec["id"] = f"final__{spec['id']}"
        spec["stage"] = "finalists"
        spec["label"] = "FINALIST_CONFIRMATION_FROM_VALIDATION"
        finalist_specs.append(spec)
    output_config = dict(source_config)
    output_config["schema"] = "PHYSICS_FIRST_GPU_FINALISTS_V1"
    output_config["selection_source"] = "INNER_EXPANDING_VALIDATION_ONLY"
    output_config["selection_quota"] = quotas
    output_config["selection_rows"] = selected
    output_config["models"] = finalist_specs
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(output_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("GPU_FINALIST_SELECTION=" + json.dumps(selected, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
