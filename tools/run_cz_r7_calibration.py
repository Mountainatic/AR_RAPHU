#!/usr/bin/env python3
"""Furnace-B 5%/10% intercept-and-scale calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np
import torch

from ar_raphu.cz_real.frozen import (
    evaluate_frozen_horizon,
    load_frozen_horizon,
)
from ar_raphu.cz_real.linear import regression_metrics
from ar_raphu.cz_real.protocol import load_furnace_b
from ar_raphu.orss.diagnostics import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fractions", nargs="+", type=float, required=True)
    parser.add_argument(
        "--frozen-model",
        default="results/cz_real_data/complete_5090/frozen_model",
    )
    parser.add_argument(
        "--outer-workbook",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw/实验数据1-张.xlsx",
    )
    parser.add_argument(
        "--results-root",
        default="results/cz_real_data/complete_5090",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.results_root)
    output = root / "R7"
    destination = output / "furnace_b_calibration.json"
    if args.resume and destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            print("FURNACE_B_CALIBRATION_RESUMED")
            return
    zero_shot = json.loads(
        (root / "R6" / "R6_STATUS.json").read_text(encoding="utf-8")
    )
    if not bool(zero_shot.get("FURNACE_B_ZERO_SHOT_COMPLETE")):
        raise RuntimeError("FURNACE_B_ZERO_SHOT_NOT_COMPLETE")
    if any(not 0.0 < fraction < 0.5 for fraction in args.fractions):
        raise ValueError("Calibration fractions must be in (0, 0.5).")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data = load_furnace_b(
        args.outer_workbook,
        protocol_frozen=True,
        stage="R8",
    )
    locked = Path(args.frozen_model)
    manifest = json.loads(
        (locked / "FROZEN_MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )
    device = torch.device(config["runtime"]["device"])
    rows: dict[str, object] = {}
    for horizon, model_row in manifest["models"].items():
        model = load_frozen_horizon(locked / model_row["path"])
        evaluated = evaluate_frozen_horizon(
            model,
            data.inputs,
            data.target,
            target_start=0,
            target_stop=len(data.target),
            device=device,
            chunk_time=int(config["cuda"]["operator_chunk_time"]),
            include_predictions=True,
        )
        prediction = np.asarray(
            evaluated.pop("direct_prediction"), dtype=np.float64
        )
        target = np.asarray(evaluated.pop("direct_target"), dtype=np.float64)
        evaluated.pop("direct_target_indices")
        horizon_rows: dict[str, object] = {}
        for fraction in args.fractions:
            count = max(2, int(np.floor(len(target) * fraction)))
            design = np.column_stack(
                (np.ones(count, dtype=np.float64), prediction[:count])
            )
            parameters = np.linalg.lstsq(
                design, target[:count], rcond=None
            )[0]
            calibrated = parameters[0] + parameters[1] * prediction[count:]
            horizon_rows[f"{fraction:.2f}"] = {
                "calibration_count": count,
                "evaluation_count": int(len(target) - count),
                "adapter": {
                    "intercept_mm": float(parameters[0]),
                    "output_scale": float(parameters[1]),
                },
                "metrics": regression_metrics(target[count:], calibrated),
                "history_retuned": False,
                "resolution_retuned": False,
                "penalty_retuned": False,
                "rank_retuned": False,
                "kernel_basis_retuned": False,
            }
        rows[horizon] = horizon_rows
        torch.cuda.empty_cache()
    write_json(
        destination,
        {
            "schema": "CZ_FURNACE_B_CALIBRATION_V1",
            "status": "COMPLETED",
            "fractions": args.fractions,
            "allowed_adapter": config["allowed_adapter"],
            "models": rows,
            "FURNACE_B_CALIBRATION_COMPLETE": True,
            "furnace_B_access_count": 2,
        },
    )
    write_json(
        output / "R7_STATUS.json",
        {
            "status": "COMPLETED",
            "FURNACE_B_CALIBRATION_COMPLETE": True,
            "next_stage": "INTERPRETABILITY",
        },
    )


if __name__ == "__main__":
    main()
