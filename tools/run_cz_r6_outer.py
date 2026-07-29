#!/usr/bin/env python3
"""Locked Furnace-B zero-shot outer evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch
import numpy as np

from ar_raphu.cz_real.frozen import (
    evaluate_frozen_horizon,
    load_frozen_horizon,
)
from ar_raphu.cz_real.protocol import load_furnace_b
from ar_raphu.orss.diagnostics import write_json


def _require(path: Path, key: str) -> None:
    if not path.exists():
        raise RuntimeError(f"MISSING_OUTER_GATE:{key}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get(key, payload.get("status") == "COMPLETED")):
        raise RuntimeError(f"OUTER_GATE_FAILED:{key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-model", required=True)
    parser.add_argument(
        "--outer-workbook",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw/实验数据1-张.xlsx",
    )
    parser.add_argument("--sheet", default="Sheet2")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--results-root",
        default="results/cz_real_data/complete_5090",
    )
    args = parser.parse_args()
    if not args.zero_shot:
        raise RuntimeError("OUTER_ZERO_SHOT_FLAG_REQUIRED")
    root = Path(args.results_root)
    output = root / "R6"
    result_path = output / "outer_zero_shot_metrics.json"
    if args.resume and result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            print("FURNACE_B_ZERO_SHOT_RESUMED")
            return
    _require(root / "R3A" / "R3A_STATUS.json", "H2_NATIVE_COMPLETE")
    _require(root / "R3B" / "R3B_STATUS.json", "R3_RESOLUTION_COMPLETE")
    _require(root / "R3C" / "R3C_STATUS.json", "R3_CONTINUATION_COMPLETE")
    _require(root / "R3D" / "R3D_STATUS.json", "R3_RANK_COMPLETE")
    _require(root / "R4" / "R4_STATUS.json", "BASELINE_MATRIX_COMPLETE")
    _require(
        root / "R5" / "R5_STATUS.json",
        "FURNACE_A_CONFIRMATION_COMPLETE",
    )
    locked = Path(args.frozen_model)
    _require(
        locked / "FROZEN_MODEL_HASH_WRITTEN.json",
        "FROZEN_MODEL_HASH_WRITTEN",
    )
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not bool(config["protocol_frozen"]):
        raise RuntimeError("PROTOCOL_NOT_FROZEN")

    # This is the first value-level access to Furnace B.
    data = load_furnace_b(
        args.outer_workbook,
        protocol_frozen=True,
        stage="R7",
    )
    manifest = json.loads(
        (locked / "FROZEN_MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )
    device = torch.device(config["runtime"]["device"])
    rows: dict[str, object] = {}
    support_shift: dict[str, object] = {}
    domain_rows: list[dict[str, object]] = []
    free_run: dict[str, object] = {}
    for horizon, row in manifest["models"].items():
        model = load_frozen_horizon(locked / row["path"])
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
        prediction = evaluated.pop("direct_prediction")
        target = evaluated.pop("direct_target")
        indices = evaluated.pop("direct_target_indices")
        output.mkdir(parents=True, exist_ok=True)
        prediction_path = output / f"h_{int(horizon):03d}_predictions.npz"
        np.savez_compressed(
            prediction_path,
            prediction=prediction,
            target=target,
            target_indices=indices,
        )
        rows[horizon] = evaluated["direct_metrics"]
        rows[horizon]["prediction_artifact"] = prediction_path.name
        free_run[horizon] = evaluated["free_run_metrics"]
        support_shift[horizon] = {
            "continuation_by_channel": evaluated["continuation_by_channel"],
            "maximum_normalized_extrapolation_distance": evaluated[
                "free_run_metrics"
            ]["maximum_normalized_extrapolation_distance"],
        }
        for item in evaluated["continuation_by_channel"]:
            domain_rows.append({"horizon": int(horizon), **item})
        print(
            f"R6 h={horizon} "
            f"RMSE={rows[horizon]['RMSE_mm']:.8g} "
            f"free_RMSE={free_run[horizon]['free_run_RMSE_mm']:.8g}",
            flush=True,
        )
        torch.cuda.empty_cache()
    write_json(
        result_path,
        {
            "schema": "CZ_FURNACE_B_ZERO_SHOT_V1",
            "status": "COMPLETED",
            "zero_shot": True,
            "models": rows,
            "FURNACE_B_ZERO_SHOT_COMPLETE": True,
            "furnace_B_access_count": 1,
        },
    )
    write_json(output / "outer_support_shift.json", support_shift)
    write_json(output / "outer_free_run.json", free_run)
    write_json(
        output / "outer_incremental_value.json",
        {
            "status": "NOT_YET_AVAILABLE",
            "reason": "Filled after the locked baseline outer evaluation.",
        },
    )
    try:
        import pandas as pd

        pd.DataFrame(domain_rows).to_csv(
            output / "outer_domain_usage.csv", index=False
        )
    except Exception:
        write_json(output / "outer_domain_usage.json", domain_rows)
    write_json(
        output / "R6_STATUS.json",
        {
            "status": "COMPLETED",
            "FURNACE_B_ZERO_SHOT_COMPLETE": True,
            "furnace_B_access_count": 1,
            "next_stage": "FURNACE_B_CALIBRATION",
        },
    )


if __name__ == "__main__":
    main()
