#!/usr/bin/env python3
"""One-time Furnace-A internal confirmation of the frozen ORSS family."""

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

from ar_raphu.cz_real.frozen import (
    evaluate_frozen_horizon,
    load_frozen_horizon,
)
from ar_raphu.cz_real.protocol import (
    assert_confirmation_access,
    confirmation_interval,
    load_furnace_a,
)
from ar_raphu.orss.diagnostics import write_json


def _furnace_a(raw_dir: Path) -> Path:
    candidates = [path for path in raw_dir.iterdir() if path.name.endswith("1.xlsx")]
    if len(candidates) != 1:
        raise RuntimeError("Furnace-A path is not uniquely identifiable.")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--locked-model", required=True)
    parser.add_argument(
        "--raw-dir",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw",
    )
    parser.add_argument(
        "--output",
        default="results/cz_real_data/complete_5090/R5",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    result_path = output / "furnace_a_confirmation.json"
    if args.resume and result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            print("FURNACE_A_CONFIRMATION_RESUMED")
            return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert_confirmation_access(
        protocol_frozen=bool(config["protocol_frozen"]), stage="R6"
    )
    locked = Path(args.locked_model)
    manifest = json.loads(
        (locked / "FROZEN_MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )
    if manifest["status"] != "COMPLETED":
        raise RuntimeError("FROZEN_MODEL_HASH_NOT_WRITTEN")
    data = load_furnace_a(_furnace_a(Path(args.raw_dir)))
    start, stop = confirmation_interval(len(data.target))
    device = torch.device(config["runtime"]["device"])
    rows: dict[str, object] = {}
    for horizon, row in manifest["models"].items():
        model = load_frozen_horizon(locked / row["path"])
        evaluated = evaluate_frozen_horizon(
            model,
            data.inputs,
            data.target,
            target_start=start,
            target_stop=stop,
            device=device,
            chunk_time=int(config["cuda"]["operator_chunk_time"]),
            include_predictions=True,
        )
        prediction = evaluated.pop("direct_prediction")
        target = evaluated.pop("direct_target")
        indices = evaluated.pop("direct_target_indices")
        output.mkdir(parents=True, exist_ok=True)
        npz_path = output / f"h_{int(horizon):03d}_predictions.npz"
        import numpy as np

        np.savez_compressed(
            npz_path,
            prediction=prediction,
            target=target,
            target_indices=indices,
        )
        evaluated["prediction_artifact"] = npz_path.name
        rows[horizon] = evaluated
        print(
            f"R5 h={horizon} "
            f"RMSE={rows[horizon]['direct_metrics']['RMSE_mm']:.8g} "
            f"free_RMSE={rows[horizon]['free_run_metrics']['free_run_RMSE_mm']:.8g}",
            flush=True,
        )
        torch.cuda.empty_cache()
    payload = {
        "schema": "CZ_FURNACE_A_CONFIRMATION_V1",
        "status": "COMPLETED",
        "protocol_frozen": True,
        "retuning_performed": False,
        "target_interval": [start, stop],
        "models": rows,
        "FURNACE_A_CONFIRMATION_COMPLETE": True,
        "furnace_B_access_count": 0,
    }
    write_json(result_path, payload)
    write_json(
        output / "R5_STATUS.json",
        {
            "status": "COMPLETED",
            "FURNACE_A_CONFIRMATION_COMPLETE": True,
            "next_stage": "FURNACE_B_ZERO_SHOT",
        },
    )


if __name__ == "__main__":
    main()
