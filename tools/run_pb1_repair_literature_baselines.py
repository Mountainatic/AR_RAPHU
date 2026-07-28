#!/usr/bin/env python3
"""Run literature-frozen ARX and pNARX PB1 Repair-V2 development baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.baselines import fit_arx_fixed_history, fit_and_select_pnarx
from ar_raphu.datasets.loaders import load_cascaded_tanks, load_silverbox
from ar_raphu.datasets.pb1_protocol import apply_pb1_repair_v2_partition


LOADERS = {
    "cascaded_tanks": load_cascaded_tanks,
    "silverbox": load_silverbox,
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(LOADERS), required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    parser.add_argument("--force-development", action="store_true")
    args = parser.parse_args()
    config_path = (
        ROOT / f"configs/public_benchmarks/pb1_{args.dataset}.yaml"
    )
    config = _json(config_path)
    raw = LOADERS[args.dataset](args.raw_root, include_test=False)
    dataset = apply_pb1_repair_v2_partition(raw, config)
    profile = config["literature_baselines"]
    nx, ny = int(profile["arx"]["nx"]), int(profile["arx"]["ny"])
    arx = fit_arx_fixed_history(dataset, nx=nx, ny=ny)
    orders = tuple(map(int, profile["pnarx"]["polynomial_orders"]))
    pnarx = fit_and_select_pnarx(dataset, nx=nx, ny=ny, orders=orders)
    arx_row = arx.candidates[0]
    selected_pnarx = next(
        row for row in pnarx.candidates if row.order == pnarx.order
    )
    payload = {
        "schema_version": 7,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "dataset": args.dataset,
        "stage": "development_repair",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "split": config["dataset"]["development_split"],
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "confirmation_allowed": False,
        "arx": {
            "profile_source": "CHAMPNEYS2024_ASSOCIATED_CODE",
            "history": {"nx": nx, "ny": ny},
            "scientific_ridge": 0.0,
            "solver": profile["arx"]["solver"],
            "validation_aic_mean": arx_row.validation_aic_mean,
            "validation_aic_by_record": list(
                arx_row.validation_aic_by_record
            ),
            "effective_rank": arx_row.effective_rank,
            "condition_number": arx_row.condition_number,
            "stable_simulation": arx_row.stable_simulation,
            "elapsed_seconds": arx.elapsed_seconds,
        },
        "pnarx": {
            "profile_source": "CHAMPNEYS2024_PEER_REVIEWED",
            "history": {"nx": nx, "ny": ny},
            "basis": profile["pnarx"]["basis"],
            "order_candidates": list(orders),
            "selected_order": pnarx.order,
            "candidates": [
                {
                    "order": row.order,
                    "validation_aic_mean": (
                        row.validation_aic_mean
                        if math.isfinite(row.validation_aic_mean)
                        else None
                    ),
                    "validation_aic_by_record": list(
                        row.validation_aic_by_record
                    ),
                    "effective_rank": row.effective_rank,
                    "condition_number": row.condition_number,
                    "stable_simulation": row.stable_simulation,
                    "status": (
                        "COMPLETED"
                        if row.stable_simulation
                        else "FAILED_UNSTABLE_SIMULATION"
                    ),
                }
                for row in pnarx.candidates
            ],
            "elapsed_seconds": pnarx.elapsed_seconds,
        },
        "status": (
            "COMPLETED"
            if arx_row.stable_simulation
            and selected_pnarx.stable_simulation
            else "FAILED"
        ),
    }
    output = (
        ROOT
        / "results/public_benchmarks/pb1_repair_v2"
        / args.dataset
        / "development/LITERATURE_BASELINES/baseline_selection.json"
    )
    if output.exists() and not args.force_development:
        raise FileExistsError(f"{output} already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        f"status={payload['status']} arx_aic={arx_row.validation_aic_mean:.8g} "
        f"pnarx_order={pnarx.order}"
    )
    return 0 if payload["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
