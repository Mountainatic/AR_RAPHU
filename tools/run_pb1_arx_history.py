#!/usr/bin/env python3
"""Run the frozen H1 Champneys-2024 ARX history selector on PB1 development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.baselines import fit_and_select_arx_history
from ar_raphu.datasets.loaders import load_pwh, load_whpn
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    load_pb1_protocol_freeze,
)


LOADERS = {"pwh": load_pwh, "whpn": load_whpn}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


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

    preflight_path = (
        ROOT
        / "results/public_benchmarks/pb1/PB1_DEVELOPMENT_PREFLIGHT.json"
    )
    preflight = _read_json(preflight_path)
    if preflight.get("overall_status") != "READY_FOR_DEVELOPMENT":
        raise RuntimeError("PB1 development preflight is not ready.")
    if preflight["datasets"][args.dataset]["status"] != "READY_FOR_DEVELOPMENT":
        raise RuntimeError(f"{args.dataset} development is not ready.")

    freeze_path = ROOT / "configs/public_benchmarks/PB1_PROTOCOL_FREEZE.json"
    freeze = load_pb1_protocol_freeze(freeze_path)
    config_path = (
        ROOT / f"configs/public_benchmarks/pb1_{args.dataset}.yaml"
    )
    config = _read_json(config_path)
    whpn_audit = None
    if args.dataset == "whpn":
        whpn_audit = _read_json(
            ROOT
            / "results/public_benchmarks/pb1/protocol_audit"
            / "whpn_realization_audit.json"
        )
    raw = LOADERS[args.dataset](args.raw_root, include_test=False)
    dataset = apply_pb1_development_partition(
        raw, freeze, whpn_audit=whpn_audit
    )
    h1 = config["task"]["xar_history_selection"]["H1_baseline_faithful"]
    nx_range = h1["nx_candidates"]
    ny_range = h1["ny_candidates"]
    if nx_range[0] != 1 or ny_range[0] != 1:
        raise ValueError("H1 ARX lag candidates must begin at one.")
    selection = fit_and_select_arx_history(
        dataset,
        max_nx=int(nx_range[1]),
        max_ny=int(ny_range[1]),
    )
    output = (
        ROOT
        / "results/public_benchmarks/pb1"
        / args.dataset
        / "development/H1_ARX_NO_FUTURE_X/history_selection.json"
    )
    if output.exists() and not args.force_development:
        raise FileExistsError(
            f"{output} exists; use --force-development only for development."
        )
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "dataset": args.dataset,
        "stage": "development",
        "lane": "H1_BASELINE_FAITHFUL",
        "model": "ARX_CHAMPNEYS2024_PROJECT_SPLIT_NO_FUTURE_X",
        "evaluation_mode": "FREE_RUNNING_SIMULATION",
        "forecast_alignment": "X_THROUGH_T_AND_Y_THROUGH_T_TO_TARGET_Y_T_PLUS_1",
        "horizon": 1,
        "selection_metric": "MEAN_RECORD_VALIDATION_AIC",
        "source_commit": _commit(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "protocol_freeze_sha256": _sha256(freeze_path),
        "preflight_sha256": _sha256(preflight_path),
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "train_only_standardization": True,
        "solver": "STREAMING_QR_THEN_SVD_MINIMUM_NORM_SUBSET",
        "scientific_ridge": 0.0,
        "selected": {
            "nx": selection.selected_nx,
            "ny": selection.selected_ny,
            "coefficients_y_standardized": selection.coefficients_y.tolist(),
            "coefficients_x_standardized": selection.coefficients_x.tolist(),
            "x_mean": selection.x_mean,
            "x_scale": selection.x_scale,
            "y_mean": selection.y_mean,
            "y_scale": selection.y_scale,
        },
        "candidates": [
            {
                "nx": candidate.nx,
                "ny": candidate.ny,
                "validation_aic_mean": candidate.validation_aic_mean,
                "validation_aic_by_record": list(
                    candidate.validation_aic_by_record
                ),
                "effective_rank": candidate.effective_rank,
                "condition_number": candidate.condition_number,
                "stable_simulation": candidate.stable_simulation,
            }
            for candidate in selection.candidates
        ],
        "runtime": {
            "elapsed_seconds": selection.elapsed_seconds,
            "dtype": "float64",
            "device": "cpu",
        },
        "status": "COMPLETED",
        "scientific_role": "DEVELOPMENT_HISTORY_SELECTION_NOT_TEST_EVIDENCE",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        f"selected nx={selection.selected_nx} ny={selection.selected_ny} "
        f"elapsed={selection.elapsed_seconds:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
