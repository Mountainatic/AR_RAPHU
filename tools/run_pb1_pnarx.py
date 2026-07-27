#!/usr/bin/env python3
"""Run the peer-reviewed Champneys-2024 pNARX baseline on PB1 development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.baselines.pnarx_champneys2024 import fit_and_select_pnarx
from ar_raphu.datasets.loaders import load_pwh, load_whpn
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    load_pb1_protocol_freeze,
)


LOADERS = {"pwh": load_pwh, "whpn": load_whpn}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        ROOT / "results/public_benchmarks/pb1/PB1_DEVELOPMENT_PREFLIGHT.json"
    )
    preflight = _json(preflight_path)
    if preflight.get("overall_status") != "READY_FOR_DEVELOPMENT":
        raise RuntimeError("PB1 development preflight is not ready.")
    freeze_path = ROOT / "configs/public_benchmarks/PB1_PROTOCOL_FREEZE.json"
    freeze = load_pb1_protocol_freeze(freeze_path)
    whpn_audit = (
        _json(
            ROOT
            / "results/public_benchmarks/pb1/protocol_audit"
            / "whpn_realization_audit.json"
        )
        if args.dataset == "whpn"
        else None
    )
    raw = LOADERS[args.dataset](args.raw_root, include_test=False)
    dataset = apply_pb1_development_partition(
        raw, freeze, whpn_audit=whpn_audit
    )
    h1_path = (
        ROOT
        / "results/public_benchmarks/pb1"
        / args.dataset
        / "development/H1_ARX_NO_FUTURE_X/history_selection.json"
    )
    h1 = _json(h1_path)
    nx, ny = int(h1["selected"]["nx"]), int(h1["selected"]["ny"])
    selection = fit_and_select_pnarx(dataset, nx=nx, ny=ny)
    output = (
        ROOT
        / "results/public_benchmarks/pb1"
        / args.dataset
        / "development/PNARX_CHAMPNEYS2024/order_selection.json"
    )
    if output.exists() and not args.force_development:
        raise FileExistsError(f"{output} already exists.")
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "dataset": args.dataset,
        "stage": "development",
        "model": "PNARX_CHAMPNEYS2024",
        "profile_source": "PEER_REVIEWED_PAPER",
        "evaluation_mode": "FREE_RUNNING_SIMULATION",
        "forecast_alignment": "X_THROUGH_T_AND_Y_THROUGH_T_TO_TARGET_Y_T_PLUS_1",
        "horizon": 1,
        "history_source": "H1_ARX_AIC_SELECTED",
        "history": {"nx": nx, "ny": ny},
        "basis": "UNIVARIATE_LEGENDRE_MONOMIAL_NO_CROSS_TERMS",
        "order_candidates": [2, 3, 4, 5, 6, 7],
        "selection_metric": "MEAN_RECORD_VALIDATION_AIC",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "h1_sha256": _sha256(h1_path),
        "protocol_freeze_sha256": _sha256(freeze_path),
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "train_only_standardization": True,
        "solver": "STREAMING_QR_THEN_SVD_MINIMUM_NORM_SUBSET",
        "scientific_ridge": 0.0,
        "selected": {
            "order": selection.order,
            "coefficients_standardized": selection.coefficients.tolist(),
            "x_mean": selection.x_mean,
            "x_scale": selection.x_scale,
            "y_mean": selection.y_mean,
            "y_scale": selection.y_scale,
        },
        "candidates": [
            {
                "order": row.order,
                "validation_aic_mean": row.validation_aic_mean,
                "validation_aic_by_record": list(
                    row.validation_aic_by_record
                ),
                "effective_rank": row.effective_rank,
                "condition_number": row.condition_number,
                "stable_simulation": row.stable_simulation,
            }
            for row in selection.candidates
        ],
        "runtime": {
            "elapsed_seconds": selection.elapsed_seconds,
            "dtype": "float64",
            "device": "cpu",
        },
        "status": "COMPLETED",
        "scientific_role": "DEVELOPMENT_BASELINE_SELECTION_NOT_TEST_EVIDENCE",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        f"selected order={selection.order} "
        f"elapsed={selection.elapsed_seconds:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
