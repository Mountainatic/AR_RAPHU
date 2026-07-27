#!/usr/bin/env python3
"""Run the frozen 250-cluster development bootstrap for PB1 spectral rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.datasets.loaders import load_pwh, load_whpn
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    load_pb1_protocol_freeze,
)
from ar_raphu.spectral.pb1_development import (
    bootstrap_external_rank_spectrum,
    fit_pb1_shared_history_spectral,
)


LOADERS = {"pwh": load_pwh, "whpn": load_whpn}


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

    base = (
        ROOT / "results/public_benchmarks/pb1" / args.dataset / "development"
    )
    pilot_path = (
        base / "H3_SHARED_HISTORY/SPECTRAL_PILOT_H1/full_spectral.json"
    )
    pilot = _json(pilot_path)
    if pilot["status"] != "COMPLETED":
        raise RuntimeError("The frozen H3 spectral pilot did not complete.")
    freeze = load_pb1_protocol_freeze(
        ROOT / "configs/public_benchmarks/PB1_PROTOCOL_FREEZE.json"
    )
    audit = (
        _json(
            ROOT
            / "results/public_benchmarks/pb1/protocol_audit"
            / "whpn_realization_audit.json"
        )
        if args.dataset == "whpn"
        else None
    )
    dataset = apply_pb1_development_partition(
        LOADERS[args.dataset](args.raw_root, include_test=False),
        freeze,
        whpn_audit=audit,
    )
    fit = fit_pb1_shared_history_spectral(
        dataset,
        L_x=int(pilot["history"]["L_x"]),
        L_y=int(pilot["history"]["L_y"]),
        horizon=1,
        lag_kind="discrete_identity",
        amplitude_count=int(pilot["basis"]["amplitude"]["count"]),
        grid_points=int(pilot["penalty"]["grid_points_per_axis"]),
        maximum_expansions=int(pilot["penalty"]["maximum_expansions"]),
    )
    original = pilot["penalty"]["selected"]
    rerun = fit.selected
    comparisons = (
        (rerun.lag_weight, original["lag_weight"]),
        (rerun.amplitude_weight, original["amplitude_weight"]),
        (rerun.ridge_weight, original["ridge_weight"]),
        (rerun.validation_mse_mean, original["validation_mse_mean"]),
    )
    if any(
        not np.isclose(left, right, rtol=1.0e-10, atol=1.0e-14)
        for left, right in comparisons
    ):
        raise RuntimeError("Frozen spectral pilot did not reproduce.")
    bootstrap = bootstrap_external_rank_spectrum(
        fit,
        replicates=250,
        seed=20240727,
    )
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "dataset": args.dataset,
        "stage": "development",
        "lane": "H3_SHARED_HISTORY_FAIRNESS",
        "model": "FULL_SPECTRAL_AR_RAPHU",
        "bootstrap_role": "FROZEN_MODEL_EXTERNAL_SPECTRUM_STABILITY",
        "resampling_unit": (
            "PHASE_CLUSTER_ALL_AMPLITUDES_AND_BOTH_PERIODS"
            if args.dataset == "pwh"
            else "WHOLE_REALIZATION"
        ),
        "retunes_penalty": False,
        "retunes_resolution": False,
        "replicates": bootstrap.replicates,
        "seed": bootstrap.seed,
        "singular_value_quantiles": {
            key: values.tolist()
            for key, values in bootstrap.singular_value_quantiles.items()
        },
        "normalized_energy_quantiles": {
            key: values.tolist()
            for key, values in bootstrap.normalized_energy_quantiles.items()
        },
        "spectral_tail_budget_rank_frequencies": (
            bootstrap.spectral_tail_budget_rank_frequencies
        ),
        "maximum_relative_kkt_residual": (
            bootstrap.maximum_relative_kkt_residual
        ),
        "runtime_seconds": fit.elapsed_seconds + bootstrap.elapsed_seconds,
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "structural_rank_claim_allowed": False,
        "status": (
            "COMPLETED"
            if bootstrap.maximum_relative_kkt_residual <= 1.0e-8
            else "FAILED"
        ),
    }
    output = pilot_path.parent / "rank_bootstrap.json"
    if output.exists() and not args.force_development:
        raise FileExistsError(f"{output} already exists.")
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        f"status={payload['status']} replicates={bootstrap.replicates} "
        f"max_kkt={bootstrap.maximum_relative_kkt_residual:.3e} "
        f"seconds={payload['runtime_seconds']:.3f}"
    )
    return 0 if payload["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
