#!/usr/bin/env python3
"""Run frozen-model Repair-V2 rank bootstrap at one direct horizon."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.datasets.loaders import (
    load_cascaded_tanks,
    load_pwh,
    load_silverbox,
    load_whpn,
)
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    apply_pb1_repair_v2_partition,
    load_pb1_protocol_freeze,
)
from ar_raphu.spectral.pb1_development import (
    _block_penalties,
    bootstrap_external_rank_spectrum,
    fit_pb1_shared_history_spectral,
)
from ar_raphu.spectral.penalty_interval import normalize_penalty_relative_to_gram


LOADERS = {
    "pwh": load_pwh,
    "whpn": load_whpn,
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
    parser.add_argument(
        "--horizon", type=int, choices=(1, 5, 10, 20), required=True
    )
    args = parser.parse_args()

    base = (
        ROOT
        / "results/public_benchmarks/pb1_repair_v2"
        / args.dataset
        / "development"
    )
    pilot_path = (
        base
        / f"H3_SHARED_HISTORY/SPECTRAL_PILOT_H{args.horizon}/full_spectral.json"
    )
    pilot = _json(pilot_path)
    if pilot["status"] != "COMPLETED":
        raise RuntimeError("The frozen H3 spectral pilot did not complete.")
    config = _json(
        ROOT / f"configs/public_benchmarks/pb1_{args.dataset}.yaml"
    )
    raw = LOADERS[args.dataset](args.raw_root, include_test=False)
    if args.dataset in {"cascaded_tanks", "silverbox"}:
        dataset = apply_pb1_repair_v2_partition(raw, config)
    else:
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
            raw, freeze, whpn_audit=audit
        )
    fit = fit_pb1_shared_history_spectral(
        dataset,
        L_x=int(pilot["history"]["L_x"]),
        L_y=int(pilot["history"]["L_y"]),
        horizon=args.horizon,
        lag_kind="discrete_identity",
        amplitude_count=int(pilot["basis"]["amplitude"]["count"]),
        grid_points=int(pilot["penalty"]["positive_grid_points_per_axis"]),
        maximum_expansions=int(pilot["penalty"]["maximum_expansions"]),
    )
    frozen = pilot["penalty"]["selected"]
    blocks = tuple(
        block for block in (fit.x_block, fit.ar_block) if block is not None
    )
    centered = fit.train_matrix - fit.train_matrix.mean(axis=0)
    gram = centered.T @ centered / len(centered)
    normalized = tuple(
        normalize_penalty_relative_to_gram(component, gram).normalized
        for component in _block_penalties(blocks)
    )
    frozen_weights = (
        float(frozen["lag_weight"]),
        float(frozen["amplitude_weight"]),
        float(frozen["ridge_weight"]),
    )
    fit = replace(
        fit,
        selected_penalty=sum(
            weight * component
            for weight, component in zip(
                frozen_weights, normalized, strict=True
            )
        ),
    )
    bootstrap = bootstrap_external_rank_spectrum(
        fit,
        replicates=250,
        seed=20240727,
    )
    payload = {
        "schema_version": 7,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "dataset": args.dataset,
        "stage": "development_repair",
        "lane": "H3_SHARED_HISTORY_FAIRNESS",
        "model": "FULL_SPECTRAL_AR_RAPHU",
        "horizon": args.horizon,
        "bootstrap_role": "FROZEN_MODEL_EXTERNAL_SPECTRUM_STABILITY",
        "resampling_unit": config["bootstrap"]["primary_unit"],
        "automatic_block_length": bootstrap.automatic_block_length,
        "retunes_penalty": False,
        "retunes_resolution": False,
        "frozen_penalty_weights": {
            "lag_weight": frozen_weights[0],
            "amplitude_weight": frozen_weights[1],
            "ridge_weight": frozen_weights[2],
        },
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
        "confirmation_allowed": False,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
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
