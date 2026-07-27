#!/usr/bin/env python3
"""Run H3 shared-history spectral development at the first declared resolution."""

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

from ar_raphu.datasets.loaders import load_pwh, load_whpn
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    load_pb1_protocol_freeze,
)
from ar_raphu.spectral.pb1_development import (
    fit_pb1_shared_history_spectral,
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
    parser.add_argument(
        "--horizon", type=int, choices=(1, 5, 10, 20), default=1
    )
    parser.add_argument(
        "--track", choices=("X", "AR", "XAR"), default="XAR"
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
    config_path = (
        ROOT / f"configs/public_benchmarks/pb1_{args.dataset}.yaml"
    )
    config = _json(config_path)
    h1_path = (
        ROOT
        / "results/public_benchmarks/pb1"
        / args.dataset
        / "development/H1_ARX_NO_FUTURE_X/history_selection.json"
    )
    h1 = _json(h1_path)
    L_x = int(h1["selected"]["nx"])
    L_y = int(h1["selected"]["ny"])
    lag_candidate = config["basis"]["lag_candidates"][0]
    if lag_candidate != {"type": "discrete_identity"}:
        raise ValueError("H3 pilot must use the first declared lag resolution.")
    amplitude_count = int(config["basis"]["amplitude_count_grid"][0])
    penalty = config["selection"]["spectral_penalty"]
    if args.horizon not in config["task"]["horizons"]:
        raise ValueError("Horizon is not frozen in the dataset config.")
    fit = fit_pb1_shared_history_spectral(
        dataset,
        L_x=L_x,
        L_y=L_y,
        horizon=args.horizon,
        lag_kind="discrete_identity",
        lag_count=None,
        amplitude_count=amplitude_count,
        grid_points=int(penalty["grid_points_per_axis"]),
        maximum_expansions=int(penalty["boundary_expansions_max"]),
        track=args.track,
    )
    output = (
        ROOT
        / "results/public_benchmarks/pb1"
        / args.dataset
        / (
            "development/H3_SHARED_HISTORY/"
            f"SPECTRAL_PILOT_H{args.horizon}"
        )
        / (
            "full_spectral.json"
            if args.track == "XAR"
            else f"spectral_{args.track}.json"
        )
    )
    if output.exists() and not args.force_development:
        raise FileExistsError(f"{output} already exists.")
    selected = fit.selected
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "dataset": args.dataset,
        "stage": "development",
        "lane": "H3_SHARED_HISTORY_FAIRNESS",
        "role": "PENALTY_CERTIFICATION_AT_FIRST_PREDECLARED_RESOLUTION",
        "model": "FULL_SPECTRAL_AR_RAPHU",
        "track": args.track,
        "horizon": args.horizon,
        "use_future_x": False,
        "history": {"L_x": L_x, "L_y": L_y, "source": "H1_ARX_AIC"},
        "basis": {
            "lag": {"type": "discrete_identity"},
            "amplitude": {"type": "cubic_bspline", "count": amplitude_count},
            "x_design_shape_train": (
                None
                if fit.x_block is None
                else list(fit.x_block.train_matrix.shape)
            ),
            "ar_design_shape_train": (
                None
                if fit.ar_block is None
                else list(fit.ar_block.train_matrix.shape)
            ),
        },
        "penalty": {
            "normalization": "POSITIVE_GENERALIZED_EIGENVALUE_MEDIAN_RELATIVE_TO_TRAIN_GRAM",
            "grid_points_per_axis": int(penalty["grid_points_per_axis"]),
            "maximum_expansions": int(penalty["boundary_expansions_max"]),
            "selection": "GROUPED_VALIDATION_ONE_SE_LOWEST_EFFECTIVE_DF",
            "status": fit.penalty_status,
            "interval_history": list(fit.interval_history),
            "selected": {
                "lag_weight": selected.lag_weight,
                "amplitude_weight": selected.amplitude_weight,
                "ridge_weight": selected.ridge_weight,
                "validation_mse_mean": selected.validation_mse_mean,
                "validation_mse_by_group": list(
                    selected.validation_mse_by_group
                ),
                "validation_mse_se": selected.validation_mse_se,
                "effective_df": selected.effective_df,
                "relative_kkt_residual": selected.relative_kkt_residual,
                "numerical_jitter": selected.numerical_jitter,
                "intercept": selected.intercept,
                "coefficients": selected.coefficients.tolist(),
            },
            "candidate_count": len(fit.candidates),
        },
        "rank_audit_after_penalty_freeze": fit.rank_audit,
        "contracts": {
            "truth_available": False,
            "k_level_certificate": False,
            "structural_rank_claim_allowed": False,
            "predictive_svd_rank_claim_allowed": True,
        },
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": _sha256(config_path),
        "h1_sha256": _sha256(h1_path),
        "protocol_freeze_sha256": _sha256(freeze_path),
        "runtime": {
            "elapsed_seconds": fit.elapsed_seconds,
            "dtype": "float64",
            "device": "cpu",
        },
        "status": (
            "COMPLETED"
            if fit.penalty_status == "PENALTY_INTERVAL_CERTIFIED"
            and selected.relative_kkt_residual <= 1.0e-8
            else "FAILED"
        ),
        "confirmation_allowed": (
            fit.penalty_status == "PENALTY_INTERVAL_CERTIFIED"
            and selected.relative_kkt_residual <= 1.0e-8
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        f"status={payload['status']} penalty={fit.penalty_status} "
        f"mse={selected.validation_mse_mean:.8g} "
        f"edf={selected.effective_df:.4f} "
        f"seconds={fit.elapsed_seconds:.3f}"
    )
    return 0 if payload["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
