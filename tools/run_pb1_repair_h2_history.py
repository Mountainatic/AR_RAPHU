#!/usr/bin/env python3
"""Run resumable PB1 Repair-V2 H2 native-history screening."""

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
from ar_raphu.spectral.amplitude_domain import AmplitudeOutOfDomainError
from ar_raphu.spectral.pb1_development import fit_pb1_shared_history_spectral
from ar_raphu.spectral.pb1_selection import (
    H2HistoryScore,
    history_complexity_key,
    select_h2_history_one_se,
)


LOADERS = {
    "pwh": load_pwh,
    "whpn": load_whpn,
    "cascaded_tanks": load_cascaded_tanks,
    "silverbox": load_silverbox,
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(child) for child in value]
    return value


def _partition(name: str, raw_root: Path, config: dict) -> object:
    raw = LOADERS[name](raw_root, include_test=False)
    if name in {"cascaded_tanks", "silverbox"}:
        return apply_pb1_repair_v2_partition(raw, config)
    freeze = load_pb1_protocol_freeze(
        ROOT / "configs/public_benchmarks/PB1_PROTOCOL_FREEZE.json"
    )
    audit = (
        _json(
            ROOT
            / "results/public_benchmarks/pb1/protocol_audit"
            / "whpn_realization_audit.json"
        )
        if name == "whpn"
        else None
    )
    return apply_pb1_development_partition(raw, freeze, whpn_audit=audit)


def _root(dataset: str, horizon: int) -> Path:
    return (
        ROOT
        / "results/public_benchmarks/pb1_repair_v2"
        / dataset
        / "development/H2_NATIVE_HISTORY"
        / f"DIRECT_H{horizon}"
    )


def _candidate(args: argparse.Namespace, config: dict) -> int:
    if args.L_x not in config["task"]["lx_grid"]:
        raise ValueError("L_x is outside the preregistered candidate set.")
    if args.L_y not in config["task"]["ly_grid"]:
        raise ValueError("L_y is outside the preregistered candidate set.")
    output = (
        _root(args.dataset, args.horizon)
        / "history_candidates"
        / f"Lx_{args.L_x:03d}_Ly_{args.L_y:03d}.json"
    )
    if output.exists() and not args.force_development:
        raise FileExistsError(f"{output} already exists.")
    dataset = _partition(args.dataset, args.raw_root, config)
    penalty = config["selection"]["spectral_penalty"]
    positive_points = int(
        penalty.get(
            "grid_points_per_positive_axis",
            penalty.get("grid_points_per_axis"),
        )
    )
    try:
        fit = fit_pb1_shared_history_spectral(
            dataset,
            L_x=args.L_x,
            L_y=args.L_y,
            horizon=args.horizon,
            lag_kind="discrete_identity",
            lag_count=None,
            amplitude_count=max(config["basis"]["amplitude_count_grid"]),
            grid_points=positive_points,
            maximum_expansions=int(penalty["boundary_expansions_max"]),
            track="XAR",
        )
        selected = fit.selected
        status = (
            "COMPLETED"
            if fit.penalty_status == "PENALTY_INTERVAL_CERTIFIED"
            and selected.relative_kkt_residual <= 1.0e-8
            else "FAILED"
        )
        payload = {
            "schema_version": 7,
            "dataset": args.dataset,
            "stage": "development_repair",
            "lane": "H2_NATIVE_HISTORY",
            "horizon": args.horizon,
            "history": {"L_x": args.L_x, "L_y": args.L_y},
            "anchor_representation": {
                "lag_basis": "discrete_identity",
                "amplitude_count": max(
                    config["basis"]["amplitude_count_grid"]
                ),
            },
            "penalty_status": fit.penalty_status,
            "validation_loss": selected.validation_mse_mean,
            "validation_se": selected.validation_mse_se,
            "effective_df": selected.effective_df,
            "selected_penalty": {
                "lag_weight": selected.lag_weight,
                "amplitude_weight": selected.amplitude_weight,
                "ridge_weight": selected.ridge_weight,
            },
            "relative_kkt_residual": selected.relative_kkt_residual,
            "solver_diagnostics": selected.solver_diagnostics,
            "candidate_count": len(fit.candidates),
            "elapsed_seconds": fit.elapsed_seconds,
            "official_test_rows_loaded": 0,
            "official_test_access_count": 0,
            "confirmation_allowed": False,
            "status": status,
        }
    except AmplitudeOutOfDomainError as error:
        payload = {
            "schema_version": 7,
            "dataset": args.dataset,
            "stage": "development_repair",
            "lane": "H2_NATIVE_HISTORY",
            "horizon": args.horizon,
            "history": {"L_x": args.L_x, "L_y": args.L_y},
            "official_test_rows_loaded": 0,
            "official_test_access_count": 0,
            "confirmation_allowed": False,
            "status": "FAILED_REPRESENTATION_COVERAGE",
            "failure": str(error),
        }
    payload["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(output)
    print(f"status={payload['status']}")
    return 0 if payload["status"] == "COMPLETED" else 1


def _aggregate(args: argparse.Namespace, config: dict) -> int:
    root = _root(args.dataset, args.horizon)
    expected = [
        (int(L_x), int(L_y))
        for L_x in config["task"]["lx_grid"]
        for L_y in config["task"]["ly_grid"]
    ]
    payloads = []
    missing = []
    for L_x, L_y in expected:
        path = (
            root
            / "history_candidates"
            / f"Lx_{L_x:03d}_Ly_{L_y:03d}.json"
        )
        if not path.exists():
            missing.append([L_x, L_y])
        else:
            payloads.append(_json(path))
    if missing:
        raise RuntimeError(f"Missing H2 history candidates: {missing}")
    rows = [
        H2HistoryScore(
            L_x=int(row["history"]["L_x"]),
            L_y=int(row["history"]["L_y"]),
            validation_loss=float(row["validation_loss"]),
            validation_se=float(row["validation_se"]),
            status=str(row["status"]),
        )
        for row in payloads
        if row["status"] == "COMPLETED"
    ]
    selected = select_h2_history_one_se(rows)
    minimum = min(rows, key=lambda row: row.validation_loss)
    output_payload = {
        "schema_version": 7,
        "dataset": args.dataset,
        "stage": "development_repair",
        "lane": "H2_NATIVE_HISTORY",
        "horizon": args.horizon,
        "selector": "VALIDATION_ONE_STANDARD_ERROR",
        "complexity_key": [
            "Lx_plus_Ly",
            "Lx_times_Ly",
            "max_Lx_Ly",
            "Lx",
            "Ly",
        ],
        "minimum": {
            "L_x": minimum.L_x,
            "L_y": minimum.L_y,
            "validation_loss": minimum.validation_loss,
            "validation_se": minimum.validation_se,
        },
        "selected": {
            "L_x": selected.L_x,
            "L_y": selected.L_y,
            "validation_loss": selected.validation_loss,
            "validation_se": selected.validation_se,
            "complexity": list(history_complexity_key(selected)),
        },
        "history_range_edge_selected": (
            selected.L_x == max(config["task"]["lx_grid"])
            or selected.L_y == max(config["task"]["ly_grid"])
        ),
        "candidate_count": len(expected),
        "official_test_access_count": 0,
        "confirmation_allowed": False,
        "status": "H2_HISTORY_FROZEN",
    }
    output = root / "history_selection.json"
    output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        f"selected=({selected.L_x},{selected.L_y}) "
        f"loss={selected.validation_loss:.8g}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("candidate", "aggregate"))
    parser.add_argument("--dataset", choices=tuple(LOADERS), required=True)
    parser.add_argument(
        "--horizon", type=int, choices=(1, 5, 10, 20), required=True
    )
    parser.add_argument("--L-x", dest="L_x", type=int)
    parser.add_argument("--L-y", dest="L_y", type=int)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    parser.add_argument("--force-development", action="store_true")
    args = parser.parse_args()
    config = _json(
        ROOT / f"configs/public_benchmarks/pb1_{args.dataset}.yaml"
    )
    if args.horizon not in config["task"]["horizons"]:
        raise ValueError("Horizon is outside the preregistered set.")
    if args.command == "candidate":
        if args.L_x is None or args.L_y is None:
            raise ValueError("candidate requires --L-x and --L-y.")
        return _candidate(args, config)
    return _aggregate(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
