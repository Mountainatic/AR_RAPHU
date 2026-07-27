#!/usr/bin/env python3
"""Summarize PB1 X/AR/XAR direct-horizon development evidence."""

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
from ar_raphu.datasets.scaling import TrainOnlyStandardizer
from ar_raphu.datasets.windowing import build_windowed_task


LOADERS = {"pwh": load_pwh, "whpn": load_whpn}
HORIZONS = (1, 5, 10, 20)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset(name: str, raw_root: Path):
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
    return apply_pb1_development_partition(
        LOADERS[name](raw_root, include_test=False),
        freeze,
        whpn_audit=audit,
    )


def _persistence_mse(dataset, *, L_x: int, L_y: int, horizon: int) -> float:
    scaler = TrainOnlyStandardizer.fit(dataset)
    task = build_windowed_task(
        scaler.transform(dataset),
        target=0,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        include_splits=("train", "validation"),
    )
    validation = task.split == "validation"
    error = task.target[validation] - task.y_history[validation, 0]
    return float(np.mean(error * error))


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
    h1 = _json(
        base / "H1_ARX_NO_FUTURE_X/history_selection.json"
    )
    L_x = int(h1["selected"]["nx"])
    L_y = int(h1["selected"]["ny"])
    dataset = _dataset(args.dataset, args.raw_root)
    rows = []
    failed_tracks = []
    for horizon in HORIZONS:
        folder = (
            base
            / "H3_SHARED_HISTORY"
            / f"SPECTRAL_PILOT_H{horizon}"
        )
        artifacts = {
            "X": _json(folder / "spectral_X.json"),
            "AR": _json(folder / "spectral_AR.json"),
            "XAR": _json(folder / "full_spectral.json"),
        }
        persistence = _persistence_mse(
            dataset, L_x=L_x, L_y=L_y, horizon=horizon
        )
        tracks = {}
        for track, artifact in artifacts.items():
            mse = float(artifact["penalty"]["selected"]["validation_mse_mean"])
            tracks[track] = {
                "status": artifact["status"],
                "penalty_status": artifact["penalty"]["status"],
                "validation_mse_group_mean": mse,
                "validation_rmse_group_mean": float(np.sqrt(mse)),
                "skill_vs_persistence": float(1.0 - mse / persistence),
                "relative_kkt_residual": artifact["penalty"]["selected"][
                    "relative_kkt_residual"
                ],
            }
            if artifact["status"] != "COMPLETED":
                failed_tracks.append(
                    {
                        "horizon": horizon,
                        "track": track,
                        "status": artifact["status"],
                        "penalty_status": artifact["penalty"]["status"],
                    }
                )
        incremental = None
        if (
            artifacts["AR"]["status"] == "COMPLETED"
            and artifacts["XAR"]["status"] == "COMPLETED"
        ):
            delta = (
                tracks["AR"]["validation_mse_group_mean"]
                - tracks["XAR"]["validation_mse_group_mean"]
            )
            incremental = {
                "delta_X_given_AR_mse": delta,
                "relative_AR_mse_reduction": float(
                    delta / tracks["AR"]["validation_mse_group_mean"]
                ),
                "status": "COMPLETED",
            }
        else:
            incremental = {
                "status": "FAILED_DEPENDENCY_TRACK_GATE",
                "delta_X_given_AR_mse": None,
                "relative_AR_mse_reduction": None,
            }
        rank = artifacts["XAR"]["rank_audit_after_penalty_freeze"]
        rows.append(
            {
                "horizon": horizon,
                "persistence_mse_pooled": persistence,
                "tracks": tracks,
                "incremental_external_value": incremental,
                "external_predictive_effective_ranks": rank[
                    "predictive_effective_ranks"
                ],
                "external_first_mode_energy": rank[
                    "normalized_spectral_energy"
                ][0],
            }
        )
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "dataset": args.dataset,
        "stage": "development",
        "lane": "H3_SHARED_HISTORY_FAIRNESS",
        "history": {"L_x": L_x, "L_y": L_y, "source": "H1_ARX_AIC"},
        "horizons": rows,
        "failed_track_gates": failed_tracks,
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "status": "COMPLETED" if not failed_tracks else "PARTIAL",
        "structural_rank_claim_allowed": False,
    }
    output = base / "H3_SHARED_HISTORY/DIRECT_HORIZON_SUMMARY.json"
    if output.exists() and not args.force_development:
        raise FileExistsError(f"{output} already exists.")
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    for row in rows:
        delta = row["incremental_external_value"]
        print(
            row["horizon"],
            row["tracks"]["AR"]["validation_mse_group_mean"],
            row["tracks"]["XAR"]["validation_mse_group_mean"],
            delta["relative_AR_mse_reduction"],
            row["external_predictive_effective_ranks"]["0.05"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
