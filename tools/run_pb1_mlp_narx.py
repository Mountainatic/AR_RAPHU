#!/usr/bin/env python3
"""Train or select peer-reviewed PB1 MLP-NARX candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.baselines.mlp_narx_champneys2024 import (
    MLPWeights,
    MinMaxScaling,
    train_mlp_narx,
    validation_aic,
)
from ar_raphu.datasets.loaders import load_pwh, load_whpn
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    load_pb1_protocol_freeze,
)


LOADERS = {"pwh": load_pwh, "whpn": load_whpn}
WIDTHS = (2, 5, 7, 10)
SEEDS = (0, 1, 2, 3, 4)


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


def _history(name: str) -> tuple[int, int]:
    h1 = _json(
        ROOT
        / "results/public_benchmarks/pb1"
        / name
        / "development/H1_ARX_NO_FUTURE_X/history_selection.json"
    )
    return int(h1["selected"]["nx"]), int(h1["selected"]["ny"])


def _candidate_dir(name: str) -> Path:
    return (
        ROOT
        / "results/public_benchmarks/pb1"
        / name
        / "development/MLPNARX_CHAMPNEYS2024/candidates"
    )


def _save_candidate(
    name: str,
    *,
    width: int,
    seed: int,
    nx: int,
    ny: int,
    result,
) -> Path:
    folder = _candidate_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"width_{width:02d}_seed_{seed:02d}"
    weights_path = folder / f"{stem}.npz"
    np.savez_compressed(
        weights_path,
        hidden_weight=result.weights.hidden_weight,
        hidden_bias=result.weights.hidden_bias,
        output_weight=result.weights.output_weight,
        output_bias=np.array(result.weights.output_bias),
    )
    payload = {
        "schema_version": 6,
        "dataset": name,
        "model": "MLPNARX_CHAMPNEYS2024",
        "width": width,
        "seed": seed,
        "history": {"nx": nx, "ny": ny},
        "iterations_completed": 20_000,
        "early_stopping": False,
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "training_loss": "RMSE",
        "final_training_rmse_scaled": result.final_training_rmse,
        "scaling": {
            "x_min": result.scaling.x_min,
            "x_max": result.scaling.x_max,
            "y_min": result.scaling.y_min,
            "y_max": result.scaling.y_max,
        },
        "weights_file": weights_path.name,
        "runtime_seconds": result.elapsed_seconds,
        "dtype": "float32",
        "device": "cpu",
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "status": "COMPLETED",
    }
    json_path = folder / f"{stem}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return json_path


def _load_weights(path: Path) -> MLPWeights:
    values = np.load(path)
    return MLPWeights(
        hidden_weight=values["hidden_weight"],
        hidden_bias=values["hidden_bias"],
        output_weight=values["output_weight"],
        output_bias=float(values["output_bias"]),
    )


def train(args) -> int:
    output = _candidate_dir(args.dataset) / (
        f"width_{args.width:02d}_seed_{args.seed:02d}.json"
    )
    if output.exists() and not args.force_development:
        raise FileExistsError(f"{output} already exists.")
    dataset = _dataset(args.dataset, args.raw_root)
    nx, ny = _history(args.dataset)
    result = train_mlp_narx(
        dataset,
        nx=nx,
        ny=ny,
        width=args.width,
        seed=args.seed,
        iterations=20_000,
        learning_rate=0.01,
    )
    path = _save_candidate(
        args.dataset,
        width=args.width,
        seed=args.seed,
        nx=nx,
        ny=ny,
        result=result,
    )
    print(path)
    print(
        f"width={args.width} seed={args.seed} "
        f"rmse={result.final_training_rmse:.8g} "
        f"seconds={result.elapsed_seconds:.3f}"
    )
    return 0


def select(args) -> int:
    dataset = _dataset(args.dataset, args.raw_root)
    nx, ny = _history(args.dataset)
    rows = []
    for width in WIDTHS:
        for seed in SEEDS:
            stem = f"width_{width:02d}_seed_{seed:02d}"
            metadata_path = _candidate_dir(args.dataset) / f"{stem}.json"
            weights_path = _candidate_dir(args.dataset) / f"{stem}.npz"
            if not metadata_path.exists() or not weights_path.exists():
                raise FileNotFoundError(f"Missing candidate {stem}.")
            metadata = _json(metadata_path)
            scaling = MinMaxScaling(**metadata["scaling"])
            scores, stable = validation_aic(
                dataset,
                nx=nx,
                ny=ny,
                weights=_load_weights(weights_path),
                scaling=scaling,
            )
            rows.append(
                {
                    "width": width,
                    "seed": seed,
                    "validation_aic_mean": float(np.mean(scores)),
                    "validation_aic_by_record": list(scores),
                    "stable_simulation": stable,
                    "final_training_rmse_scaled": metadata[
                        "final_training_rmse_scaled"
                    ],
                    "runtime_seconds": metadata["runtime_seconds"],
                    "metadata_file": str(metadata_path.relative_to(ROOT)),
                    "weights_file": str(weights_path.relative_to(ROOT)),
                }
            )
    selected = min(
        rows,
        key=lambda row: (
            row["validation_aic_mean"],
            row["width"],
            row["seed"],
        ),
    )
    output = _candidate_dir(args.dataset).parent / "selection.json"
    if output.exists() and not args.force_development:
        raise FileExistsError(f"{output} already exists.")
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "dataset": args.dataset,
        "stage": "development",
        "model": "MLPNARX_CHAMPNEYS2024",
        "profile_source": "PEER_REVIEWED_PAPER",
        "history_source": "H1_ARX_AIC_SELECTED",
        "history": {"nx": nx, "ny": ny},
        "widths": list(WIDTHS),
        "initializations": list(SEEDS),
        "iterations": 20_000,
        "early_stopping": False,
        "selection_metric": "MEAN_RECORD_VALIDATION_AIC",
        "forecast_alignment": "X_THROUGH_T_AND_Y_THROUGH_T_TO_TARGET_Y_T_PLUS_1",
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "selected": selected,
        "candidates": rows,
        "status": (
            "COMPLETED"
            if np.isfinite(selected["validation_aic_mean"])
            else "FAILED"
        ),
        "scientific_role": "DEVELOPMENT_BASELINE_SELECTION_NOT_TEST_EVIDENCE",
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        f"selected width={selected['width']} seed={selected['seed']} "
        f"aic={selected['validation_aic_mean']:.8g}"
    )
    return 0 if payload["status"] == "COMPLETED" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(LOADERS), required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    parser.add_argument("--force-development", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--width", type=int, choices=WIDTHS, required=True)
    train_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    subparsers.add_parser("select")
    args = parser.parse_args()
    return train(args) if args.command == "train" else select(args)


if __name__ == "__main__":
    raise SystemExit(main())
