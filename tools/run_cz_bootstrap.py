#!/usr/bin/env python3
"""Blocked bootstrap for CZ incremental predictive value."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np
import torch

from ar_raphu.cz_real.frozen import load_frozen_horizon
from ar_raphu.cz_real.linear import TrainScaler, fit_ols, target_indices
from ar_raphu.cz_real.orss_r3 import evaluate_frozen_configuration
from ar_raphu.cz_real.protocol import (
    build_development_folds,
    load_furnace_a,
    load_furnace_b,
)
from ar_raphu.orss.diagnostics import write_json
from ar_raphu.orss.penalties import PenaltyWeights


def _furnace_a(raw_dir: Path) -> Path:
    candidates = [path for path in raw_dir.iterdir() if path.name.endswith("1.xlsx")]
    if len(candidates) != 1:
        raise RuntimeError("Furnace-A path is not uniquely identifiable.")
    return candidates[0]


def _ar_fit(
    train_data,
    evaluation_y: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    train_stop: int,
    horizon: int,
    L_y: int,
) -> np.ndarray:
    scaler = TrainScaler.fit(
        train_data.inputs, train_data.target, train_stop
    )
    train_indices = target_indices(
        start=0,
        stop=train_stop,
        horizon=horizon,
        max_history=L_y,
    )
    train_origins = train_indices - horizon
    offsets = np.arange(L_y, dtype=np.int64)
    train_windows = (
        train_data.target[
            train_origins[:, None] - offsets[None, :]
        ]
        - scaler.y_mean
    ) / scaler.y_scale
    train_target = (
        train_data.target[train_indices] - scaler.y_mean
    ) / scaler.y_scale
    fitted = fit_ols(train_windows, train_target)
    origins = evaluation_indices - horizon
    windows = (
        evaluation_y[origins[:, None] - offsets[None, :]]
        - scaler.y_mean
    ) / scaler.y_scale
    return scaler.y_mean + scaler.y_scale * fitted.predict(windows)


def _block_length(residual: np.ndarray) -> int:
    centered = residual - residual.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 16
    integrated = 1.0
    maximum = min(512, len(centered) // 4)
    for lag in range(1, maximum + 1):
        correlation = float(
            np.dot(centered[:-lag], centered[lag:]) / denominator
        )
        if correlation <= 0.0:
            break
        integrated += 2.0 * correlation
    return int(np.clip(np.ceil(integrated), 16, max(16, maximum)))


def _sample_indices(
    length: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    starts = rng.integers(0, length, size=int(np.ceil(length / block)))
    offsets = np.arange(block)
    return ((starts[:, None] + offsets[None, :]) % length).reshape(-1)[:length]


def _bootstrap(
    target_groups: list[np.ndarray],
    ours_groups: list[np.ndarray],
    ar_groups: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    blocks = [
        _block_length(ours - target)
        for target, ours in zip(target_groups, ours_groups, strict=True)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 2), dtype=np.float64)
    for replicate in range(replicates):
        ours_squared = []
        ar_squared = []
        for target, ours, ar, block in zip(
            target_groups, ours_groups, ar_groups, blocks, strict=True
        ):
            indices = _sample_indices(len(target), block, rng)
            ours_squared.append((ours[indices] - target[indices]) ** 2)
            ar_squared.append((ar[indices] - target[indices]) ** 2)
        ours_mse = float(np.mean(np.concatenate(ours_squared)))
        ar_mse = float(np.mean(np.concatenate(ar_squared)))
        samples[replicate, 0] = np.sqrt(ar_mse) - np.sqrt(ours_mse)
        samples[replicate, 1] = (ar_mse - ours_mse) / max(ar_mse, 1.0e-15)
    point_ours = float(
        np.mean(
            np.concatenate(
                [
                    (ours - target) ** 2
                    for target, ours in zip(
                        target_groups, ours_groups, strict=True
                    )
                ]
            )
        )
    )
    point_ar = float(
        np.mean(
            np.concatenate(
                [
                    (ar - target) ** 2
                    for target, ar in zip(
                        target_groups, ar_groups, strict=True
                    )
                ]
            )
        )
    )
    return {
        "replicates": replicates,
        "block_lengths": blocks,
        "Delta_RMSE_AR_minus_ours_mm": {
            "point": np.sqrt(point_ar) - np.sqrt(point_ours),
            "CI95": [
                float(np.quantile(samples[:, 0], 0.025)),
                float(np.quantile(samples[:, 0], 0.975)),
            ],
        },
        "Delta_U_given_AR": {
            "point": (point_ar - point_ours) / max(point_ar, 1.0e-15),
            "CI95": [
                float(np.quantile(samples[:, 1], 0.025)),
                float(np.quantile(samples[:, 1], 0.975)),
            ],
        },
    }, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--development-replicates", type=int, required=True)
    parser.add_argument("--confirmation-replicates", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--raw-dir",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw",
    )
    args = parser.parse_args()
    root = Path(args.results_root)
    output = root / "bootstrap"
    status = output / "BOOTSTRAP_STATUS.json"
    if args.resume and status.exists():
        existing = json.loads(status.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            print("BOOTSTRAP_RESUMED")
            return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    continuation = json.loads(
        (root / "R3C" / "continuation_selection.json").read_text(
            encoding="utf-8"
        )
    )
    furnace_a = load_furnace_a(_furnace_a(Path(args.raw_dir)))
    furnace_b = load_furnace_b(
        Path(args.raw_dir) / "实验数据1-张.xlsx",
        protocol_frozen=True,
        stage="R9",
    )
    device = torch.device(config["runtime"]["device"])
    locked = root / "frozen_model"
    manifest = json.loads(
        (locked / "FROZEN_MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )
    summary: dict[str, object] = {}
    output.mkdir(parents=True, exist_ok=True)
    for horizon_text, frozen in continuation["selections"].items():
        horizon = int(horizon_text)
        L_x = int(frozen["history"]["L_x"])
        L_y = int(frozen["history"]["L_y"])
        development = evaluate_frozen_configuration(
            furnace_a.inputs,
            furnace_a.target,
            horizon=horizon,
            L_x=L_x,
            L_y=L_y,
            M_tau=int(frozen["resolution"]["M_tau"]),
            M_x=int(frozen["resolution"]["M_x"]),
            normalized_weights=PenaltyWeights(**frozen["penalty"]),
            c_rho=float(
                frozen["selected_CONTINUATION_SCALE_COEFFICIENT"]
            ),
            device=device,
            chunk_time=int(config["cuda"]["operator_chunk_time"]),
            maximum_iterations=int(config["krylov"]["maximum_iterations"]),
            include_predictions=True,
        )
        dev_targets = []
        dev_ours = []
        dev_ar = []
        folds = build_development_folds(L_x=L_x, L_y=L_y)
        for fold, row in zip(folds, development["folds"], strict=True):
            indices = np.asarray(
                row.pop("validation_target_indices"), dtype=np.int64
            )
            target = np.asarray(row.pop("validation_target"))
            ours = np.asarray(row.pop("teacher_forced_prediction"))
            dev_targets.append(target)
            dev_ours.append(ours)
            dev_ar.append(
                _ar_fit(
                    furnace_a,
                    furnace_a.target,
                    indices,
                    train_stop=fold.effective_train_stop,
                    horizon=horizon,
                    L_y=L_y,
                )
            )
        development_summary, development_samples = _bootstrap(
            dev_targets,
            dev_ours,
            dev_ar,
            replicates=args.development_replicates,
            seed=8100 + horizon,
        )
        model = load_frozen_horizon(
            locked / manifest["models"][horizon_text]["path"]
        )
        scopes = {}
        raw_samples = {"development": development_samples}
        for name, dataset, artifact in (
            (
                "furnace_A_confirmation",
                furnace_a,
                root / "R5" / f"h_{horizon:03d}_predictions.npz",
            ),
            (
                "furnace_B_zero_shot",
                furnace_b,
                root / "R6" / f"h_{horizon:03d}_predictions.npz",
            ),
        ):
            values = np.load(artifact)
            indices = values["target_indices"]
            target = values["target"]
            ours = values["prediction"]
            ar = _ar_fit(
                furnace_a,
                dataset.target,
                indices,
                train_stop=int(model["effective_train_stop"]),
                horizon=horizon,
                L_y=L_y,
            )
            scope_summary, scope_samples = _bootstrap(
                [target],
                [ours],
                [ar],
                replicates=args.confirmation_replicates,
                seed=9100 + horizon + (100 if name.startswith("furnace_B") else 0),
            )
            scopes[name] = scope_summary
            raw_samples[name] = scope_samples
        summary[horizon_text] = {
            "development": development_summary,
            **scopes,
        }
        np.savez_compressed(
            output / f"h_{horizon:03d}_bootstrap_samples.npz",
            **raw_samples,
        )
        torch.cuda.empty_cache()
    write_json(
        output / "bootstrap_summary.json",
        {
            "schema": "CZ_BLOCK_BOOTSTRAP_V1",
            "status": "COMPLETED",
            "development_replicates": args.development_replicates,
            "confirmation_replicates": args.confirmation_replicates,
            "unit": "CONTIGUOUS_CIRCULAR_TIME_BLOCK",
            "models": summary,
            "BOOTSTRAP_COMPLETE": True,
        },
    )
    write_json(
        status,
        {
            "status": "COMPLETED",
            "BOOTSTRAP_COMPLETE": True,
            "next_stage": "FINAL_REPORT_AND_PACKAGE",
        },
    )


if __name__ == "__main__":
    main()
