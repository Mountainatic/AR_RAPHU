#!/usr/bin/env python3
"""Execute frozen early stages of CZ real-data protocol v1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ar_raphu.cz_real.linear import (
    TrainScaler,
    fit_ols,
    regression_metrics,
    target_indices,
    window_designs,
)
from ar_raphu.cz_real.protocol import (
    DIRECT_HORIZONS,
    EXPECTED_A_HEADERS,
    EXPECTED_B_HEADERS,
    FURNACE_A_SAMPLES,
    FURNACE_A_SHA256,
    FURNACE_B_SAMPLES,
    FURNACE_B_SHA256,
    PRIMARY_INPUTS,
    PROTOCOL_SCHEMA,
    TARGET,
    build_development_folds,
    confirmation_interval,
    file_sha256,
    load_furnace_a,
)
from ar_raphu.cz_real.spectral import fit_shared_history_smoke
from ar_raphu.spectral.amplitude_domain import AmplitudeOutOfDomainError


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _runtime() -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pid": os.getpid(),
    }


def run_r0(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir).resolve()
    output = Path(args.output).resolve() / "R0"
    a_path = raw_dir / "实验数据1.xlsx"
    b_path = raw_dir / "实验数据1-张.xlsx"
    a_hash = file_sha256(a_path)
    b_hash = file_sha256(b_path)
    if a_hash != FURNACE_A_SHA256 or b_hash != FURNACE_B_SHA256:
        raise RuntimeError("Frozen workbook hash mismatch.")

    # Only furnace A values are opened here. Furnace B is verified by whole-file
    # hash; its Sheet2 values remain unopened until the R7 lock-box transition.
    furnace_a = load_furnace_a(a_path)
    folds = build_development_folds(L_x=32, L_y=32)
    confirmation = confirmation_interval()
    gates = {
        "NO_RANDOM_SPLIT": True,
        "TRAIN_ONLY_SCALER": "ENFORCED_BY_R1_IMPLEMENTATION",
        "PURGE_PASS": all(
            fold.nominal_train_stop - fold.effective_train_stop == fold.purge_gap
            for fold in folds
        ),
        "FURNACE_B_NOT_USED_IN_TUNING": True,
        "FURNACE_B_VALUES_OPENED": False,
    }
    manifest = {
        "schema": "CZ_R0_DATA_AUDIT_V1",
        "protocol_schema": PROTOCOL_SCHEMA,
        "status": "COMPLETED",
        "runtime": _runtime(),
        "private_sources": {
            "furnace_A": {
                "filename": a_path.name,
                "sha256": a_hash,
                "sheet": "Sheet1",
                "samples": FURNACE_A_SAMPLES,
                "columns": len(EXPECTED_A_HEADERS),
                "missing_or_nonfinite_cells": 0,
                "values_opened": True,
            },
            "furnace_B": {
                "filename": b_path.name,
                "sha256": b_hash,
                "sheet_reserved": "Sheet2",
                "expected_samples_from_frozen_audit": FURNACE_B_SAMPLES,
                "expected_columns_from_frozen_audit": len(EXPECTED_B_HEADERS),
                "values_opened": False,
                "lock_state": "LOCKED_UNTIL_R7",
            },
        },
        "deduplication": {
            "furnace_B_Sheet1_equals_A": "VERIFIED_BY_FROZEN_PROTOCOL_AUDIT",
            "unique_sequences": 2,
        },
        "canonical_schema": {
            "primary_inputs": list(PRIMARY_INPUTS),
            "target": TARGET,
            "all_A_headers": list(furnace_a.canonical_headers),
        },
        "split_index_rule": "floor(N*fraction), 0-based, left-closed/right-open",
        "furnace_A_development_folds_R1_Lx32_Ly32": [
            fold.to_dict() for fold in folds
        ],
        "furnace_A_internal_confirmation": {
            "interval": list(confirmation),
            "values_opened_by_modeling": False,
            "lock_state": "LOCKED_UNTIL_R6",
        },
        "gates": gates,
        "physical_time_labels": "PROVISIONAL_PENDING_SAMPLING_CONFIRMATION",
    }
    _write_json(output / "R0_DATA_AUDIT.json", manifest)
    _write_json(
        output / "R0_STATUS.json",
        {
            "stage": "CZ_R0_DATA_AUDIT",
            "status": "COMPLETED",
            "next_stage": "CZ_R1_LINEAR_SMOKE",
            "furnace_B_lock": "LOCKED_UNTIL_R7",
            "furnace_A_confirmation_lock": "LOCKED_UNTIL_R6",
            "gates": gates,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _fold_model_results(
    x: np.ndarray,
    y: np.ndarray,
    *,
    horizon: int,
    fold: object,
    L_x: int,
    L_y: int,
) -> dict[str, object]:
    scaler = TrainScaler.fit(x, y, fold.effective_train_stop)
    train_targets = target_indices(
        start=0,
        stop=fold.effective_train_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    validation_targets = target_indices(
        start=fold.validation_start,
        stop=fold.validation_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    train_x, train_y_hist = window_designs(
        x,
        y,
        targets=train_targets,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        scaler=scaler,
    )
    val_x, val_y_hist = window_designs(
        x,
        y,
        targets=validation_targets,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        scaler=scaler,
    )
    train_target_scaled = (y[train_targets] - scaler.y_mean) / scaler.y_scale
    validation_target = y[validation_targets]

    ar_fit = fit_ols(train_y_hist, train_target_scaled)
    x_fit = fit_ols(train_x, train_target_scaled)
    arx_fit = fit_ols(
        np.column_stack((train_x, train_y_hist)), train_target_scaled
    )
    predictions = {
        "mean": np.full(len(validation_targets), scaler.y_mean),
        "persistence": y[validation_targets - horizon],
        "AR": scaler.y_mean + scaler.y_scale * ar_fit.predict(val_y_hist),
        "X_only": scaler.y_mean + scaler.y_scale * x_fit.predict(val_x),
        "ARX": scaler.y_mean
        + scaler.y_scale
        * arx_fit.predict(np.column_stack((val_x, val_y_hist))),
    }
    metrics = {
        model: regression_metrics(validation_target, prediction)
        for model, prediction in predictions.items()
    }
    ar_mse = metrics["AR"]["MSE_mm2"]
    for model in ("X_only", "ARX"):
        metrics[model]["relative_MSE_reduction_vs_AR"] = (
            (ar_mse - metrics[model]["MSE_mm2"]) / ar_mse
            if ar_mse > 0.0
            else float("nan")
        )
        metrics[model]["RMSE_delta_y"] = (
            metrics["AR"]["RMSE_mm"] - metrics[model]["RMSE_mm"]
        )
    return {
        "fold": fold.fold,
        "horizon_samples": horizon,
        "train_target_count": int(len(train_targets)),
        "validation_target_count": int(len(validation_targets)),
        "max_train_input_index": int(train_targets[-1] - horizon),
        "min_validation_target_index": int(validation_targets[0]),
        "purge_gap": fold.purge_gap,
        "train_only_scaler_stop": fold.effective_train_stop,
        "fit_rank": {
            "AR": ar_fit.rank,
            "X_only": x_fit.rank,
            "ARX": arx_fit.rank,
        },
        "metrics": metrics,
    }


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    models = ("mean", "persistence", "AR", "X_only", "ARX")
    for horizon in DIRECT_HORIZONS:
        horizon_records = [
            record for record in records if record["horizon_samples"] == horizon
        ]
        by_model: dict[str, object] = {}
        for model in models:
            keys = horizon_records[0]["metrics"][model].keys()
            by_model[model] = {
                key: {
                    "mean_across_folds": float(
                        np.mean(
                            [record["metrics"][model][key] for record in horizon_records]
                        )
                    ),
                    "std_across_folds": float(
                        np.std(
                            [record["metrics"][model][key] for record in horizon_records],
                            ddof=1,
                        )
                    ),
                }
                for key in keys
            }
        result[str(horizon)] = by_model
    return result


def run_r1(args: argparse.Namespace) -> None:
    start = time.monotonic()
    raw_dir = Path(args.raw_dir).resolve()
    output = Path(args.output).resolve() / "R1"
    data = load_furnace_a(raw_dir / "实验数据1.xlsx")
    L_x = L_y = 32
    folds = build_development_folds(L_x=L_x, L_y=L_y)
    records: list[dict[str, object]] = []
    for fold in folds:
        for horizon in DIRECT_HORIZONS:
            record = _fold_model_results(
                data.inputs,
                data.target,
                horizon=horizon,
                fold=fold,
                L_x=L_x,
                L_y=L_y,
            )
            records.append(record)
            print(
                f"R1 fold={fold.fold} h={horizon} "
                f"AR={record['metrics']['AR']['RMSE_mm']:.6f} "
                f"ARX={record['metrics']['ARX']['RMSE_mm']:.6f}",
                flush=True,
            )
    result = {
        "schema": "CZ_R1_LINEAR_SMOKE_V1",
        "protocol_schema": PROTOCOL_SCHEMA,
        "status": "COMPLETED",
        "scope": "FURNACE_A_DEVELOPMENT_ONLY",
        "runtime": {
            **_runtime(),
            "elapsed_seconds": time.monotonic() - start,
        },
        "configuration": {
            "L_x": L_x,
            "L_y": L_y,
            "horizons_samples": list(DIRECT_HORIZONS),
            "linear_solver": "numpy.linalg.lstsq_minimum_norm_OLS",
            "selection_or_tuning": False,
            "target_scaling": "fold_train_only_zscore",
            "input_scaling": "fold_train_only_zscore",
            "furnace_A_confirmation_accessed": False,
            "furnace_B_accessed": False,
        },
        "fold_records": records,
        "aggregate": _aggregate(records),
        "gates": {
            "NO_RANDOM_SPLIT": True,
            "TRAIN_ONLY_SCALER": True,
            "PURGE_PASS": True,
            "FURNACE_B_NOT_USED_IN_TUNING": True,
        },
    }
    _write_json(output / "R1_LINEAR_SMOKE_RESULTS.json", result)
    _write_json(
        output / "R1_STATUS.json",
        {
            "stage": "CZ_R1_LINEAR_SMOKE",
            "status": "COMPLETED",
            "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            "next_stage": "CZ_R2_H3_SPECTRAL_SMOKE",
            "furnace_B_lock": "LOCKED_UNTIL_R7",
            "furnace_A_confirmation_lock": "LOCKED_UNTIL_R6",
        },
    )


def run_r2(args: argparse.Namespace) -> None:
    start = time.monotonic()
    raw_dir = Path(args.raw_dir).resolve()
    output = Path(args.output).resolve() / "R2"
    data = load_furnace_a(raw_dir / "实验数据1.xlsx")
    fold = build_development_folds(L_x=32, L_y=32)[0]
    try:
        result = fit_shared_history_smoke(
            data.inputs,
            data.target,
            fold=fold,
            horizon=1,
            L_shared=32,
            lag_basis_count=16,
            amplitude_basis_count=16,
            smoothness_weight=1.0e-3,
            ridge_weight=1.0e-8,
        )
    except AmplitudeOutOfDomainError as error:
        blocked = {
            "stage": "CZ_R2_H3_SPECTRAL_SMOKE",
            "status": "BLOCKED_BY_MISSING_METADATA",
            "elapsed_seconds": time.monotonic() - start,
            "reason": str(error),
            "missing_prerequisite": (
                "The protocol names v4.1 bounded C1 continuation but does not "
                "freeze its continuation length/bound or provide an implementation."
            ),
            "why_not_clipped": (
                "Silent clipping or an invented continuation scale would change "
                "validation predictions and violate the interpretation firewall."
            ),
            "furnace_B_lock": "LOCKED_UNTIL_R7",
            "furnace_A_confirmation_lock": "LOCKED_UNTIL_R6",
        }
        _write_json(output / "R2_STATUS.json", blocked)
        _write_json(
            output / "R2_H3_SPECTRAL_SMOKE_BLOCKER.json",
            {
                "schema": "CZ_R2_H3_SPECTRAL_SMOKE_BLOCKER_V1",
                **blocked,
                "attempted_configuration": {
                    "fold": 1,
                    "horizon_samples": 1,
                    "L_x": 32,
                    "L_y": 32,
                    "lag_basis_count": 16,
                    "amplitude_basis_count": 16,
                    "pilot_smoothness_weight": 1.0e-3,
                    "pilot_ridge_weight": 1.0e-8,
                },
                "scientific_data_opened": {
                    "furnace_A_development": True,
                    "furnace_A_confirmation": False,
                    "furnace_B": False,
                },
            },
        )
        print(json.dumps(blocked, ensure_ascii=False, indent=2), flush=True)
        return
    kkt_passed = result.relative_kkt_residual <= 1.0e-8
    payload = {
        "schema": "CZ_R2_H3_SPECTRAL_SMOKE_V1",
        "protocol_schema": PROTOCOL_SCHEMA,
        "status": "COMPLETED" if kkt_passed else "FAILED",
        "scope": "FURNACE_A_DEVELOPMENT_FOLD1_H1_ENGINEERING_SMOKE_ONLY",
        "runtime": {
            **_runtime(),
            "elapsed_seconds": time.monotonic() - start,
        },
        "configuration": {
            "history_policy": "H3_SHARED",
            "L_x": 32,
            "L_y": 32,
            "lag_basis_count": 16,
            "amplitude_basis_count": 16,
            "pilot_smoothness_weight": 1.0e-3,
            "pilot_ridge_weight": 1.0e-8,
            "hyperparameter_selection": False,
            "scientific_evidence": False,
            "furnace_A_confirmation_accessed": False,
            "furnace_B_accessed": False,
        },
        "result": {
            "metrics": result.metrics,
            "relative_kkt_residual": result.relative_kkt_residual,
            "kkt_threshold": 1.0e-8,
            "kkt_passed": kkt_passed,
            "coefficients": result.coefficients,
            "train_target_count": result.train_target_count,
            "validation_target_count": result.validation_target_count,
        },
        "gates": {
            "NO_RANDOM_SPLIT": True,
            "TRAIN_ONLY_BASIS": True,
            "PURGE_PASS": True,
            "FURNACE_B_NOT_USED_IN_TUNING": True,
            "R2_KKT_PASS": kkt_passed,
        },
    }
    _write_json(output / "R2_H3_SPECTRAL_SMOKE_RESULTS.json", payload)
    _write_json(
        output / "R2_STATUS.json",
        {
            "stage": "CZ_R2_H3_SPECTRAL_SMOKE",
            "status": payload["status"],
            "elapsed_seconds": payload["runtime"]["elapsed_seconds"],
            "next_stage": (
                "CZ_R3_H2_NATIVE_SELECTION" if kkt_passed else "BLOCKED"
            ),
            "furnace_B_lock": "LOCKED_UNTIL_R7",
            "furnace_A_confirmation_lock": "LOCKED_UNTIL_R6",
        },
    )
    print(
        f"R2 H3 smoke RMSE={result.metrics['RMSE_mm']:.6f} "
        f"KKT={result.relative_kkt_residual:.3e}",
        flush=True,
    )
    if not kkt_passed:
        raise RuntimeError("R2 original-coordinate KKT gate failed.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("stage", choices=("r0", "r1", "r2"))
    result.add_argument("--raw-dir", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.stage == "r0":
        run_r0(args)
    elif args.stage == "r1":
        run_r1(args)
    else:
        run_r2(args)


if __name__ == "__main__":
    main()
