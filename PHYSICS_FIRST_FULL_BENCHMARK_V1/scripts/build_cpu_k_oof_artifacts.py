#!/usr/bin/env python3
"""Build leakage-safe rolling-OOF K predictions on the frozen CPU FP64 path.

This is deliberately a CPU-side artifact builder.  It reuses the frozen L6
shared matrices, the already selected per-direction ridge alpha, and the same
four expanding-window folds used by the registered CPU benchmark.  The GPU
code only consumes the resulting immutable predictions and never fits K.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.benchmark import (  # noqa: E402
    _custom_ridge_api,
    _folds_for_direction,
    _load_direction,
)
from src.common import atomic_json, sha256_array, sha256_file  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _kernel_alpha(results_root: Path, direction: str) -> float:
    path = results_root / "KERNELS" / f"{direction}_lift_kernel.json"
    payload = _load_json(path)
    if payload.get("nonlinear_block") != "EXACT_ZERO":
        raise RuntimeError(f"UNEXPECTED_NONLINEAR_K:{direction}")
    return float(payload["alpha"])


def build_direction(
    *,
    shared_root: Path,
    results_root: Path,
    output_root: Path,
    protocol: dict,
    direction: str,
) -> dict:
    fit_ridge = _custom_ridge_api(REPO_ROOT)
    data = _load_direction(shared_root, direction)
    train, test, metadata = data["train"], data["test"], data["metadata"]
    y_train = np.asarray(train["target_z"], dtype=np.float64)
    lag_count = len(metadata["lag_blocks"])
    lift_train = np.asarray(
        train["multiresolution_u"][:, :lag_count], dtype=np.float64
    )
    lift_test = np.asarray(
        test["multiresolution_u"][:, :lag_count], dtype=np.float64
    )
    alpha = _kernel_alpha(results_root, direction)
    folds = _folds_for_direction(
        np.asarray(train["origin_raw_index"]), protocol, REPO_ROOT
    )
    oof = np.full(len(y_train), np.nan, dtype=np.float64)
    fold_records = []
    for fold_index, fold in enumerate(folds):
        model = fit_ridge(
            lift_train[fold.train_indices],
            y_train[fold.train_indices],
            alpha=alpha,
        )
        values = np.asarray(
            model.predict(lift_train[fold.validation_indices]), dtype=np.float64
        )
        if np.any(np.isfinite(oof[fold.validation_indices])):
            raise RuntimeError(f"OVERLAPPING_OOF_FOLDS:{direction}:{fold_index}")
        oof[fold.validation_indices] = values
        fold_records.append(
            {
                "fold": fold_index,
                "train_rows": int(len(fold.train_indices)),
                "validation_rows": int(len(fold.validation_indices)),
                "validation_sample_id_sha256": sha256_array(
                    np.asarray(train["sample_id"])[fold.validation_indices]
                ),
            }
        )
    full_model = fit_ridge(lift_train, y_train, alpha=alpha)
    test_prediction = np.asarray(full_model.predict(lift_test), dtype=np.float64)
    registered_path = (
        results_root / "CPU_MODEL_PREDICTIONS" / direction / "K-only.npz"
    )
    with np.load(registered_path, allow_pickle=False) as stored:
        registered_ids = stored["sample_id"]
        registered_prediction = np.asarray(stored["prediction"], dtype=np.float64)
    if not np.array_equal(registered_ids, np.asarray(test["sample_id"])):
        raise RuntimeError(f"REGISTERED_TEST_ID_MISMATCH:{direction}")
    max_abs_difference = float(
        np.max(np.abs(test_prediction - registered_prediction))
    )
    if max_abs_difference > 1e-12:
        raise RuntimeError(
            f"FROZEN_K_REPRODUCTION_MISMATCH:{direction}:{max_abs_difference}"
        )
    target = output_root / direction / "K-only.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        train_sample_id=np.asarray(train["sample_id"]),
        train_oof_prediction=oof,
        train_oof_available=np.isfinite(oof),
        test_sample_id=np.asarray(test["sample_id"]),
        test_prediction=test_prediction,
        alpha=np.asarray(alpha, dtype=np.float64),
    )
    return {
        "direction": direction,
        "alpha": alpha,
        "train_rows": int(len(oof)),
        "train_oof_rows": int(np.isfinite(oof).sum()),
        "train_oof_fraction": float(np.isfinite(oof).mean()),
        "test_rows": int(len(test_prediction)),
        "registered_test_max_abs_difference": max_abs_difference,
        "artifact": str(target.relative_to(output_root)),
        "artifact_sha256": sha256_file(target),
        "folds": fold_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", required=True)
    parser.add_argument(
        "--cpu-results", default=str(ROOT / "results_cpu")
    )
    parser.add_argument(
        "--protocol", default=str(ROOT / "configs" / "protocol_frozen_l6.yaml")
    )
    parser.add_argument(
        "--output", default=str(ROOT / "results_cpu" / "CPU_K_OOF_ARTIFACTS")
    )
    args = parser.parse_args()
    shared_root = Path(args.shared).expanduser().resolve()
    results_root = Path(args.cpu_results).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    protocol = _load_json(protocol_path)
    shared_protocol = _load_json(shared_root / "BENCHMARK_PROTOCOL.json")
    if shared_protocol.get("config_sha256") != sha256_file(protocol_path):
        raise RuntimeError("SHARED_PROTOCOL_HASH_MISMATCH")
    if not shared_protocol.get("frozen", False):
        raise RuntimeError("PROTOCOL_NOT_FROZEN")
    directions = [
        f"{train.lower()}_to_{test.lower()}"
        for train, test in protocol["outer_directions"]
    ]
    records = [
        build_direction(
            shared_root=shared_root,
            results_root=results_root,
            output_root=output_root,
            protocol=protocol,
            direction=direction,
        )
        for direction in directions
    ]
    manifest = {
        "schema": "FROZEN_CPU_K_ROLLING_OOF_V1",
        "status": "PASS",
        "generator_device": "CPU_FP64",
        "gpu_k_refit_forbidden": True,
        "protocol_sha256": sha256_file(protocol_path),
        "shared_manifest_sha256": sha256_file(
            shared_root / "DATA_AND_SPLIT_HASHES.json"
        ),
        "records": records,
    }
    atomic_json(output_root / "MANIFEST.json", manifest)
    print("CPU_K_OOF_RESULT=" + json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
