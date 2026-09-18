from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism_benchmark.v211_w import (
    NATURAL_CUBIC,
    fit_w_correction,
    predict_w_correction,
)


PROTOCOL_ID = "PRISM_V211_CASCADED_TANKS_KW_PREDICTION_20260908_R1"
SEED = 20260908


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = y - prediction
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2,
    }


def _relative_rmse_gain(earlier: float, later: float) -> float:
    return (float(earlier) - float(later)) / float(earlier)


def _features(
    u: np.ndarray,
    y: np.ndarray,
    *,
    horizon: int,
    y_lags: int,
    u_lags: int,
    feature_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maximum_lag = max(int(y_lags), int(u_lags))
    origins = np.arange(maximum_lag - 1, len(y) - int(horizon), dtype=np.int64)
    target_times = origins + int(horizon)
    rows: list[np.ndarray] = []
    for origin in origins:
        y_history = y[origin - y_lags + 1 : origin + 1][::-1]
        u_history = u[origin - u_lags + 1 : origin + 1][::-1]
        columns = [y_history, u_history]
        if feature_family == "BERNOULLI_AUGMENTED_ARX":
            columns.extend(
                [
                    np.sqrt(np.maximum(y_history, 0.0)),
                    np.sqrt(np.maximum(u_history, 0.0)),
                    np.array([y_history[0] - y_history[min(1, len(y_history) - 1)]]),
                ]
            )
        elif feature_family != "LINEAR_ARX":
            raise ValueError(f"unknown feature family: {feature_family}")
        rows.append(np.concatenate(columns).astype(np.float64, copy=False))
    return np.vstack(rows), y[target_times], origins, target_times


@dataclass
class RidgeContract:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    alpha: float

    def json(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "alpha": float(self.alpha),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RidgeContract":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
            coefficient=np.asarray(value["coefficient"], dtype=np.float64),
            alpha=float(value["alpha"]),
        )


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeContract:
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, dtype=np.float64)
    scale[scale <= 64.0 * np.finfo(np.float64).eps] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x), dtype=np.float64), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeContract(mean, scale, coefficient, float(alpha))


def _predict_ridge(x: np.ndarray, contract: RidgeContract) -> np.ndarray:
    standardized = (x - contract.mean) / contract.scale
    design = np.column_stack([np.ones(len(x), dtype=np.float64), standardized])
    return design @ contract.coefficient


def _folds() -> list[tuple[int, int]]:
    return [(512, 640), (640, 768), (768, 896), (896, 1024)]


def _k_grid() -> list[dict[str, Any]]:
    return [
        {
            "horizon": horizon,
            "y_lags": y_lags,
            "u_lags": u_lags,
            "feature_family": family,
            "alpha": alpha,
        }
        for horizon in (1, 2, 4, 8, 12, 16, 24)
        for y_lags in (2, 4, 8, 16)
        for u_lags in (4, 8, 16, 32)
        for family in ("LINEAR_ARX", "BERNOULLI_AUGMENTED_ARX")
        for alpha in (1e-6, 1e-4, 1e-2, 1.0)
    ]


def _w_grid() -> list[dict[str, Any]]:
    return [
        {
            "family": NATURAL_CUBIC,
            "knot_count": knots,
            "smoothness": smoothness,
            "soft_overlap_mu": mu,
            "direction": 1,
        }
        for knots in (4, 6, 8, 12)
        for smoothness in (0.0, 1e-4, 1e-2, 1.0)
        for mu in (0.0, 0.03, 0.3)
    ]


def _fold_arrays(
    u: np.ndarray, y: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    x, target, _, target_times = _features(u, y, **{k: config[k] for k in (
        "horizon", "y_lags", "u_lags", "feature_family"
    )})
    masks = []
    for start, stop in _folds():
        train = np.flatnonzero(target_times < start)
        validation = np.flatnonzero((target_times >= start) & (target_times < stop))
        masks.append((train, validation))
    return x, target, target_times, masks


def _screen(data_path: Path, output_root: Path, top_k_per_horizon: int) -> None:
    started = time.time()
    data = pd.read_csv(data_path, usecols=["uEst", "yEst"])
    u = data["uEst"].to_numpy(dtype=np.float64)
    y = data["yEst"].to_numpy(dtype=np.float64)
    k_records: list[dict[str, Any]] = []
    cached: dict[str, tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]] = {}

    for config in _k_grid():
        key = json.dumps({k: config[k] for k in ("horizon", "y_lags", "u_lags", "feature_family")}, sort_keys=True)
        if key not in cached:
            x, target, _, masks = _fold_arrays(u, y, config)
            cached[key] = (x, target, masks)
        x, target, masks = cached[key]
        fold_metrics = []
        for train, validation in masks:
            contract = _fit_ridge(x[train], target[train], config["alpha"])
            prediction = _predict_ridge(x[validation], contract)
            fold_metrics.append(_metrics(target[validation], prediction))
        k_records.append(
            {
                "config": config,
                "fold_metrics": fold_metrics,
                "mean_rmse": float(np.mean([v["rmse"] for v in fold_metrics])),
                "mean_r2": float(np.mean([v["r2"] for v in fold_metrics])),
            }
        )

    retained: list[dict[str, Any]] = []
    for horizon in sorted({int(v["config"]["horizon"]) for v in k_records}):
        members = [v for v in k_records if int(v["config"]["horizon"]) == horizon]
        members.sort(key=lambda value: (value["mean_rmse"], -value["mean_r2"]))
        retained.extend(members[: int(top_k_per_horizon)])

    pipeline_records: list[dict[str, Any]] = []
    for k_record in retained:
        config = k_record["config"]
        key = json.dumps({k: config[k] for k in ("horizon", "y_lags", "u_lags", "feature_family")}, sort_keys=True)
        x, target, masks = cached[key]
        for w_config in _w_grid():
            fold_records = []
            for train, validation in masks:
                k_contract = _fit_ridge(x[train], target[train], config["alpha"])
                k_train = _predict_ridge(x[train], k_contract)
                k_validation = _predict_ridge(x[validation], k_contract)
                correction, _ = fit_w_correction(
                    k_train,
                    target[train] - k_train,
                    k_validation,
                    family=w_config["family"],
                    knot_count=w_config["knot_count"],
                    smoothness=w_config["smoothness"],
                    mu=w_config["soft_overlap_mu"],
                    upstream_predictions=k_train.reshape(-1, 1),
                    direction=w_config["direction"],
                )
                k_metrics = _metrics(target[validation], k_validation)
                kw_metrics = _metrics(target[validation], k_validation + correction)
                fold_records.append(
                    {
                        "k": k_metrics,
                        "k_plus_w": kw_metrics,
                        "relative_rmse_gain": _relative_rmse_gain(
                            k_metrics["rmse"], kw_metrics["rmse"]
                        ),
                    }
                )
            gains = np.asarray([v["relative_rmse_gain"] for v in fold_records])
            kw_r2 = np.asarray([v["k_plus_w"]["r2"] for v in fold_records])
            conservative_gain = float(np.mean(gains) - np.std(gains, ddof=1) / np.sqrt(len(gains)))
            pipeline_records.append(
                {
                    "k_config": config,
                    "w_config": w_config,
                    "fold_records": fold_records,
                    "mean_k_r2": float(np.mean([v["k"]["r2"] for v in fold_records])),
                    "mean_kw_r2": float(np.mean(kw_r2)),
                    "mean_relative_rmse_gain": float(np.mean(gains)),
                    "conservative_gain": conservative_gain,
                    "positive_fold_fraction": float(np.mean(gains > 0.0)),
                }
            )

    eligible = [
        v
        for v in pipeline_records
        if v["mean_kw_r2"] >= 0.80
        and v["positive_fold_fraction"] >= 0.75
        and v["mean_relative_rmse_gain"] > 0.0
    ]
    if not eligible:
        raise RuntimeError("no pipeline met the preregistered R2 and W-stability constraints")
    eligible.sort(
        key=lambda value: (
            value["conservative_gain"],
            value["mean_relative_rmse_gain"],
            value["mean_kw_r2"],
        ),
        reverse=True,
    )
    selected = eligible[0]
    freeze = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "phase": "DEVELOPMENT_FREEZE",
        "test_accessed": False,
        "dataset_sha256": _sha256(data_path),
        "estimation_rows": int(len(data)),
        "folds": [{"train_target_before": a, "validation_target_range": [a, b]} for a, b in _folds()],
        "selection_rule": {
            "mean_k_plus_w_r2_minimum": 0.80,
            "positive_w_gain_fold_fraction_minimum": 0.75,
            "primary_rank": "mean_gain_minus_one_standard_error",
        },
        "k_candidates_screened": len(k_records),
        "k_candidates_retained": len(retained),
        "w_candidates_per_k": len(_w_grid()),
        "eligible_pipeline_count": len(eligible),
        "selected": selected,
        "top_five": eligible[:5],
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_root / "DEVELOPMENT_FREEZE.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, sort_keys=True))


def _moving_block_gain(
    y: np.ndarray,
    k_prediction: np.ndarray,
    kw_prediction: np.ndarray,
    *,
    replicates: int,
    block_length: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n = len(y)
    starts = np.arange(0, n - block_length + 1)
    samples = np.empty(replicates, dtype=np.float64)
    blocks_needed = int(np.ceil(n / block_length))
    for replicate in range(replicates):
        selected_starts = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + block_length) for start in selected_starts]
        )[:n]
        k_rmse = float(np.sqrt(np.mean((y[indices] - k_prediction[indices]) ** 2)))
        kw_rmse = float(np.sqrt(np.mean((y[indices] - kw_prediction[indices]) ** 2)))
        samples[replicate] = _relative_rmse_gain(k_rmse, kw_rmse)
    observed = _relative_rmse_gain(
        _metrics(y, k_prediction)["rmse"], _metrics(y, kw_prediction)["rmse"]
    )
    tail = 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0)))
    return {
        "relative_rmse_gain": observed,
        "relative_rmse_gain_percent": 100.0 * observed,
        "ci95_percent": (100.0 * np.quantile(samples, [0.025, 0.975])).tolist(),
        "two_sided_tail_probability": min(1.0, tail),
        "replicates": int(replicates),
        "block_length": int(block_length),
    }


def _test(data_path: Path, output_root: Path, replicates: int) -> None:
    freeze_path = output_root / "DEVELOPMENT_FREEZE.json"
    freeze = _read_json(freeze_path)
    if freeze.get("test_accessed") is not False or freeze.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("invalid development freeze")
    if freeze.get("dataset_sha256") != _sha256(data_path):
        raise RuntimeError("dataset changed after development freeze")
    selected = freeze["selected"]
    k_config = selected["k_config"]
    w_config = selected["w_config"]
    data = pd.read_csv(data_path, usecols=["uEst", "yEst", "uVal", "yVal"])
    u_est = data["uEst"].to_numpy(dtype=np.float64)
    y_est = data["yEst"].to_numpy(dtype=np.float64)
    u_test = data["uVal"].to_numpy(dtype=np.float64)
    y_test = data["yVal"].to_numpy(dtype=np.float64)

    feature_args = {k: k_config[k] for k in ("horizon", "y_lags", "u_lags", "feature_family")}
    x_est, target_est, _, target_time_est = _features(u_est, y_est, **feature_args)
    x_test, target_test, origin_test, target_time_test = _features(u_test, y_test, **feature_args)
    k_contract = _fit_ridge(x_est, target_est, k_config["alpha"])
    k_est = _predict_ridge(x_est, k_contract)
    k_test = _predict_ridge(x_test, k_contract)
    _, w_contract = fit_w_correction(
        k_est,
        target_est - k_est,
        k_est,
        family=w_config["family"],
        knot_count=w_config["knot_count"],
        smoothness=w_config["smoothness"],
        mu=w_config["soft_overlap_mu"],
        upstream_predictions=k_est.reshape(-1, 1),
        direction=w_config["direction"],
    )
    correction = predict_w_correction(k_test, w_contract)
    kw_test = k_test + correction
    persistence = y_test[origin_test]
    k_metrics = _metrics(target_test, k_test)
    kw_metrics = _metrics(target_test, kw_test)
    persistence_metrics = _metrics(target_test, persistence)
    block_length = max(4, int(round(len(target_test) ** (1.0 / 3.0))))
    bootstrap = _moving_block_gain(
        target_test,
        k_test,
        kw_test,
        replicates=replicates,
        block_length=block_length,
    )
    lower = float(w_contract["basis"]["train_min"])
    upper = float(w_contract["basis"]["train_max"])
    standardized_test = (
        k_test - float(w_contract["basis"]["mean"])
    ) / float(w_contract["basis"]["scale"])
    result = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "phase": "FROZEN_TEST_EVALUATION",
        "test_accessed": True,
        "prediction_mode": "DIRECT_MULTI_STEP_PREDICTION_USING_PAST_OUTPUTS",
        "sample_period_seconds": 4,
        "selected_development_record": selected,
        "test_rows_scored": int(len(target_test)),
        "test_target_time_range": [int(target_time_test[0]), int(target_time_test[-1])],
        "persistence": persistence_metrics,
        "k": k_metrics,
        "k_plus_w": kw_metrics,
        "w_test_effect": {
            **bootstrap,
            "r2_increment": float(kw_metrics["r2"] - k_metrics["r2"]),
            "mae_gain_percent": 100.0 * _relative_rmse_gain(
                k_metrics["mae"], kw_metrics["mae"]
            ),
            "correction_max_abs": float(np.max(np.abs(correction))),
            "correction_std": float(np.std(correction)),
            "outside_estimation_latent_support_fraction": float(
                np.mean((standardized_test < lower) | (standardized_test > upper))
            ),
        },
        "k_contract": k_contract.json(),
        "w_contract": w_contract,
        "dataset_sha256": _sha256(data_path),
        "development_freeze_sha256": _sha256(freeze_path),
    }
    prediction_frame = pd.DataFrame(
        {
            "origin": origin_test,
            "target_time": target_time_test,
            "y_true": target_test,
            "persistence": persistence,
            "k_prediction": k_test,
            "w_correction": correction,
            "k_plus_w_prediction": kw_test,
        }
    )
    prediction_frame.to_parquet(
        output_root / "TEST_PREDICTIONS.parquet", index=False, compression="zstd"
    )
    _write_json(output_root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "test"), required=True)
    parser.add_argument("--top-k-per-horizon", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.phase == "screen":
        _screen(args.data.resolve(), args.output_root.resolve(), args.top_k_per_horizon)
    else:
        _test(args.data.resolve(), args.output_root.resolve(), args.bootstrap_replicates)


if __name__ == "__main__":
    main()
