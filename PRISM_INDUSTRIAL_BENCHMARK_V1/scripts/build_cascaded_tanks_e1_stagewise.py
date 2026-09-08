from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prism_benchmark.stagewise_ablation_reporting import (
    holm_adjust,
    paired_moving_block_gain,
)
from prism_benchmark.stagewise_ablation_runner import development_residual_block_length
from prism_benchmark.v211_w import fit_w_correction, predict_w_correction


PROTOCOL_ID = "PRISM_V211_CASCADED_TANKS_E1_STAGEWISE_20260908_R1"
SOURCE_PROTOCOL_ID = "PRISM_V211_CASCADED_TANKS_KW_PREDICTION_20260908_R1"
UNIT = "Cascaded Tanks H16 prediction"
STAGES = ("K", "K+C", "K+C+DELTA_W", "K+C+DELTA_W+A")
BOOTSTRAP_REPLICATES = 500


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


def _support_hash(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["y_true"].to_numpy(dtype=np.float64)
    prediction = frame["y_pred"].to_numpy(dtype=np.float64)
    residual = truth - prediction
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "rows": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2_level": 1.0 - float(np.sum(residual**2)) / denominator,
        "sample_id_order_hash": _support_hash(frame["sample_id"]),
    }


def _load_tanks_module(code_dir: Path):
    sys.path.insert(0, str(code_dir))
    import run_cascaded_tanks_w_experiment as tanks

    return tanks


def _stage_frame(source: pd.DataFrame, stage: str) -> pd.DataFrame:
    prediction_column = {
        "K": "k_prediction",
        "K+C": "k_prediction",
        "K+C+DELTA_W": "k_plus_w_prediction",
        "K+C+DELTA_W+A": "k_plus_w_prediction",
    }[stage]
    return pd.DataFrame(
        {
            "sample_id": source["target_time"].map(lambda value: f"CT_VAL_{int(value):04d}"),
            "entity_id": "CASCADED_TANKS_VALIDATION",
            "origin": source["target_time"].to_numpy(dtype=np.int64),
            "y_true": source["y_true"].to_numpy(dtype=np.float64),
            "y_pred": source[prediction_column].to_numpy(dtype=np.float64),
            "stage": stage,
        }
    )


def _seed(increment: str) -> int:
    digest = hashlib.sha256(f"{PROTOCOL_ID}\0{UNIT}\0{increment}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _freeze(data_path: Path, source_root: Path, output_root: Path) -> None:
    source_freeze_path = source_root / "results" / "DEVELOPMENT_FREEZE.json"
    source_freeze = _read_json(source_freeze_path)
    if source_freeze.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise RuntimeError("source development protocol mismatch")
    if source_freeze.get("test_accessed") is not False:
        raise RuntimeError("source development freeze is not test-sealed")
    if source_freeze.get("dataset_sha256") != _sha256(data_path):
        raise RuntimeError("dataset/source-freeze hash mismatch")

    tanks = _load_tanks_module(source_root / "code")
    selected = source_freeze["selected"]
    k_config = selected["k_config"]
    w_config = selected["w_config"]
    data = pd.read_csv(data_path, usecols=["uEst", "yEst"])
    u = data["uEst"].to_numpy(dtype=np.float64)
    y = data["yEst"].to_numpy(dtype=np.float64)
    feature_args = {
        key: k_config[key]
        for key in ("horizon", "y_lags", "u_lags", "feature_family")
    }
    x, target, _, target_time = tanks._features(u, y, **feature_args)
    k_contract = tanks._fit_ridge(x, target, k_config["alpha"])
    k_prediction = tanks._predict_ridge(x, k_contract)
    _, w_contract = fit_w_correction(
        k_prediction,
        target - k_prediction,
        k_prediction,
        family=w_config["family"],
        knot_count=w_config["knot_count"],
        smoothness=w_config["smoothness"],
        mu=w_config["soft_overlap_mu"],
        upstream_predictions=k_prediction.reshape(-1, 1),
        direction=w_config["direction"],
    )
    kw_prediction = k_prediction + predict_w_correction(k_prediction, w_contract)
    development_frame = pd.DataFrame(
        {
            "entity_id": "CASCADED_TANKS_ESTIMATION",
            "origin": target_time,
            "y_true": target,
            "y_pred": kw_prediction,
        }
    )
    block_audit = development_residual_block_length(development_frame)

    per_fold_rows = []
    for fold, fold_record in enumerate(selected["fold_records"], start=1):
        for stage, metric_source in (
            ("K", fold_record["k"]),
            ("K+C", fold_record["k"]),
            ("K+C+DELTA_W", fold_record["k_plus_w"]),
            ("K+C+DELTA_W+A", fold_record["k_plus_w"]),
        ):
            per_fold_rows.append(
                {
                    "unit": UNIT,
                    "fold": fold,
                    "stage": stage,
                    "rmse": float(metric_source["rmse"]),
                    "mae": float(metric_source["mae"]),
                    "r2_level": float(metric_source["r2"]),
                }
            )
    development_dir = output_root / "development"
    development_dir.mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame(per_fold_rows)
    per_fold.to_csv(development_dir / "per_fold.csv", index=False)
    aggregate = (
        per_fold.groupby(["unit", "stage"], sort=False)[["rmse", "mae", "r2_level"]]
        .mean()
        .reset_index()
    )
    aggregate.to_csv(development_dir / "aggregate.csv", index=False)

    manifest = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "protocol_class": "E1_COMPATIBLE_SINGLE_TASK_EXTENSION",
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "test_accessed": False,
        "ood_accessed": False,
        "unit": UNIT,
        "information_set": "dynamic",
        "prediction_mode": "DIRECT_MULTI_STEP_PREDICTION_USING_PAST_OUTPUTS",
        "sample_period_seconds": 4,
        "horizon_steps": int(k_config["horizon"]),
        "stage_order": list(STAGES),
        "stage_contracts": {
            "K": {"status": "ADMITTED", "family": k_config["feature_family"], "contract": k_contract.json()},
            "C": {"status": "REJECTED_IDENTITY", "family": "IDENTITY_COMPRESSION", "reason": "NO_SEPARATE_REGISTERED_C_ROUTE"},
            "DELTA_W": {"status": "ADMITTED", "family": w_config["family"], "contract": w_contract},
            "A": {"status": "REJECTED_IDENTITY", "family": "EXACT_ZERO", "reason": "NO_SEPARATE_REGISTERED_A_ROUTE"},
        },
        "w_development_gate": {
            "minimum_mean_relative_rmse_gain": 0.01,
            "minimum_positive_fold_fraction": 0.75,
            "observed_mean_relative_rmse_gain": float(selected["mean_relative_rmse_gain"]),
            "observed_positive_fold_fraction": float(selected["positive_fold_fraction"]),
            "pass": bool(
                selected["mean_relative_rmse_gain"] >= 0.01
                and selected["positive_fold_fraction"] >= 0.75
            ),
        },
        "development_mean_k_plus_w_r2_minimum": 0.80,
        "development_mean_k_plus_w_r2_observed": float(selected["mean_kw_r2"]),
        "bootstrap_block_length_freeze": block_audit,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "source_dataset_sha256": _sha256(data_path),
        "source_development_freeze": str(source_freeze_path),
        "source_development_freeze_sha256": _sha256(source_freeze_path),
    }
    if not manifest["w_development_gate"]["pass"]:
        raise RuntimeError("W failed the registered E1 development activation gate")
    _write_json(output_root / "STAGEWISE_CHECKPOINT_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def _report(data_path: Path, source_root: Path, output_root: Path) -> None:
    checkpoint_path = output_root / "STAGEWISE_CHECKPOINT_MANIFEST.json"
    checkpoint = _read_json(checkpoint_path)
    if checkpoint.get("protocol_id") != PROTOCOL_ID or checkpoint.get("test_accessed") is not False:
        raise RuntimeError("invalid E1 checkpoint manifest")
    if checkpoint.get("source_dataset_sha256") != _sha256(data_path):
        raise RuntimeError("dataset changed after E1 freeze")
    source_prediction_path = source_root / "results" / "TEST_PREDICTIONS.parquet"
    source_result_path = source_root / "results" / "RESULT.json"
    source_result = _read_json(source_result_path)
    if source_result.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise RuntimeError("source inference protocol mismatch")
    source = pd.read_parquet(source_prediction_path)

    prediction_root = output_root / "predictions"
    frames: dict[str, pd.DataFrame] = {}
    metric_rows = []
    for stage in STAGES:
        frame = _stage_frame(source, stage)
        frames[stage] = frame
        destination = prediction_root / stage
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination / "prediction.parquet", index=False, compression="zstd")
        metric_rows.append({"unit": UNIT, "stage": stage, **_metrics(frame)})
    support_hashes = {stage: _support_hash(frame["sample_id"]) for stage, frame in frames.items()}
    if len(set(support_hashes.values())) != 1:
        raise RuntimeError("stagewise scoring support differs across stages")

    block_length = int(checkpoint["bootstrap_block_length_freeze"]["block_length"])
    adjacent = (
        ("K", "K+C", "C"),
        ("K+C", "K+C+DELTA_W", "DELTA_W"),
        ("K+C+DELTA_W", "K+C+DELTA_W+A", "A"),
    )
    gains = []
    p_values = []
    for earlier, later, increment in adjacent:
        gain = paired_moving_block_gain(
            frames[earlier],
            frames[later],
            block_length=block_length,
            replicates=BOOTSTRAP_REPLICATES,
            seed=_seed(increment),
        )
        p_values.append(float(gain["bootstrap_tail_probability_two_sided"]))
        gains.append(
            {
                "unit": UNIT,
                "earlier_stage": earlier,
                "later_stage": later,
                "increment": increment,
                **gain,
            }
        )
    adjusted = holm_adjust(p_values)
    for row, value in zip(gains, adjusted, strict=True):
        row["p_adjusted_holm"] = float(value)
        row["significant_holm_0_05"] = bool(value <= 0.05)

    metrics = pd.DataFrame(metric_rows)
    gain_frame = pd.DataFrame(gains)
    admission = pd.DataFrame(
        [
            {"unit": UNIT, "stage": "K", "status": "ADMITTED"},
            {"unit": UNIT, "stage": "C", "status": "REJECTED_IDENTITY"},
            {"unit": UNIT, "stage": "DELTA_W", "status": "ADMITTED"},
            {"unit": UNIT, "stage": "A", "status": "REJECTED_IDENTITY"},
        ]
    )
    report_root = output_root / "public_aggregate_report"
    report_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_root / "STAGEWISE_METRICS_LONG.csv", index=False)
    wide = metrics.pivot(index="unit", columns="stage", values=["rmse", "mae", "r2_level"])
    wide.columns = [f"{stage}_{metric}" for metric, stage in wide.columns]
    wide.reset_index().to_csv(report_root / "STAGEWISE_METRICS_WIDE.csv", index=False)
    gain_frame.to_csv(report_root / "STAGEWISE_INCREMENTAL_GAINS.csv", index=False)
    admission.to_csv(report_root / "STAGE_ADMISSION_STATUS.csv", index=False)

    colors = ["#9ca3af", "#9ca3af", "#2563eb"]
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(
        gain_frame["increment"],
        gain_frame["relative_delta_rmse_percent"],
        color=colors,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Incremental RMSE gain (%)")
    axis.set_title("Cascaded Tanks — E1 stagewise incremental gains")
    figure.tight_layout()
    figure.savefig(report_root / "STAGEWISE_INCREMENTAL_GAINS_FIGURE.png", dpi=180)
    figure.savefig(report_root / "STAGEWISE_INCREMENTAL_GAINS_FIGURE.pdf")
    plt.close(figure)

    audit = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "protocol_class": "E1_COMPATIBLE_SINGLE_TASK_EXTENSION",
        "test_accessed": True,
        "unit_count": 1,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_block_length": block_length,
        "multiplicity": "HOLM_ACROSS_3_PRIMARY_ADJACENT_STAGE_COMPARISONS",
        "stage_order": list(STAGES),
        "same_scoring_support": True,
        "scoring_support_hash": next(iter(support_hashes.values())),
        "checkpoint_manifest": str(checkpoint_path),
        "checkpoint_manifest_sha256": _sha256(checkpoint_path),
        "source_prediction_sha256": _sha256(source_prediction_path),
        "source_result_sha256": _sha256(source_result_path),
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(report_root.iterdir())
            if path.is_file()
        },
    }
    _write_json(report_root / "STAGEWISE_REPORT_AUDIT.json", audit)
    _write_json(
        output_root / "STAGEWISE_INFERENCE_MANIFEST.json",
        {
            "status": "PASS",
            "protocol_id": PROTOCOL_ID,
            "test_accessed": True,
            "records": metric_rows,
            "incremental_gains": gains,
            "report_audit": str(report_root / "STAGEWISE_REPORT_AUDIT.json"),
        },
    )
    print(json.dumps({"audit": audit, "metrics": metric_rows, "gains": gains}, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "report"), required=True)
    args = parser.parse_args()
    if args.phase == "freeze":
        _freeze(args.data.resolve(), args.source_root.resolve(), args.output_root.resolve())
    else:
        _report(args.data.resolve(), args.source_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
