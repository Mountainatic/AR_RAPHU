"""Validated aggregate reporting for the hybrid-h/w stagewise ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cpu_data import sha256_file
from .level_reconstruction import support_hash
from .stage0 import write_json
from .stagewise_ablation_runner import BOOTSTRAP_BLOCK_REGISTRY, PROTOCOL_ID
from .v211_a import EXACT_ZERO
from .v211_w import IDENTITY


STAGES = {
    "K": "PRISM_V2_1_1_K",
    "K+C": "PRISM_V2_1_1_K_C_DYNAMIC",
    "K+C+DELTA_W": "PRISM_V2_1_1_K_C_W_DYNAMIC",
    "K+C+DELTA_W+A": "PRISM_V2_1_1_PHYSICS_FIRST",
    "J": "PRISM_V2_1_1_JOINT_KWA",
}
INCREMENTS = (
    ("C", "K", "K+C"),
    ("DELTA_W", "K+C", "K+C+DELTA_W"),
    ("A", "K+C+DELTA_W", "K+C+DELTA_W+A"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _unit_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["target_head"]),
        str(record["information_set"]),
        str(record["availability_scenario"]),
        str(record["proxy_policy"]),
    )


def _manifest_record(unit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = Path(unit["selection_run_root"])
    prediction = Path(unit["prediction_run_root"])
    inference_path = prediction / "STAGEWISE_INFERENCE_COMPLETE.json"
    inference = _read_json(inference_path)
    if inference.get("status") != "PASS" or inference.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"invalid stagewise inference manifest: {inference_path}")
    key = (
        str(unit["head_id"]),
        str(unit["information_set"]),
        str(unit["availability_scenario"]),
        str(unit["proxy_policy"]),
    )
    block_path = selection / "freeze" / BOOTSTRAP_BLOCK_REGISTRY
    blocks = _read_json(block_path)
    if blocks.get("status") != "PASS" or blocks.get("test_accessed") is not False:
        raise RuntimeError(f"invalid development-only block freeze: {block_path}")
    block_matches = [record for record in blocks["records"] if _unit_key(record) == key]
    if len(block_matches) != 1:
        raise RuntimeError(f"bootstrap block record is not unique: {unit['label']}")
    return inference, block_matches[0]


def _stage_records(
    unit: dict[str, Any], inference: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    wanted = {
        **STAGES,
        "K+C": (
            "PRISM_V2_1_1_K_C"
            if unit["information_set"] == "input_only"
            else STAGES["K+C"]
        ),
        "K+C+DELTA_W": (
            "PRISM_V2_1_1_K_C_W"
            if unit["information_set"] == "input_only"
            else STAGES["K+C+DELTA_W"]
        ),
    }
    records: dict[str, dict[str, Any]] = {}
    statuses: dict[str, str] = {}
    for stage, model in wanted.items():
        matches = [record for record in inference["records"] if record.get("model") == model]
        if len(matches) != 1:
            raise RuntimeError(f"model record is not unique: {unit['label']}/{model}")
        record = matches[0]
        statuses[stage] = str(record.get("status"))
        if record.get("status") == "PASS":
            records[stage] = record
    if unit["information_set"] == "dynamic":
        for stage in ("K", "K+C", "K+C+DELTA_W", "K+C+DELTA_W+A"):
            if statuses.get(stage) != "PASS":
                raise RuntimeError(f"required stage is not PASS: {unit['label']}/{stage}")
    return records, statuses


def _prediction_frame(unit: dict[str, Any], record: dict[str, Any]) -> pd.DataFrame:
    root = Path(unit["prediction_run_root"])
    path = root / str(record["prediction_path"])
    if sha256_file(path) != record["prediction_sha256"]:
        raise RuntimeError(f"prediction hash mismatch: {path}")
    columns = ["sample_id", "entity_id", "origin", "y_true", "y_pred"]
    frame = pd.read_parquet(path, columns=columns)
    if len(frame) != int(record["rows"]):
        raise RuntimeError(f"prediction row count mismatch: {path}")
    if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"non-finite prediction content: {path}")
    observed_order_hash = support_hash(frame["sample_id"].astype(str).tolist())
    if observed_order_hash != record["sample_id_order_hash"]:
        raise RuntimeError(f"prediction sample-order hash mismatch: {path}")
    truth = frame["y_true"].to_numpy(dtype=np.float64)
    error = frame["y_pred"].to_numpy(dtype=np.float64) - truth
    denominator = float(
        np.sum(np.square(truth - np.mean(truth, dtype=np.float64)), dtype=np.float64)
    )
    recomputed = {
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "r2_delta": (
            float(1.0 - np.sum(np.square(error), dtype=np.float64) / denominator)
            if denominator
            else float("nan")
        ),
    }
    for key, value in recomputed.items():
        if not np.isclose(
            value,
            float(record[key]),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(f"recomputed metric mismatch: {path}/{key}")
    return frame


def _validate_alignment(frames: dict[str, pd.DataFrame], label: str) -> None:
    reference = frames["K"]
    for stage, frame in frames.items():
        if not reference["sample_id"].equals(frame["sample_id"]):
            raise RuntimeError(f"sample order mismatch: {label}/{stage}")
        if not np.array_equal(
            reference["y_true"].to_numpy(dtype=np.float64),
            frame["y_true"].to_numpy(dtype=np.float64),
        ):
            raise RuntimeError(f"truth mismatch: {label}/{stage}")


def _moving_block_indices(
    frame: pd.DataFrame, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for _, part in frame.groupby("entity_id", sort=False):
        indices = part.index.to_numpy(dtype=np.int64)
        count = len(indices)
        width = min(max(1, int(block_length)), count)
        starts = np.arange(count - width + 1, dtype=np.int64)
        blocks: list[np.ndarray] = []
        while sum(len(block) for block in blocks) < count:
            start = int(rng.choice(starts))
            blocks.append(indices[start : start + width])
        sampled.append(np.concatenate(blocks)[:count])
    return np.concatenate(sampled)


def paired_moving_block_gain(
    earlier: pd.DataFrame,
    later: pd.DataFrame,
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    _validate_alignment({"K": earlier, "later": later}, "bootstrap_pair")
    ordered_earlier = earlier.sort_values(["entity_id", "origin"], kind="stable").reset_index(drop=True)
    ordered_later = later.set_index("sample_id").loc[ordered_earlier["sample_id"]].reset_index()
    truth = ordered_earlier["y_true"].to_numpy(dtype=np.float64)
    earlier_error = ordered_earlier["y_pred"].to_numpy(dtype=np.float64) - truth
    later_error = ordered_later["y_pred"].to_numpy(dtype=np.float64) - truth
    earlier_rmse = float(np.sqrt(np.mean(np.square(earlier_error), dtype=np.float64)))
    later_rmse = float(np.sqrt(np.mean(np.square(later_error), dtype=np.float64)))
    observed = earlier_rmse - later_rmse
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        selected = _moving_block_indices(ordered_earlier, block_length, rng)
        values[index] = float(
            np.sqrt(np.mean(np.square(earlier_error[selected]), dtype=np.float64))
            - np.sqrt(np.mean(np.square(later_error[selected]), dtype=np.float64))
        )
    lower, upper = np.quantile(values, [0.025, 0.975])
    lower_tail = (1 + int(np.count_nonzero(values <= 0.0))) / (len(values) + 1)
    upper_tail = (1 + int(np.count_nonzero(values >= 0.0))) / (len(values) + 1)
    return {
        "rmse_earlier": earlier_rmse,
        "rmse_later": later_rmse,
        "delta_rmse": observed,
        "relative_delta_rmse_percent": 100.0 * observed / earlier_rmse if earlier_rmse else float("nan"),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "relative_ci95_low_percent": 100.0 * float(lower) / earlier_rmse if earlier_rmse else float("nan"),
        "relative_ci95_high_percent": 100.0 * float(upper) / earlier_rmse if earlier_rmse else float("nan"),
        "bootstrap_tail_probability_two_sided": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "replicates": int(replicates),
        "block_length": int(block_length),
        "paired": True,
        "entity_boundaries_crossed": False,
    }


def holm_adjust(values: Iterable[float]) -> list[float]:
    p = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, float((len(p) - rank) * p[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _seed(label: str, increment: str) -> int:
    digest = hashlib.sha256(f"{PROTOCOL_ID}\0{label}\0{increment}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _checkpoint_admission(unit: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checkpoint = Path(records["K"]["checkpoint_dir"]) / "checkpoint.json"
    state = _read_json(checkpoint)
    w_family = str(state["w_contract"]["family"])
    a_family = str(state.get("a_contract", {}).get("family", "NOT_APPLICABLE"))
    return {
        "unit": unit["label"],
        "K": "ADMITTED" if state["k_contract"].get("channel") else "REJECTED_EXACT_ZERO",
        "C": "ADMITTED" if state["physical"].get("channels") else "REJECTED_NO_ACTIVE_CHANNEL",
        "DELTA_W": "REJECTED_IDENTITY" if w_family == IDENTITY else "ADMITTED",
        "A": "REJECTED_EXACT_ZERO" if a_family == EXACT_ZERO else "ADMITTED",
        "J": "ADMITTED" if state.get("joint_status") == "PASS" else "NOT_RUN_PROTOCOL_INCOMPATIBLE",
        "W_family": w_family,
        "A_family": a_family,
        "J_reason": state.get("joint_reason"),
    }


def _plot_gains(gains: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = gains[gains["primary"]].copy()
    labels = list(dict.fromkeys(primary["unit"].tolist()))
    palette = {"C": "#356CA5", "DELTA_W": "#D39A28", "A": "#D4743B"}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 6.8), sharey=True)
    y = np.arange(len(labels))
    for axis, (increment, _, _) in zip(axes, INCREMENTS):
        part = primary[primary["increment"] == increment].set_index("unit").loc[labels]
        x = part["relative_delta_rmse_percent"].to_numpy(dtype=float)
        low = part["relative_ci95_low_percent"].to_numpy(dtype=float)
        high = part["relative_ci95_high_percent"].to_numpy(dtype=float)
        axis.hlines(y, low, high, color=palette[increment], linewidth=1.2)
        axis.scatter(
            x,
            y,
            s=28,
            color=palette[increment],
            edgecolor="#25313C",
            linewidth=0.7,
            zorder=3,
        )
        axis.axvline(0.0, color="#4A535B", linewidth=0.9)
        axis.grid(axis="x", color="#DCE1E5", linewidth=0.7)
        axis.set_title({"C": "C gain", "DELTA_W": "ΔW gain", "A": "A gain"}[increment])
        axis.set_xlabel("Relative RMSE reduction (%)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Incremental gain of each admitted PRISM stage", x=0.06, ha="left", fontsize=14)
    fig.text(
        0.06,
        0.93,
        "Main hybrid-h/w tasks; dots are observed paired gains and bars are 95% moving-block bootstrap intervals",
        ha="left",
        fontsize=9,
        color="#4A535B",
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.86, bottom=0.12, wspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_stagewise_report(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("report input manifest protocol mismatch")
    replicates = int(manifest.get("bootstrap_replicates", 500))
    if replicates != 500:
        raise RuntimeError("frozen bootstrap replicate count must equal 500")
    metric_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    admission_rows: list[dict[str, Any]] = []
    for unit in manifest["units"]:
        inference, block = _manifest_record(unit)
        records, statuses = _stage_records(unit, inference)
        frames = {stage: _prediction_frame(unit, record) for stage, record in records.items()}
        _validate_alignment({key: value for key, value in frames.items() if key != "J"}, unit["label"])
        for stage in STAGES:
            record = records.get(stage)
            row = {
                "unit": unit["label"],
                "dataset": unit["dataset"],
                "head_id": unit["head_id"],
                "direction": unit.get("direction"),
                "primary": bool(unit.get("primary", True)),
                "information_set": unit["information_set"],
                "availability_scenario": unit["availability_scenario"],
                "proxy_policy": unit["proxy_policy"],
                "stage": stage,
                "status": statuses.get(stage, "ABSENT"),
            }
            if record is not None:
                row.update(
                    {
                        "rows": record["rows"],
                        "rmse": record["rmse"],
                        "mae": record["mae"],
                        "r2_level": record["r2_level_reconstructed"],
                        "r2_delta": record["r2_delta"],
                        "persistence_skill": record["persistence_skill"],
                    }
                )
            metric_rows.append(row)
        if unit["information_set"] == "dynamic":
            for increment, earlier, later in INCREMENTS:
                result = paired_moving_block_gain(
                    frames[earlier],
                    frames[later],
                    block_length=int(block["block_length"]),
                    replicates=replicates,
                    seed=_seed(unit["label"], increment),
                )
                gain_rows.append(
                    {
                        "unit": unit["label"],
                        "dataset": unit["dataset"],
                        "head_id": unit["head_id"],
                        "direction": unit.get("direction"),
                        "primary": bool(unit.get("primary", True)),
                        "increment": increment,
                        "earlier_stage": earlier,
                        "later_stage": later,
                        **result,
                    }
                )
        admission_rows.append(_checkpoint_admission(unit, records))
    metrics = pd.DataFrame(metric_rows)
    gains = pd.DataFrame(gain_rows)
    primary_index = gains.index[gains["primary"]].tolist()
    adjusted = holm_adjust(
        gains.loc[primary_index, "bootstrap_tail_probability_two_sided"].tolist()
    )
    gains["holm_adjusted_probability"] = np.nan
    gains.loc[primary_index, "holm_adjusted_probability"] = adjusted
    gains["significant_positive_after_holm"] = (
        (gains["ci95_low"] > 0.0)
        & (gains["holm_adjusted_probability"] < 0.05)
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "STAGEWISE_METRICS_LONG.csv", index=False)
    wide = metrics.pivot(index="unit", columns="stage", values=["rmse", "mae", "r2_level"])
    wide.columns = [f"{stage}_{metric}" for metric, stage in wide.columns]
    wide.reset_index().to_csv(output / "STAGEWISE_METRICS_WIDE.csv", index=False)
    gains.to_csv(output / "STAGEWISE_INCREMENTAL_GAINS.csv", index=False)
    pd.DataFrame(admission_rows).to_csv(output / "STAGE_ADMISSION_STATUS.csv", index=False)
    _plot_gains(gains, output / "STAGEWISE_INCREMENTAL_GAINS_FIGURE")
    result = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "units": len(manifest["units"]),
        "primary_units": int(sum(bool(unit.get("primary", True)) for unit in manifest["units"])),
        "primary_comparisons": len(primary_index),
        "expected_primary_comparisons": 27,
        "bootstrap_replicates": replicates,
        "multiplicity": "HOLM_ACROSS_27_PRIMARY_ADJACENT_STAGE_COMPARISONS",
        "public_outputs_aggregate_only": True,
        "private_prediction_paths_emitted": False,
        "input_manifest_sha256": sha256_file(manifest_path),
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    if result["primary_comparisons"] != result["expected_primary_comparisons"]:
        raise RuntimeError("primary comparison family is not exactly 9 x 3")
    write_json(output / "STAGEWISE_REPORT_AUDIT.json", result)
    return result
