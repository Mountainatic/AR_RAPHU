"""Private C1 adapter for the native PRISM v2.1.1 K/C/W/A pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import HeadSpec, ViewSpec
from prism_benchmark.v211_support import SUPPORT_CONTRACT


REGIMES = ("S1_K_ONLY", "S2_K_C", "S3_K_C_W", "S4_K_C_W_A")
CHANNELS = tuple(f"x{index}" for index in range(8))
DECOY_CHANNEL = 4
HISTORY_BY_SCALE = {"fast": 8, "mid": 32, "slow": 128}
SCALE_BY_CHANNEL = {0: "fast", 1: "mid", 2: "slow", 3: "mid"}
HEAD_ID = "TIM_E2_NATIVE_H0_W1"
TASK_ID = "TIM_E2_NATIVE"
DATASET = "debutanizer"
TARGET = "synthetic_target"
PROXY_POLICY = "tim_e2_private"
AVAILABILITY = "record_time"
MAXIMUM_REGISTERED_HISTORY = 1024


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def head() -> HeadSpec:
    return HeadSpec(
        head_id=HEAD_ID,
        task_id=TASK_ID,
        dataset=DATASET,
        target=TARGET,
        cadence_seconds=1.0,
        h_steps=0,
        w_steps=1,
        w0_steps=1,
        primary=True,
    )


def view(information_set: str) -> ViewSpec:
    if information_set not in {"input_only", "dynamic"}:
        raise ValueError(information_set)
    return ViewSpec(head(), information_set, AVAILABILITY, PROXY_POLICY)


@dataclass(frozen=True)
class GeneratedSeries:
    frame: pd.DataFrame
    truth: dict[str, Any]
    sample_origins: dict[str, np.ndarray]


def _exponential_filter(values: np.ndarray, history: int) -> np.ndarray:
    alpha = 2.0 / (history + 1.0)
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def generate_series(
    regime: str,
    seed: int,
    *,
    n_samples: int = 3000,
    noise_level: str = "very_low",
) -> GeneratedSeries:
    """Generate causal truth outside the PRISM spline/block candidate basis."""
    if regime not in REGIMES:
        raise ValueError(regime)
    if n_samples < 500:
        raise ValueError("native E2 requires at least 500 scored samples")
    noise_sd = {
        "clean": 0.005,
        "very_low": 0.02,
        "low": 0.06,
        "medium": 0.12,
        "high": 0.24,
    }[noise_level]
    rng = np.random.default_rng(seed)
    total = MAXIMUM_REGISTERED_HISTORY + n_samples
    x = rng.normal(size=(total, len(CHANNELS)))
    for index in range(1, total):
        x[index] = 0.72 * x[index - 1] + np.sqrt(1.0 - 0.72**2) * x[index]
    x[:, DECOY_CHANNEL] = x[:, 0] + 0.15 * rng.normal(size=total)

    filtered = {
        channel: _exponential_filter(x[:, channel], HISTORY_BY_SCALE[scale])
        for channel, scale in SCALE_BY_CHANNEL.items()
    }
    components = {
        0: 0.85 * np.tanh(filtered[0]),
        1: 0.55 * filtered[1] / np.sqrt(1.0 + filtered[1] ** 2),
        2: 0.70 * (1.0 / (1.0 + np.exp(-1.4 * filtered[2])) - 0.5),
        3: 0.45 * np.arctan(1.2 * filtered[3]),
    }
    active_channels = [0] if regime == "S1_K_ONLY" else [0, 1, 2, 3]
    latent = sum((components[index] for index in active_channels), np.zeros(total))
    c_active = regime != "S1_K_ONLY"
    if c_active:
        # C is chiefly multi-channel fusion in native PRISM.  The mild product
        # deliberately prevents an exact inverse-crime match.
        interaction = filtered[0] * filtered[3]
        latent = latent + 0.15 * interaction / np.sqrt(1.0 + interaction**2)

    w_active = regime in {"S3_K_C_W", "S4_K_C_W_A"}
    w = 0.35 * np.tanh(0.9 * latent) if w_active else np.zeros(total)
    a_active = regime == "S4_K_C_W_A"
    a = np.zeros(total, dtype=np.float64)
    if a_active:
        for index in range(2, total):
            a[index] = 0.55 * a[index - 1] - 0.20 * a[index - 2] + 0.10 * rng.normal()
    target = latent + w + a + noise_sd * rng.normal(size=total)

    origins = np.arange(MAXIMUM_REGISTERED_HISTORY, total, dtype=np.int64)
    train_stop = int(0.60 * n_samples)
    validation_stop = int(0.80 * n_samples)
    sample_origins = {
        "train": origins[:train_stop],
        "validation": origins[train_stop:validation_stop],
        "test": origins[validation_stop:],
    }
    truth = {
        "regime": regime,
        "seed": int(seed),
        "generator": "TIM_E2_EXTERNAL_CAUSAL_NONLINEAR_V2",
        "generator_basis": [
            "exponential_filter",
            "tanh",
            "soft_saturation",
            "sigmoid",
            "arctan",
            "bilinear_soft_saturation",
            "latent_AR_2",
        ],
        "inverse_crime": False,
        "native_prism_stage_semantics": {
            "K": "independent channel structure and history profile",
            "C": "multi-channel fusion with mild unmatched interaction",
            "DeltaW": "static correction of the fused latent prediction",
            "A": "mature residual temporal state",
        },
        "active_channels": active_channels,
        "zero_channels": sorted(set(range(len(CHANNELS))) - set(active_channels)),
        "correlated_decoy_channel": DECOY_CHANNEL,
        "scale_by_channel": {
            str(index): SCALE_BY_CHANNEL[index] for index in active_channels
        },
        "history_by_scale": HISTORY_BY_SCALE,
        "stage_truth": {
            "K": True,
            "C": c_active,
            "W": w_active,
            "A": a_active,
        },
        "noise_level": noise_level,
        "noise_sd": noise_sd,
        "split": "chronological_60_20_20",
        "test_materialization": "after_selection_and_checkpoint_seal_only",
    }
    frame = pd.DataFrame(x, columns=CHANNELS)
    frame.insert(0, "row_in_entity", np.arange(total, dtype=np.int64))
    frame.insert(0, "entity_id", f"regime={regime};seed={seed}")
    frame[TARGET] = target
    return GeneratedSeries(frame=frame, truth=truth, sample_origins=sample_origins)


def _sample_frame(series: GeneratedSeries, split: str, information_set: str) -> pd.DataFrame:
    origins = series.sample_origins[split]
    entity = str(series.frame["entity_id"].iloc[0])
    base_ids = [f"{entity};origin={int(origin)}" for origin in origins]
    return pd.DataFrame(
        {
            "base_origin_id": base_ids,
            "view_sample_id": [f"{value};view={information_set}" for value in base_ids],
            "dataset": DATASET,
            "entity_id": entity,
            "task_id": TASK_ID,
            "target_head": HEAD_ID,
            "split": split,
            "origin": origins,
            "dependency_start": origins,
            "dependency_stop_exclusive": origins + 1,
            "latest_available_target_index": origins - 1,
            "y_true": series.frame[TARGET].to_numpy(dtype=np.float64)[origins],
            "causal_history_floor": np.zeros(len(origins), dtype=np.int64),
            "anchor_history_steps": np.full(
                len(origins), MAXIMUM_REGISTERED_HISTORY, dtype=np.int64
            ),
            "sample_support_contract": SUPPORT_CONTRACT,
        }
    )


def _write_metadata(shared: Path, truth: dict[str, Any]) -> None:
    _write_json(
        shared / "TASK_REGISTRY.json",
        {
            "dataset": DATASET,
            "sample_support_contract": SUPPORT_CONTRACT,
            "heads": [head().__dict__],
        },
    )
    _write_json(
        shared / "dataset_views" / "VIEW_REGISTRY.json",
        [
            {
                "task_id": TASK_ID,
                "dataset": DATASET,
                "head_id": HEAD_ID,
                "information_set": information_set,
                "availability_scenario": AVAILABILITY,
                "proxy_policy": PROXY_POLICY,
                "input_columns": list(CHANNELS),
                "target_history_column": TARGET if information_set == "dynamic" else None,
            }
            for information_set in ("input_only", "dynamic")
        ],
    )
    _write_json(
        shared / "PROTOCOL.json",
        {
            "protocol_id": "TIM_E2_NATIVE_SYNTHETIC_C1_V2",
            "support_contract": SUPPORT_CONTRACT,
            "tasks": [{"task_id": TASK_ID, "proxy_policies": [PROXY_POLICY]}],
            "selection": "validation_only",
            "test_materialization": "after_selection_and_checkpoint_seal_only",
        },
    )
    _write_json(shared / "GENERATOR_TRUTH.json", truth)


def build_development(
    shared: Path,
    regime: str,
    seed: int,
    *,
    n_samples: int = 3000,
    noise_level: str = "very_low",
) -> dict[str, Any]:
    if shared.exists() and any(shared.iterdir()):
        raise RuntimeError(f"refusing nonempty synthetic shared root: {shared}")
    series = generate_series(regime, seed, n_samples=n_samples, noise_level=noise_level)
    _write_metadata(shared, series.truth)
    validation_start = int(series.sample_origins["validation"][0])
    train_base = series.frame.iloc[:validation_start].copy()
    validation_base = series.frame.iloc[validation_start : int(series.sample_origins["test"][0])].copy()
    for split, frame in (("train", train_base), ("validation", validation_base)):
        destination = shared / "base_data" / DATASET / f"{split}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False, compression="zstd")
    for information_set in ("input_only", "dynamic"):
        current_view = view(information_set)
        for split in ("train", "validation"):
            destination = shared / "sample_ids" / current_view.relative_root / f"{split}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            _sample_frame(series, split, information_set).to_parquet(
                destination, index=False, compression="zstd"
            )
    files = sorted(path for path in shared.rglob("*") if path.is_file())
    result = {
        "status": "PASS",
        "stage": "TIM_E2_DEVELOPMENT_C1",
        "regime": regime,
        "seed": seed,
        "n_samples": n_samples,
        "noise_level": noise_level,
        "test_materialized": False,
        "test_y_read": False,
        "files": [
            {"path": str(path.relative_to(shared)), "sha256": _sha256(path)}
            for path in files
        ],
    }
    _write_json(shared / "C1_DEVELOPMENT_AUDIT.json", result)
    return result


def materialize_test(
    shared: Path,
    regime: str,
    seed: int,
    *,
    selection_freeze: Path,
    checkpoint_seal: Path,
    n_samples: int = 3000,
    noise_level: str = "very_low",
) -> dict[str, Any]:
    freeze = json.loads(selection_freeze.read_text(encoding="utf-8"))
    seal = json.loads(checkpoint_seal.read_text(encoding="utf-8"))
    if freeze.get("status") != "SELECTION_FROZEN" or freeze.get("sealed") is not True:
        raise RuntimeError("selection freeze is not sealed")
    if seal.get("status") != "CHECKPOINTS_SEALED" or seal.get("sealed") is not True:
        raise RuntimeError("checkpoint seal is not sealed")
    series = generate_series(regime, seed, n_samples=n_samples, noise_level=noise_level)
    expected = json.loads((shared / "GENERATOR_TRUTH.json").read_text(encoding="utf-8"))
    if series.truth != expected:
        raise RuntimeError("synthetic generator replay drifted")
    test_start = int(series.sample_origins["test"][0])
    base_path = shared / "base_data" / DATASET / "test.parquet"
    if base_path.exists():
        raise RuntimeError(f"refusing existing test partition: {base_path}")
    series.frame.iloc[test_start:].to_parquet(base_path, index=False, compression="zstd")
    for information_set in ("input_only", "dynamic"):
        destination = shared / "sample_ids" / view(information_set).relative_root / "test.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _sample_frame(series, "test", information_set).to_parquet(
            destination, index=False, compression="zstd"
        )
    result = {
        "status": "PASS",
        "stage": "TIM_E2_TEST_C1",
        "test_materialized": True,
        "test_accessed_after_selection_freeze": True,
        "selection_freeze_sha256": _sha256(selection_freeze),
        "checkpoint_seal_sha256": _sha256(checkpoint_seal),
        "test_rows_per_view": len(series.sample_origins["test"]),
        "test_base_sha256": _sha256(base_path),
    }
    _write_json(shared / "C1_TEST_ACCESS_AUDIT.json", result)
    return result
