"""Private CZ semi-synthetic materialization using frozen authority operators.

The module only constructs E2 inputs.  Model fitting and stage recovery remain
the responsibility of the unmodified authority K/C/W/A/Joint implementation.
No target-rod or formal-test partition is read.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor
from .cz_l256_nowcast import TARGET_COLUMN
from .portable_checkpoints import INFERENCE_ONLY_ENV, load_portable_checkpoint
from .stage0 import write_json
from .v2_k import profile_values
from .v2_urysohn import predict_contract


RAW_INPUT_COLUMNS = (
    "main_heater_power",
    "crystal_lift",
    "crucible_lift",
    "crystal_rotation",
    "crucible_rotation",
)
HISTORY_STEPS = 256
H_STEPS = 4
BLOCK_SHIFT_STEPS = 256
S2_REGISTRY_FORMAT = "CZ_RAW2S_H4_E2_TRUTH_OPERATOR_REGISTRY_V1"
S2_REGISTRY_SHA256 = "e4501d1325b726a41916da2907c342067abb0edbc814f837037b37a0a60f58a1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    previous = os.environ.get(INFERENCE_ONLY_ENV)
    os.environ[INFERENCE_ONLY_ENV] = "1"
    try:
        state, _, manifest = load_portable_checkpoint(path)
    finally:
        if previous is None:
            os.environ.pop(INFERENCE_ONLY_ENV, None)
        else:
            os.environ[INFERENCE_ONLY_ENV] = previous
    return state, manifest


def _assert_source_only_template(shared: Path) -> dict[str, Any]:
    audit_path = shared / "C1_NATIVE_SUPPORT_AUDIT.json"
    if not audit_path.is_file():
        raise RuntimeError("STOP_E2_SOURCE_ONLY_TEMPLATE_AUDIT_MISSING")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "PASS"
        or audit.get("target_sheet_read_for_this_direction") is not False
        or audit.get("test_accessed") is not False
    ):
        raise RuntimeError("STOP_E2_TEMPLATE_HAS_TARGET_ACCESS")
    for split in ("test", "ood"):
        for path in shared.glob(f"sample_ids/**/{split}.parquet"):
            if len(pd.read_parquet(path, columns=["base_origin_id"])):
                raise RuntimeError(f"STOP_E2_TEMPLATE_{split.upper()}_IS_NOT_EMPTY")
        for path in shared.glob(f"targets/**/{split}.parquet"):
            if len(pd.read_parquet(path, columns=["base_origin_id"])):
                raise RuntimeError(f"STOP_E2_TEMPLATE_{split.upper()}_TARGET_IS_NOT_EMPTY")
    return audit


def _block_circular_shift(
    values: np.ndarray, rng: np.random.Generator, block_steps: int
) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return array.copy(), 0
    block_steps = max(1, int(block_steps))
    # The last partial block is a legal block.  Using floor division would
    # silently leave 257--511-row eligible segments unshifted.
    block_count = max(1, (len(array) + block_steps - 1) // block_steps)
    block_offset = int(rng.integers(0, block_count))
    offset = block_offset * block_steps
    if offset == 0 and block_count > 1:
        offset = block_steps
    return np.roll(array, offset), offset


def _shift_inputs(
    base: pd.DataFrame,
    pca: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    missing = sorted(set((*RAW_INPUT_COLUMNS, "entity_id", "row_in_entity")) - set(base))
    if missing:
        raise RuntimeError(f"STOP_E2_SOURCE_BASE_COLUMNS_MISSING:{missing}")
    result = base.copy()
    audits: list[dict[str, Any]] = []
    for entity, group in base.groupby("entity_id", sort=False):
        ordered = group.sort_values("row_in_entity")
        index = ordered.index
        for channel in RAW_INPUT_COLUMNS:
            shifted, offset = _block_circular_shift(
                ordered[channel].to_numpy(dtype=np.float64), rng, BLOCK_SHIFT_STEPS
            )
            result.loc[index, channel] = shifted
            audits.append(
                {
                    "entity_id": str(entity),
                    "channel": channel,
                    "rows": int(len(shifted)),
                    "block_steps": BLOCK_SHIFT_STEPS,
                    "circular_offset_steps": int(offset),
                }
            )
    mean = np.asarray(pca["mean"], dtype=np.float64)
    scale = np.asarray(pca["scale"], dtype=np.float64)
    loading = np.asarray(pca["pc1_loading"], dtype=np.float64)
    lift = result[["crystal_lift", "crucible_lift"]].to_numpy(dtype=np.float64)
    result["joint_lift"] = ((lift - mean) / scale) @ loading
    return result, audits


def _all_origins(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entity, group in base.groupby("entity_id", sort=False):
        maximum = int(group["row_in_entity"].max())
        for origin in range(HISTORY_STEPS, maximum - H_STEPS + 2):
            rows.append(
                {
                    "entity_id": str(entity),
                    "origin": int(origin),
                    "latest_available_target_index": int(origin - 1),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("STOP_E2_NO_COMPLETE_L256_H4_ORIGINS")
    return result


def _lookup(values: pd.DataFrame, column: str) -> dict[tuple[str, int], float]:
    return {
        (str(row.entity_id), int(row.origin)): float(getattr(row, column))
        for row in values[["entity_id", "origin", column]].itertuples(index=False)
    }


def _load_s2_registry(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError("STOP_E2_S2_TRUTH_REGISTRY_MISSING")
    digest = _sha256(path)
    if digest != S2_REGISTRY_SHA256:
        raise RuntimeError("STOP_E2_S2_TRUTH_REGISTRY_HASH_MISMATCH")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if (
        registry.get("format") != S2_REGISTRY_FORMAT
        or registry.get("status") != "FROZEN_BEFORE_S2_SCREENING"
        or registry.get("authority_base_commit")
        != "2ee6273b8f915cbcdff2f46d56bc80047ddae4a7"
    ):
        raise RuntimeError("STOP_E2_S2_TRUTH_REGISTRY_INVALID")
    s2 = registry.get("S2_KC", {})
    c = s2.get("C_increment", {})
    if (
        c.get("stage_semantics") != "ADDITIVE_COMPRESSED_SECOND_ACTIVE_K_CHANNEL"
        or c.get("channel") not in RAW_INPUT_COLUMNS
        or c.get("family") != "LINEAR_DISTRIBUTED_LAG"
        or len(c.get("interval_weights", [])) != int(c.get("m_tau", -1))
        or float(c.get("stage_rms_fraction", 0.0)) <= 0.0
        or float(s2.get("innovation_snr", 0.0)) <= 0.0
    ):
        raise RuntimeError("STOP_E2_S2_TRUTH_REGISTRY_CONTENT_INVALID")
    return registry, digest


def _registered_linear_signal(
    accessor: BaseAccessor,
    origins: pd.DataFrame,
    *,
    channel: str,
    profile: tuple[int, int],
    m_tau: int,
    interval_weights: Iterable[float],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    values, intervals = profile_values(accessor, origins, channel, profile, m_tau)
    weights = np.asarray(tuple(interval_weights), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(weights):
        raise RuntimeError("STOP_E2_REGISTERED_LINEAR_OPERATOR_SHAPE_MISMATCH")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise RuntimeError("STOP_E2_REGISTERED_LINEAR_OPERATOR_NONFINITE")
    return values @ weights, intervals


def _subset_ids(frame: pd.DataFrame, rows: int) -> set[str]:
    ordered = frame.sort_values(["entity_id", "origin"]).drop_duplicates(
        "base_origin_id"
    )
    if rows >= len(ordered):
        return set(ordered["base_origin_id"].astype(str))
    positions = np.unique(
        np.rint(np.linspace(0, len(ordered) - 1, rows)).astype(np.int64)
    )
    return set(ordered.iloc[positions]["base_origin_id"].astype(str))


def _update_samples(
    shared: Path,
    levels: dict[str, np.ndarray],
    sample_size: int,
) -> dict[str, int]:
    input_files = sorted(shared.glob("sample_ids/*/input_only/record_time/primary/*.parquet"))
    by_split = {path.stem: path for path in input_files if path.stem in {"train", "validation"}}
    if set(by_split) != {"train", "validation"}:
        raise RuntimeError("STOP_E2_SOURCE_SAMPLE_PARTITIONS_MISSING")
    originals = {split: pd.read_parquet(path) for split, path in by_split.items()}
    available = sum(len(frame) for frame in originals.values())
    requested = min(int(sample_size), available)
    train_rows = min(len(originals["train"]), max(1, int(np.floor(0.8 * requested))))
    validation_rows = min(len(originals["validation"]), requested - train_rows)
    if validation_rows < 1:
        validation_rows = 1
        train_rows = requested - 1
    keep = {
        "train": _subset_ids(originals["train"], train_rows),
        "validation": _subset_ids(originals["validation"], validation_rows),
    }
    counts: dict[str, int] = {}
    for split in ("train", "validation"):
        selected: pd.DataFrame | None = None
        for information_set in ("input_only", "dynamic"):
            paths = list(
                shared.glob(
                    f"sample_ids/*/{information_set}/record_time/primary/{split}.parquet"
                )
            )
            if len(paths) != 1:
                raise RuntimeError(
                    f"STOP_E2_SAMPLE_PATH_AMBIGUOUS:{information_set}:{split}"
                )
            frame = pd.read_parquet(paths[0])
            frame = frame[frame["base_origin_id"].astype(str).isin(keep[split])].copy()
            entity = frame["entity_id"].astype(str).to_numpy()
            origin = frame["origin"].to_numpy(dtype=np.int64)
            current = np.asarray(
                [levels[e][int(t) - 1] for e, t in zip(entity, origin, strict=True)],
                dtype=np.float64,
            )
            future = np.asarray(
                [levels[e][int(t) + H_STEPS - 1] for e, t in zip(entity, origin, strict=True)],
                dtype=np.float64,
            )
            frame["current_level"] = current
            frame["future_level_true"] = future
            frame["y_true"] = future - current
            frame.to_parquet(paths[0], index=False, compression="zstd")
            if information_set == "input_only":
                selected = frame
        assert selected is not None
        target_paths = list(shared.glob(f"targets/*/{split}.parquet"))
        if len(target_paths) != 1:
            raise RuntimeError(f"STOP_E2_TARGET_PATH_AMBIGUOUS:{split}")
        target = selected[
            [
                "base_origin_id",
                "entity_id",
                "origin",
                "current_level",
                "future_level_true",
                "y_true",
            ]
        ].copy()
        target.to_parquet(target_paths[0], index=False, compression="zstd")
        counts[split] = int(len(selected))
    return counts


def materialize_s1_k_pilot(
    template_shared: Path,
    authority_checkpoint: Path,
    output_shared: Path,
    *,
    seed: int,
    sample_size: int = 2048,
    innovation_snr: float = 4.0,
) -> dict[str, Any]:
    """Materialize one non-authoritative S1-K pilot on source-rod data only."""

    if output_shared.exists():
        raise RuntimeError(f"STOP_E2_REFUSE_EXISTING_UNIT:{output_shared}")
    if sample_size < 512 or innovation_snr <= 0:
        raise ValueError("E2 pilot requires sample_size>=512 and innovation_snr>0")
    source_audit = _assert_source_only_template(template_shared)
    state, manifest = _load_checkpoint(authority_checkpoint)
    c_contract = state["c_contract"]
    if c_contract.get("family") != "BEST_ACTIVE_K_CHANNEL":
        raise RuntimeError("STOP_E2_S1_REQUIRES_FROZEN_BEST_ACTIVE_K_CHANNEL")
    channel = str(c_contract["channel"])
    physical = state["physical"]
    matches = [item for item in physical["channel_contracts"] if item["channel"] == channel]
    if len(matches) != 1:
        raise RuntimeError("STOP_E2_S1_AUTHORITY_K_CONTRACT_AMBIGUOUS")
    operator = matches[0]
    shutil.copytree(template_shared, output_shared)
    base_path = output_shared / "base_data" / "cz_czochralski" / "train.parquet"
    original_base = pd.read_parquet(base_path)
    original_levels = {
        str(entity): group.sort_values("row_in_entity")[TARGET_COLUMN].to_numpy(
            dtype=np.float64
        )
        for entity, group in original_base.groupby("entity_id", sort=False)
    }
    pca = json.loads(
        (output_shared / "JOINT_LIFT_PCA_CONTRACT.json").read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(int(seed))
    shifted, shift_audit = _shift_inputs(original_base, pca, rng)
    shifted.to_parquet(base_path, index=False, compression="zstd")
    origins = _all_origins(shifted)
    accessor = BaseAccessor(output_shared, "cz_czochralski", "train", [channel])
    values, intervals = profile_values(
        accessor,
        origins,
        channel,
        tuple(int(value) for value in operator["profile"]),
        int(operator["m_tau"]),
    )
    raw_signal = predict_contract(values, dict(operator["k_contract"]))
    signal_frame = origins.assign(raw_signal=raw_signal)
    train_paths = list(
        output_shared.glob("sample_ids/*/input_only/record_time/primary/train.parquet")
    )
    if len(train_paths) != 1:
        raise RuntimeError("STOP_E2_TRAIN_SAMPLE_PATH_AMBIGUOUS")
    source_train = pd.read_parquet(train_paths[0])
    source_signal = _lookup(signal_frame, "raw_signal")
    train_signal = np.asarray(
        [
            source_signal[(str(entity), int(origin))]
            for entity, origin in zip(
                source_train["entity_id"], source_train["origin"], strict=True
            )
        ],
        dtype=np.float64,
    )
    signal_mean = float(train_signal.mean(dtype=np.float64))
    signal_std = float(train_signal.std(dtype=np.float64))
    target_std = float(source_train["y_true"].to_numpy(dtype=np.float64).std())
    if signal_std <= np.finfo(np.float64).eps or target_std <= 0:
        raise RuntimeError("STOP_E2_DEGENERATE_SOURCE_SCALE")
    effect_scale = target_std / signal_std
    signal_frame["signal"] = (signal_frame["raw_signal"] - signal_mean) * effect_scale
    innovation_std = target_std / float(innovation_snr)
    signal_frame["innovation"] = rng.normal(0.0, innovation_std, len(signal_frame))
    signal_frame["target_delta"] = signal_frame["signal"] + signal_frame["innovation"]
    target_lookup = _lookup(signal_frame, "target_delta")
    synthetic_levels: dict[str, np.ndarray] = {}
    for entity, original in original_levels.items():
        level = original.copy()
        for origin in range(HISTORY_STEPS, len(level) - H_STEPS + 1):
            level[origin + H_STEPS - 1] = (
                level[origin - 1] + target_lookup[(entity, origin)]
            )
        synthetic_levels[entity] = level
    for entity, group in shifted.groupby("entity_id", sort=False):
        ordered = group.sort_values("row_in_entity")
        shifted.loc[ordered.index, TARGET_COLUMN] = synthetic_levels[str(entity)]
    shifted.to_parquet(base_path, index=False, compression="zstd")
    counts = _update_samples(output_shared, synthetic_levels, int(sample_size))
    audit = {
        "status": "PASS",
        "role": "P1_TINY_PILOT_NON_SELECTION_AUTHORITY",
        "regime": "S1_K",
        "source_rod_only": True,
        "source_template_audit_status": source_audit["status"],
        "source_template_target_sheet_read": False,
        "formal_target_or_ood_accessed": False,
        "seed": int(seed),
        "sample_size_requested": int(sample_size),
        "sample_rows": counts,
        "history_steps": HISTORY_STEPS,
        "h_steps": H_STEPS,
        "w_steps": 1,
        "w0_steps": 1,
        "target_formula": "D[t+3]-D[t-1]",
        "block_shift_steps": BLOCK_SHIFT_STEPS,
        "shift_audit": shift_audit,
        "truth_operator": {
            "stage": "K",
            "channel": channel,
            "profile": list(operator["profile"]),
            "m_tau": int(operator["m_tau"]),
            "family": operator["k_contract"]["family"],
            "intervals": [list(value) for value in intervals],
            "authority_checkpoint_hash": manifest["checkpoint_hash"],
            "authority_checkpoint_manifest_sha256": _sha256(
                authority_checkpoint / "MANIFEST.json"
            ),
        },
        "pilot_effect_scale": effect_scale,
        "source_train_target_std": target_std,
        "source_train_raw_signal_mean": signal_mean,
        "source_train_raw_signal_std": signal_std,
        "innovation_snr": float(innovation_snr),
        "innovation_std": innovation_std,
        "complete_synthetic_diameter_trajectory": True,
        "statistical_claims_allowed": False,
    }
    write_json(output_shared / "E2_SEMISYNTHETIC_DERIVATION.json", audit)
    return audit


def materialize_s2_kc_pilot(
    template_shared: Path,
    authority_checkpoint: Path,
    truth_registry_path: Path,
    output_shared: Path,
    *,
    seed: int,
    sample_size: int = 2048,
) -> dict[str, Any]:
    """Materialize one non-authoritative S2-KC pilot on source-rod data only.

    The C truth is an additive fusion increment from a second registered K
    channel.  This matches the authority industrial C stage, whose parent is
    the best active K channel and whose nonzero families fuse active channel
    contributions.  It deliberately does not reuse a BEST_ACTIVE_K_CHANNEL
    fallback as a nonzero C operator.
    """

    if output_shared.exists():
        raise RuntimeError(f"STOP_E2_REFUSE_EXISTING_UNIT:{output_shared}")
    if sample_size < 512:
        raise ValueError("E2 pilot requires sample_size>=512")
    registry, registry_sha256 = _load_s2_registry(truth_registry_path)
    s2 = registry["S2_KC"]
    c_spec = s2["C_increment"]
    innovation_snr = float(s2["innovation_snr"])
    source_audit = _assert_source_only_template(template_shared)
    state, manifest = _load_checkpoint(authority_checkpoint)
    c_contract = state["c_contract"]
    if c_contract.get("family") != "BEST_ACTIVE_K_CHANNEL":
        raise RuntimeError("STOP_E2_S2_REQUIRES_FROZEN_BEST_ACTIVE_K_CHANNEL_PARENT")
    k_channel = str(c_contract["channel"])
    matches = [
        item
        for item in state["physical"]["channel_contracts"]
        if item["channel"] == k_channel
    ]
    if len(matches) != 1:
        raise RuntimeError("STOP_E2_S2_AUTHORITY_K_CONTRACT_AMBIGUOUS")
    k_operator = matches[0]
    c_channel = str(c_spec["channel"])
    if c_channel == k_channel:
        raise RuntimeError("STOP_E2_S2_C_INCREMENT_MUST_USE_SECOND_CHANNEL")

    shutil.copytree(template_shared, output_shared)
    base_path = output_shared / "base_data" / "cz_czochralski" / "train.parquet"
    original_base = pd.read_parquet(base_path)
    original_levels = {
        str(entity): group.sort_values("row_in_entity")[TARGET_COLUMN].to_numpy(
            dtype=np.float64
        )
        for entity, group in original_base.groupby("entity_id", sort=False)
    }
    pca = json.loads(
        (output_shared / "JOINT_LIFT_PCA_CONTRACT.json").read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(int(seed))
    shifted, shift_audit = _shift_inputs(original_base, pca, rng)
    shifted.to_parquet(base_path, index=False, compression="zstd")
    origins = _all_origins(shifted)

    k_accessor = BaseAccessor(output_shared, "cz_czochralski", "train", [k_channel])
    k_values, k_intervals = profile_values(
        k_accessor,
        origins,
        k_channel,
        tuple(int(value) for value in k_operator["profile"]),
        int(k_operator["m_tau"]),
    )
    k_raw = predict_contract(k_values, dict(k_operator["k_contract"]))
    c_accessor = BaseAccessor(output_shared, "cz_czochralski", "train", [c_channel])
    c_raw, c_intervals = _registered_linear_signal(
        c_accessor,
        origins,
        channel=c_channel,
        profile=tuple(int(value) for value in c_spec["profile"]),
        m_tau=int(c_spec["m_tau"]),
        interval_weights=c_spec["interval_weights"],
    )
    signal_frame = origins.assign(k_raw=k_raw, c_raw=c_raw)

    train_paths = list(
        output_shared.glob("sample_ids/*/input_only/record_time/primary/train.parquet")
    )
    if len(train_paths) != 1:
        raise RuntimeError("STOP_E2_TRAIN_SAMPLE_PATH_AMBIGUOUS")
    source_train = pd.read_parquet(train_paths[0])
    source_k = _lookup(signal_frame, "k_raw")
    source_c = _lookup(signal_frame, "c_raw")
    keys = [
        (str(entity), int(origin))
        for entity, origin in zip(
            source_train["entity_id"], source_train["origin"], strict=True
        )
    ]
    train_k = np.asarray([source_k[key] for key in keys], dtype=np.float64)
    train_c = np.asarray([source_c[key] for key in keys], dtype=np.float64)
    target_std = float(source_train["y_true"].to_numpy(dtype=np.float64).std())
    k_mean = float(train_k.mean(dtype=np.float64))
    c_mean = float(train_c.mean(dtype=np.float64))
    k_std = float(train_k.std(dtype=np.float64))
    c_std = float(train_c.std(dtype=np.float64))
    if min(k_std, c_std, target_std) <= np.finfo(np.float64).eps:
        raise RuntimeError("STOP_E2_S2_DEGENERATE_SOURCE_SCALE")
    k_rms_fraction = float(s2["K_parent"]["stage_rms_fraction"])
    c_rms_fraction = float(c_spec["stage_rms_fraction"])
    signal_frame["k_signal"] = (k_raw - k_mean) * (
        k_rms_fraction * target_std / k_std
    )
    signal_frame["c_signal"] = (c_raw - c_mean) * (
        c_rms_fraction * target_std / c_std
    )
    innovation_std = target_std / innovation_snr
    signal_frame["innovation"] = rng.normal(0.0, innovation_std, len(signal_frame))
    signal_frame["target_delta"] = (
        signal_frame["k_signal"]
        + signal_frame["c_signal"]
        + signal_frame["innovation"]
    )

    target_lookup = _lookup(signal_frame, "target_delta")
    synthetic_levels: dict[str, np.ndarray] = {}
    for entity, original in original_levels.items():
        level = original.copy()
        for origin in range(HISTORY_STEPS, len(level) - H_STEPS + 1):
            level[origin + H_STEPS - 1] = (
                level[origin - 1] + target_lookup[(entity, origin)]
            )
        synthetic_levels[entity] = level
    for entity, group in shifted.groupby("entity_id", sort=False):
        ordered = group.sort_values("row_in_entity")
        shifted.loc[ordered.index, TARGET_COLUMN] = synthetic_levels[str(entity)]
    shifted.to_parquet(base_path, index=False, compression="zstd")
    counts = _update_samples(output_shared, synthetic_levels, int(sample_size))
    audit = {
        "status": "PASS",
        "role": "P1_TINY_PILOT_NON_SELECTION_AUTHORITY",
        "regime": "S2_KC",
        "source_rod_only": True,
        "source_template_audit_status": source_audit["status"],
        "source_template_target_sheet_read": False,
        "formal_target_or_ood_accessed": False,
        "seed": int(seed),
        "sample_size_requested": int(sample_size),
        "sample_rows": counts,
        "history_steps": HISTORY_STEPS,
        "h_steps": H_STEPS,
        "w_steps": 1,
        "w0_steps": 1,
        "target_formula": "D[t+3]-D[t-1]",
        "block_shift_steps": BLOCK_SHIFT_STEPS,
        "shift_audit": shift_audit,
        "truth_registry": {
            "path": str(truth_registry_path),
            "sha256": registry_sha256,
            "format": registry["format"],
            "status": registry["status"],
        },
        "truth_operators": {
            "K": {
                "channel": k_channel,
                "profile": list(k_operator["profile"]),
                "m_tau": int(k_operator["m_tau"]),
                "family": k_operator["k_contract"]["family"],
                "intervals": [list(value) for value in k_intervals],
                "stage_rms_fraction": k_rms_fraction,
                "authority_checkpoint_hash": manifest["checkpoint_hash"],
            },
            "C": {
                "stage_semantics": c_spec["stage_semantics"],
                "authority_c_family": c_spec["authority_c_family"],
                "channel": c_channel,
                "profile": list(c_spec["profile"]),
                "m_tau": int(c_spec["m_tau"]),
                "family": c_spec["family"],
                "interval_weights": list(c_spec["interval_weights"]),
                "intervals": [list(value) for value in c_intervals],
                "stage_rms_fraction": c_rms_fraction,
            },
        },
        "source_train_target_std": target_std,
        "source_train_k_raw_mean": k_mean,
        "source_train_k_raw_std": k_std,
        "source_train_c_raw_mean": c_mean,
        "source_train_c_raw_std": c_std,
        "innovation_snr": innovation_snr,
        "innovation_std": innovation_std,
        "complete_synthetic_diameter_trajectory": True,
        "statistical_claims_allowed": False,
    }
    write_json(output_shared / "E2_SEMISYNTHETIC_DERIVATION.json", audit)
    return audit
