#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.c1_contracts import (  # noqa: E402
    realize_heads,
    stable_identifier,
    target_change,
    valid_anchor_origins_for_interval,
)


PRIMARY_INPUTS = ["ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
FORBIDDEN_PRIMARY_PROXIES = ["pm", "stator_tooth", "stator_yoke", "stator_winding"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class ParquetAppender:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.rows: dict[str, int] = {}

    def append(self, relative: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = self.writers.get(relative)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd", use_dictionary=True)
            self.writers[relative] = writer
            self.rows[relative] = 0
        elif writer.schema != table.schema:
            raise RuntimeError(f"parquet schema drift for {relative}")
        writer.write_table(table)
        self.rows[relative] += len(frame)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def sample_frame(*, entity_id: str, split: str, head: Any, origins: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    values = target_change(y, origins, head.h_steps, head.w_steps, head.w0_steps)
    base_ids = [
        stable_identifier("BASE_ORIGIN_V1", "pmsm", entity_id, head.head_id, int(origin))
        for origin in origins
    ]
    view_ids = [
        stable_identifier(
            "VIEW_SAMPLE_V1",
            base_id,
            "dynamic",
            "record_time",
            "proxy_excluded",
            "DEPENDENCY_INTERVAL_V1",
        )
        for base_id in base_ids
    ]
    return pd.DataFrame(
        {
            "base_origin_id": base_ids,
            "view_sample_id": view_ids,
            "dataset": "pmsm",
            "entity_id": entity_id,
            "task_id": head.task_id,
            "target_head": head.head_id,
            "split": split,
            "origin": origins,
            "current_start": origins - head.w0_steps,
            "current_stop_exclusive": origins,
            "target_start": origins + head.h_steps,
            "target_stop_exclusive": origins + head.h_steps + head.w_steps,
            "dependency_start": origins - head.w0_steps,
            "dependency_stop_exclusive": origins + head.h_steps + head.w_steps,
            "latest_available_target_index": origins - 1,
            "availability_delay_steps": 0,
            "availability_scenario": "record_time",
            "proxy_policy": "proxy_excluded",
            "information_set": "dynamic",
            "causal_history_floor": 0,
            "anchor_history_steps": head.w0_steps,
            "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
            "lmax_steps": head.w0_steps,
            "y_true": values,
        }
    )


def moments(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    values = frame[columns].to_numpy(dtype=np.float64, copy=False)
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.std(values, axis=0, dtype=np.float64)
    std = np.where(std > 0, std, 1.0)
    return {
        "fit_partition": "train_only",
        "dtype": "float64",
        "columns": columns,
        "count": [int(len(frame))] * len(columns),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PMSM stator-winding train/validation-only C1 shared data")
    parser.add_argument("--data", required=True, type=Path, help="CSV already physically stripped of registered test profiles")
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    split_registry = json.loads(args.split_registry.read_text(encoding="utf-8"))
    config, heads = realize_heads(args.task_config)
    if len(config["tasks"]) != 1 or config["tasks"][0]["task_id"] != "PMSM_SW":
        raise RuntimeError("isolated task config must contain only PMSM_SW")
    if any(head.status != "PASS" for head in heads):
        raise RuntimeError("one or more PMSM_SW heads are unsupported by cadence")
    if [head.head_id for head in heads] != ["PMSM_SW__H60__W60", "PMSM_SW__H600__W60", "PMSM_SW__H2400__W60"]:
        raise RuntimeError([head.head_id for head in heads])

    frame = pd.read_csv(args.data)
    required = {"profile_id", "stator_winding", *PRIMARY_INPUTS, *FORBIDDEN_PRIMARY_PROXIES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing PMSM columns: {missing}")

    frame["profile_id"] = pd.to_numeric(frame["profile_id"], errors="raise").astype(np.int64)
    train_profiles = set(int(v) for v in split_registry["train_profile_ids"])
    validation_profiles = set(int(v) for v in split_registry["validation_profile_ids"])
    test_profiles = set(int(v) for v in split_registry["test_profile_ids"])
    observed_profiles = set(int(v) for v in frame["profile_id"].unique())
    if observed_profiles & test_profiles:
        raise RuntimeError(f"registered test profiles physically present in development CSV: {sorted(observed_profiles & test_profiles)}")
    expected = train_profiles | validation_profiles
    if observed_profiles != expected:
        raise RuntimeError({"missing_development_profiles": sorted(expected - observed_profiles), "unexpected_profiles": sorted(observed_profiles - expected)})

    if freeze["input_contract"]["primary_inputs"] != PRIMARY_INPUTS:
        raise RuntimeError("freeze primary input contract drift")
    if freeze["input_contract"]["forbidden_primary_proxy_inputs"] != FORBIDDEN_PRIMARY_PROXIES:
        raise RuntimeError("freeze proxy exclusion drift")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    writer = ParquetAppender(args.output)

    train_base_parts: list[pd.DataFrame] = []
    sample_counts: dict[str, dict[str, int]] = {head.head_id: {"train": 0, "validation": 0} for head in heads}
    profile_counts = {"train": 0, "validation": 0}

    for profile_id, group in frame.groupby("profile_id", sort=True):
        profile = int(profile_id)
        if profile in train_profiles:
            split_name = "train"
        elif profile in validation_profiles:
            split_name = "validation"
        else:
            raise RuntimeError(f"unassigned development profile {profile}")
        profile_counts[split_name] += 1

        entity = group.reset_index(drop=True).copy()
        entity_id = str(profile)
        entity["entity_id"] = entity_id
        entity["row_in_entity"] = np.arange(len(entity), dtype=np.int64)
        writer.append(f"base_data/pmsm/{split_name}.parquet", entity)
        if split_name == "train":
            train_base_parts.append(entity[PRIMARY_INPUTS])

        y = entity["stator_winding"].to_numpy(dtype=np.float64, copy=False)
        for head in heads:
            origins = valid_anchor_origins_for_interval(
                0,
                len(entity),
                anchor_history_steps=head.w0_steps,
                h=head.h_steps,
                w=head.w_steps,
                delay=0,
                left_buffer=0,
            )
            if not len(origins):
                continue
            samples = sample_frame(entity_id=entity_id, split=split_name, head=head, origins=origins, y=y)
            relative = f"sample_ids/{head.head_id}/dynamic/record_time/proxy_excluded/{split_name}.parquet"
            writer.append(relative, samples)
            writer.append(
                f"targets/{head.head_id}/{split_name}.parquet",
                samples[["base_origin_id", "entity_id", "origin", "y_true"]],
            )
            sample_counts[head.head_id][split_name] += len(samples)

    writer.close()

    task_registry = {
        "contract_status": config["contract_status"],
        "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        "heads": [asdict(head) | {"head_id": head.head_id} for head in heads],
    }
    write_json(args.output / "TASK_REGISTRY.json", task_registry)
    write_json(args.output / "PROTOCOL.json", config)
    write_json(
        args.output / "dataset_views/VIEW_REGISTRY.json",
        [
            {
                "dataset": "pmsm",
                "task_id": "PMSM_SW",
                "proxy_policy": "proxy_excluded",
                "input_columns": PRIMARY_INPUTS,
                "dynamic_addition": "STRICT_PAST_TARGET_ONLY",
            }
        ],
    )
    write_json(args.output / "SPLIT_REGISTRY.json", {"pmsm": split_registry})
    write_json(
        args.output / "DATASET_HASHES.json",
        {"pmsm": {"source_full_raw_sha256": freeze["dataset"]["raw_sha256"], "development_csv_sha256": sha256_file(args.data)}},
    )
    write_json(
        args.output / "sequence_views/VIEW_SPEC.json",
        {"storage": "lazy_from_base_data_and_sample_ids", "index_contract": "HALF_OPEN_V1", "future_inputs": False},
    )
    write_json(
        args.output / "multiresolution_tabular_views/VIEW_SPEC.json",
        {
            "storage": "lazy",
            "source": "immutable_base_data",
            "profile_selection": "training_only",
            "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
            "candidate_native_history_mask_required_downstream": True,
        },
    )
    write_json(
        args.output / "masks/PURGE_CONTRACT.json",
        {
            "contract": "DEPENDENCY_INTERVAL_V1",
            "interval": "[t-W0,t+h+W) at anchor level; candidate-native K history is applied downstream",
            "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
            "test_profiles_physically_absent": True,
        },
    )

    train_base = pd.concat(train_base_parts, ignore_index=True)
    write_json(args.output / "scaler_metadata/pmsm/PMSM_SW/proxy_excluded.json", moments(train_base, PRIMARY_INPUTS))
    for head in heads:
        path = args.output / f"targets/{head.head_id}/train.parquet"
        target = pd.read_parquet(path, columns=["y_true"])["y_true"].to_numpy(dtype=np.float64)
        write_json(
            args.output / f"scaler_metadata/targets/{head.head_id}.json",
            {
                "fit_partition": "train_only",
                "count": int(len(target)),
                "mean": float(np.mean(target, dtype=np.float64)),
                "std": float(np.std(target, dtype=np.float64)),
            },
        )

    lockbox = {
        "contract": "PRISM_V2_2_PMSM_SW_PRELOCKBOX_NO_TEST_PRESENT_V1",
        "protocol_frozen": True,
        "registered_test_profiles": sorted(test_profiles),
        "test_profiles_physically_absent": True,
        "test_target_rows_materialized": False,
        "test_metrics_available": False,
        "unlock_rule": "separate final confirmatory workflow after train-validation implementation audit freeze",
    }
    write_json(args.output / "LOCKBOX.json", lockbox)

    files = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name not in {"SAMPLE_ID_REGISTRY.json", "VALIDATION_REPORT.md"}:
            item: dict[str, Any] = {
                "path": path.relative_to(args.output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.suffix == ".parquet":
                item["rows"] = pq.ParquetFile(path).metadata.num_rows
            files.append(item)
    registry = {
        "contract": "IMMUTABLE_SAMPLE_IDS_V1",
        "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        "anchor_universe": "MAXIMALLY_PERMISSIVE_HEAD_LEGAL_ANCHORS_TRAIN_VALIDATION_ONLY",
        "candidate_native_history_mask_required_downstream": True,
        "test_profiles_physically_absent": True,
        "files": files,
    }
    registry_sha = write_json(args.output / "SAMPLE_ID_REGISTRY.json", registry)

    manifest = {
        "status": "PASS",
        "protocol_id": freeze["protocol_id"],
        "task_id": "PMSM_SW",
        "raw_source_sha256": freeze["dataset"]["raw_sha256"],
        "development_csv_sha256": sha256_file(args.data),
        "development_rows": int(len(frame)),
        "development_profiles": int(len(observed_profiles)),
        "profile_counts": profile_counts,
        "test_profile_count_present": 0,
        "test_target_rows_materialized": 0,
        "test_metrics_computed": False,
        "sample_counts": sample_counts,
        "sample_registry_sha256": registry_sha,
        "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        "candidate_native_history_mask_required_downstream": True,
    }
    write_json(args.output / "BUILD_MANIFEST.json", manifest)
    (args.output / "VALIDATION_REPORT.md").write_text(
        "# PMSM SW pre-lockbox shared-data validation\n\n"
        "Status: `PASS`\n\n"
        "- Only registered train and validation profiles are present.\n"
        "- Registered test profiles and their target rows are physically absent.\n"
        "- Target construction uses `HALF_OPEN_V1` future-minus-current window change semantics.\n"
        "- Sample support is `NATIVE_K_COMMON_ASSEMBLY_R1`; candidate-native history masks remain mandatory downstream.\n"
        "- No model was fit and no target performance metric was computed.\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
