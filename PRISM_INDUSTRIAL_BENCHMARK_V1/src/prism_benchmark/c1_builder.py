from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadr

from .c1_contracts import (
    RealizedHead,
    ceil_steps,
    realize_heads,
    stable_identifier,
    target_change,
    valid_anchor_origins_for_interval,
)
from .stage0 import canonical_json_bytes, read_numeric_text, sha256_file, write_json


def _write_canonical(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return __import__("hashlib").sha256(payload).hexdigest()


class BufferedParquet:
    def __init__(self, root: Path, flush_rows: int = 100_000) -> None:
        self.root = root
        self.flush_rows = flush_rows
        self.buffers: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)
        self.total_rows: dict[str, int] = defaultdict(int)
        self.writers: dict[str, pq.ParquetWriter] = {}

    def append(self, relative: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.buffers[relative].append(frame)
        self.counts[relative] += len(frame)
        self.total_rows[relative] += len(frame)
        if self.counts[relative] >= self.flush_rows:
            self.flush(relative)

    def flush(self, relative: str) -> None:
        if not self.buffers[relative]:
            return
        frame = pd.concat(self.buffers[relative], ignore_index=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative not in self.writers:
            self.writers[relative] = pq.ParquetWriter(
                path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        self.writers[relative].write_table(table)
        self.buffers[relative].clear()
        self.counts[relative] = 0

    def close(self) -> None:
        for relative in list(self.buffers):
            self.flush(relative)
        for writer in self.writers.values():
            writer.close()


class OnlineMoments:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.count = np.zeros(len(columns), dtype=np.int64)
        self.sum = np.zeros(len(columns), dtype=np.float64)
        self.sumsq = np.zeros(len(columns), dtype=np.float64)

    def update(self, frame: pd.DataFrame) -> None:
        values = frame[self.columns].to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        self.count += finite.sum(axis=0)
        safe = np.where(finite, values, 0.0)
        self.sum += safe.sum(axis=0, dtype=np.float64)
        self.sumsq += np.square(safe).sum(axis=0, dtype=np.float64)

    def result(self) -> dict[str, Any]:
        mean = self.sum / np.maximum(self.count, 1)
        variance = self.sumsq / np.maximum(self.count, 1) - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std > 0, std, 1.0)
        return {
            "fit_partition": "train_only",
            "dtype": "float64",
            "columns": self.columns,
            "count": self.count.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }


def _task_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in config["tasks"]}


def _input_columns(dataset: str, task_id: str, columns: list[str], proxy_policy: str) -> list[str]:
    metadata = {"faultNumber", "simulationRun", "sample", "profile_id", "timestamp", "Unnamed: 0", "row_in_entity", "entity_id"}
    if dataset == "tep":
        excluded = {f"xmeas_{index}" for index in range(37, 42)}
        return [column for column in columns if column not in metadata | excluded]
    if dataset == "debutanizer":
        return [column for column in columns if column.startswith("u")]
    if dataset == "sru":
        return [column for column in columns if column.startswith("u")]
    if dataset == "pmsm":
        primary = ["ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
        secondary = ["stator_winding", "stator_tooth", "stator_yoke"]
        return primary + (secondary if proxy_policy == "full_sensor_secondary" else [])
    if dataset == "metropt":
        excluded = metadata | ({"Reservoirs", "TP3"} if task_id == "METRO_P60" else {"Oil_temperature"})
        result = [column for column in columns if column not in excluded]
        if task_id == "METRO_P60" and proxy_policy == "full_sensor_secondary":
            result.append("TP3")
        return list(dict.fromkeys(result))
    raise KeyError(dataset)


def _availability_name(dataset: str, delay: int) -> str:
    if dataset == "tep":
        return "record_time" if delay == 0 else "analyzer_maturity_5_steps"
    if dataset == "debutanizer":
        return "record_time" if delay == 0 else "delay_10_steps"
    return "record_time"


def _base_origin_ids(dataset: str, entity_id: str, head_id: str, origins: np.ndarray) -> list[str]:
    return [stable_identifier("BASE_ORIGIN_V1", dataset, entity_id, head_id, int(origin)) for origin in origins]


def _sample_frame(
    *,
    dataset: str,
    entity_id: str,
    split: str,
    head: RealizedHead,
    delay: int,
    proxy_policy: str,
    information_set: str,
    causal_history_floor: int,
    anchor_history_steps: int,
    origins: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    targets = target_change(y, origins, head.h_steps, head.w_steps, head.w0_steps)
    base_ids = _base_origin_ids(dataset, entity_id, head.head_id, origins)
    availability = _availability_name(dataset, delay)
    view_ids = [
        stable_identifier(
            "VIEW_SAMPLE_V1",
            base_id,
            information_set,
            availability,
            proxy_policy,
            "DEPENDENCY_INTERVAL_V1",
        )
        for base_id in base_ids
    ]
    return pd.DataFrame(
        {
            "base_origin_id": base_ids,
            "view_sample_id": view_ids,
            "dataset": dataset,
            "entity_id": entity_id,
            "task_id": head.task_id,
            "target_head": head.head_id,
            "split": split,
            "origin": origins,
            "current_start": origins - head.w0_steps,
            "current_stop_exclusive": origins,
            "target_start": origins + head.h_steps,
            "target_stop_exclusive": origins + head.h_steps + head.w_steps,
            "dependency_start": origins - anchor_history_steps,
            "dependency_stop_exclusive": origins + head.h_steps + head.w_steps + delay,
            "latest_available_target_index": origins - 1 - delay,
            "availability_delay_steps": delay,
            "availability_scenario": availability,
            "proxy_policy": proxy_policy,
            "information_set": information_set,
            "causal_history_floor": causal_history_floor,
            "anchor_history_steps": anchor_history_steps,
            "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
            # Compatibility-only anchor history.  This is not the maximum
            # registered K history under NATIVE_K_COMMON_ASSEMBLY_R1.
            "lmax_steps": anchor_history_steps,
            "y_true": targets,
        }
    )


def _split_lookup(registry_root: Path, dataset: str) -> dict[str, Any]:
    return json.loads((registry_root / dataset / "SPLIT_REGISTRY.json").read_text(encoding="utf-8"))


def _tep_entities(raw_root: Path) -> Iterator[tuple[str, str, pd.DataFrame]]:
    paths = sorted((raw_root / "tep_rieth").glob("*.RData"))
    for path in paths:
        objects = pyreadr.read_r(path)
        _, frame = next(iter(objects.items()))
        frame = normalize_tep_index_dtypes(frame)
        partition = "Training" if "Training" in path.name else "Testing"
        for (fault, run), group in frame.groupby(["faultNumber", "simulationRun"], sort=True):
            entity_id = f"{partition}|fault={int(fault)}|run={int(run)}"
            yield entity_id, path.stem, group.sort_values("sample").reset_index(drop=True)


def normalize_tep_index_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("faultNumber", "simulationRun", "sample"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"TEP index column {column} contains non-integer values")
        frame[column] = values.astype(np.int64)
    return frame


def _pmsm_entities(raw_root: Path) -> Iterator[tuple[str, str, pd.DataFrame]]:
    frame = pd.read_csv(raw_root / "pmsm_original/measures_v2.csv")
    for profile, group in frame.groupby("profile_id", sort=True):
        yield str(int(profile)), "measures_v2", group.reset_index(drop=True)


def _single_entity(dataset: str, raw_root: Path) -> tuple[str, str, pd.DataFrame]:
    if dataset == "debutanizer":
        values, _ = read_numeric_text(raw_root / "industrial-debutanizer-soft-sensor/data/debutanizer_data.txt", 8)
        return "debutanizer_1", "debutanizer_data", pd.DataFrame(values, columns=[f"u{i}" for i in range(1, 8)] + ["y"])
    if dataset == "sru":
        values, _ = read_numeric_text(raw_root / "sru/SRU_data.txt", 7)
        return "sru_line4", "SRU_data", pd.DataFrame(values, columns=[f"u{i}" for i in range(1, 6)] + ["y1", "y2"])
    if dataset == "metropt":
        with zipfile.ZipFile(raw_root / "metropt/metropt_3.zip") as archive, archive.open("MetroPT3(AirCompressor).csv") as handle:
            frame = pd.read_csv(handle)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        return "metropt_3", "MetroPT3", frame
    raise KeyError(dataset)


def _metro_labels(frame: pd.DataFrame, split: dict[str, Any]) -> np.ndarray:
    timestamps = frame["timestamp"]
    months = timestamps.dt.to_period("M").astype(str)
    labels = np.full(len(frame), "unused", dtype=object)
    for label, months_for_split in (
        ("train", split["train"]),
        ("validation", split["validation"]),
        ("test", split["test"]),
    ):
        labels[months.isin(months_for_split).to_numpy()] = label
    for start, stop in split["fault_windows"]:
        mask = ((timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(stop))).to_numpy()
        labels[mask] = "ood"
    return labels


def _segments(labels: np.ndarray) -> list[tuple[int, int, str]]:
    if len(labels) == 0:
        return []
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(labels)]))
    return [(int(a), int(b), str(labels[a])) for a, b in zip(boundaries[:-1], boundaries[1:])]


def _continuous_intervals(dataset: str, frame: pd.DataFrame, split: dict[str, Any], b_steps: int) -> list[tuple[int, int, str, int]]:
    if dataset in {"debutanizer", "sru"}:
        return [
            (split["train"][0], split["train"][1], "train", 0),
            (split["validation"][0], split["validation"][1], "validation", b_steps),
            (split["test"][0], split["test"][1], "test", b_steps),
        ]
    labels = _metro_labels(frame, split)
    return [(start, stop, label, 0 if start == 0 else b_steps) for start, stop, label in _segments(labels)]


def _dataset_entities(dataset: str, raw_root: Path) -> Iterator[tuple[str, str, pd.DataFrame]]:
    if dataset == "tep":
        yield from _tep_entities(raw_root)
    elif dataset == "pmsm":
        yield from _pmsm_entities(raw_root)
    else:
        yield _single_entity(dataset, raw_root)


def _entity_split(dataset: str, entity_id: str, split: dict[str, Any]) -> str:
    if dataset == "tep":
        for label, key in (
            ("train", "train_run_ids"),
            ("validation", "validation_run_ids"),
            ("test", "main_test_run_ids"),
            ("ood", "unseen_disturbance_ood_run_ids"),
            ("discarded", "discarded_run_ids"),
        ):
            if entity_id in _entity_split._cache[(dataset, key)]:
                return label
    elif dataset == "pmsm":
        profile = int(entity_id)
        for label, key in (("train", "train_profile_ids"), ("validation", "validation_profile_ids"), ("test", "test_profile_ids")):
            if profile in _entity_split._cache[(dataset, key)]:
                return label
    raise KeyError(f"unassigned entity {dataset}:{entity_id}")


_entity_split._cache = {}  # type: ignore[attr-defined]


def _prepare_split_cache(dataset: str, split: dict[str, Any]) -> None:
    if dataset == "tep":
        keys = ("train_run_ids", "validation_run_ids", "main_test_run_ids", "unseen_disturbance_ood_run_ids", "discarded_run_ids")
    elif dataset == "pmsm":
        keys = ("train_profile_ids", "validation_profile_ids", "test_profile_ids")
    else:
        return
    for key in keys:
        _entity_split._cache[(dataset, key)] = set(split[key])  # type: ignore[attr-defined]


def _view_specs(dataset: str, task: dict[str, Any], columns: list[str]) -> list[dict[str, Any]]:
    specs = []
    for policy in task["proxy_policies"]:
        specs.append(
            {
                "dataset": dataset,
                "task_id": task["task_id"],
                "proxy_policy": policy,
                "input_columns": _input_columns(dataset, task["task_id"], columns, policy),
                "dynamic_addition": "STRICT_PAST_TARGET_ONLY",
            }
        )
    return specs


def _lockbox_paths(output_root: Path) -> list[str]:
    return [
        path.relative_to(output_root).as_posix()
        for path in sorted(output_root.rglob("*.parquet"))
        if path.name in {"test.parquet", "ood.parquet"}
    ]


def write_lockbox_contract(output_root: Path) -> str:
    paths = _lockbox_paths(output_root)
    contract = {
        "contract": "PRISM_C1_TEST_LOCKBOX_V1",
        "protocol_frozen": False,
        "access_rule": "MODEL_HYPERPARAMETER_PROFILE_AND_THRESHOLDS_FROZEN",
        "metric_access_before_freeze": False,
        "locked_files": paths,
    }
    digest = _write_canonical(output_root / "LOCKBOX.json", contract)
    for relative in paths:
        os.chmod(output_root / relative, 0o400)
    return digest


def finalize_existing_shared_data(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve(strict=True)
    lockbox_hash = write_lockbox_contract(output_root)
    registry_path = output_root / "SAMPLE_ID_REGISTRY.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    files = [entry for entry in registry["files"] if entry["path"] != "LOCKBOX.json"]
    lockbox_path = output_root / "LOCKBOX.json"
    files.append({"path": "LOCKBOX.json", "bytes": lockbox_path.stat().st_size, "sha256": lockbox_hash})
    registry["files"] = sorted(files, key=lambda entry: entry["path"])
    registry_hash = _write_canonical(registry_path, registry)
    os.chmod(registry_path, 0o444)
    report_path = output_root / "VALIDATION_REPORT.md"
    lines = [line for line in report_path.read_text(encoding="utf-8").splitlines() if not line.startswith("- Registered files:") and not line.startswith("- Sample registry SHA256:") and not line.startswith("- Lockbox contract:")]
    lines.extend(
        [
            f"- Registered files: {len(registry['files'])}",
            f"- Sample registry SHA256: `{registry_hash}`",
            f"- Lockbox contract: `PRISM_C1_TEST_LOCKBOX_V1` ({len(registry['files'])} registry entries; {len(_lockbox_paths(output_root))} locked parquet files).",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "PASS", "sample_registry_sha256": registry_hash, "lockbox_sha256": lockbox_hash}


def build_shared_data(raw_root: Path, registry_root: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    raw_root = raw_root.resolve(strict=True)
    registry_root = registry_root.resolve(strict=True)
    config, heads = realize_heads(config_path)
    if any(head.status != "PASS" for head in heads):
        raise RuntimeError("one or more registered heads are unsupported by cadence")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    writers = BufferedParquet(output_root)
    task_by_id = _task_map(config)
    heads_by_dataset: dict[str, list[RealizedHead]] = defaultdict(list)
    for head in heads:
        heads_by_dataset[head.dataset].append(head)

    all_view_specs: list[dict[str, Any]] = []
    scaler_accumulators: dict[tuple[str, str, str], OnlineMoments] = {}
    target_moments: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    for dataset in ("tep", "debutanizer", "sru", "pmsm", "metropt"):
        split = _split_lookup(registry_root, dataset)
        _prepare_split_cache(dataset, split)
        entities = _dataset_entities(dataset, raw_root)
        first_columns: list[str] | None = None
        for entity_id, source_name, frame in entities:
            frame = frame.copy()
            frame["entity_id"] = entity_id
            frame["row_in_entity"] = np.arange(len(frame), dtype=np.int64)
            if first_columns is None:
                first_columns = list(frame.columns)
                for task in (task_by_id[head.task_id] for head in heads_by_dataset[dataset]):
                    for spec in _view_specs(dataset, task, first_columns):
                        if spec not in all_view_specs:
                            all_view_specs.append(spec)
                            key = (dataset, spec["task_id"], spec["proxy_policy"])
                            scaler_accumulators[key] = OnlineMoments(spec["input_columns"])

            if dataset in {"tep", "pmsm"}:
                split_label = _entity_split(dataset, entity_id, split)
                intervals = [(0, len(frame), split_label, 0)]
            else:
                b_steps = ceil_steps(config["extra_dependency_interval_seconds"], float(heads_by_dataset[dataset][0].cadence_seconds))
                intervals = _continuous_intervals(dataset, frame, split, b_steps)

            for start, stop, split_label, _ in intervals:
                base_chunk = frame.iloc[start:stop].copy()
                writers.append(f"base_data/{dataset}/{split_label}.parquet", base_chunk)
                if split_label == "train":
                    for spec in all_view_specs:
                        if spec["dataset"] == dataset:
                            scaler_accumulators[(dataset, spec["task_id"], spec["proxy_policy"])].update(base_chunk)

            for head in heads_by_dataset[dataset]:
                task = task_by_id[head.task_id]
                y = frame[head.target].to_numpy(dtype=np.float64, copy=False)
                for delay in task["availability_delays_steps"]:
                    information_sets = ["dynamic"]
                    if delay == 0:
                        information_sets.insert(0, "input_only")
                    for start, stop, split_label, left_buffer in intervals:
                        if split_label in {"discarded", "unused"}:
                            continue
                        causal_history_floor = start + left_buffer
                        origins = valid_anchor_origins_for_interval(
                            start,
                            stop,
                            anchor_history_steps=head.w0_steps,
                            h=head.h_steps,
                            w=head.w_steps,
                            delay=int(delay),
                            left_buffer=left_buffer,
                        )
                        if len(origins) == 0:
                            continue
                        for policy_index, proxy_policy in enumerate(task["proxy_policies"]):
                            for information_set in information_sets:
                                sample = _sample_frame(
                                    dataset=dataset,
                                    entity_id=entity_id,
                                    split=split_label,
                                    head=head,
                                    delay=int(delay),
                                    proxy_policy=proxy_policy,
                                    information_set=information_set,
                                    causal_history_floor=causal_history_floor,
                                    anchor_history_steps=head.w0_steps,
                                    origins=origins,
                                    y=y,
                                )
                                availability = _availability_name(dataset, int(delay))
                                relative = f"sample_ids/{head.head_id}/{information_set}/{availability}/{proxy_policy}/{split_label}.parquet"
                                writers.append(relative, sample)
                                if delay == 0 and policy_index == 0 and information_set == "input_only":
                                    target_relative = f"targets/{head.head_id}/{split_label}.parquet"
                                    writers.append(target_relative, sample[["base_origin_id", "entity_id", "origin", "y_true"]])
                                    if split_label == "train":
                                        stats = target_moments[head.head_id]
                                        values = sample["y_true"].to_numpy(dtype=np.float64)
                                        stats[0] += len(values)
                                        stats[1] += values.sum(dtype=np.float64)
                                        stats[2] += np.square(values).sum(dtype=np.float64)

        if first_columns is None:
            raise RuntimeError(f"no entities loaded for {dataset}")

    writers.close()

    task_registry = {
        "contract_status": config["contract_status"],
        "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        "heads": [asdict(head) | {"head_id": head.head_id} for head in heads],
    }
    _write_canonical(output_root / "TASK_REGISTRY.json", task_registry)
    _write_canonical(output_root / "PROTOCOL.json", config)
    dataset_hashes = {}
    split_registry = {}
    for dataset in ("tep", "debutanizer", "sru", "pmsm", "metropt"):
        dataset_hashes[dataset] = json.loads((registry_root / dataset / "RAW_FILE_HASHES.json").read_text())
        split_registry[dataset] = json.loads((registry_root / dataset / "SPLIT_REGISTRY.json").read_text())
    _write_canonical(output_root / "DATASET_HASHES.json", dataset_hashes)
    _write_canonical(output_root / "SPLIT_REGISTRY.json", split_registry)
    _write_canonical(output_root / "dataset_views/VIEW_REGISTRY.json", all_view_specs)
    _write_canonical(
        output_root / "sequence_views/VIEW_SPEC.json",
        {"storage": "lazy_from_base_data_and_sample_ids", "index_contract": "HALF_OPEN_V1", "future_inputs": False},
    )
    _write_canonical(
        output_root / "multiresolution_tabular_views/VIEW_SPEC.json",
        {
            "storage": "lazy",
            "source": "immutable_base_data",
            "profile_selection": "outer_training_only_C4",
            "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        },
    )
    _write_canonical(
        output_root / "graph_views/VIEW_SPEC.json",
        {"storage": "lazy", "source": "immutable_base_data", "graph_fit": "outer_training_only_GPU_stage"},
    )
    _write_canonical(
        output_root / "masks/PURGE_CONTRACT.json",
        {
            "contract": "DEPENDENCY_INTERVAL_V1",
            "interval": "[t-W0,t+h+W+D) at anchor level; candidate-native K history is applied downstream",
            "extra_buffer_seconds": 600,
            "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        },
    )
    for key, moments in scaler_accumulators.items():
        dataset, task_id, proxy_policy = key
        _write_canonical(output_root / f"scaler_metadata/{dataset}/{task_id}/{proxy_policy}.json", moments.result())
    for head_id, (count, total, total_sq) in target_moments.items():
        mean = total / max(count, 1.0)
        variance = total_sq / max(count, 1.0) - mean * mean
        _write_canonical(
            output_root / f"scaler_metadata/targets/{head_id}.json",
            {"fit_partition": "train_only", "count": int(count), "mean": mean, "std": float(np.sqrt(max(variance, 0.0)))},
        )

    write_lockbox_contract(output_root)

    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"SAMPLE_ID_REGISTRY.json", "VALIDATION_REPORT.md"}:
            relative = path.relative_to(output_root).as_posix()
            entry = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if path.suffix == ".parquet":
                entry["rows"] = pq.ParquetFile(path).metadata.num_rows
                entry["dtype"] = "schema_in_parquet"
            files.append(entry)
    registry = {
        "contract": "IMMUTABLE_SAMPLE_IDS_V1",
        "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        "anchor_universe": "MAXIMALLY_PERMISSIVE_HEAD_LEGAL_ANCHORS",
        "protocol_sha256": sha256_file(output_root / "PROTOCOL.json"),
        "files": files,
    }
    registry_hash = _write_canonical(output_root / "SAMPLE_ID_REGISTRY.json", registry)
    report = [
        "# C1 shared-data validation report",
        "",
        "Status: `PASS`",
        "",
        "- Target index contract: `HALF_OPEN_V1`",
        "- Time realization contract: `ROUND_HALF_UP_V1`",
        "- Purge contract: `DEPENDENCY_INTERVAL_V1`",
        "- Sample support contract: `NATIVE_K_COMMON_ASSEMBLY_R1` (maximal anchor universe; native K masks downstream).",
        "- Raw test targets were materialized into immutable lockbox artifacts but were not summarized or used for selection.",
        f"- Registered files: {len(files)}",
        f"- Sample registry SHA256: `{registry_hash}`",
        "- All scaler metadata are train-only.",
        "- Expanded windows are lazy and must be reconstructed only from immutable base data plus sample registries.",
        f"- Lockbox contract: `PRISM_C1_TEST_LOCKBOX_V1` ({len(_lockbox_paths(output_root))} locked parquet files).",
    ]
    (output_root / "VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    os.chmod(output_root / "SAMPLE_ID_REGISTRY.json", 0o444)
    return {"status": "PASS", "files": len(files), "sample_registry_sha256": registry_hash}
