from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .c1_contracts import stable_identifier
from .stage0 import sha256_file, write_json


SAMPLE_COLUMNS = [
    "base_origin_id",
    "view_sample_id",
    "dataset",
    "entity_id",
    "target_head",
    "split",
    "origin",
    "current_start",
    "current_stop_exclusive",
    "target_start",
    "target_stop_exclusive",
    "dependency_start",
    "dependency_stop_exclusive",
    "latest_available_target_index",
    "availability_delay_steps",
    "availability_scenario",
    "proxy_policy",
    "information_set",
    "causal_history_floor",
    "anchor_history_steps",
    "sample_support_contract",
    "lmax_steps",
    "y_true",
]

GENERATED_CSV_ALLOWLIST = {"C1_SAMPLE_COUNTS.csv"}


def _forbidden_raw_files(root: Path) -> list[Path]:
    forbidden_suffixes = {".rdata", ".xlsx", ".xls", ".zip"}
    result: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in forbidden_suffixes:
            result.append(path)
        elif path.suffix.lower() == ".csv" and str(path.relative_to(root)) not in GENERATED_CSV_ALLOWLIST:
            result.append(path)
    return result


def _sample_rows(path: Path) -> list[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    table = parquet.read_row_group(0, columns=SAMPLE_COLUMNS)
    if table.num_rows == 0:
        return []
    indices = sorted({0, table.num_rows // 2, table.num_rows - 1})
    return [
        {name: table[name][index].as_py() for name in SAMPLE_COLUMNS}
        for index in indices
    ]


def _validate_sample_row(row: dict[str, Any], head: dict[str, Any]) -> None:
    origin = row["origin"]
    delay = row["availability_delay_steps"]
    assert row["current_start"] == origin - head["w0_steps"]
    assert row["current_stop_exclusive"] == origin
    assert row["target_start"] == origin + head["h_steps"]
    assert row["target_stop_exclusive"] == origin + head["h_steps"] + head["w_steps"]
    assert row["dependency_start"] == origin - row["lmax_steps"]
    assert row["dependency_start"] == origin - row["anchor_history_steps"]
    assert row["anchor_history_steps"] == head["w0_steps"]
    assert row["sample_support_contract"] == "NATIVE_K_COMMON_ASSEMBLY_R1"
    assert origin - row["anchor_history_steps"] >= row["causal_history_floor"]
    assert row["dependency_stop_exclusive"] == origin + head["h_steps"] + head["w_steps"] + delay
    assert row["latest_available_target_index"] == origin - 1 - delay
    expected_base = stable_identifier("BASE_ORIGIN_V1", row["dataset"], row["entity_id"], row["target_head"], origin)
    expected_view = stable_identifier(
        "VIEW_SAMPLE_V1",
        expected_base,
        row["information_set"],
        row["availability_scenario"],
        row["proxy_policy"],
        "DEPENDENCY_INTERVAL_V1",
    )
    assert row["base_origin_id"] == expected_base
    assert row["view_sample_id"] == expected_view
    assert np.isfinite(row["y_true"])


def validate_shared_data(root: Path, output: Path | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    registry = json.loads((root / "SAMPLE_ID_REGISTRY.json").read_text(encoding="utf-8"))
    task_registry = json.loads((root / "TASK_REGISTRY.json").read_text(encoding="utf-8"))
    heads = {head["head_id"]: head for head in task_registry["heads"]}
    lockbox = json.loads((root / "LOCKBOX.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    if registry.get("sample_support_contract") != "NATIVE_K_COMMON_ASSEMBLY_R1":
        failures.append("SAMPLE_SUPPORT_CONTRACT")
    if task_registry.get("sample_support_contract") != "NATIVE_K_COMMON_ASSEMBLY_R1":
        failures.append("TASK_SAMPLE_SUPPORT_CONTRACT")
    checked_parquet = 0
    checked_rows = 0
    sample_counts: dict[tuple[str, str, str, str, str], int] = {}
    target_counts: dict[tuple[str, str], int] = {}

    for entry in registry["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"MISSING:{entry['path']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append(f"SIZE:{entry['path']}")
        if sha256_file(path) != entry["sha256"]:
            failures.append(f"SHA256:{entry['path']}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            checked_parquet += 1
            if parquet.metadata.num_rows != entry["rows"]:
                failures.append(f"ROWS:{entry['path']}")
            if entry["path"].startswith("sample_ids/"):
                parts = Path(entry["path"]).parts
                sample_counts[(parts[1], parts[2], parts[3], parts[4], Path(parts[5]).stem)] = entry["rows"]
                missing = set(SAMPLE_COLUMNS).difference(parquet.schema.names)
                if missing:
                    failures.append(f"SCHEMA:{entry['path']}:{sorted(missing)}")
                else:
                    try:
                        rows = _sample_rows(path)
                        for row in rows:
                            _validate_sample_row(row, heads[row["target_head"]])
                        checked_rows += len(rows)
                    except Exception as error:  # validation must retain the exact failing file
                        failures.append(f"SEMANTICS:{entry['path']}:{type(error).__name__}:{error}")
            elif entry["path"].startswith("targets/"):
                parts = Path(entry["path"]).parts
                target_counts[(parts[1], Path(parts[2]).stem)] = entry["rows"]

    for (head, info, availability, policy, split), count in sample_counts.items():
        if availability == "record_time" and info == "input_only":
            dynamic = sample_counts.get((head, "dynamic", availability, policy, split))
            if dynamic != count:
                failures.append(f"INFO_SET_SAMPLE_MISMATCH:{head}:{policy}:{split}:{count}:{dynamic}")
        same_view_policy_counts = {
            other_count
            for (other_head, other_info, other_availability, _other_policy, other_split), other_count in sample_counts.items()
            if (other_head, other_info, other_availability, other_split) == (head, info, availability, split)
        }
        if len(same_view_policy_counts) != 1:
            failures.append(f"PROXY_SAMPLE_MISMATCH:{head}:{info}:{availability}:{split}")
    for (head, split), count in target_counts.items():
        candidates = [
            sample_count
            for (sample_head, info, availability, _policy, sample_split), sample_count in sample_counts.items()
            if (sample_head, info, availability, sample_split) == (head, "input_only", "record_time", split)
        ]
        if not candidates or any(candidate != count for candidate in candidates):
            failures.append(f"TARGET_SAMPLE_MISMATCH:{head}:{split}:{count}:{candidates}")
    required_splits = {
        "tep": {"train", "validation", "test", "ood"},
        "debutanizer": {"train", "validation", "test"},
        "sru": {"train", "validation", "test"},
        "pmsm": {"train", "validation", "test"},
        "metropt": {"train", "validation", "test", "ood"},
    }
    for head_id, head in heads.items():
        present = {split for (candidate_head, split), count in target_counts.items() if candidate_head == head_id and count > 0}
        missing = required_splits[head["dataset"]].difference(present)
        if missing:
            failures.append(f"MISSING_REQUIRED_SPLIT:{head_id}:{sorted(missing)}")

    expected_locked = sorted(
        entry["path"]
        for entry in registry["files"]
        if entry["path"].endswith("/test.parquet") or entry["path"].endswith("/ood.parquet")
    )
    if sorted(lockbox["locked_files"]) != expected_locked:
        failures.append("LOCKBOX_FILE_SET")
    if lockbox["protocol_frozen"] is not False or lockbox["metric_access_before_freeze"] is not False:
        failures.append("LOCKBOX_STATE")
    for relative in expected_locked:
        if os.stat(root / relative).st_mode & 0o222:
            failures.append(f"LOCKBOX_WRITABLE:{relative}")
    forbidden = _forbidden_raw_files(root)
    if forbidden:
        failures.append("RAW_FILES_PRESENT:" + ",".join(path.name for path in forbidden))

    result = {
        "status": "PASS" if not failures else "FAIL",
        "registry_sha256": sha256_file(root / "SAMPLE_ID_REGISTRY.json"),
        "registered_files": len(registry["files"]),
        "checked_parquet_files": checked_parquet,
        "sample_semantic_spot_checks": checked_rows,
        "locked_files": len(expected_locked),
        "failures": failures,
    }
    if output is not None:
        write_json(output, result)
    return result
