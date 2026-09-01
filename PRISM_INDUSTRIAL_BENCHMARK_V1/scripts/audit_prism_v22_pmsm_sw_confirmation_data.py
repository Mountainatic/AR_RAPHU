#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_INPUTS = ["ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
EXPECTED_FORBIDDEN = ["pm", "stator_tooth", "stator_yoke", "stator_winding"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--c1-tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    split = json.loads(args.split_registry.read_text(encoding="utf-8"))
    c1 = json.loads(args.c1_tasks.read_text(encoding="utf-8"))

    # Lockbox hygiene: read only the header and profile_id column.  The new
    # stator_winding target values are deliberately never loaded here.
    columns = list(pd.read_csv(args.data, nrows=0).columns)
    profiles = pd.read_csv(args.data, usecols=["profile_id"], dtype={"profile_id": "int64"})["profile_id"]

    digest = sha256_file(args.data)
    assert digest == freeze["dataset"]["raw_sha256"], (digest, freeze["dataset"]["raw_sha256"])
    assert freeze["task"]["target"] == "stator_winding"
    assert freeze["input_contract"]["primary_inputs"] == EXPECTED_INPUTS
    assert freeze["input_contract"]["forbidden_primary_proxy_inputs"] == EXPECTED_FORBIDDEN
    assert all(name in columns for name in ["profile_id", "stator_winding", *EXPECTED_INPUTS, *EXPECTED_FORBIDDEN])
    assert "stator_winding" not in EXPECTED_INPUTS

    registered_targets = {task["target"] for task in c1["tasks"]}
    assert "stator_winding" not in registered_targets, registered_targets

    train = set(int(value) for value in split["train_profile_ids"])
    validation = set(int(value) for value in split["validation_profile_ids"])
    test = set(int(value) for value in split["test_profile_ids"])
    assert not (train & validation or train & test or validation & test)
    expected_profiles = train | validation | test
    observed_profiles = set(int(value) for value in profiles.unique())
    assert expected_profiles == observed_profiles, {
        "missing": sorted(expected_profiles - observed_profiles),
        "unexpected": sorted(observed_profiles - expected_profiles),
    }
    assert sorted(test) == sorted(freeze["dataset"]["test_profile_ids"])

    row_counts = {
        "train": int(profiles.isin(train).sum()),
        "validation": int(profiles.isin(validation).sum()),
        "test": int(profiles.isin(test).sum()),
    }
    assert sum(row_counts.values()) == len(profiles)

    output = {
        "audit_id": "PRISM_V2_2_PMSM_SW_ZERO_TARGET_VALUE_DATA_AUDIT_20260901_V1",
        "status": "PASS",
        "protocol_id": freeze["protocol_id"],
        "raw_sha256": digest,
        "raw_bytes": args.data.stat().st_size,
        "row_count": int(len(profiles)),
        "columns": columns,
        "sampling_hz": split.get("sampling_hz"),
        "profile_count": int(len(observed_profiles)),
        "row_counts_by_split": row_counts,
        "profile_counts_by_split": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "registered_targets_before_confirmation": sorted(registered_targets),
        "new_target": "stator_winding",
        "new_target_values_read_by_this_audit": False,
        "test_target_metrics_computed": False,
        "test_target_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
