#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.c1_contracts import realize_heads  # noqa: E402


FAST = {"u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"}
SLOW = {"ambient", "coolant"}
CHANNELS = ["ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]


def profiles_for(head, channel: str, c4: dict) -> list[tuple[int, int]]:
    if channel in FAST:
        category = "FAST"
    elif channel in SLOW:
        category = "SLOW"
    else:
        raise KeyError(channel)
    deltas = [int(v) for v in c4["delta_ratio_by_class"][category]]
    maximum_delta = max(1, int(head.h_steps + head.w_steps))
    deltas = [delta for delta in deltas if delta <= maximum_delta] or [min(deltas)]
    if head.h_steps <= 0:
        raise RuntimeError("PMSM_SW confirmation has only positive horizons")
    histories = [max(1, int(multiplier * head.h_steps)) for multiplier in c4["history_for_positive_h"]]
    return sorted({(delta, history) for delta in deltas for history in histories if delta <= history}, key=lambda x: (x[1], -x[0]))


def support_rows(length: int, head, history: int) -> int:
    first = max(int(head.w0_steps), int(history))
    last = int(length) - int(head.h_steps) - int(head.w_steps)
    return max(0, last - first + 1)


def grouped_chunks(profile_ids: list[int], count: int = 5) -> list[list[int]]:
    ordered = sorted((str(v) for v in profile_ids), key=lambda value: (len(value), value))
    return [list(map(int, chunk.tolist())) for chunk in np.array_split(np.asarray(ordered, dtype=object), count)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--cpu-freeze", required=True, type=Path)
    parser.add_argument("--confirmation-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    _, heads = realize_heads(args.task_config)
    split = json.loads(args.split_registry.read_text(encoding="utf-8"))
    cpu = json.loads(args.cpu_freeze.read_text(encoding="utf-8"))
    confirmation = json.loads(args.confirmation_freeze.read_text(encoding="utf-8"))
    c4 = cpu["c4"]
    lengths = {int(k): int(v) for k, v in split["profile_rows"].items()}
    train_ids = [int(v) for v in split["train_profile_ids"]]
    validation_ids = [int(v) for v in split["validation_profile_ids"]]
    test_ids = [int(v) for v in split["test_profile_ids"]]
    chunks = grouped_chunks(train_ids, 5)
    evaluation_chunks = chunks[1:]

    if confirmation["input_contract"]["primary_inputs"] != CHANNELS:
        raise RuntimeError("confirmation primary input order drift")
    if set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
        raise RuntimeError("split overlap")

    audit = {
        "audit_id": "PRISM_V2_2_PMSM_SW_NATIVE_SUPPORT_AUDIT_20260901_V1",
        "status": "PASS",
        "target_values_read": False,
        "test_target_values_read": False,
        "candidate_universe_changed": False,
        "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        "grouped_train_selection_chunks": chunks,
        "heads": {},
    }
    warnings = []

    for head in heads:
        head_record = {
            "h_steps": int(head.h_steps),
            "w_steps": int(head.w_steps),
            "w0_steps": int(head.w0_steps),
            "primary": bool(head.primary),
            "channels": {},
        }
        for channel in CHANNELS:
            candidates = profiles_for(head, channel, c4)
            channel_record = []
            for delta, history in candidates:
                train_by_profile = {pid: support_rows(lengths[pid], head, history) for pid in train_ids}
                validation_by_profile = {pid: support_rows(lengths[pid], head, history) for pid in validation_ids}
                fold_rows = [sum(train_by_profile[pid] for pid in fold_ids) for fold_ids in evaluation_chunks]
                positive_folds = sum(value > 0 for value in fold_rows)
                record = {
                    "delta_steps": int(delta),
                    "history_steps": int(history),
                    "history_seconds": float(history * head.cadence_seconds),
                    "train_support_rows": int(sum(train_by_profile.values())),
                    "validation_support_rows": int(sum(validation_by_profile.values())),
                    "train_profiles_with_support": int(sum(v > 0 for v in train_by_profile.values())),
                    "validation_profiles_with_support": int(sum(v > 0 for v in validation_by_profile.values())),
                    "selection_evaluation_fold_support_rows": [int(v) for v in fold_rows],
                    "selection_positive_evaluation_folds": int(positive_folds),
                    "usable_under_frozen_minimum_3_fold_rule": bool(positive_folds >= 3),
                }
                if positive_folds < 3:
                    warnings.append({"head": head.head_id, "channel": channel, "delta": delta, "history": history, "positive_folds": positive_folds})
                channel_record.append(record)
            head_record["channels"][channel] = channel_record
        audit["heads"][head.head_id] = head_record

    audit["support_warnings"] = warnings
    audit["all_candidates_have_at_least_3_positive_selection_folds"] = not warnings
    # Lack of native support is not a protocol failure and does not remove a
    # frozen candidate. It must be represented as unsupported/NaN downstream.
    audit["downstream_rule"] = "APPLY_CANDIDATE_NATIVE_MASK; NEVER_BACKFILL_ACROSS_PROFILE_START; RETAIN_UNSUPPORTED_CANDIDATE_AS_UNAVAILABLE"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "all_candidates_have_at_least_3_positive_selection_folds": audit["all_candidates_have_at_least_3_positive_selection_folds"],
        "warning_count": len(warnings),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
