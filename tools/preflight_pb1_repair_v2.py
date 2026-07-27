#!/usr/bin/env python3
"""Machine-check PB1 Repair V2 without loading any official test records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("pwh", "whpn", "cascaded_tanks", "silverbox")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_preflight_stage(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^stage:\s*["\']?([^"\'\s]+)', text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing stage in {path}")
    return match.group(1)


def _scan_test_access(results_root: Path) -> tuple[int, list[str]]:
    maximum = 0
    violations: list[str] = []
    if not results_root.exists():
        return maximum, violations
    for path in results_root.rglob("*.json"):
        try:
            payload = _read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "official_test_access_count":
                        count = int(child)
                        maximum = max(maximum, count)
                        if count != 0:
                            violations.append(str(path.relative_to(ROOT)))
                    elif key == "official_test_rows_loaded" and int(child) != 0:
                        violations.append(str(path.relative_to(ROOT)))
                    else:
                        stack.append(child)
            elif isinstance(value, list):
                stack.extend(value)
    return maximum, sorted(set(violations))


def validate(config_dir: Path, require_test_access_zero: bool) -> dict:
    stage = _read_preflight_stage(ROOT / "PB1_REPAIR_PREFLIGHT_V2.yaml")
    patch = _read_json(config_dir / "PB1_PROTOCOL_PATCH_V2.json")
    failures: list[str] = []
    configs: dict[str, dict] = {}
    for dataset in DATASETS:
        path = config_dir / f"pb1_{dataset}.yaml"
        configs[dataset] = _read_json(path)
        if configs[dataset].get("schema_version") != 7:
            failures.append(f"{dataset}:SCHEMA_NOT_7")
        if configs[dataset].get("repair_v2", {}).get("exact_zero_penalty") is not True:
            failures.append(f"{dataset}:EXACT_ZERO_NOT_ENABLED")
        if configs[dataset]["solver"].get("kkt_relative_residual") != 1.0e-8:
            failures.append(f"{dataset}:KKT_THRESHOLD_CHANGED")
        if configs[dataset]["solver"].get(
            "numerical_jitter_is_scientific_ridge"
        ) is not False:
            failures.append(f"{dataset}:JITTER_RIDGE_CONFLATED")
    tanks = configs["cascaded_tanks"]
    if tanks["dataset"]["development_split"].get("train_rows") != [0, 700]:
        failures.append("cascaded_tanks:TRAIN_SPLIT_NOT_700")
    if tanks["dataset"]["development_split"].get("validation_rows") != [
        700,
        "end",
    ]:
        failures.append("cascaded_tanks:VALIDATION_SPLIT_NOT_REST")
    silverbox = configs["silverbox"]
    if silverbox["dataset"]["development_split"].get("train_fraction") != [
        0.0,
        0.5,
    ]:
        failures.append("silverbox:TRAIN_SPLIT_NOT_HALF")
    if silverbox["dataset"]["development_split"].get(
        "validation_fraction"
    ) != [0.5, 1.0]:
        failures.append("silverbox:VALIDATION_SPLIT_NOT_HALF")
    if patch.get("confirmation_allowed") is not False:
        failures.append("PATCH_ALLOWS_CONFIRMATION")
    maximum_access, access_violations = _scan_test_access(
        ROOT / "results" / "public_benchmarks"
    )
    if require_test_access_zero and (maximum_access != 0 or access_violations):
        failures.append("OFFICIAL_TEST_ACCESS_NONZERO")
    return {
        "schema_version": "PB1_REPAIR_PREFLIGHT_STATUS_V2",
        "stage": stage,
        "datasets": list(DATASETS),
        "protocol_patch_freeze_status": patch.get("freeze_status"),
        "official_test_access_count": maximum_access,
        "official_test_access_violations": access_violations,
        "gates": {
            "SCHEMA_7_ALL_DATASETS": all(
                config.get("schema_version") == 7 for config in configs.values()
            ),
            "TANKS_SPLIT_700_REST": (
                tanks["dataset"]["development_split"].get("train_rows")
                == [0, 700]
            ),
            "SILVERBOX_SPLIT_HALF_HALF": (
                silverbox["dataset"]["development_split"].get("train_fraction")
                == [0.0, 0.5]
            ),
            "OFFICIAL_TEST_ACCESS_COUNT_ZERO": maximum_access == 0,
            "CONFIRMATION_FORBIDDEN": patch.get("confirmation_allowed") is False,
        },
        "failures": failures,
        "status": "COMPLETED" if not failures else "FAILED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "configs" / "public_benchmarks",
    )
    parser.add_argument("--require-test-access-zero", action="store_true")
    arguments = parser.parse_args()
    payload = validate(arguments.config_dir, arguments.require_test_access_zero)
    output = (
        ROOT
        / "results"
        / "public_benchmarks"
        / "pb1_repair_v2"
        / "PB1_REPAIR_PREFLIGHT_STATUS.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "COMPLETED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
