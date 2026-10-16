from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage0 import write_json
from .v2_config import V2Paths, load_frozen_config, sha256_file


def run_v0_audit(paths: V2Paths, c6_summary: Path, *, full_registry_check: bool = True) -> dict[str, Any]:
    config = load_frozen_config(paths.project)
    inheritance = json.loads(paths.inheritance_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def record(name: str, expected: Any, observed: Any) -> None:
        checks.append({"name": name, "expected": expected, "observed": observed, "pass": observed == expected})

    validation = json.loads((paths.shared / "C1_VALIDATION.json").read_text(encoding="utf-8"))
    record("C1_VALIDATION_STATUS", "PASS", validation.get("status"))
    protocol = json.loads((paths.shared / "PROTOCOL.json").read_text(encoding="utf-8"))
    for key, expected in inheritance["contracts"].items():
        observed_key = {
            "target_index": "target_index_contract", "time_realization": "time_realization_contract",
            "purge": "purge_contract", "debutanizer_availability": "debutanizer_availability_contract",
            "tep_availability": "tep_availability_contract",
        }.get(key, key)
        record(f"CONTRACT_{key}", expected, protocol.get(observed_key))
    for dataset, expected in inheritance["split_registry_sha256"].items():
        path = paths.project / "dataset_registry" / dataset / "SPLIT_REGISTRY.json"
        record(f"SPLIT_REGISTRY_{dataset}", expected, sha256_file(path) if path.is_file() else "MISSING")
    registry = json.loads((paths.shared / "SAMPLE_ID_REGISTRY.json").read_text(encoding="utf-8"))
    file_checks = []
    if full_registry_check:
        for entry in registry["files"]:
            path = paths.shared / entry["path"]
            observed = sha256_file(path) if path.is_file() else "MISSING"
            file_checks.append({"path": entry["path"], "expected": entry["sha256"], "observed": observed, "pass": observed == entry["sha256"]})
    record("C6_V2_SUMMARY", inheritance["source_c6_v2_summary"]["sha256"], sha256_file(c6_summary) if c6_summary.is_file() else "MISSING")
    record("NUMERICAL_PROTOCOL_ID", "PRISM_V2_MODULAR_ASSEMBLY_NUMERICAL_FREEZE_V1", config["protocol_id"])
    passed = all(item["pass"] for item in checks) and all(item["pass"] for item in file_checks)
    result = {
        "status": "PASS" if passed else "STOP_INHERITANCE_MISMATCH",
        "stage": "V0_INHERITANCE_AUDIT", "checks": checks, "registered_file_checks": file_checks,
        "registered_file_count": len(file_checks), "test_accessed": False,
        "config_sha256": sha256_file(paths.config_path), "inheritance_sha256": sha256_file(paths.inheritance_path),
    }
    destination = paths.output / "FREEZE"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "DATA_INHERITANCE_AUDIT.json", result)
    (destination / "ASSEMBLY_CONFIG.json").write_bytes(paths.config_path.read_bytes())
    if not passed:
        failures = [item for item in [*checks, *file_checks] if not item["pass"]]
        raise RuntimeError(f"V0 inheritance audit failed: {failures[:5]}")
    return result

