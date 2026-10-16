from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v2_config import sha256_file
from .v211_metro_config import load_metro_config
from .v211_support import SUPPORT_CONTRACT


PUBLIC_ALL_PROTOCOL = "public_all"
PROTOCOL_ID = "PRISM_V2_1_1_NATIVE_SUPPORT_PUBLIC_ALL_RERUN_V1"
EVIDENCE_CLASS = (
    "PROSPECTIVE_NATIVE_SUPPORT_PROTOCOL_RERUN_WITH_PRIOR_HISTORICAL_CONTEXT"
)
SOURCE_BRANCH = "prism-v2-1-1-metro-p60-joint-stability-final"
SOURCE_COMMIT = "e47542a319640bc045ca0d31ae9b40763182dde8"
EXECUTION_BRANCH = "prism-v2-1-1-native-support-public-all-20260815"
CONFIG_RELATIVE_PATH = Path("configs/prism_v211_native_public_all.json")
PRIMARY_TASKS = frozenset(
    {
        "TEP_G12",
        "DEB_C4",
        "SRU_H2S",
        "SRU_SO2",
        "PMSM_PM5",
        "METRO_P60",
        "METRO_OIL20",
    }
)
ACTIVE_DATASETS = ("tep", "debutanizer", "sru", "pmsm", "metropt")


@dataclass(frozen=True)
class PublicAllPaths:
    project: Path
    shared: Path
    run_root: Path

    @property
    def output(self) -> Path:
        return self.run_root / "results"

    @property
    def freeze(self) -> Path:
        return self.run_root / "freeze"

    @property
    def final(self) -> Path:
        return self.run_root / "final"

    @property
    def logs(self) -> Path:
        return self.run_root / "logs"

    @property
    def return_root(self) -> Path:
        return self.run_root / "return"

    @property
    def config_path(self) -> Path:
        return self.project / CONFIG_RELATIVE_PATH

    @property
    def development_freeze_path(self) -> Path:
        return self.freeze / "PUBLIC_ALL_V211_NATIVE_DEVELOPMENT_FREEZE.json"

    @property
    def leaderboard_support_path(self) -> Path:
        return self.freeze / "TASK_LEADERBOARD_COMMON_SUPPORT.json"

    @property
    def test_access_audit_path(self) -> Path:
        return self.final / "PUBLIC_ALL_TEST_OOD_ACCESS_AUDIT.json"


def _require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(f"public-all frozen config mismatch: {label}")


def _require_bound_file(project: Path, record: dict[str, Any], label: str) -> None:
    path = project / str(record["path"])
    _require(path.is_file(), f"{label}.path")
    _require(sha256_file(path) == str(record["sha256"]), f"{label}.sha256")


def load_public_all_descriptor(project: Path) -> dict[str, Any]:
    path = project / CONFIG_RELATIVE_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("protocol_id") == PROTOCOL_ID, "protocol_id")
    _require(value.get("evidence_class") == EVIDENCE_CLASS, "evidence_class")
    _require(value.get("source_branch") == SOURCE_BRANCH, "source_branch")
    _require(value.get("source_commit") == SOURCE_COMMIT, "source_commit")
    _require(value.get("execution_branch") == EXECUTION_BRANCH, "execution_branch")
    _require(value.get("support_contract") == SUPPORT_CONTRACT, "support_contract")
    _require(tuple(value.get("active_datasets", ())) == ACTIVE_DATASETS, "active_datasets")
    _require(frozenset(value.get("primary_tasks", ())) == PRIMARY_TASKS, "primary_tasks")
    _require(value.get("full_multihorizon_scale_sweep") is False, "scale_sweep")
    _require(value.get("gpu_baselines") == "OUT_OF_SCOPE", "gpu_baselines")
    _require(value.get("test_access_before_global_freeze") is False, "test_access")
    _require(value.get("ood_access_before_global_freeze") is False, "ood_access")
    for label in ("algorithm_config", "joint_config", "sru_patch", "cpu_model_freeze"):
        _require_bound_file(project, value[label], label)
    resource = value["resource"]
    _require(int(resource["outer_workers"]) >= 1, "resource.outer_workers")
    _require(int(resource["k_inner_workers"]) >= 1, "resource.k_inner_workers")
    _require(int(resource["stage_inner_workers"]) >= 1, "resource.stage_inner_workers")
    _require(int(resource["blas_threads"]) == 1, "resource.blas_threads")
    _require(0.0 < float(resource["memory_safety_fraction"]) < 1.0, "resource.memory_safety_fraction")
    value["config_sha256"] = sha256_file(path)
    return value


def load_public_all_algorithm_config(project: Path) -> dict[str, Any]:
    descriptor = load_public_all_descriptor(project)
    inherited = load_metro_config(project)
    merged = dict(inherited)
    merged.update(
        {
            "protocol_id": PROTOCOL_ID,
            "evidence_class": EVIDENCE_CLASS,
            "source_branch": SOURCE_BRANCH,
            "source_commit": SOURCE_COMMIT,
            "execution_branch": EXECUTION_BRANCH,
            "support_contract": SUPPORT_CONTRACT,
            "active_datasets": list(ACTIVE_DATASETS),
            "primary_tasks": sorted(PRIMARY_TASKS),
            "full_multihorizon_scale_sweep": False,
            "gpu_baselines": "OUT_OF_SCOPE",
            "public_all_config_sha256": descriptor["config_sha256"],
        }
    )
    merged["resource"] = {
        **inherited["resource"],
        "workers": int(descriptor["resource"]["outer_workers"]),
        "blas_threads": 1,
        "prediction_chunk_rows": int(descriptor["resource"]["prediction_chunk_rows"]),
    }
    return merged
