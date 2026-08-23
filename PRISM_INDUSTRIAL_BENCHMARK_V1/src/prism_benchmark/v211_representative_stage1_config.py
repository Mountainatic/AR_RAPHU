from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v2_config import sha256_file
from .v211_metro_config import load_metro_config
from .v211_support import SUPPORT_CONTRACT


PROTOCOL = "representative_stage1"
PROTOCOL_ID = "REPRESENTATIVE_HORIZON_STAGE1_TEP_SRU_CPU_DEVELOPMENT_V1"
EVIDENCE_CLASS = "POST_HOC_HORIZON_EXTENSION_WITH_PREDECLARED_SETTINGS"
BASE_COMMIT = "70537060b4d6713db7a557fc37e6771fd07b6a6b"
EXECUTION_BRANCH = (
    "prism-v2-1-1-representative-horizon-stage1-six-datasets-20260818"
)
CONFIG_RELATIVE_PATH = Path(
    "configs/representative_horizon_stage1_tep_sru_cpu.json"
)
PRIMARY_TASKS = frozenset(
    {"TEP_G_REP_H1", "SRU_H2S_REP_H1", "SRU_SO2_REP_H1"}
)
ACTIVE_DATASETS = ("tep", "sru")


@dataclass(frozen=True)
class RepresentativeStage1Paths:
    project: Path
    shared: Path
    run_root: Path

    @property
    def output(self) -> Path:
        return self.run_root / "results"

    @property
    def logs(self) -> Path:
        return self.run_root / "logs"

    @property
    def partial_evidence(self) -> Path:
        return self.run_root / "partial_development_evidence"

    @property
    def config_path(self) -> Path:
        return self.project / CONFIG_RELATIVE_PATH


def _require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(f"representative Stage-1 config mismatch: {label}")


def _require_bound_file(
    project: Path, record: dict[str, Any], label: str
) -> None:
    path = project / str(record["path"])
    _require(path.is_file(), f"{label}.path")
    _require(sha256_file(path) == str(record["sha256"]), f"{label}.sha256")


def load_representative_stage1_descriptor(project: Path) -> dict[str, Any]:
    path = project / CONFIG_RELATIVE_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("protocol_id") == PROTOCOL_ID, "protocol_id")
    _require(value.get("evidence_class") == EVIDENCE_CLASS, "evidence_class")
    _require(value.get("base_commit") == BASE_COMMIT, "base_commit")
    _require(value.get("execution_branch") == EXECUTION_BRANCH, "execution_branch")
    _require(value.get("support_contract") == SUPPORT_CONTRACT, "support_contract")
    _require(tuple(value.get("active_datasets", ())) == ACTIVE_DATASETS, "active_datasets")
    _require(frozenset(value.get("primary_tasks", ())) == PRIMARY_TASKS, "primary_tasks")
    _require(int(value.get("expected_input_views", 0)) == 3, "expected_input_views")
    _require(int(value.get("expected_dynamic_views", 0)) == 4, "expected_dynamic_views")
    _require(int(value.get("expected_k_channel_jobs", 0)) == 57, "expected_k_channel_jobs")
    _require(value.get("development_only") is True, "development_only")
    _require(value.get("neural_in_scope") is False, "neural_in_scope")
    _require(value.get("global_freeze_created") is False, "global_freeze_created")
    _require(value.get("test_access_before_global_freeze") is False, "test_access")
    _require(value.get("ood_access_before_global_freeze") is False, "ood_access")
    for label in (
        "c1_config",
        "algorithm_config",
        "joint_config",
        "sru_patch",
        "cpu_model_freeze",
    ):
        _require_bound_file(project, value[label], label)
    resource = value["resource"]
    _require(int(resource["outer_workers"]) >= 1, "resource.outer_workers")
    _require(int(resource["k_inner_workers"]) >= 1, "resource.k_inner_workers")
    _require(int(resource["stage_inner_workers"]) >= 1, "resource.stage_inner_workers")
    _require(int(resource["blas_threads"]) == 1, "resource.blas_threads")
    _require(
        0.0 < float(resource["memory_safety_fraction"]) < 1.0,
        "resource.memory_safety_fraction",
    )
    value["config_sha256"] = sha256_file(path)
    return value


def load_representative_stage1_algorithm_config(project: Path) -> dict[str, Any]:
    descriptor = load_representative_stage1_descriptor(project)
    inherited = load_metro_config(project)
    merged = dict(inherited)
    merged.update(
        {
            "protocol_id": PROTOCOL_ID,
            "evidence_class": EVIDENCE_CLASS,
            "source_commit": BASE_COMMIT,
            "execution_branch": EXECUTION_BRANCH,
            "support_contract": SUPPORT_CONTRACT,
            "active_datasets": list(ACTIVE_DATASETS),
            "primary_tasks": sorted(PRIMARY_TASKS),
            "representative_stage1_config_sha256": descriptor["config_sha256"],
        }
    )
    merged["resource"] = {
        **inherited["resource"],
        "workers": int(descriptor["resource"]["outer_workers"]),
        "blas_threads": 1,
        "prediction_chunk_rows": int(
            descriptor["resource"]["prediction_chunk_rows"]
        ),
    }
    return merged
