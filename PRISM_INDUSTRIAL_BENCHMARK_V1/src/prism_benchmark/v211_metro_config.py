from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v2_config import sha256_file


PROTOCOL_ID = "PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_V1"
EVIDENCE_CLASS = "RETROSPECTIVE_TRANSFER_AND_DEGRADATION_AUDIT"
OUTPUT_DIRECTORY = "results_prism_v2_1_1_metro_p60_w_audit"
PLAN_DIRECTORY = "PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_PACKAGE"
CONFIG_NAME = "PRISM_V2_1_1_METRO_P60_CONFIG_FROZEN_PROPOSED.json"
CONFIG_SHA256 = "764cd2b7d8a94a3a2e0391629f75afce79df1ffcbebad0fdce8b7cd514106bba"
ACTIVE_DATASET = "metropt"
ACTIVE_HEAD = "METRO_P60__H6__W1"
RECOMMENDED_BRANCH = "prism-v2-1-1-metro-p60-w-audit"
SOURCE_COMMIT = "5b3a971c8eea5127e01e607208ca5d3ea69517a8"
DEVELOPMENT_FREEZE_NAME = "METRO_P60_V211_DEVELOPMENT_FREEZE.json"
DEVELOPMENT_DECISION_NAME = "METRO_P60_V211_DEVELOPMENT_DECISION.json"
TEST_ACCESS_AUDIT_NAME = "METRO_P60_V211_TEST_OOD_ACCESS_AUDIT.json"


@dataclass(frozen=True)
class MetroV211Paths:
    project: Path
    shared: Path
    output: Path

    @property
    def plan(self) -> Path:
        return self.project / PLAN_DIRECTORY

    @property
    def config_path(self) -> Path:
        return self.plan / CONFIG_NAME

    @property
    def theory_path(self) -> Path:
        return (
            self.plan
            / "reference"
            / "PRISM_Theory_v2_1_1_Implementation_Safe_Stagewise_Routed_Modular_Assembly_Theory_Only.md"
        )

    @property
    def historical_reference_path(self) -> Path:
        return self.plan / "reference" / "METRO_P60_V1_3_HISTORICAL_AGGREGATES.json"

    @property
    def development_freeze_path(self) -> Path:
        return self.output / "FREEZE" / DEVELOPMENT_FREEZE_NAME

    @property
    def development_decision_path(self) -> Path:
        return self.output / "FREEZE" / DEVELOPMENT_DECISION_NAME

    @property
    def test_access_audit_path(self) -> Path:
        return self.output / "FINAL" / TEST_ACCESS_AUDIT_NAME


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"Metro-P60 frozen config mismatch: {label}")


def load_metro_config(project: Path) -> dict[str, Any]:
    path = project / PLAN_DIRECTORY / CONFIG_NAME
    if sha256_file(path) != CONFIG_SHA256:
        raise RuntimeError("Metro-P60 frozen config SHA256 mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    fixed = {
        "protocol_id": PROTOCOL_ID,
        "evidence_class": EVIDENCE_CLASS,
        "inherits_theory": "PRISM_Theory_v2_1_1_Implementation_Safe_Stagewise_Routed_Modular_Assembly_Theory_Only.md",
        "inherits_implementation_contract": "PRISM_V2_1_1_SRU_IMPLEMENTATION_CORRECTION",
        "output_root": OUTPUT_DIRECTORY,
        "recommended_branch": RECOMMENDED_BRANCH,
        "active_datasets": [ACTIVE_DATASET],
        "active_heads": [ACTIVE_HEAD],
        "write_shared_data": False,
        "rebuild_or_resplit_c1": False,
        "algorithm_change_allowed": False,
        "historical_test_ood_known": True,
        "historical_aggregates_selection_use_forbidden": True,
        "test_access_before_freeze": False,
        "ood_access_before_freeze": False,
        "dtype": "float64",
    }
    for key, expected in fixed.items():
        _require_equal(config.get(key), expected, key)
    _require_equal(
        config.get("resource"),
        {
            "workers": 2,
            "blas_threads": 1,
            "prediction_chunk_rows": 50000,
            "release_candidate_matrices_after_use": True,
            "cache_all_candidate_design_matrices": False,
        },
        "resource",
    )
    _require_equal(
        config.get("row_caps"),
        {
            "single_channel_k_fit": 100000,
            "validation_selection_per_fold": 50000,
            "joint_physical_fit": 250000,
            "wiener_fit": 250000,
            "state_fit": 250000,
            "joint_predictive_fit": 250000,
            "final_prediction": "ALL_IMMUTABLE_ROWS",
            "subsample_order": "ASCENDING_SHA256_OF_BASE_ORIGIN_ID",
            "nested_candidates_share_rows": True,
        },
        "row_caps",
    )
    _require_equal(
        config["W"].get("candidates"),
        [
            "IDENTITY_CORRECTION",
            "MONOTONE_I_SPLINE_CORRECTION",
            "NATURAL_CUBIC_CORRECTION",
        ],
        "W.candidates",
    )
    _require_equal(config["W"].get("minimum_usable_folds"), 3, "W.minimum_usable_folds")
    _require_equal(config["W"].get("raw_absolute_variance_gate"), False, "W.raw_absolute_variance_gate")
    _require_equal(config["W"].get("identity_prediction_equivalence_required"), True, "W.identity_prediction_equivalence_required")
    _require_equal(config["W"].get("identity_residual_equivalence_required"), True, "W.identity_residual_equivalence_required")
    _require_equal(
        config["J"].get("candidates"),
        ["J_K", "J_KW", "J_KA", "J_KWA"],
        "J.candidates",
    )
    _require_equal(config["J"].get("allow_k_zero"), False, "J.allow_k_zero")
    _require_equal(config["J"].get("allow_ar_only"), False, "J.allow_ar_only")
    _require_equal(
        config["J"].get("jointly_fit_w_basis_coefficients"),
        True,
        "J.jointly_fit_w_basis_coefficients",
    )
    _require_equal(config["J"].get("share_pf_input_path_gate"), True, "J.share_pf_input_path_gate")
    _require_equal(
        config["post_freeze_materialized_candidates"],
        {
            "physics_first": ["KC", "KCW", "KCA", "KCWA", "PF_SELECTED"],
            "joint": ["J_K", "J_KW", "J_KA", "J_KWA", "J_SELECTED"],
        },
        "post_freeze_materialized_candidates",
    )
    _require_equal(config["statistics"].get("paired_moving_block_bootstrap_replicates"), 500, "statistics.bootstrap")
    _require_equal(config["statistics"].get("holm_alpha"), 0.05, "statistics.holm_alpha")
    return config


def git_value(project: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def require_metro_test_freeze(paths: MetroV211Paths) -> dict[str, Any]:
    if not paths.development_freeze_path.is_file():
        raise RuntimeError("Metro-P60 test/OOD access requires development freeze")
    if paths.test_access_audit_path.exists():
        raise RuntimeError("Metro-P60 test/OOD was already accessed")
    manifest = json.loads(paths.development_freeze_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "METRO_P60_V2_1_1_DEVELOPMENT_FROZEN":
        raise RuntimeError("Metro-P60 development freeze is not valid")
    if manifest.get("test_accessed") is not False or manifest.get("ood_accessed") is not False:
        raise RuntimeError("Metro-P60 freeze records early test/OOD access")
    if manifest.get("config_sha256") != sha256_file(paths.config_path):
        raise RuntimeError("Metro-P60 config changed after freeze")
    if manifest.get("development_decision_sha256") != sha256_file(
        paths.development_decision_path
    ):
        raise RuntimeError("Metro-P60 development decision changed after freeze")
    if manifest.get("code_commit") != git_value(paths.project, "rev-parse", "HEAD"):
        raise RuntimeError("Metro-P60 code commit changed after freeze")
    if git_value(paths.project, "status", "--porcelain=v1"):
        raise RuntimeError("Metro-P60 worktree is dirty after freeze")
    load_metro_config(paths.project)
    return manifest
