from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v2_config import load_frozen_config, sha256_file


PROTOCOL_ID = "PRISM_V2_1_SRU_STAGEWISE_ROUTED_PROPOSED_V1"
PROTOCOL_STATUS = "PROPOSED_BEFORE_IMPLEMENTATION_AND_TEST_ACCESS"
ACTIVE_DATASET = "sru"
ACTIVE_HEADS = frozenset({"SRU_H2S__H5__W1", "SRU_SO2__H5__W1"})
OUTPUT_DIRECTORY = "results_prism_v2_1_sru"
BASELINE_AMENDMENT_ID = "PRISM_V2_1_SRU_BASELINE_REPLAY_AMENDMENT_20260806"
BASELINE_AMENDMENT_STATUS = "FROZEN_USER_AUTHORIZED_BEFORE_BASELINE_REPLAY"
BASELINE_AMENDMENT_JSON = "PRISM_V2_1_SRU_BASELINE_REPLAY_AMENDMENT_20260806.json"


@dataclass(frozen=True)
class V21Paths:
    project: Path
    shared: Path
    output: Path

    @property
    def plan(self) -> Path:
        return self.project / "PRISM_V2_1_SRU_STAGEWISE_ROUTED"

    @property
    def config_path(self) -> Path:
        return self.plan / "PRISM_V2_1_SRU_CONFIG_PROPOSED.json"

    @property
    def final_freeze_path(self) -> Path:
        return self.output / "FREEZE" / "V21_SRU_FINAL_FREEZE_MANIFEST.json"

    @property
    def baseline_amendment_path(self) -> Path:
        return self.plan / BASELINE_AMENDMENT_JSON

    @property
    def baseline_replay_root(self) -> Path:
        return self.output / "BASELINES" / "REPLAY"


def load_baseline_replay_amendment(project: Path) -> dict[str, Any]:
    path = project / "PRISM_V2_1_SRU_STAGEWISE_ROUTED" / BASELINE_AMENDMENT_JSON
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("amendment_id") != BASELINE_AMENDMENT_ID:
        raise RuntimeError("PRISM v2.1 baseline replay amendment_id mismatch")
    if amendment.get("status") != BASELINE_AMENDMENT_STATUS:
        raise RuntimeError("PRISM v2.1 baseline replay amendment is not frozen")
    if amendment.get("historical_baseline_parquet") != "NOT_AVAILABLE_NOT_SEARCHED_NOT_REQUIRED":
        raise RuntimeError("historical baseline parquet must not be searched or required")
    if amendment.get("active_datasets") != [ACTIVE_DATASET]:
        raise RuntimeError("baseline replay is restricted to SRU")
    if frozenset(amendment.get("active_heads", ())) != ACTIVE_HEADS:
        raise RuntimeError("baseline replay active SRU heads mismatch")
    if amendment.get("splits_materialized_by_b0") != ["validation", "test"]:
        raise RuntimeError("baseline replay must materialize validation and test only")
    if any(
        amendment.get(key) is not False
        for key in ("ood_enabled", "other_datasets_enabled", "private_cz_enabled")
    ):
        raise RuntimeError("baseline replay scope expansion is forbidden")
    if amendment.get("external_baseline_root_allowed") is not False:
        raise RuntimeError("external baseline roots are forbidden by the amendment")
    if amendment.get("historical_prediction_search_allowed") is not False:
        raise RuntimeError("historical prediction search is forbidden by the amendment")
    baseline_test = amendment.get("baseline_test_access", {})
    if (
        baseline_test.get("allowed_in_b0") is not True
        or baseline_test.get("metrics_computed_in_b0") is not False
        or baseline_test.get("selection_exposure") is not False
        or baseline_test.get("separate_subprocess_required") is not True
    ):
        raise RuntimeError("baseline-only test access contract mismatch")
    return amendment


def load_v21_config(project: Path) -> dict[str, Any]:
    path = project / "PRISM_V2_1_SRU_STAGEWISE_ROUTED" / "PRISM_V2_1_SRU_CONFIG_PROPOSED.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("PRISM v2.1 protocol_id mismatch")
    if config.get("status") != PROTOCOL_STATUS:
        raise RuntimeError("PRISM v2.1 configuration status mismatch")
    if config.get("active_datasets") != [ACTIVE_DATASET]:
        raise RuntimeError("PRISM v2.1 is restricted to SRU")
    if frozenset(config.get("active_heads", ())) != ACTIVE_HEADS:
        raise RuntimeError("PRISM v2.1 active SRU heads mismatch")
    if config.get("output_root") != OUTPUT_DIRECTORY:
        raise RuntimeError("PRISM v2.1 output namespace mismatch")
    if config.get("write_shared_data") is not False:
        raise RuntimeError("PRISM v2.1 must not write shared C1 data")
    if config.get("preserve_all_c1_data_bases") is not True:
        raise RuntimeError("PRISM v2.1 must preserve all C1 data bases")
    if 0.0 not in [float(value) for value in config["W"]["soft_overlap_mu"]]:
        raise RuntimeError("W soft-overlap grid must contain exact zero")
    if 0.0 not in [float(value) for value in config["A"]["soft_overlap_mu"]]:
        raise RuntimeError("A soft-overlap grid must contain exact zero")
    if config["J"].get("allow_k_zero") is not False or config["J"].get("allow_ar_only") is not False:
        raise RuntimeError("Joint-KWA must retain a mandatory input path")
    return config


def load_v21_and_v2_config(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the v2.1 routing contract and inherited numerically frozen v2 grids."""
    return load_v21_config(project), load_frozen_config(project)


def require_test_freeze(paths: V21Paths) -> dict[str, Any]:
    if not paths.final_freeze_path.is_file():
        raise RuntimeError("SRU test access requires V21_SRU_FINAL_FREEZE_MANIFEST.json")
    manifest = json.loads(paths.final_freeze_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ASSEMBLY_FROZEN":
        raise RuntimeError("v2.1 final freeze manifest is not ASSEMBLY_FROZEN")
    if manifest.get("test_accessed") is not False:
        raise RuntimeError("v2.1 SRU test was already accessed")
    expected = manifest.get("config_sha256")
    observed = sha256_file(paths.config_path)
    if expected != observed:
        raise RuntimeError("v2.1 config changed after final freeze")
    amendment_expected = manifest.get("baseline_replay_amendment_sha256")
    if amendment_expected != sha256_file(paths.baseline_amendment_path):
        raise RuntimeError("baseline replay amendment changed after final freeze")
    replay_path = paths.output / "BASELINES" / "BASELINE_REPLAY_MANIFEST.json"
    if manifest.get("baseline_replay_manifest_sha256") != sha256_file(replay_path):
        raise RuntimeError("baseline replay manifest changed after final freeze")
    if manifest.get("baseline_test_metrics_exposed_to_selection") is not False:
        raise RuntimeError("baseline test metrics reached v2.1 selection")
    if manifest.get("v21_candidate_test_accessed") is not False:
        raise RuntimeError("v2.1 candidate test was accessed before E7")
    return manifest
