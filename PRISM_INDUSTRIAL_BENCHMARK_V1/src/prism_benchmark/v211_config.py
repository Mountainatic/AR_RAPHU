from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v2_config import load_frozen_config, sha256_file
from .v21_config import ACTIVE_HEADS, load_v21_config


PROTOCOL_ID = "PRISM_V2_1_1_SRU_IMPLEMENTATION_CORRECTION_V1"
OUTPUT_DIRECTORY = "results_prism_v2_1_1_sru"
PLAN_DIRECTORY = "PRISM_V2_1_1_SRU_IMPLEMENTATION_CORRECTION"
CONFIG_NAME = "PRISM_V2_1_1_SRU_CONFIG_PATCH.json"
FINAL_FREEZE_NAME = "V211_SRU_FINAL_FREEZE_MANIFEST.json"
DEVELOPMENT_DECISION_NAME = "V211_DEVELOPMENT_DECISION.json"
SRU_PROTOCOL = "sru"
METRO_P60_PROTOCOL = "metro_p60"


PUBLIC_ALL_PROTOCOL = "public_all"
REPRESENTATIVE_STAGE1_PROTOCOL = "representative_stage1"
TEP_CPU_HISTORY_EXTENSION_PROTOCOL = "tep_cpu_history_l256"
TEP_CPU_NOWCAST_HISTORY_EXTENSION_PROTOCOL = "tep_cpu_nowcast_history_l256"
@dataclass(frozen=True)
class V211Paths:
    project: Path
    shared: Path
    output: Path
    baseline_source: Path | None = None

    @property
    def plan(self) -> Path:
        return self.project / PLAN_DIRECTORY

    @property
    def config_path(self) -> Path:
        return self.plan / CONFIG_NAME

    @property
    def inherited_plan(self) -> Path:
        return self.project / "PRISM_V2_1_SRU_STAGEWISE_ROUTED"

    @property
    def baseline_amendment_path(self) -> Path:
        return (
            self.inherited_plan
            / "PRISM_V2_1_SRU_BASELINE_REPLAY_AMENDMENT_20260806.json"
        )

    @property
    def baseline_replay_root(self) -> Path:
        return self.output / "BASELINES" / "REPLAY"

    @property
    def resolved_baseline_source(self) -> Path:
        return (
            self.baseline_source
            if self.baseline_source is not None
            else self.project / "results_prism_v2_1_sru"
        ).resolve()

    @property
    def development_decision_path(self) -> Path:
        return self.output / "FREEZE" / DEVELOPMENT_DECISION_NAME

    @property
    def final_freeze_path(self) -> Path:
        return self.output / "FREEZE" / FINAL_FREEZE_NAME


def load_v211_config(project: Path) -> dict[str, Any]:
    path = project / PLAN_DIRECTORY / CONFIG_NAME
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("PRISM v2.1.1 protocol_id mismatch")
    if config.get("inherits") != "PRISM_V2_1_SRU_STAGEWISE_ROUTED_PROPOSED_V1":
        raise RuntimeError("PRISM v2.1.1 inheritance mismatch")
    if config.get("output_root") != OUTPUT_DIRECTORY:
        raise RuntimeError("PRISM v2.1.1 output namespace mismatch")
    if config.get("active_datasets") != ["sru"]:
        raise RuntimeError("PRISM v2.1.1 is restricted to SRU")
    if frozenset(config.get("active_heads", ())) != ACTIVE_HEADS:
        raise RuntimeError("PRISM v2.1.1 active SRU heads mismatch")
    if config.get("reuse_frozen_baseline_predictions") is not True:
        raise RuntimeError("PRISM v2.1.1 must reuse frozen baseline predictions")
    if config.get("retune_baselines") is not False:
        raise RuntimeError("PRISM v2.1.1 baseline retuning is forbidden")
    if config.get("write_shared_data") is not False:
        raise RuntimeError("PRISM v2.1.1 must not write shared C1 data")
    if config.get("test_access_before_freeze") is not False:
        raise RuntimeError("PRISM v2.1.1 test access before freeze is forbidden")
    if config["K"]["profile_selection"].get(
        "maximum_relative_regret_vs_best"
    ) != 0.02:
        raise RuntimeError("PRISM v2.1.1 K profile regret guard changed")
    if config["K"]["profile_selection"].get("maximum_retained_profiles") != 2:
        raise RuntimeError("PRISM v2.1.1 retained-profile limit changed")
    if config["J"].get("allow_k_zero") is not False:
        raise RuntimeError("PRISM v2.1.1 Joint cannot register K-zero")
    if config["J"].get("allow_ar_only") is not False:
        raise RuntimeError("PRISM v2.1.1 Joint cannot register AR-only")
    if config["J"].get("share_pf_input_path_gate") is not True:
        raise RuntimeError("PF and Joint must share the input-path gate")
    return config


def load_v211_configs(
    project: Path,
    protocol: str = SRU_PROTOCOL,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the selected v2.1.1 patch and its inherited frozen contracts.

    The default remains the published SRU correction.  The Metro-P60 audit is
    an explicit protocol value rather than an environment override so a worker
    cannot silently switch scientific contracts.
    """
    if protocol == SRU_PROTOCOL:
        patch = load_v211_config(project)
    elif protocol == METRO_P60_PROTOCOL:
        from .v211_metro_config import load_metro_config

        patch = load_metro_config(project)
    elif protocol == PUBLIC_ALL_PROTOCOL:
        from .v211_public_all_config import load_public_all_algorithm_config

        patch = load_public_all_algorithm_config(project)
    elif protocol == REPRESENTATIVE_STAGE1_PROTOCOL:
        from .v211_representative_stage1_config import (
            load_representative_stage1_algorithm_config,
        )

        patch = load_representative_stage1_algorithm_config(project)
    elif protocol == TEP_CPU_HISTORY_EXTENSION_PROTOCOL:
        from .v211_tep_cpu_history_config import (
            load_tep_cpu_history_algorithm_config,
        )

        patch = load_tep_cpu_history_algorithm_config(project)
    elif protocol == TEP_CPU_NOWCAST_HISTORY_EXTENSION_PROTOCOL:
        from .v211_tep_cpu_history_config import (
            load_tep_cpu_nowcast_history_algorithm_config,
        )

        patch = load_tep_cpu_nowcast_history_algorithm_config(project)
    else:
        raise ValueError(f"unsupported PRISM v2.1.1 protocol: {protocol}")
    return patch, load_v21_config(project), load_frozen_config(project)


def input_views_for_protocol(shared: Path, protocol: str):
    if protocol == PUBLIC_ALL_PROTOCOL:
        from .v211_public_all_views import public_all_input_views

        return public_all_input_views(shared)
    if protocol == REPRESENTATIVE_STAGE1_PROTOCOL:
        from .v211_representative_stage1_views import (
            representative_stage1_input_views,
        )

        return representative_stage1_input_views(shared)
    if protocol == TEP_CPU_HISTORY_EXTENSION_PROTOCOL:
        from .v2_views import development_input_views

        return [
            view
            for view in development_input_views(shared)
            if view.head.task_id == "TEP_G_REP_H1"
        ]
    if protocol == TEP_CPU_NOWCAST_HISTORY_EXTENSION_PROTOCOL:
        from .v2_views import development_input_views

        return [
            view
            for view in development_input_views(shared)
            if view.head.task_id == "TEP_G_NOWCAST_H0"
        ]
    from .v21_views import sru_input_views

    return sru_input_views(shared)


def dynamic_views_for_protocol(shared: Path, protocol: str):
    if protocol == PUBLIC_ALL_PROTOCOL:
        from .v211_public_all_views import public_all_dynamic_views

        return public_all_dynamic_views(shared)
    if protocol == REPRESENTATIVE_STAGE1_PROTOCOL:
        from .v211_representative_stage1_views import (
            representative_stage1_dynamic_views,
        )

        return representative_stage1_dynamic_views(shared)
    if protocol == TEP_CPU_HISTORY_EXTENSION_PROTOCOL:
        from .v2_views import development_dynamic_views

        return [
            view
            for view in development_dynamic_views(shared)
            if view.head.task_id == "TEP_G_REP_H1"
        ]
    if protocol == TEP_CPU_NOWCAST_HISTORY_EXTENSION_PROTOCOL:
        from .v2_views import development_dynamic_views

        return [
            view
            for view in development_dynamic_views(shared)
            if view.head.task_id == "TEP_G_NOWCAST_H0"
        ]
    from .v21_views import sru_dynamic_views

    return sru_dynamic_views(shared)


def require_v211_test_freeze(paths: V211Paths) -> dict[str, Any]:
    if not paths.final_freeze_path.is_file():
        raise RuntimeError("SRU test access requires V211_SRU_FINAL_FREEZE_MANIFEST.json")
    manifest = json.loads(paths.final_freeze_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "V2_1_1_ASSEMBLY_FROZEN":
        raise RuntimeError("v2.1.1 final freeze manifest is not valid")
    if manifest.get("test_accessed") is not False:
        raise RuntimeError("v2.1.1 SRU test was already accessed")
    if manifest.get("development_continue_gate") != "PASS":
        raise RuntimeError("v2.1.1 development continue gate did not pass")
    load_v211_config(paths.project)
    if manifest.get("config_sha256") != sha256_file(paths.config_path):
        raise RuntimeError("v2.1.1 config changed after E6R freeze")
    frozen_config = paths.output / "FREEZE" / "V211_SRU_CONFIG_FROZEN.json"
    if manifest.get("frozen_config_sha256") != sha256_file(frozen_config):
        raise RuntimeError("v2.1.1 frozen config changed after E6R")
    if manifest.get("development_decision_sha256") != sha256_file(
        paths.development_decision_path
    ):
        raise RuntimeError("v2.1.1 development decision changed after E6R")
    baseline_inventory = paths.output / "BASELINES" / "FROZEN_BASELINE_INVENTORY.json"
    if manifest.get("baseline_inventory_sha256") != sha256_file(baseline_inventory):
        raise RuntimeError("v2.1.1 baseline inventory changed after E6R")
    baseline_replay = paths.output / "BASELINES" / "BASELINE_REPLAY_MANIFEST.json"
    if manifest.get("baseline_replay_manifest_sha256") != sha256_file(
        baseline_replay
    ):
        raise RuntimeError("v2.1.1 baseline replay manifest changed after E6R")
    if manifest.get("baseline_replay_amendment_sha256") != sha256_file(
        paths.baseline_amendment_path
    ):
        raise RuntimeError("v2.1.1 baseline amendment changed after E6R")
    if manifest.get("baseline_test_metrics_exposed_to_selection") is not False:
        raise RuntimeError("baseline test metrics reached v2.1.1 selection")
    if manifest.get("v211_candidate_test_accessed") is not False:
        raise RuntimeError("v2.1.1 candidate test was accessed before E7R")
    return manifest
