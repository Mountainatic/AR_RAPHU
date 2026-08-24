from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from prism_benchmark.c1_builder import _selected_datasets
from prism_benchmark.c1_contracts import realize_heads
from prism_benchmark.cz_only_pilot import (
    PILOT_ACTIVE_DATASETS,
    PILOT_PROTOCOL_ID,
    pilot_path_views,
)
from prism_benchmark.representative_formal import checkpoint_namespace_root
from prism_benchmark.v211_representative_stage1_config import (
    load_representative_stage1_descriptor,
)


CONFIG = PROJECT / "configs/representative_horizon_stage1_tep_sru_c1_tasks.json"
PARENT_CONFIG = PROJECT / "configs/c1_tasks.json"


def test_representative_stage1_realizes_exact_h1_heads() -> None:
    _, heads = realize_heads(CONFIG)
    observed = {
        head.head_id: (head.h_steps, head.w_steps, head.w0_steps)
        for head in heads
    }
    assert observed == {
        "TEP_G_REP_H1__H1__W2": (1, 2, 2),
        "SRU_H2S_REP_H1__H1__W1": (1, 1, 1),
        "SRU_SO2_REP_H1__H1__W1": (1, 1, 1),
    }
    assert all(head.primary and head.status == "PASS" for head in heads)


def test_representative_stage1_changes_only_horizon() -> None:
    parents = {
        task["task_id"]: task
        for task in json.loads(PARENT_CONFIG.read_text(encoding="utf-8"))["tasks"]
    }
    tasks = json.loads(CONFIG.read_text(encoding="utf-8"))["tasks"]
    invariant_fields = (
        "dataset",
        "target",
        "cadence_seconds",
        "window_seconds",
        "availability_delays_steps",
        "proxy_policies",
    )
    for task in tasks:
        parent = parents[task["parent_task_id"]]
        assert task["changed_field"] == "horizon_only"
        assert task["W_unchanged"] is True
        assert task["W0_unchanged"] is True
        for field in invariant_fields:
            assert task[field] == parent[field]
        assert task["new_horizon"]["seconds"] == task["horizons_seconds"][0]
        assert task["new_horizon"]["seconds"] != task["parent_horizon"]["seconds"]


def test_representative_stage1_subset_selects_only_tep_and_sru() -> None:
    _, heads = realize_heads(CONFIG)
    grouped = defaultdict(list)
    for head in heads:
        grouped[head.dataset].append(head)
    assert _selected_datasets(grouped) == ("tep", "sru")
    with pytest.raises(ValueError, match="no tasks"):
        _selected_datasets({})
    with pytest.raises(ValueError, match="unsupported"):
        _selected_datasets({"unknown": heads})


def test_representative_stage1_refuses_existing_c1_output() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["output_policy"] == "REFUSE_EXISTING"


def test_formal_scope_registers_only_tep_sru_cz_as_active() -> None:
    value = load_representative_stage1_descriptor(PROJECT)
    assert value["active_datasets"] == ["tep", "sru", "cz_czochralski"]
    assert set(value["primary_tasks"]) == {
        "TEP_G_REP_H1",
        "SRU_H2S_REP_H1",
        "SRU_SO2_REP_H1",
        "CZ_DIAM_RAW2S_CURRENT_L256",
    }
    assert value["reserved_datasets"] == {
        "debutanizer": "NOT_RUN_BY_USER_SCOPE",
        "pmsm": "NOT_RUN_BY_USER_SCOPE",
        "metropt": "NOT_RUN_BY_USER_SCOPE",
    }
    assert value["neural3_status"] == "NOT_RUN_BY_USER_SCOPE"


def test_formal_checkpoint_namespaces_keep_cz_directions_disjoint(tmp_path: Path) -> None:
    rod_1_to_rod_2 = checkpoint_namespace_root(
        tmp_path, "cz:Rod_1_to_Rod_2"
    )
    rod_2_to_rod_1 = checkpoint_namespace_root(
        tmp_path, "cz:Rod_2_to_Rod_1"
    )
    assert rod_1_to_rod_2 != rod_2_to_rod_1
    assert rod_1_to_rod_2 == tmp_path / "cz" / "Rod_1_to_Rod_2"
    assert rod_2_to_rod_1 == tmp_path / "cz" / "Rod_2_to_Rod_1"
    assert checkpoint_namespace_root(tmp_path, "public") == tmp_path / "public"
    with pytest.raises(RuntimeError, match="UNKNOWN_CZ_CHECKPOINT_DIRECTION"):
        checkpoint_namespace_root(tmp_path, "cz:invalid")


def test_cz_pilot_contains_both_directions_and_is_single_dataset(tmp_path: Path) -> None:
    views = pilot_path_views(PROJECT, tmp_path)
    assert PILOT_PROTOCOL_ID == "CZ_ONLY_L256_DELTA_NOWCAST_PILOT_V1"
    assert PILOT_ACTIVE_DATASETS == ("cz_czochralski",)
    assert [namespace for namespace, _, _ in views] == [
        "cz:Rod_1_to_Rod_2",
        "cz:Rod_2_to_Rod_1",
    ]
    assert all(len(candidates) == 2 for _, _, candidates in views)
