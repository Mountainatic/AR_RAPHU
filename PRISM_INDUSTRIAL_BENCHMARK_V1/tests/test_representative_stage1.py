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
