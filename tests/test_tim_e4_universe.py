from __future__ import annotations

import json
from pathlib import Path

from prism_benchmark.tim_e4_universe import (
    apply_candidate_universe,
    history_grid_for_universe,
    universe_manifest,
)
from prism_benchmark.v211_config import PUBLIC_ALL_PROTOCOL, load_v211_configs


PROJECT = Path(__file__).resolve().parents[1] / "PRISM_INDUSTRIAL_BENCHMARK_V1"


def test_standard_is_exact_and_universes_are_nested() -> None:
    v211, v21, v2 = load_v211_configs(PROJECT, PUBLIC_ALL_PROTOCOL)
    standard = apply_candidate_universe(v211, v21, v2, "standard")
    assert standard == (v211, v21, v2)
    manifest = universe_manifest(v211, v21, v2)
    for axis in manifest["universes"]["standard"]["axes"]:
        coarse = set(map(str, manifest["universes"]["coarse"]["axes"][axis]))
        frozen = set(map(str, manifest["universes"]["standard"]["axes"][axis]))
        expanded = set(map(str, manifest["universes"]["expanded"]["axes"][axis]))
        assert coarse <= frozen <= expanded, axis
    assert manifest["test_accessed"] is False


def test_manifest_is_json_serializable() -> None:
    configs = load_v211_configs(PROJECT, PUBLIC_ALL_PROTOCOL)
    json.dumps(universe_manifest(*configs), sort_keys=True)


def test_task_history_grid_is_mechanically_nested() -> None:
    assert history_grid_for_universe([128, 256], "coarse") == [128]
    assert history_grid_for_universe([128, 256], "standard") == [128, 256]
    assert history_grid_for_universe([128, 256], "expanded") == [64, 128, 256, 512]
