from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.tim_validation.e4_candidate.runner import _common_support_metrics
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


def test_common_support_metrics_use_strict_intersection(tmp_path: Path) -> None:
    results = {}
    rows = {
        "coarse": (["a", "b", "c"], [0.0, 1.0, 2.0]),
        "standard": (["b", "c", "d"], [1.0, 2.0, 3.0]),
        "expanded": (["b", "c"], [1.0, 2.0]),
    }
    for universe, (ids, truth) in rows.items():
        root = tmp_path / "demo" / universe
        path = root / "final" / "prediction.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame(
            {"base_origin_id": ids, "y_true": truth, "y_pred": truth}
        ).to_parquet(path, index=False)
        results[universe] = {
            "record": {
                "prediction_path": "final/prediction.parquet",
                "scoring_support_hash": universe,
            }
        }
    metrics, audit = _common_support_metrics(tmp_path, "demo", results)
    assert audit["rows"] == 2
    assert audit["identical_target_rows"] is True
    assert all(value["RMSE"] == 0.0 for value in metrics.values())
