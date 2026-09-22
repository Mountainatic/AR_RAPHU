from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from prism_benchmark.unified_hw_protocol import (
    load_registry,
    protocols,
    registered_levels,
    registered_target,
    report_delta_predictions,
    report_direct_level_predictions,
)


def test_registry_contains_six_public_dataset_families_and_private_cz() -> None:
    registry = load_registry()
    assert registry["protocol_id"].endswith("R2_CORRECTED_20260922")
    names = {item["dataset"] for item in registry["heads"]}
    assert {"TEP", "Debutanizer", "SRU", "PMSM", "MetroPT", "Cascaded Tanks"} <= names
    assert "Private CZ" in names


def test_registry_code_map_points_to_committed_files() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_registry()
    for relative in registry["code_map"].values():
        assert (root / relative).is_file(), relative


def test_registered_public_window_indices_are_half_open() -> None:
    values = np.arange(1000, dtype=np.float64)
    p = protocols()["PMSM_PM5__H600__W60"]
    current, future = registered_levels(values, 100, p)
    assert current == np.mean(values[40:100])
    assert future == np.mean(values[700:760])
    assert registered_target(values, 100, p) == future - current


def test_tep_h0_is_current_point_nowcast_without_current_input() -> None:
    values = np.arange(20, dtype=np.float64)
    p = protocols()["TEP_G_NOWCAST_H0__H0__W1"]
    assert registered_levels(values, 7, p) == (6.0, 7.0)
    assert registered_target(values, 7, p) == 1.0


def test_cz_raw2s_h_is_counted_from_last_observed_anchor() -> None:
    values = np.arange(400, dtype=np.float64)
    p = protocols()["CZ_ROD1_TO_ROD2_RAW2S_L256_H4"]
    assert p.horizon_seconds == 8
    assert registered_levels(values, 300, p) == (299.0, 303.0)
    assert registered_target(values, 300, p) == 4.0


def test_tanks_is_direct_level_at_origin_plus_16() -> None:
    values = np.arange(40, dtype=np.float64)
    p = protocols()["TANKS_H16__H16__W1"]
    assert p.horizon_seconds == 64
    assert registered_levels(values, 7, p) == (None, 23.0)
    assert registered_target(values, 7, p) == 23.0


def test_origins_must_be_exact_and_cannot_use_negative_slices() -> None:
    values = np.arange(20, dtype=np.float64)
    p = protocols()["SRU_H2S_REP_H1__H1__W1"]
    with np.testing.assert_raises(ValueError):
        registered_levels(values, 7.5, p)
    with np.testing.assert_raises(IndexError):
        registered_levels(values, 0, p)


def test_constant_direct_level_target_reports_json_safe_null_r2() -> None:
    metrics = report_direct_level_predictions([1.0, 1.0], [1.0, 2.0])
    assert metrics["r2_level"] is None


def test_reporting_names_both_persistence_skill_definitions() -> None:
    metrics = report_delta_predictions(
        delta_true=[1.0, -1.0, 2.0, -2.0],
        delta_pred=[0.5, -0.5, 1.0, -1.0],
        current_level=[10.0, 10.0, 10.0, 10.0],
    )
    assert metrics["persistence_skill"] == metrics["persistence_skill_mse"]
    assert np.isclose(float(metrics["persistence_skill_mse"]), 0.75)
    assert np.isclose(float(metrics["persistence_skill_rmse"]), 0.5)
    assert np.isclose(float(metrics["rmse"]), float(metrics["rmse_delta"]))


def test_registered_public_results_match_committed_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    table_path = (
        root
        / "results"
        / "prism_strict_oof_finalization_20260915"
        / "TABLES_FOR_PAPER"
        / "table_stagewise.csv"
    )
    with table_path.open("r", encoding="utf-8", newline="") as stream:
        rows = {row["task"]: row for row in csv.DictReader(stream)}
    registry = {item["head_id"]: item for item in load_registry()["heads"]}
    mapping = {
        "TEP_G_NOWCAST_H0__H0__W1": "TEP record-time",
        "DEB_C4__H5__W1": "Debutanizer",
        "SRU_H2S_REP_H1__H1__W1": "SRU H2S",
        "SRU_SO2_REP_H1__H1__W1": "SRU SO2",
        "PMSM_PM5__H600__W60": "PMSM proxy-excluded",
        "METRO_P60__H6__W1": "MetroPT P60",
        "METRO_OIL20__H120__W12": "MetroPT Oil20",
    }
    for head_id, task in mapping.items():
        frozen = registry[head_id]["result"]
        source = rows[task]
        assert int(frozen["rows"]) == int(source["rows"])
        assert np.isclose(float(frozen["rmse"]), float(source["J_RMSE"]))
        assert np.isclose(float(frozen["r2_delta"]), float(source["J_Delta_R2"]))
        assert np.isclose(
            float(frozen["r2_level_reconstructed"]), float(source["J_Level_R2"])
        )
        assert np.isclose(
            float(frozen["persistence_skill_mse"]),
            float(source["J_persistence_skill"]),
        )

    tanks_path = (
        root.parent
        / "PRISM_TANKS_CZ_SUPPLEMENTARY_E1_E6"
        / "TANKS"
        / "E1_STAGEWISE"
        / "STAGEWISE_METRICS_WIDE.csv"
    )
    with tanks_path.open("r", encoding="utf-8", newline="") as stream:
        tanks = next(csv.DictReader(stream))
    frozen_tanks = registry["TANKS_H16__H16__W1"]["result"]
    assert np.isclose(
        float(frozen_tanks["rmse"]), float(tanks["K+C+DELTA_W_rmse"])
    )
    assert np.isclose(
        float(frozen_tanks["r2_level"]), float(tanks["K+C+DELTA_W_r2_level"])
    )


def test_registered_cz_results_and_skill_names_match_derived_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = (
        root / "results" / "cz_raw2s_hscan_20260920" / "selected_dynamic_metrics.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    records = {
        (item["direction"], int(item["h_steps"])): item
        for item in evidence["records"]
    }
    registry = {item["head_id"]: item for item in load_registry()["heads"]}
    mapping = {
        "CZ_ROD1_TO_ROD2_RAW2S_L256_H4": "Rod_1_to_Rod_2",
        "CZ_ROD2_TO_ROD1_RAW2S_L256_H4": "Rod_2_to_Rod_1",
    }
    for head_id, direction in mapping.items():
        source = records[(direction, 4)]
        frozen = registry[head_id]["result"]
        for key in (
            "rows",
            "rmse",
            "r2_level_reconstructed",
            "persistence_skill_mse",
            "persistence_skill_rmse",
        ):
            assert np.isclose(float(frozen[key]), float(source[key]))
        assert frozen["model"] == source["model"]
        expected_rmse_skill = 1.0 - np.sqrt(
            1.0 - float(source["persistence_skill_mse"])
        )
        assert np.isclose(float(source["persistence_skill_rmse"]), expected_rmse_skill)
