from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.c1_contracts import (
    RealizedHead,
    dependency_bounds,
    latest_available_target_index,
    maximum_registered_history,
    round_half_up_steps,
    target_change,
    valid_origins_for_interval,
)
import pandas as pd

from prism_benchmark.c1_builder import _sample_frame, normalize_tep_index_dtypes


def test_half_open_target_has_exact_w0_and_w_samples() -> None:
    y = np.arange(20, dtype=np.float64)
    origins = np.asarray([5, 8])
    actual = target_change(y, origins, h=2, w=3, w0=2)
    expected = np.asarray([
        np.mean(y[7:10]) - np.mean(y[3:5]),
        np.mean(y[10:13]) - np.mean(y[6:8]),
    ])
    np.testing.assert_allclose(actual, expected)


def test_h_zero_never_uses_same_sample_target_history() -> None:
    assert latest_available_target_index(origin=10, delay=0) == 9
    assert latest_available_target_index(origin=10, delay=5) == 4
    assert latest_available_target_index(origin=10, delay=10) == -1


def test_round_half_up_and_zero_contract() -> None:
    assert round_half_up_steps(0, 6, minimum=0) == (0, 0.0)
    assert round_half_up_steps(15, 10, minimum=1) == (2, 1 / 3)
    assert round_half_up_steps(14, 10, minimum=1) == (1, 4 / 14)


def test_dependency_interval_and_boundary_purge_are_half_open() -> None:
    origins = valid_origins_for_interval(100, 200, lmax=20, h=5, w=3, delay=2, left_buffer=7)
    assert origins[0] == 127
    assert origins[-1] == 190
    left, right = dependency_bounds(origins[[0, -1]], lmax=20, h=5, w=3, delay=2)
    np.testing.assert_array_equal(left, [107, 170])
    np.testing.assert_array_equal(right, [137, 200])


def test_registered_history_uses_eight_h_or_largest_feasible_h_zero_candidate() -> None:
    assert maximum_registered_history(12, 2, [500]) == 96
    assert maximum_registered_history(0, 1, [1436]) == 1024
    assert maximum_registered_history(0, 1, [6048]) == 4096


def test_view_id_changes_with_availability_and_proxy_but_base_origin_does_not() -> None:
    head = RealizedHead(
        task_id="T",
        dataset="debutanizer",
        target="y",
        cadence_seconds=360,
        requested_horizon_seconds=0,
        requested_window_seconds=360,
        h_steps=0,
        w_steps=1,
        w0_steps=1,
        horizon_error=0,
        window_error=0,
        status="PASS",
        primary=False,
    )
    common = {
        "dataset": "debutanizer",
        "entity_id": "one",
        "split": "train",
        "head": head,
        "information_set": "dynamic",
        "lmax": 2,
        "origins": np.asarray([3]),
        "y": np.arange(10, dtype=np.float64),
    }
    main = _sample_frame(delay=0, proxy_policy="primary", **common)
    delayed = _sample_frame(delay=10, proxy_policy="primary", **common)
    proxy = _sample_frame(delay=0, proxy_policy="full", **common)

    assert main.loc[0, "base_origin_id"] == delayed.loc[0, "base_origin_id"] == proxy.loc[0, "base_origin_id"]
    assert len({main.loc[0, "view_sample_id"], delayed.loc[0, "view_sample_id"], proxy.loc[0, "view_sample_id"]}) == 3
    assert main.loc[0, "latest_available_target_index"] == 2
    assert delayed.loc[0, "latest_available_target_index"] == -8


def test_tep_index_schema_is_stable_across_rdata_numeric_decoders() -> None:
    float_frame = pd.DataFrame({"faultNumber": [0.0], "simulationRun": [1.0], "sample": [1.0], "xmeas_1": [2.0]})
    int_frame = pd.DataFrame({"faultNumber": np.asarray([0], dtype=np.int32), "simulationRun": [1.0], "sample": np.asarray([1], dtype=np.int32), "xmeas_1": [2.0]})

    normalized_float = normalize_tep_index_dtypes(float_frame)
    normalized_int = normalize_tep_index_dtypes(int_frame)

    assert normalized_float.dtypes.to_dict() == normalized_int.dtypes.to_dict()
    assert all(str(normalized_float[column].dtype) == "int64" for column in ("faultNumber", "simulationRun", "sample"))
