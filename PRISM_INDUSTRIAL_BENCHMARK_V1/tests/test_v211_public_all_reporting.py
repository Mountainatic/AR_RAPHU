from __future__ import annotations

import pandas as pd

from prism_benchmark.v211_public_all_reporting import _add_ranks, _holm


def test_reporting_ranks_only_pass_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "task_id": "TASK",
                "target_head": "HEAD",
                "information_set": "dynamic",
                "availability_scenario": "record_time",
                "proxy_policy": "primary",
                "split": "test",
                "status": "PASS",
                "model": "A",
                "mse": 1.0,
            },
            {
                "task_id": "TASK",
                "target_head": "HEAD",
                "information_set": "dynamic",
                "availability_scenario": "record_time",
                "proxy_policy": "primary",
                "split": "test",
                "status": "PASS",
                "model": "B",
                "mse": 2.0,
            },
            {
                "task_id": "TASK",
                "target_head": "HEAD",
                "information_set": "dynamic",
                "availability_scenario": "record_time",
                "proxy_policy": "primary",
                "split": "test",
                "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE",
                "model": "C",
                "mse": None,
            },
        ]
    )
    ranked = _add_ranks(frame)
    assert ranked.loc[ranked["model"] == "A", "rank"].item() == 1
    assert ranked.loc[ranked["model"] == "B", "rank"].item() == 2
    assert pd.isna(ranked.loc[ranked["model"] == "C", "rank"].item())


def test_reporting_holm_uses_finite_sample_p_values() -> None:
    rows = [
        {
            "task_id": "TASK",
            "information_set": "dynamic",
            "comparison_family": "PAIR",
            "raw_p_value": 2.0 / 501.0,
            "holm_p_value": None,
        },
        {
            "task_id": "TASK",
            "information_set": "dynamic",
            "comparison_family": "PAIR",
            "raw_p_value": 4.0 / 501.0,
            "holm_p_value": None,
        },
    ]
    _holm(rows)
    assert all(row["holm_p_value"] is not None for row in rows)
    assert all(row["holm_p_value"] > 0.0 for row in rows)
