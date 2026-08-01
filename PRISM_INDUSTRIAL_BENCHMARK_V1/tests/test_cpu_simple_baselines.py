from __future__ import annotations

import numpy as np
import pandas as pd

from prism_benchmark.cpu_simple_baselines import _metrics, _prediction_frame


def test_persistence_is_exact_zero_change_prediction() -> None:
    samples = pd.DataFrame(
        {
            "view_sample_id": ["a", "b"],
            "base_origin_id": ["A", "B"],
            "dataset": ["demo", "demo"],
            "task_id": ["task", "task"],
            "target_head": ["head", "head"],
            "split": ["validation", "validation"],
            "y_true": [1.0, -2.0],
            "information_set": ["dynamic", "dynamic"],
            "entity_id": ["e", "e"],
            "availability_scenario": ["record_time", "record_time"],
            "proxy_policy": ["primary", "primary"],
        }
    )
    frame = _prediction_frame(samples, "PERSISTENCE", 0.0)
    np.testing.assert_array_equal(frame["y_pred"].to_numpy(), np.zeros(2))
    assert frame["dtype"].eq("float64").all()
    assert frame["parameter_count"].eq(0).all()


def test_metrics_are_fp64_reference_values() -> None:
    result = _metrics(np.array([1.0, 3.0]), np.array([0.0, 2.0]))
    assert result["mse"] == 1.0
    assert result["rmse"] == 1.0
    assert result["mae"] == 1.0
    assert result["r2"] == 0.0
    assert result["nrmse"] == 1.0
