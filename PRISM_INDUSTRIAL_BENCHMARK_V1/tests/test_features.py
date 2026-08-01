import numpy as np
import pandas as pd

from prism_cpu.config import TaskSpec
from prism_cpu.features import attach_series, build_task_data, target_lags
from prism_cpu.io import DatasetFrame


def test_target_alignment_and_no_future_input():
    n = 80
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float) * 2})
    ds = DatasetFrame("toy", df, ["y"], ["x"], np.zeros(n, dtype=int), None, [], {"cadence_seconds": 1})
    task = TaskSpec("T", "toy", "y", 3, 1, True, "all")
    d = attach_series(build_task_data(ds, task, max_history_steps=8), ds)
    assert np.all(d.y == d.metadata["target_series"][d.origins + d.horizon_steps])
    assert np.max(d.origins) < np.max(d.origins + d.horizon_steps)
    Y, _ = target_lags(d, [0, 1, 2])
    assert np.all(Y[:, 0] == d.metadata["target_series"][d.origins])

