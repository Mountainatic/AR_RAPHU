import numpy as np

from prism_cpu.metrics import metric_row, paired_block_bootstrap


def test_metrics_and_bootstrap_are_finite():
    y = np.arange(40, dtype=float)
    p = y + .1
    b = y + .5
    row = metric_row(y, p, baseline=b)
    assert row["rmse"] > 0
    boot = paired_block_bootstrap(y, p, b, reps=20, block=4)
    assert boot["reps"] == 20
    assert np.isfinite(boot["mean_gain"])

