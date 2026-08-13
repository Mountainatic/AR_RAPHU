import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.neurobem_sampling_rate_scaling.history_scaling import (
    registered_conditions,
    resync_intervals,
)
from experiments.neurobem_sampling_rate_scaling.resampling import (
    audit_resampling,
    resample_track_b,
)
from prism_benchmark.neurobem_literature import MOTOR_COLUMNS, TRACK_B_STATE_COLUMNS, resample_track_b_100hz


ROOT = Path(__file__).parents[1]


def fixture_frame(rows=401):
    t = np.arange(rows) / 400.0
    values = {"t": t}
    for index, name in enumerate(TRACK_B_STATE_COLUMNS + MOTOR_COLUMNS):
        values[name] = np.sin(t * (index + 1)) + index
    return pd.DataFrame(values)


def test_100hz_resampling_exactly_calls_r3_semantics():
    frame = fixture_frame()
    pd.testing.assert_frame_equal(resample_track_b(frame, 100), resample_track_b_100hz(frame))


def test_native_400hz_timestamp_spacing():
    frame = fixture_frame()
    sampled = resample_track_b(frame, 400)
    audit = audit_resampling(frame, sampled, 400)
    assert np.isclose(audit.median_dt_seconds, 0.0025)
    assert audit.operator == "NATIVE_2P5MS_NO_DOWNSAMPLING"


def test_200hz_timestamp_spacing():
    frame = fixture_frame()
    sampled = resample_track_b(frame, 200)
    audit = audit_resampling(frame, sampled, 200)
    assert np.isclose(audit.median_dt_seconds, 0.005)
    assert audit.maximum_dt_error_seconds < 1e-12


def test_history_does_not_cross_trajectory_boundary():
    source = (ROOT / "run_experiment.py").read_text()
    assert "for name in names:" in source
    assert "LiteratureTrajectory(name, partition, sampled)" in source
    assert "np.concatenate" not in (ROOT / "resampling.py").read_text()


def test_fixed_step_history_is_20_at_all_rates():
    values = [value for value in registered_conditions() if value.history_mode == "FIXED_STEP"]
    assert [value.history_steps for value in values] == [20, 20, 20]
    assert [value.history_duration_ms for value in values] == [200.0, 100.0, 50.0]


def test_fixed_time_history_is_200ms_at_all_rates():
    values = [value for value in registered_conditions() if value.history_mode == "FIXED_TIME"]
    assert [value.history_steps for value in values] == [20, 40, 80]
    assert [value.history_duration_ms for value in values] == [200.0, 200.0, 200.0]


def test_no_test_before_calibration_freeze():
    source = (ROOT / "run_experiment.py").read_text()
    assert "TEST_REQUIRES_RATE_CALIBRATION_FREEZE" in source
    assert "RATE_CONFIG_HASH_MISMATCH" in source
    assert 'partition, names = "test", source.names("test")' in source
    assert "split_train_without_test_reads" in source


def test_deterministic_ordered_fork_execution():
    source = (ROOT / "run_experiment.py").read_text()
    assert 'mp.get_context("fork").Pool' in source
    assert "pool.imap(_condition_task, tasks, chunksize=1)" in source


def test_rate_specific_config_and_adapter_hash_audit():
    source = (ROOT / "run_experiment.py").read_text()
    assert "RATE_CONFIG_HASH_MISMATCH" in source
    assert "RATE_ADAPTER_HASH_MISMATCH" in source
    cfg = json.loads((ROOT / "configs" / "calibration.yaml").read_text())
    assert cfg["sampling_rates_hz"] == [100, 200, 400]


def test_results_report_steps_and_milliseconds():
    source = (ROOT / "run_experiment.py").read_text()
    for field in ("t_diverge_steps", "t_diverge_ms", "reliable_horizon_steps", "reliable_horizon_ms", "resync_interval_steps", "resync_interval_ms"):
        assert field in source


def test_resync_grids_are_registered_before_test():
    assert resync_intervals(100) == [1, 5, 10, 20, 50, 100, "infinity"]
    assert 40 in resync_intervals(200)
    assert 80 in resync_intervals(400)


def test_no_prism_core_or_stabilization_change():
    cfg = json.loads((ROOT / "configs" / "calibration.yaml").read_text())
    assert cfg["stabilization_added"] is False
    source = (ROOT / "run_experiment.py").read_text()
    for forbidden in ("clip(", "spectral_radius", "lyapunov_penalty", "state_saturation"):
        assert forbidden not in source.lower()
