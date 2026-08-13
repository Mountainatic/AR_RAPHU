import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.neurobem_sampling_rate_scaling.composition_consistency import (
    HORIZONS_MS, composition_failure_class, exact_time_index,
    one_step_order_supported, registered_anchor_times, state_norms,
)
from prism_benchmark.neurobem_literature import TRACK_B_STATE_COLUMNS


ROOT = Path(__file__).parents[1]


def state(value=0.0):
    out = np.zeros(10); out[3] = 1.0; out[0] = value
    return out


def test_physical_horizon_grid_is_frozen():
    cfg = json.loads((ROOT / "configs" / "composition_consistency.yaml").read_text())
    assert tuple(cfg["physical_horizons_ms"]) == HORIZONS_MS == (10, 20, 50, 100, 200, 500)


def test_exact_shared_physical_timestamp_alignment():
    frame = pd.DataFrame({"t": np.arange(401) / 400.0})
    assert exact_time_index(frame, 0.5, 400) == 200


def test_anchor_grid_respects_history_and_terminal_support():
    frame = pd.DataFrame({"t": np.arange(801) / 400.0})
    anchors = registered_anchor_times(frame, maximum_anchors=16)
    assert min(anchors) >= 0.2
    assert max(anchors) + 0.5 <= 2.0 + 1e-12
    assert len(anchors) <= 16


def test_quaternion_aware_state_norm():
    a = state(); b = state(3.0)
    value = state_norms(a, b)
    assert value["velocity"] == 3.0
    assert value["attitude"] == 0.0
    assert value["full_state"] == 3.0


def test_one_step_order_contract():
    assert one_step_order_supported({100: 3.0, 200: 2.0, 400: 1.0})
    assert not one_step_order_supported({100: 1.0, 200: 2.0, 400: 3.0})


def test_composition_failure_class_is_predeclared():
    assert composition_failure_class({100: 1.0, 200: 2.0, 400: 3.0}) == "COMPOSITION_CONSISTENCY_FAILURE_SUPPORTED"
    assert composition_failure_class({100: 3.0, 200: 2.0, 400: 1.0}) == "COMPOSITION_CONSISTENCY_FAILURE_NOT_SUPPORTED"
    assert composition_failure_class({100: 1.0, 200: 3.0, 400: 2.0}) == "COMPOSITION_CONSISTENCY_MIXED"


def test_frozen_models_are_loaded_without_fit():
    source = (ROOT / "run_composition_audit.py").read_text()
    assert "_load_frozen_adapters" in source
    assert "fit_local_adapter" not in source
    assert '"model_retrained": False' in source


def test_test_requires_composition_freeze():
    source = (ROOT / "run_composition_audit.py").read_text()
    assert "COMPOSITION_TEST_REQUIRES_FREEZE" in source
    assert "COMPOSITION_FREEZE_HASH_MISMATCH" in source
    assert 'names = source.names("test")' in source


def test_existing_r3_r4_reproduction_gate_is_audited():
    source = (ROOT / "run_composition_audit.py").read_text()
    assert "audit_frozen_r3_r4_reproduction_gate" in source
    assert "R3_100HZ_REPRODUCTION.csv" in source
    assert "R3_100HZ_RESYNCHRONIZATION_REPRODUCTION.csv" in source
    assert "COMPOSITION_R3_R4_REPRODUCTION_VALUE_MISMATCH" in source


def test_no_stabilization_or_clipping():
    text = (ROOT / "run_composition_audit.py").read_text().lower() + (ROOT / "composition_consistency.py").read_text().lower()
    for forbidden in ("clip(", "spectral_radius", "lyapunov_penalty", "multi_step_training"):
        assert forbidden not in text


def test_rate_models_share_native_initial_and_terminal_state():
    source = (ROOT / "run_composition_audit.py").read_text()
    assert "initial = native_state[exact_time_index(native, anchor, 400)]" in source
    assert "target = native_state[exact_time_index(native, target_time, 400)]" in source
    assert "common_initial_state" in (ROOT / "composition_consistency.py").read_text()


def test_channel_decomposition_fields_are_present():
    source = (ROOT / "run_composition_audit.py").read_text()
    for field in ("full_state", "velocity", "attitude", "body_rate"):
        assert f'"{field}"' in (ROOT / "composition_consistency.py").read_text()
        assert f'"gt_{field}_error"' in source or "f\"gt_{key}_error\"" in source


def test_164hz_exclusion_is_inherited():
    cfg = json.loads((ROOT / "configs" / "composition_consistency.yaml").read_text())
    assert cfg["excluded_native_164hz_train_segments"] == 4
    assert cfg["model_retraining"] is False
