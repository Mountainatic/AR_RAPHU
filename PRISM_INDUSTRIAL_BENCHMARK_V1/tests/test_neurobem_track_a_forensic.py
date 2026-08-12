from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.neurobem_track_a_forensic import (
    FROZEN_INERTIA_KG_M2,
    FROZEN_MASS_KG,
    GroundTruthContract,
    assert_forensic_stage_has_no_training,
    manifest_identity,
    neuromhe_metric,
    reconstruct_force_torque_gt,
    rss21_metric,
)


def physical_frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "t": np.arange(rows, dtype=float),
        "acc x": np.ones(rows), "acc y": np.ones(rows) * 2, "acc z": np.ones(rows) * 3,
        "ang acc x": np.ones(rows), "ang acc y": np.ones(rows) * 2, "ang acc z": np.ones(rows) * 3,
        "ang vel x": np.ones(rows) * .1, "ang vel y": np.ones(rows) * .2, "ang vel z": np.ones(rows) * .3,
        "error_residual_fx": np.ones(rows) * 1e99,
    })


def test_force_gt_does_not_use_released_residual_force():
    source = inspect.getsource(reconstruct_force_torque_gt)
    assert "error_residual" not in source.lower()
    a = physical_frame()
    b = a.copy(); b["error_residual_fx"] *= -7
    np.testing.assert_array_equal(reconstruct_force_torque_gt(a), reconstruct_force_torque_gt(b))


def test_force_gt_uses_frozen_mass_0772():
    target = reconstruct_force_torque_gt(physical_frame())
    expected = np.tile(FROZEN_MASS_KG * np.array([[1., 2., 3.]]), (len(target), 1))
    np.testing.assert_allclose(target[:, :3], expected)
    with pytest.raises(ValueError, match="FROZEN_RIGID_BODY"):
        reconstruct_force_torque_gt(physical_frame(), GroundTruthContract(mass_kg=.752))


def test_torque_gt_uses_frozen_inertia():
    target = reconstruct_force_torque_gt(physical_frame())
    w = np.array([.1, .2, .3]); j = np.asarray(FROZEN_INERTIA_KG_M2)
    expected = np.array([1., 2., 3.]) * j + np.cross(w, w * j)
    np.testing.assert_allclose(target[:, 3:], np.tile(expected, (len(target), 1)))


def test_force_frame_matches_reference():
    assert GroundTruthContract().force_frame == "BODY_FRONT_LEFT_UP"
    assert "INCLUDING_GRAVITY" in GroundTruthContract().acceleration_semantics


def test_torque_frame_matches_reference():
    assert GroundTruthContract().torque_frame == "BODY_FRONT_LEFT_UP"
    assert GroundTruthContract().timestamp_alignment == "SAME_ROW_NO_SHIFT"


def test_rss_metric_contract():
    y = np.zeros((2, 6)); p = np.ones((2, 6))
    m = rss21_metric(y, p)
    assert m["Fxy"] == pytest.approx(1.0) and m["F"] == pytest.approx(1.0)


def test_neuromhe_metric_contract():
    y = np.zeros((2, 6)); p = np.ones((2, 6))
    m = neuromhe_metric(y, p)
    assert m["Fxy"] == pytest.approx(np.sqrt(2)) and m["F"] == pytest.approx(np.sqrt(3))


def test_rss_and_neuromhe_group_metrics_are_not_conflated():
    y = np.zeros((2, 6)); p = np.ones((2, 6))
    assert rss21_metric(y, p)["F"] != neuromhe_metric(y, p)["F"]


def test_testset_manifest_is_audited():
    row = manifest_identity("flight_seg_1", {"flight_seg_1": "3D Circle_1"})
    assert row["NeuroMHE_match"] is True and row["RSS_match_if_known"] is False


def test_no_prism_retraining_in_forensic_stage():
    root = Path(__file__).resolve().parents[1]
    assert_forensic_stage_has_no_training([root / "src/prism_benchmark/neurobem_track_a_forensic.py"])


def test_no_test_driven_model_change():
    root = Path(__file__).resolve().parents[1]
    assert_forensic_stage_has_no_training([
        root / "src/prism_benchmark/neurobem_track_a_forensic.py",
        root / "tools/run_neurobem_track_a_forensic.py",
    ])
