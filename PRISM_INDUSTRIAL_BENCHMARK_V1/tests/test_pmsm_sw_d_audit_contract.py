from __future__ import annotations

import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark import pmsm_sw_d_audit as audit  # noqa: E402


def test_preregistered_d_freezes_are_consistent() -> None:
    implementation, c4 = audit.assert_freeze_consistency(PROJECT_ROOT)
    assert implementation["sample_support"]["contract"] == "NATIVE_K_COMMON_ASSEMBLY_R1"
    assert c4["complexity_ladder"] == audit.D_LADDER
    assert c4["rank_candidates"] == [0, 1, 2, 3, 4]
    assert c4["penalties"]["selection"] == "sequential_validation_only_lambda0_then_lambda_tau_then_lambda_x"


def test_validation_materialization_occurs_after_selection_freeze() -> None:
    source = inspect.getsource(audit.run_d_channel)
    boundary = source.index("selection_frozen =")
    validation_load = source.index('load_native_samples(shared, view, "validation")')
    validation_accessor = source.index("validation_accessor = BaseAccessor(")
    accessor_block = source[validation_accessor : validation_accessor + 240]
    assert '"validation"' in accessor_block
    assert validation_load > boundary
    assert validation_accessor > boundary
    before_boundary = source[:boundary]
    assert 'load_native_samples(shared, view, "validation")' not in before_boundary
    assert "validation_accessor = BaseAccessor(" not in before_boundary


def test_test_partition_is_never_requested_by_d_runner() -> None:
    source = inspect.getsource(audit.run_d_channel)
    assert 'load_native_samples(shared, view, "test")' not in source
    assert 'BaseAccessor(shared, view.head.dataset, "test"' not in source
    assert audit.EXPECTED_INPUTS == [
        "ambient",
        "coolant",
        "u_d",
        "u_q",
        "i_d",
        "i_q",
        "motor_speed",
        "torque",
    ]
