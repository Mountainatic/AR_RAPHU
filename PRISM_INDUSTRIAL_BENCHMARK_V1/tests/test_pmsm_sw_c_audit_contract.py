from __future__ import annotations

import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark import pmsm_sw_c_audit as audit  # noqa: E402


def test_validation_is_materialized_only_after_c_structure_freeze() -> None:
    source = inspect.getsource(audit.run_primary_corrected_c)
    boundary = source.index("# Holdout boundary: validation is materialized only after structure_frozen.")
    load = source.index('load_native_samples(shared, view, "validation")')
    assert load > boundary
    before = source[:boundary]
    assert 'load_native_samples(shared, view, "validation")' not in before
    assert 'BaseAccessor(\n            shared,\n            view.head.dataset,\n            "validation"' not in before


def test_train_fold_assembly_support_precedes_row_caps() -> None:
    source = inspect.getsource(audit.run_primary_corrected_c)
    fit_support = source.index("fit_common = apply_assembly_support(fit_raw, active)")
    eval_support = source.index("evaluation_common = apply_assembly_support(")
    fit_cap = source.index("fit = _cap_after_support(fit_common, fit_cap)")
    eval_cap = source.index("evaluation = _cap_after_support(evaluation_common, evaluation_cap)")
    assert fit_support < fit_cap
    assert eval_support < eval_cap


def test_c_replay_never_reselects_k_structure() -> None:
    source = inspect.getsource(audit.run_primary_corrected_c)
    assert "active_d_descriptors(d_output)" in source
    for forbidden in (
        "one_se_select(",
        "practical_activation(",
        "channel_profiles(",
        "_candidate_losses(",
    ):
        assert forbidden not in source


def test_test_partition_is_never_requested_by_c_runner() -> None:
    source = inspect.getsource(audit.run_primary_corrected_c)
    assert 'load_native_samples(shared, view, "test")' not in source
    assert '"test", [item["channel"]' not in source


def test_inherited_gate_and_smallest_stable_rule_are_explicit() -> None:
    source = inspect.getsource(audit.run_primary_corrected_c)
    assert "input_path_preservation_gate(" in source
    assert "if selected_alpha is None and sum(passed) >= minimum_folds" in source
    assert "COMPRESSED if bool(gate.get(\"pass\")) else BEST_ACTIVE_K" in source
