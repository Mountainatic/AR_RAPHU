from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from prism_benchmark.v211_joint import J_KW, predict_joint_candidate
from prism_benchmark.v211_joint_stability import (
    fit_joint_candidate_stability,
    fit_joint_candidate_v22,
)
from prism_benchmark.v211_joint_stability_config import (
    JOINT_ESTIMATOR_SEMANTICS,
    MODEL_VERSION,
    OUTPUT_DIRECTORY,
    PRACTICE_REVISION,
    PROTOCOL_ID,
    load_joint_stability_config,
)
from prism_benchmark.v211_joint_stability_runner import (
    _canonical_ids,
    run_m5_forbidden,
    run_m7_preflight,
)
from prism_benchmark.v211_metro_final import materialize_view
from prism_benchmark.v211_metro_reporting import JOINT_VS_PF_COMPARISON, run_m8


def _fixture() -> tuple[dict[str, np.ndarray], np.ndarray]:
    x = np.linspace(-1.0, 1.0, 40)
    blocks = {
        "K": np.column_stack([x, x * x]),
        "W": np.column_stack([np.sin(x), np.cos(x)]),
        "A": x[:, None],
    }
    return blocks, 0.5 * x + 0.2 * np.sin(x)


def test_canonical_version_and_practice_metadata() -> None:
    assert MODEL_VERSION == "PRISM_V2_1_1"
    assert PROTOCOL_ID == "PRISM_V2_1_1_METRO_P60_JOINT_STABILITY_FINAL_V1"
    assert PRACTICE_REVISION == "PRISM_V211_JOINT_PREDICTIVE_STABILITY_PRACTICE_R1"
    assert JOINT_ESTIMATOR_SEMANTICS == "PREDICTIVE_STABILITY_RIDGE_R1"
    assert OUTPUT_DIRECTORY == "results_prism_v2_1_1_metro_p60_joint_stability_final"


def test_config_has_no_canonical_v22_metadata() -> None:
    project = Path(__file__).resolve().parents[1]
    config = load_joint_stability_config(project)
    assert config["model_version"] == "PRISM_V2_1_1"
    assert "V2_2" not in config["protocol_id"]
    assert config["historical_aggregates_selection_use_forbidden"] is True


def test_estimator_semantics_dispatch_is_explicit_and_hard_stops() -> None:
    source = inspect.getsource(materialize_view)
    assert 'semantics == JOINT_ESTIMATOR_SEMANTICS' in source
    assert 'semantics == "LEGACY_V211_JOINT"' in source
    assert "STOP_ESTIMATOR_SEMANTICS_UNBOUND" in source
    assert 'get("estimator_version") == "PRISM_V2_2"' not in source


def test_renamed_estimator_is_exact_abf7_alias() -> None:
    blocks, target = _fixture()
    kwargs = dict(
        candidate=J_KW, k_representation="FULL_BASIS",
        numerical_alpha=1e-4, predictive_eta=0.1,
        raw_k_support=("k0", "k1"),
    )
    canonical = fit_joint_candidate_stability(blocks, target, blocks, **kwargs)
    historical = fit_joint_candidate_v22(blocks, target, blocks, **kwargs)
    np.testing.assert_array_equal(canonical[0], historical[0])
    assert canonical[1] == historical[1]
    for key in canonical[2]:
        np.testing.assert_array_equal(canonical[2][key], historical[2][key])


def test_predictive_eta_is_not_reapplied_at_prediction() -> None:
    blocks, target = _fixture()
    prediction, contract, _ = fit_joint_candidate_stability(
        blocks, target, blocks, candidate=J_KW, k_representation="CHANNEL_COMPRESSED",
        numerical_alpha=0.0, predictive_eta=1.0, raw_k_support=("k0", "k1"),
    )
    replay, _ = predict_joint_candidate(blocks, contract)
    np.testing.assert_allclose(prediction, replay, rtol=0.0, atol=1e-15)
    assert "predictive_eta" not in inspect.getsource(predict_joint_candidate)


def test_full_and_compressed_final_materialization_are_explicit() -> None:
    source = inspect.getsource(materialize_view)
    assert "CHANNEL_COMPRESSED" in source
    assert "FULL_BASIS" in source
    assert 'blocks["compressed"]' in source
    assert 'blocks["joint"]' in source


def test_m5_recomputation_is_forbidden() -> None:
    try:
        run_m5_forbidden()
    except RuntimeError as error:
        assert "M5_RECOMPUTATION_FORBIDDEN" in str(error)
    else:
        raise AssertionError("M5 recomputation unexpectedly allowed")


def test_candidate_id_namespace_is_canonical() -> None:
    source = inspect.getsource(_canonical_ids)
    assert '"FINAL_V211_JOINT_STABILITY"' in source
    assert '"METRO_P60_FINAL_V211_JOINT_STABILITY"' not in source
    assert "FINAL_V22_JOINT" not in inspect.getsource(_canonical_ids)


def test_m7_preflight_does_not_load_test_or_ood() -> None:
    source = inspect.getsource(run_m7_preflight)
    assert "load_samples" not in source
    assert '"test_or_ood_read_by_preflight": False' in source


def test_primary_joint_vs_pf_is_registered_before_lockbox() -> None:
    assert JOINT_VS_PF_COMPARISON == {
        "comparison_id": "JOINT_SELECTED_VS_PF_SELECTED",
        "candidate": "J_SELECTED",
        "comparator": "PF_SELECTED",
        "comparison_family": "JOINT_VS_PHYSICS_FIRST",
    }


def test_m8_reports_pf_and_joint_without_reselection() -> None:
    source = inspect.getsource(run_m8)
    assert "METRO_P60_V211_PRIMARY_JOINT_VS_PF.csv" in source
    assert "METRO_P60_V211_FINAL_EVIDENCE_SUMMARY.json" in source
    assert '"post_test_reselection": False' in source


def test_withdrawn_theory_is_not_canonical() -> None:
    project = Path(__file__).resolve().parents[1]
    archived = project / "archive/legacy_execution_notes/PRISM_Theory_v2_2_Joint_Predictive_Stability_Extension_Theory_Only.md"
    canonical = project / "PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_PACKAGE/reference/PRISM_Theory_v2_1_1_Implementation_Safe_Stagewise_Routed_Modular_Assembly_Theory_Only.md"
    assert "WITHDRAWN AS CANONICAL THEORY" in archived.read_text(encoding="utf-8")
    assert "11A.11 Joint 预测稳定性实践 Tips" in canonical.read_text(encoding="utf-8")
