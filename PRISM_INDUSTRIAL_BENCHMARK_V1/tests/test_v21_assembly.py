import pandas as pd

from prism_benchmark.v21_a import EXACT_ZERO
from prism_benchmark.v21_assembly import build_physics_first_card
from prism_benchmark.v21_c import _target_mean
from prism_benchmark.v21_w import IDENTITY


def _module(candidate):
    return {
        "status": "PASS",
        "final_selected_candidate": candidate,
        "final_selected_fold_losses": [1.0, 1.0, 1.0],
        "final_selected_prediction_path": "validation.parquet",
        "final_selected_contract": {"family": candidate},
    }


def test_pf_is_stagewise_and_never_falls_back_to_a_only():
    card = build_physics_first_card(
        {"status": "PASS", "active": True, "final_selected_candidate": "K"},
        {"status": "PASS", "final_selected_candidate": "C"},
        _module(IDENTITY), _module(EXACT_ZERO),
    )
    assert card["assembly"] == "PRISM_V2_1_PF_K"
    assert card["global_assembly_one_se"] is False
    assert card["a_only_fallback_allowed"] is False


def test_pf_reports_not_supported_when_input_path_is_zero():
    card = build_physics_first_card(
        {"status": "K_EXACT_ZERO", "active": False},
        {"status": "PASS"}, _module(IDENTITY), _module(EXACT_ZERO),
    )
    assert card["status"] == "PHYSICS_ROUTE_NOT_SUPPORTED"
    assert card["assembly"] is None


def test_c_exact_zero_intercept_accepts_pandas_series():
    assert _target_mean(pd.Series([1.0, 2.0, 3.0])) == 2.0
