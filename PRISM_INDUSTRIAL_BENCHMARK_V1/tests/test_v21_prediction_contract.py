import pytest
import json

from prism_benchmark.v21_config import V21Paths
from prism_benchmark.v21_final import run_e7_test
from prism_benchmark.v21_selection import assert_final_prediction_contract


def test_prediction_contract_rejects_missing_materialized_path():
    result = {"final_selected_candidate": "K", "final_selected_fold_losses": [1.0, 0.9, 0.8], "final_selected_prediction_path": "", "final_selected_contract": {"family": "K"}}
    with pytest.raises(RuntimeError, match="path is empty"):
        assert_final_prediction_contract(result)


def test_prediction_contract_accepts_one_materialized_selection():
    result = {"final_selected_candidate": "K", "final_selected_fold_losses": [1.0, 0.9, 0.8], "final_selected_prediction_path": "DEVELOPMENT/K/validation.parquet", "final_selected_contract": {"family": "K"}}
    assert_final_prediction_contract(result)


def test_e7_failure_leaves_one_time_access_sentinel(tmp_path, monkeypatch):
    paths = V21Paths(
        project=tmp_path / "project",
        shared=tmp_path / "shared",
        output=tmp_path / "results_prism_v2_1_sru",
    )
    freeze = paths.output / "FREEZE" / "V21_SRU_FINAL_FREEZE_MANIFEST.json"
    freeze.parent.mkdir(parents=True)
    freeze.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "prism_benchmark.v21_final.require_test_freeze",
        lambda _paths: {"code_commit": "commit"},
    )
    monkeypatch.setattr(
        "prism_benchmark.v21_final.sru_input_views",
        lambda _shared: [object()],
    )
    monkeypatch.setattr(
        "prism_benchmark.v21_final._fit_final_input_view",
        lambda _paths, _view: (_ for _ in ()).throw(RuntimeError("failure")),
    )
    with pytest.raises(RuntimeError, match="failure"):
        run_e7_test(paths)
    sentinel = json.loads(
        (paths.output / "FINAL" / "TEST_ACCESS_AUDIT.json").read_text(
            encoding="utf-8"
        )
    )
    assert sentinel["status"] == "TEST_ACCESS_STARTED"
    assert sentinel["test_accessed"] is True
