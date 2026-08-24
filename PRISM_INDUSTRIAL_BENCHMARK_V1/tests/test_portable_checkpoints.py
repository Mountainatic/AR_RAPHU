from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.level_reconstruction import metric_bundle_delta_and_level
from prism_benchmark.portable_checkpoints import (
    INFERENCE_ONLY_ENV,
    assert_fitting_allowed,
    fit_pls_codec,
    fit_rbf_svr_codec,
    fit_standardized_ridge,
    load_portable_checkpoint,
    predict_codec,
    write_portable_checkpoint,
)
from prism_benchmark.representative_formal import (
    _assert_no_out_of_scope_artifacts,
    _rankings,
    _support_acceptance,
)
from prism_benchmark import representative_baseline_checkpoints as baseline_checkpoints
from prism_benchmark.representative_prism_checkpoints import _predict_joint


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260824)
    x = rng.normal(size=(96, 5))
    y = x @ np.asarray([0.5, -1.0, 0.25, 0.0, 0.75]) + 0.1
    return x, y


@pytest.mark.parametrize("family", ["ridge", "pls", "svr"])
def test_portable_codec_round_trip(tmp_path, monkeypatch, family: str) -> None:
    x, y = _fixture()
    if family == "ridge":
        metadata, arrays = fit_standardized_ridge(x, y, 0.01)
    elif family == "pls":
        pytest.importorskip("sklearn")
        metadata, arrays = fit_pls_codec(x, y, 3)
    else:
        pytest.importorskip("sklearn")
        metadata, arrays = fit_rbf_svr_codec(
            x, y, c_value=2.0, gamma=0.125, epsilon=0.01
        )
    expected = predict_codec(x, metadata, arrays)
    root = tmp_path / family
    written = write_portable_checkpoint(root, metadata, arrays)
    monkeypatch.setenv(INFERENCE_ONLY_ENV, "1")
    loaded_metadata, loaded_arrays, manifest = load_portable_checkpoint(root)
    observed = predict_codec(x, loaded_metadata, loaded_arrays)
    assert manifest["checkpoint_hash"] == written["checkpoint_hash"]
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)


def test_inference_fit_guard_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv(INFERENCE_ONLY_ENV, "1")
    with pytest.raises(RuntimeError, match="FIT_REFIT_SELECT_FORBIDDEN"):
        assert_fitting_allowed()


def test_level_delta_metric_identity_and_persistence() -> None:
    current = np.asarray([10.0, 11.0, 10.5, 12.0])
    truth = np.asarray([0.5, -0.25, 1.0, 0.0])
    prediction = np.asarray([0.4, -0.1, 0.8, 0.1])
    result = metric_bundle_delta_and_level(truth, prediction, current)
    assert result["mse"] == pytest.approx(result["mse_delta"], abs=1e-15)
    assert result["rmse"] == pytest.approx(result["rmse_delta"], abs=1e-15)
    assert result["mae"] == pytest.approx(result["mae_delta"], abs=1e-15)
    assert "r2_level_reconstructed" in result
    assert "r2_delta" in result
    assert "r2_level_persistence" in result
    assert "persistence_skill" in result
    assert "variance_ratio" in result


def _formal_record(model: str, level_r2: float, support: str = "support") -> dict:
    return {
        "status": "PASS",
        "namespace": "public",
        "dataset": "sru",
        "target_head": "SRU_H2S_REP_H1__H1__W1",
        "information_set": "input_only",
        "availability_scenario": "record_time",
        "proxy_policy": "primary",
        "model": model,
        "rows": 10,
        "scoring_support_hash": support,
        "sample_id_order_hash": "order",
        "r2_level_reconstructed": level_r2,
        "r2_delta": level_r2 - 0.1,
    }


def test_rankings_are_separate_per_information_set_and_view() -> None:
    rankings = _rankings([_formal_record("A", 0.5), _formal_record("B", 0.8)])
    assert len(rankings["input_only"]) == 1
    assert [row["model"] for row in rankings["input_only"][0]["leaderboard"]] == [
        "B",
        "A",
    ]
    assert rankings["dynamic"] == []


def test_support_acceptance_rejects_method_row_or_order_drift() -> None:
    records = [_formal_record("A", 0.5), _formal_record("B", 0.6, "different")]
    with pytest.raises(RuntimeError, match="SUPPORT_MISMATCH"):
        _support_acceptance(records)


def test_out_of_scope_artifact_guard(tmp_path) -> None:
    _assert_no_out_of_scope_artifacts(tmp_path)
    (tmp_path / "stage2_output.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="OUT_OF_SCOPE_ARTIFACT"):
        _assert_no_out_of_scope_artifacts(tmp_path)


def test_joint_replay_uses_frozen_slice_order_not_json_key_order() -> None:
    blocks = {
        "K": np.array([[1.0], [2.0]]),
        "W": np.array([[10.0], [20.0]]),
        "A": np.array([[100.0], [200.0]]),
    }
    contract = {
        # Deliberately mimic alphabetically sorted checkpoint JSON.
        "block_slices": {"A": [2, 3], "K": [0, 1], "W": [1, 2]},
        "blocks": {
            name: {"mean": [0.0], "scale": [1.0]} for name in blocks
        },
        "coefficient": [1.0, 2.0, 3.0],
        "intercept": 4.0,
    }
    observed = _predict_joint(blocks, contract)
    expected = np.array([325.0, 646.0])
    assert np.array_equal(observed, expected)


def test_target_state_checkpoint_replay_materializes_target_column(monkeypatch) -> None:
    requested: dict[str, list[str]] = {}

    class StubAccessor:
        def __init__(self, shared, dataset, split, columns):
            requested["columns"] = list(columns)

        def target_state(self, samples, target, *profile):
            assert target == "crystal_diameter"
            return np.zeros((len(samples), 2), dtype=np.float64)

    monkeypatch.setattr(baseline_checkpoints, "BaseAccessor", StubAccessor)
    paths = SimpleNamespace(shared=object())
    view = SimpleNamespace(
        head=SimpleNamespace(dataset="cz", target="crystal_diameter")
    )
    state = {
        "feature": {
            "family": "TARGET_STATE",
            # Common development metadata can retain non-target columns.
            "columns": ["pull_rate", "heater_power"],
            "profile": [1, 2],
        }
    }
    observed = baseline_checkpoints._evaluation_features(
        paths,
        view,
        pd.DataFrame(index=range(3)),
        "test",
        state,
        {},
    )
    assert requested["columns"] == [
        "crystal_diameter",
        "pull_rate",
        "heater_power",
    ]
    assert observed.shape == (3, 2)
