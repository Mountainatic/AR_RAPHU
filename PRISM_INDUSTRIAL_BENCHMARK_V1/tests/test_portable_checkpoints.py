from __future__ import annotations

import numpy as np
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
