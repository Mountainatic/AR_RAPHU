from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.cpu_data import HeadSpec, ViewSpec
from prism_benchmark.c3_models import _narx_expand, _ridge_block_predict
from prism_benchmark.v211_public_all_baseline_materialization import (
    _arx_model,
    _expand_narx_block,
    _fit_narx_expansion,
    _fit_ridge_block_model,
    _narx_model,
    _ridge_blockwise_predict,
    _static_native,
    _static_model,
    _write,
    materialize_baseline_view,
)
from prism_benchmark.v211_public_all_closure import FINAL_SUCCESS_STATUSES
from prism_benchmark.v211_public_all_materialization import (
    _prior_test_residuals,
    _write_prediction,
)
from prism_benchmark.v211_support import SUPPORT_CONTRACT


def _view(information_set: str = "input_only") -> ViewSpec:
    return ViewSpec(
        head=HeadSpec(
            head_id="TEST_HEAD__H1__W1",
            task_id="TEST_HEAD",
            dataset="synthetic",
            target="y",
            cadence_seconds=600.0,
            h_steps=1,
            w_steps=1,
            w0_steps=1,
            primary=True,
        ),
        information_set=information_set,
        availability_scenario="record_time",
        proxy_policy="primary",
    )


def _samples(split: str, rows: int = 3) -> pd.DataFrame:
    origins = np.arange(10, 10 + rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "base_origin_id": [f"b{value}" for value in origins],
            "view_sample_id": [f"v{value}" for value in origins],
            "dataset": "synthetic",
            "entity_id": "entity=1",
            "task_id": "TEST_HEAD",
            "target_head": "TEST_HEAD__H1__W1",
            "split": split,
            "origin": origins,
            "latest_available_target_index": origins - 1,
            "y_true": origins.astype(np.float64),
            "causal_history_floor": np.zeros(rows, dtype=np.int64),
            "anchor_history_steps": np.ones(rows, dtype=np.int64),
            "sample_support_contract": SUPPORT_CONTRACT,
        }
    )


def test_materialization_inventory_keeps_simple_and_protocol_not_run_models(
    monkeypatch,
) -> None:
    from prism_benchmark import v211_public_all_baseline_materialization as module

    monkeypatch.setattr(
        module,
        "_result",
        lambda *args: {
            "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE",
            "reason": "TEST_PROTOCOL_REASON",
        },
    )
    audits = materialize_baseline_view(SimpleNamespace(), _view())
    assert [item["model"] for item in audits[:3]] == [
        "MEAN",
        "PERSISTENCE",
        "SEASONAL_PERSISTENCE",
    ]
    assert all(item["status"] in FINAL_SUCCESS_STATUSES for item in audits)


def test_static_materialization_uses_the_requested_evaluation_accessor(
    monkeypatch,
) -> None:
    from prism_benchmark import v211_public_all_baseline_materialization as module

    accessor_splits: list[str] = []

    class StubAccessor:
        def __init__(self, shared, dataset, split, columns) -> None:
            del shared, dataset, columns
            accessor_splits.append(split)

        def snapshot(self, samples, columns) -> np.ndarray:
            del columns
            return np.ones((len(samples), 1), dtype=np.float64)

    monkeypatch.setattr(module, "BaseAccessor", StubAccessor)
    monkeypatch.setattr(module, "_static_native", lambda *args: _samples("validation"))
    monkeypatch.setattr(module, "_common_test", lambda *args: _samples("test"))
    monkeypatch.setattr(module, "input_columns", lambda *args: ["x"])
    monkeypatch.setattr(module, "_freeze", lambda *args: {})
    monkeypatch.setattr(
        module,
        "_ridge_predictions",
        lambda x_fit, y_fit, x_test, alpha: (
            np.zeros(len(x_test), dtype=np.float64),
            2,
        ),
    )
    paths = SimpleNamespace(project=Path("project"), shared=Path("shared"))
    _static_model(
        paths,
        _view(),
        "RIDGE",
        {"selection": {"selected_alpha": 1.0}},
        split="test",
    )
    assert accessor_splits == ["validation", "test"]


def test_static_materialization_uses_frozen_model_specific_fit_caps(
    monkeypatch,
) -> None:
    from prism_benchmark import v211_public_all_baseline_materialization as module

    observed_caps: list[int] = []
    monkeypatch.setattr(
        module,
        "_development",
        lambda *args: _samples("validation", rows=10),
    )
    monkeypatch.setattr(
        module,
        "_freeze",
        lambda *args: {
            "selection": {
                "fit_row_cap_default": 250_000,
                "fit_row_cap_svr": 5_000,
                "fit_row_cap_xgboost": 100_000,
            }
        },
    )

    def capture_cap(fit, cap):
        observed_caps.append(cap)
        return fit

    monkeypatch.setattr(module, "_cap_after_support", capture_cap)
    paths = SimpleNamespace(project=Path("project"), shared=Path("shared"))

    for model in ("RIDGE", "RBF_SVR", "XGBOOST"):
        _static_native(paths, _view(), model)

    assert observed_caps == [250_000, 5_000, 100_000]


def test_blockwise_ridge_prediction_matches_frozen_reference(monkeypatch) -> None:
    from prism_benchmark import v211_public_all_baseline_materialization as module

    monkeypatch.setattr(module, "release_process_memory", lambda: None)
    rng = np.random.default_rng(20260817)
    x_train = rng.normal(size=(37, 7))
    y_train = rng.normal(size=37)
    x_eval = rng.normal(size=(23, 7))
    penalties = np.linspace(0.01, 0.07, x_train.shape[1])

    expected, expected_certificate = _ridge_block_predict(
        x_train, y_train, x_eval, penalties
    )
    scaler, coefficient, y_mean, certificate = _fit_ridge_block_model(
        x_train, y_train, penalties
    )
    observed = _ridge_blockwise_predict(
        iter((x_eval[:5], x_eval[5:17], x_eval[17:])),
        len(x_eval),
        scaler,
        coefficient,
        y_mean,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    assert certificate["solver"] == expected_certificate["solver"]
    np.testing.assert_allclose(
        certificate["relative_kkt"], expected_certificate["relative_kkt"]
    )
    np.testing.assert_allclose(
        certificate["condition_number"], expected_certificate["condition_number"]
    )


def test_blockwise_narx_expansion_matches_frozen_reference() -> None:
    rng = np.random.default_rng(20260817)
    raw_train = rng.normal(size=(41, 9))
    y_train = rng.normal(size=41)
    raw_eval = rng.normal(size=(19, 9))

    expected_train, expected_eval, expected_order = _narx_expand(
        raw_train, y_train, raw_eval, 5
    )
    observed_train, scaler, observed_order = _fit_narx_expansion(raw_train, y_train, 5)
    observed_eval = _expand_narx_block(raw_eval, scaler, observed_order)

    np.testing.assert_array_equal(observed_order, expected_order)
    np.testing.assert_allclose(observed_train, expected_train, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed_eval, expected_eval, rtol=0.0, atol=0.0)


def test_arx_and_narx_materialization_bound_test_feature_frames(
    monkeypatch,
) -> None:
    from prism_benchmark import v211_public_all_baseline_materialization as module

    fit = _samples("validation", rows=3)
    test = _samples("test", rows=11)
    observed_test_rows: list[int] = []

    class StubAccessor:
        def __init__(self, *args) -> None:
            del args

    def arx_features(accessor, samples, view, columns, profile, maximum_lags):
        del accessor, view, columns, profile, maximum_lags
        if samples["split"].iloc[0] == "test":
            observed_test_rows.append(len(samples))
        origin = samples["origin"].to_numpy(dtype=np.float64)
        return np.column_stack((origin, origin * 0.5 + 1.0, np.ones(len(samples)))), 1

    monkeypatch.setattr(module, "_MATERIALIZATION_PREDICTION_BLOCK_ROWS", 4)
    monkeypatch.setattr(module, "release_process_memory", lambda: None)
    monkeypatch.setattr(module, "_development", lambda *args: fit)
    monkeypatch.setattr(module, "_cap_after_support", lambda samples, cap: samples)
    monkeypatch.setattr(module, "_common_test", lambda *args: test)
    monkeypatch.setattr(module, "input_columns", lambda *args: ["x"])
    monkeypatch.setattr(module, "BaseAccessor", StubAccessor)
    monkeypatch.setattr(module, "_arx_features", arx_features)
    monkeypatch.setattr(
        module,
        "_freeze",
        lambda *args: {
            "selection": {"fit_row_cap_default": 100},
            "c3": {
                "arx": {"maximum_input_lags_per_channel": 2},
                "linear_narx": {"maximum_linear_state_features_before_expansion": 2},
            },
        },
    )
    paths = SimpleNamespace(project=Path("project"), shared=Path("shared"))
    view = _view("dynamic")

    arx_prediction, _, _, _ = _arx_model(
        paths,
        view,
        "ARX",
        {
            "selection": {
                "selected_profile": [1, 1],
                "ar_alpha": 0.1,
                "selected_x_penalty_ratio": 1.0,
            }
        },
        split="test",
    )
    assert len(arx_prediction) == len(test)
    assert observed_test_rows == [4, 4, 3]

    observed_test_rows.clear()
    narx_prediction, _, _, _ = _narx_model(
        paths,
        view,
        {
            "selection": {
                "selected_profile": [1, 1],
                "selected_alpha": 0.1,
            }
        },
        split="test",
    )
    assert len(narx_prediction) == len(test)
    assert observed_test_rows == [4, 4, 3]


def test_prediction_writer_persists_real_parameter_count(tmp_path: Path) -> None:
    paths = SimpleNamespace(final=tmp_path / "final", run_root=tmp_path)
    samples = _samples("test")
    result = _write(
        paths,
        _view(),
        "RIDGE",
        samples,
        np.array([1.0, 2.0, 3.0]),
        _samples("validation"),
        0.0,
        selection={"selected_alpha": 1.0},
        parameter_count=7,
        split="test",
    )
    prediction_path = tmp_path / result["prediction_path"]
    frame = pd.read_parquet(prediction_path)
    assert result["parameter_count"] == 7
    assert frame["parameter_count"].unique().tolist() == [7]
    assert len(result["prediction_sha256"]) == 64


def test_prediction_writer_routes_ood_separately(tmp_path: Path) -> None:
    paths = SimpleNamespace(final=tmp_path / "final", run_root=tmp_path)
    result = _write(
        paths,
        _view(),
        "RIDGE",
        _samples("ood"),
        np.array([1.0, 2.0, 3.0]),
        _samples("validation"),
        0.0,
        selection={"selected_alpha": 1.0},
        parameter_count=7,
        split="ood",
    )
    assert result["prediction_path"].startswith("final/baseline_ood_predictions/")
    assert result["test_accessed"] is False
    assert result["ood_accessed"] is True


def test_prism_prediction_writer_routes_ood_separately(tmp_path: Path) -> None:
    paths = SimpleNamespace(final=tmp_path / "final", run_root=tmp_path)
    result = _write_prediction(
        paths,
        _view(),
        "PRISM_V2_1_1_K_C",
        _samples("ood"),
        np.array([1.0, 2.0, 3.0]),
        7,
        0.0,
        _samples("validation"),
        split="ood",
    )
    assert result["prediction_path"].startswith("final/ood_predictions/")
    assert result["test_accessed"] is False
    assert result["ood_accessed"] is True


def test_ood_state_reuses_frozen_test_residuals(tmp_path: Path) -> None:
    paths = SimpleNamespace(final=tmp_path / "final", run_root=tmp_path)
    _write_prediction(
        paths,
        _view(),
        "PRISM_V2_1_1_K_C_DYNAMIC",
        _samples("test"),
        np.array([1.0, 2.0, 3.0]),
        7,
        0.0,
        _samples("validation"),
        split="test",
    )
    residuals = _prior_test_residuals(
        paths,
        _view(),
        "ood",
        "PRISM_V2_1_1_K_C_DYNAMIC",
    )
    assert residuals["residual"].tolist() == [9.0, 9.0, 9.0]
    assert _prior_test_residuals(
        paths,
        _view(),
        "test",
        "PRISM_V2_1_1_K_C_DYNAMIC",
    ).empty
