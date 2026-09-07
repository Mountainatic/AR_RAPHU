import json

import numpy as np
import pandas as pd

from experiments.tim_validation.e2_stagewise.synthetic_c1 import (
    DECOY_CHANNEL,
    MAXIMUM_REGISTERED_HISTORY,
    REGIMES,
    TARGET,
    build_development,
    generate_series,
    view,
)


def test_truth_vectors_follow_native_stage_semantics() -> None:
    expected = (
        {"K": True, "C": False, "W": False, "A": False},
        {"K": True, "C": True, "W": False, "A": False},
        {"K": True, "C": True, "W": True, "A": False},
        {"K": True, "C": True, "W": True, "A": True},
    )
    for regime, vector in zip(REGIMES, expected, strict=True):
        generated = generate_series(regime, 7, n_samples=500)
        assert generated.truth["stage_truth"] == vector
        assert generated.truth["inverse_crime"] is False
        assert DECOY_CHANNEL not in generated.truth["active_channels"]


def test_correlated_decoy_and_chronological_splits() -> None:
    generated = generate_series(REGIMES[-1], 2, n_samples=500)
    correlation = np.corrcoef(
        generated.frame["x0"], generated.frame[f"x{DECOY_CHANNEL}"]
    )[0, 1]
    assert 0.9 < correlation < 1.0
    train, validation, test = (
        generated.sample_origins[name] for name in ("train", "validation", "test")
    )
    assert train[-1] < validation[0] < validation[-1] < test[0]
    assert train[0] == MAXIMUM_REGISTERED_HISTORY


def test_development_c1_does_not_materialize_test(tmp_path) -> None:
    shared = tmp_path / "shared"
    result = build_development(shared, REGIMES[0], 0, n_samples=500)
    assert result["status"] == "PASS"
    assert result["test_materialized"] is False
    truth = json.loads((shared / "GENERATOR_TRUTH.json").read_text(encoding="utf-8"))
    assert truth["test_materialization"] == "after_selection_and_checkpoint_seal_only"
    for information_set in ("input_only", "dynamic"):
        root = shared / "sample_ids" / view(information_set).relative_root
        assert (root / "train.parquet").is_file()
        assert (root / "validation.parquet").is_file()
        assert not (root / "test.parquet").exists()
        samples = pd.read_parquet(root / "train.parquet")
        assert np.all(samples["latest_available_target_index"] < samples["origin"])
        assert np.all(samples["dependency_stop_exclusive"] == samples["origin"] + 1)
    assert not (shared / "base_data" / "debutanizer" / "test.parquet").exists()
    assert TARGET in pd.read_parquet(
        shared / "base_data" / "debutanizer" / "train.parquet"
    ).columns
