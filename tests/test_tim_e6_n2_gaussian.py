from __future__ import annotations

import numpy as np
import pandas as pd

import json

from experiments.tim_validation.e6_robustness.n2_gaussian import (
    _full_record,
    _jaccard,
    _perturb_split,
)


def test_n2_development_and_test_streams_are_independent_and_nested(tmp_path) -> None:
    frame = pd.DataFrame(
        {"x": np.arange(8.0), "target": np.arange(8.0), "entity_id": "e", "row_in_entity": range(8)}
    )
    source = tmp_path / "test.parquet"
    frame.to_parquet(source, index=False)
    dev_low = tmp_path / "dev_low.parquet"
    dev_high = tmp_path / "dev_high.parquet"
    test_low = tmp_path / "test_low.parquet"
    for destination, alpha, realization in (
        (dev_low, 0.05, "development"),
        (dev_high, 0.10, "development"),
        (test_low, 0.05, "test_independent"),
    ):
        _perturb_split(
            source, destination, task="T", seed=9, alpha=alpha,
            sigma={"x": 1.0}, realization=realization, target="target",
        )
    low = pd.read_parquet(dev_low)["x"].to_numpy() - frame["x"].to_numpy()
    high = pd.read_parquet(dev_high)["x"].to_numpy() - frame["x"].to_numpy()
    independent = pd.read_parquet(test_low)["x"].to_numpy() - frame["x"].to_numpy()
    assert np.allclose(high, 2.0 * low)
    assert not np.allclose(independent, low)
    assert pd.read_parquet(dev_low)["target"].equals(frame["target"])


def test_n2_empty_agreement_is_not_unqualified() -> None:
    assert _jaccard(set(), set()) == (1.0, True)


def test_n2_uses_terminal_input_only_model_when_full_model_is_absent(tmp_path) -> None:
    path = tmp_path / "STAGEWISE_INFERENCE_COMPLETE.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {"status": "PASS", "model": "PRISM_V2_1_1_K"},
                    {"status": "PASS", "model": "PRISM_V2_1_1_K_C_W"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _full_record(path)["model"] == "PRISM_V2_1_1_K_C_W"
