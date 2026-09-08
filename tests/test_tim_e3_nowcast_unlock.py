from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.tim_validation.e3_multiscale.nowcast_unlock import (
    COMMON_SUPPORT_HISTORY_STEPS,
    HEAD,
    _sample_frame,
    _stable_id,
)


def test_sample_frame_matches_frozen_combined_nowcast_identity_contract() -> None:
    support = pd.DataFrame(
        {
            "entity_id": ["run-1", "run-1"],
            "origin": [256, 257],
            "causal_history_floor": [0, 0],
        }
    )
    index = pd.MultiIndex.from_tuples(
        [("run-1", 255), ("run-1", 256), ("run-1", 257)]
    )
    lookup = pd.Series(np.asarray([1.0, 1.25, 2.0]), index=index)
    frame = _sample_frame(support, lookup, "dynamic", "record_time", 0)

    assert frame["base_origin_id"].tolist() == [
        _stable_id("NOWCAST_H0_BASE", "tep", "run-1", 256),
        _stable_id("NOWCAST_H0_BASE", "tep", "run-1", 257),
    ]
    assert frame["view_sample_id"].iloc[0] == _stable_id(
        "NOWCAST_H0_VIEW",
        frame["base_origin_id"].iloc[0],
        HEAD,
        "dynamic",
        "record_time",
        "proxy_excluded",
    )
    assert frame["y_true"].tolist() == [0.25, 0.75]
    assert (frame["dependency_start"] == frame["origin"] - 256).all()
    assert (frame["latest_available_target_index"] < frame["origin"]).all()
    assert (frame["lmax_steps"] == COMMON_SUPPORT_HISTORY_STEPS).all()


def test_maturity_delay_uses_nested_stricter_support() -> None:
    support = pd.DataFrame(
        {
            "entity_id": ["run-1", "run-1"],
            "origin": [260, 261],
            "causal_history_floor": [0, 0],
        }
    )
    index = pd.MultiIndex.from_tuples(
        [("run-1", 260), ("run-1", 261)]
    )
    lookup = pd.Series(np.asarray([1.0, 2.0]), index=index)
    frame = _sample_frame(
        support, lookup, "dynamic", "analyzer_maturity_5_steps", 5
    )
    assert frame["origin"].tolist() == [261]
    assert frame["latest_available_target_index"].tolist() == [255]
