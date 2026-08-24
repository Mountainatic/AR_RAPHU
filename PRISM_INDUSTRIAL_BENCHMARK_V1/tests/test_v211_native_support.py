from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark import cz_k_support
from prism_benchmark.cz_k_support import (
    _available_profiles,
    _profile_support_availability,
)
from prism_benchmark.v211_support import (
    LEGACY_SUPPORT_ERROR,
    SUPPORT_CONTRACT,
    apply_native_support,
    intersection_support_by_base_origin_id,
    require_native_support_contract,
    support_id_hash,
)


def _anchors(count: int = 8) -> pd.DataFrame:
    origins = list(range(1, count + 1))
    return pd.DataFrame(
        {
            "base_origin_id": [f"b{value}" for value in origins],
            "view_sample_id": [f"v{value}" for value in origins],
            "entity_id": ["run-1"] * len(origins),
            "origin": origins,
            "dependency_start": [0] * len(origins),
            "causal_history_floor": [0] * len(origins),
            "anchor_history_steps": [1] * len(origins),
            "sample_support_contract": [SUPPORT_CONTRACT] * len(origins),
        }
    )


def test_short_k_history_keeps_its_native_early_anchors() -> None:
    anchors = _anchors()
    short = apply_native_support(anchors, 2)
    long = apply_native_support(anchors, 5)
    assert short["origin"].tolist() == list(range(2, 9))
    assert long["origin"].tolist() == list(range(5, 9))


def test_local_common_scoring_is_the_native_intersection() -> None:
    anchors = _anchors()
    common = intersection_support_by_base_origin_id(anchors, [2, 5])
    long = apply_native_support(anchors, 5)
    assert support_id_hash(common) == support_id_hash(long)


def test_legacy_shared_data_cannot_claim_native_support() -> None:
    legacy = _anchors().drop(columns=["causal_history_floor"])
    with pytest.raises(RuntimeError, match=LEGACY_SUPPORT_ERROR):
        require_native_support_contract(legacy)


def test_k_profile_with_empty_registered_fold_support_is_unavailable() -> None:
    anchors = _anchors(12)
    anchors["dependency_start"] = anchors["origin"]
    folds = [(np.arange(0, 9), np.arange(9, 12))]

    audit = _profile_support_availability(
        anchors,
        folds,
        [(1, 2), (1, 4)],
        fit_cap=100,
        evaluation_cap=100,
    )

    assert audit[0]["status"] == "AVAILABLE"
    assert audit[0]["scoring_rows_by_fold"] == [1]
    assert audit[1]["status"] == "UNAVAILABLE_BY_SUPPORT"
    assert audit[1]["zero_scoring_folds"] == [0]


def test_cz_available_profiles_uses_unpatched_registered_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"sample_id": ["x"]})
    monkeypatch.setattr(
        cz_k_support,
        "load_v211_configs",
        lambda project, protocol: (
            {},
            {"selection": {"inner_folds": 1}},
            {
                "row_caps": {
                    "single_channel_k_fit": 10,
                    "validation_selection_per_fold": 10,
                }
            },
        ),
    )
    monkeypatch.setattr(
        cz_k_support, "load_native_samples", lambda *args, **kwargs: frame
    )
    monkeypatch.setattr(
        cz_k_support, "inner_folds", lambda *args, **kwargs: [(None, None)]
    )
    monkeypatch.setattr(
        cz_k_support,
        "registered_channel_profiles",
        lambda *args, **kwargs: [(1, 2)],
    )
    monkeypatch.setattr(
        cz_k_support,
        "registered_fold_native_masks",
        lambda *args, **kwargs: [
            {"fit_native": frame, "evaluation_common": frame}
        ],
    )

    def recursive_provider(*args: object, **kwargs: object) -> object:
        raise AssertionError("mutable provider must not be called")

    monkeypatch.setattr(cz_k_support.v211_k, "channel_profiles", recursive_provider)
    profiles, audit = _available_profiles(
        Path("shared"), Path("project"), object(), "channel", "protocol"
    )
    assert profiles == [(1, 2)]
    assert audit[0]["status"] == "AVAILABLE"
