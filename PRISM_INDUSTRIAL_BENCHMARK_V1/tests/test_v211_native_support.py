from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.v211_support import (
    LEGACY_SUPPORT_ERROR,
    SUPPORT_CONTRACT,
    apply_native_support,
    intersection_support_by_base_origin_id,
    require_native_support_contract,
    support_id_hash,
)


def _anchors() -> pd.DataFrame:
    origins = list(range(1, 9))
    return pd.DataFrame(
        {
            "base_origin_id": [f"b{value}" for value in origins],
            "view_sample_id": [f"v{value}" for value in origins],
            "origin": origins,
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
