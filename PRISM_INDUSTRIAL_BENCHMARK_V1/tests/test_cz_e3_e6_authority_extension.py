from __future__ import annotations

import numpy as np
import pandas as pd

from prism_benchmark.cz_e3_e6_authority_extension import (
    BLOCK_EMBARGO,
    BLOCK_ROWS,
    CZ_HISTORIES,
    _four_embargoed_blocks,
)
from prism_benchmark.e1e6_phase_b import _e3_candidates


def test_single_channel_e3_candidate_universes_are_identical() -> None:
    uniform = set(_e3_candidates(("joint_lift",), CZ_HISTORIES, 7, "UNIFORM_SCALE"))
    multiscale = set(
        _e3_candidates(
            ("joint_lift",), CZ_HISTORIES, 7, "CHANNEL_SPECIFIC_MULTISCALE"
        )
    )
    assert uniform == multiscale


def test_four_blocks_have_registered_embargo() -> None:
    rows = 4 * BLOCK_ROWS + 3 * BLOCK_EMBARGO + 100
    frame = pd.DataFrame(
        {
            "origin": np.arange(rows, dtype=np.int64),
            "base_origin_id": [f"r{value}" for value in range(rows)],
        }
    )
    selected, labels = _four_embargoed_blocks(frame)
    assert len(selected) == 4 * BLOCK_ROWS
    assert set(labels) == {0, 1, 2, 3}
    for group in range(3):
        left = selected.loc[labels == group, "origin"].max()
        right = selected.loc[labels == group + 1, "origin"].min()
        assert right - left > BLOCK_EMBARGO
