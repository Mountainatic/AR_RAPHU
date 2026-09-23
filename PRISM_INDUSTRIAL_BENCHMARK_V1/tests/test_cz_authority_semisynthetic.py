from __future__ import annotations

import numpy as np
import pandas as pd

from prism_benchmark.cz_authority_semisynthetic import (
    BLOCK_SHIFT_STEPS,
    _all_origins,
    _block_circular_shift,
)


def test_block_circular_shift_is_deterministic_segment_local_permutation() -> None:
    values = np.arange(1024, dtype=np.float64)
    shifted_a, offset_a = _block_circular_shift(
        values, np.random.default_rng(19), BLOCK_SHIFT_STEPS
    )
    shifted_b, offset_b = _block_circular_shift(
        values, np.random.default_rng(19), BLOCK_SHIFT_STEPS
    )
    assert offset_a == offset_b
    assert offset_a % BLOCK_SHIFT_STEPS == 0
    assert np.array_equal(shifted_a, shifted_b)
    assert np.array_equal(np.sort(shifted_a), values)


def test_all_origins_obeys_l256_h4_target_boundary() -> None:
    rows = 900
    base = pd.DataFrame(
        {
            "entity_id": "source_segment_0",
            "row_in_entity": np.arange(rows, dtype=np.int64),
        }
    )
    samples = _all_origins(base)
    assert int(samples["origin"].min()) == 256
    assert int(samples["origin"].max()) == rows - 4
    assert int((samples["origin"] + 3).max()) == rows - 1
    assert bool((samples["latest_available_target_index"] < samples["origin"]).all())


def test_partial_terminal_block_still_permits_nonzero_shift() -> None:
    values = np.arange(334, dtype=np.float64)
    shifted, offset = _block_circular_shift(
        values, np.random.default_rng(0), BLOCK_SHIFT_STEPS
    )
    assert offset == BLOCK_SHIFT_STEPS
    assert not np.array_equal(shifted, values)
    assert np.array_equal(np.sort(shifted), values)
