from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism_benchmark.cz_authority_semisynthetic import (
    BLOCK_SHIFT_STEPS,
    _all_origins,
    _block_circular_shift,
    _load_s2_registry,
    _registered_linear_signal,
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


def test_s2_truth_registry_is_invalidated_and_loader_refuses_it() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "cz_raw2s_h4_e2_truth_operator_registry_v1.json"
    )
    registry = json.loads(path.read_text(encoding="utf-8"))
    assert registry["status"] == "INVALIDATED_DO_NOT_RUN_OR_CITE"
    with pytest.raises(RuntimeError, match="TRUTH_REGISTRY_HASH_MISMATCH"):
        _load_s2_registry(path)


def test_registered_linear_signal_applies_only_frozen_interval_weights(monkeypatch) -> None:
    values = np.arange(24, dtype=np.float64).reshape(3, 8)
    intervals = [(4 * index, 4 * (index + 1)) for index in range(8)]

    def fake_profile_values(*args, **kwargs):
        return values, intervals

    monkeypatch.setattr(
        "prism_benchmark.cz_authority_semisynthetic.profile_values",
        fake_profile_values,
    )
    signal, actual_intervals = _registered_linear_signal(
        object(),
        pd.DataFrame(),
        channel="main_heater_power",
        profile=(4, 32),
        m_tau=8,
        interval_weights=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert np.array_equal(signal, values[:, 0])
    assert actual_intervals == intervals
