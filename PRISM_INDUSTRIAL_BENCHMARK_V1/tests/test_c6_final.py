from __future__ import annotations

import numpy as np

import pandas as pd

from prism_benchmark.c6_final import _entity_groups, _holm, _markdown_table, _paired_bootstrap, _paired_frames


def _legacy_paired_bootstrap(diff: np.ndarray, entities: np.ndarray, block: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = pd.unique(entities)
    entity_blocks = {
        str(entity): [values[start : start + block] for start in range(0, len(values), block)]
        for entity in unique
        for values in [diff[entities == entity]]
    }
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_entities = rng.choice(unique, size=len(unique), replace=True) if len(unique) > 1 else unique
        total = 0.0
        count = 0
        for entity in selected_entities:
            blocks = entity_blocks[str(entity)]
            selected = rng.integers(0, len(blocks), size=len(blocks))
            for index in selected:
                values = blocks[int(index)]
                total += float(np.sum(values, dtype=np.float64))
                count += len(values)
        draws[replicate] = total / max(count, 1)
    return draws


def test_paired_bootstrap_preserves_positive_difference() -> None:
    diff = np.ones(100, dtype=np.float64)
    entities = np.array(["a"] * 50 + ["b"] * 50)
    draws = _paired_bootstrap(diff, entities, 10, 50, 7)
    np.testing.assert_allclose(draws, 1.0)


def test_grouped_bootstrap_matches_legacy_draws() -> None:
    rng = np.random.default_rng(42)
    entities = np.array(["z"] * 13 + ["a"] * 8 + ["m"] * 17)
    diff = rng.normal(size=len(entities))
    expected = _legacy_paired_bootstrap(diff, entities, 5, 40, 99)
    actual = _paired_bootstrap(diff, entities, 5, 40, 99)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_entity_groups_keep_first_seen_order_and_positions() -> None:
    entities = np.array(["z", "a", "z", "m", "a"])
    labels, groups = _entity_groups(entities)
    np.testing.assert_array_equal(labels, ["z", "a", "m"])
    np.testing.assert_array_equal(groups[0], [0, 2])
    np.testing.assert_array_equal(groups[1], [1, 4])
    np.testing.assert_array_equal(groups[2], [3])


def test_paired_frames_skip_sort_when_order_matches_and_recover_when_needed() -> None:
    left = pd.DataFrame({"sample_id": ["b", "a"], "value": [2, 1]})
    same = pd.DataFrame({"sample_id": ["b", "a"], "value": [20, 10]})
    actual_left, actual_same = _paired_frames(left, same)
    assert actual_left is left
    assert actual_same is same
    reversed_frame = same.iloc[::-1]
    ordered_left, ordered_right = _paired_frames(left, reversed_frame)
    np.testing.assert_array_equal(ordered_left["sample_id"], ordered_right["sample_id"])


def test_holm_is_monotone_and_bounded() -> None:
    rows = [{"p_value": 0.01}, {"p_value": 0.03}, {"p_value": 0.2}]
    _holm(rows, 0.05)
    adjusted = [row["holm_adjusted_p"] for row in rows]
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_markdown_table_has_no_optional_dependency_and_escapes_cells() -> None:
    frame = pd.DataFrame({"name": ["a|b"], "score": [1.5]})
    rendered = _markdown_table(frame)
    assert "| name | score |" in rendered
    assert "| a\\|b | 1.5 |" in rendered
