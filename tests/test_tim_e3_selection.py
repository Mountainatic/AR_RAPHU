from __future__ import annotations

import pytest

from experiments.tim_validation.e3_multiscale.selection import (
    parse_profile_losses,
    select_uniform_history,
)


def _channel(name: str, short: list[float], long: list[float]) -> dict:
    return {
        "channel": name,
        "profile_fold_losses": {
            "(2, 8)": short,
            "(4, 8)": [value + 0.1 for value in short],
            "(2, 32)": long,
        },
    }


def test_parse_profile_losses_rejects_non_profile_key() -> None:
    with pytest.raises(ValueError, match="invalid profile key"):
        parse_profile_losses({"profile_fold_losses": {"8": [1.0]}})


def test_uniform_history_uses_fold_losses_and_prefers_simpler_one_se() -> None:
    result = select_uniform_history(
        [
            _channel("x1", [1.00, 1.10, 0.90, 1.00], [0.99, 1.09, 0.89, 0.99]),
            _channel("x2", [0.80, 0.90, 0.70, 0.80], [0.79, 0.89, 0.69, 0.79]),
        ],
        maximum_relative_regret=0.02,
        minimum_usable_folds=3,
    )
    assert result["best_history"] == 32
    assert result["selected_history"] == 8
    assert result["test_accessed"] is False


def test_uniform_history_requires_common_histories() -> None:
    first = _channel("x1", [1.0] * 4, [0.9] * 4)
    second = {
        "channel": "x2",
        "profile_fold_losses": {"(2, 16)": [1.0] * 4},
    }
    with pytest.raises(ValueError, match="no common"):
        select_uniform_history(
            [first, second],
            maximum_relative_regret=0.02,
            minimum_usable_folds=3,
        )
