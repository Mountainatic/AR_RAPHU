"""Validation-only selection for the E3 uniform-history arm."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def parse_profile_losses(result: Mapping[str, Any]) -> dict[tuple[int, int], list[float]]:
    parsed: dict[tuple[int, int], list[float]] = {}
    for raw_profile, raw_losses in result["profile_fold_losses"].items():
        profile = ast.literal_eval(str(raw_profile))
        if not isinstance(profile, tuple) or len(profile) != 2:
            raise ValueError(f"invalid profile key: {raw_profile}")
        parsed[tuple(int(value) for value in profile)] = [
            float(value) for value in raw_losses
        ]
    return parsed


def select_uniform_history(
    channel_results: Sequence[Mapping[str, Any]],
    *,
    maximum_relative_regret: float,
    minimum_usable_folds: int,
) -> dict[str, Any]:
    """Select one history for every channel without consulting outer test data."""
    if not channel_results:
        raise ValueError("at least one channel result is required")
    if maximum_relative_regret < 0.0:
        raise ValueError("maximum_relative_regret must be nonnegative")

    by_channel = {
        str(result["channel"]): parse_profile_losses(result)
        for result in channel_results
    }
    common_histories = set.intersection(
        *[
            {int(profile[1]) for profile in losses}
            for losses in by_channel.values()
        ]
    )
    if not common_histories:
        raise ValueError("channels have no common registered history")

    history_fold_losses: dict[int, list[float]] = {}
    chosen_profiles: dict[int, dict[str, list[int]]] = {}
    for history in sorted(common_histories):
        channel_vectors = []
        chosen_profiles[history] = {}
        for channel, profile_losses in by_channel.items():
            candidates = {
                profile: np.asarray(losses, dtype=np.float64)
                for profile, losses in profile_losses.items()
                if int(profile[1]) == history
            }
            usable = {
                profile: losses
                for profile, losses in candidates.items()
                if int(np.isfinite(losses).sum()) >= minimum_usable_folds
            }
            if not usable:
                raise ValueError(
                    f"history {history} has no usable profile for {channel}"
                )
            selected = min(
                usable,
                key=lambda profile: (
                    float(np.mean(usable[profile][np.isfinite(usable[profile])])),
                    int(profile[0]),
                ),
            )
            vector = usable[selected]
            if not np.isfinite(vector).all():
                raise ValueError(
                    "E3 requires the same finite inner folds for every channel"
                )
            chosen_profiles[history][channel] = list(selected)
            channel_vectors.append(vector)
        aggregate = np.mean(np.stack(channel_vectors, axis=0), axis=0)
        history_fold_losses[history] = aggregate.tolist()

    means = {
        history: float(np.mean(losses, dtype=np.float64))
        for history, losses in history_fold_losses.items()
    }
    best = min(means, key=lambda history: (means[history], history))
    best_losses = np.asarray(history_fold_losses[best], dtype=np.float64)
    standard_error = (
        float(np.std(best_losses, ddof=1) / math.sqrt(len(best_losses)))
        if len(best_losses) > 1
        else 0.0
    )
    one_se_limit = means[best] + standard_error
    denominator = max(abs(means[best]), np.finfo(np.float64).eps)
    eligible = [
        history
        for history in sorted(means)
        if means[history] <= one_se_limit
        and (means[history] - means[best]) / denominator
        <= maximum_relative_regret
    ]
    selected_history = min(eligible)
    return {
        "status": "SELECTION_FROZEN",
        "selection_scope": "INNER_FOLD_DEVELOPMENT_ONLY",
        "test_accessed": False,
        "common_histories": sorted(common_histories),
        "history_fold_losses": {
            str(history): losses for history, losses in history_fold_losses.items()
        },
        "history_mean_losses": {
            str(history): means[history] for history in sorted(means)
        },
        "best_history": int(best),
        "best_standard_error": standard_error,
        "one_se_limit": one_se_limit,
        "maximum_relative_regret": maximum_relative_regret,
        "eligible_histories": eligible,
        "selected_history": int(selected_history),
        "chosen_delta_profile_by_history": {
            str(history): values for history, values in chosen_profiles.items()
        },
    }
