"""Validation-only one-standard-error selection utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def validation_one_se_select(
    rows: Iterable[dict[str, Any]],
    *,
    declared_config_order: list[str],
    complexity_key: Callable[[str], tuple],
) -> dict[str, Any]:
    """Select the simplest eligible configuration from independent units.

    Each row must contain ``config_id``, ``unit_id``, and ``validation_loss``.
    Phase-1 synthetic experiments use independent seed replicates as units.
    For datasets with outer folds, callers must first average seeds within each
    fold and pass one row per fold/configuration.
    """

    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        config_id = str(row["config_id"])
        unit_id = str(row["unit_id"])
        loss = float(row["validation_loss"])
        if not np.isfinite(loss):
            raise ValueError("Validation losses must be finite.")
        if unit_id in grouped[config_id]:
            raise ValueError(f"Duplicate unit {unit_id!r} for {config_id!r}.")
        grouped[config_id][unit_id] = loss

    if set(grouped) != set(declared_config_order):
        raise ValueError("Observed configurations do not match the declaration.")
    expected_units = set(next(iter(grouped.values())))
    if len(expected_units) < 2:
        raise ValueError("At least two independent selection units are required.")
    if any(set(values) != expected_units for values in grouped.values()):
        raise ValueError("Every configuration must use the same selection units.")

    order_index = {
        config_id: index
        for index, config_id in enumerate(declared_config_order)
    }
    stats: list[dict[str, Any]] = []
    for config_id in declared_config_order:
        values = np.asarray(
            [grouped[config_id][unit] for unit in sorted(expected_units)],
            dtype=np.float64,
        )
        stats.append(
            {
                "config_id": config_id,
                "mean_validation_loss": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
                "unit_count": int(len(values)),
            }
        )

    minimum = min(
        stats,
        key=lambda item: (
            item["mean_validation_loss"],
            order_index[item["config_id"]],
        ),
    )
    threshold = (
        minimum["mean_validation_loss"] + minimum["standard_error"]
    )
    eligible = [
        item
        for item in stats
        if item["mean_validation_loss"] <= threshold
    ]
    selected = min(
        eligible,
        key=lambda item: (
            complexity_key(item["config_id"]),
            order_index[item["config_id"]],
        ),
    )
    return {
        "status": "COMPLETED",
        "minimum_config_id": minimum["config_id"],
        "minimum_mean_validation_loss": minimum["mean_validation_loss"],
        "minimum_standard_error": minimum["standard_error"],
        "one_se_threshold": threshold,
        "eligible_config_ids": [item["config_id"] for item in eligible],
        "selected_config_id": selected["config_id"],
        "all_config_stats": stats,
        "selection_data": "validation_only",
        "test_used": False,
    }
