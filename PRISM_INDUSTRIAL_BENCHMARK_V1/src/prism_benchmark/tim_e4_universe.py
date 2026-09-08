"""Mechanically generated nested candidate universes for TIM E4 only."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from typing import Any, Mapping, Sequence


ENVIRONMENT_VARIABLE = "PRISM_TIM_E4_CANDIDATE_UNIVERSE"
UNIVERSES = ("coarse", "standard", "expanded")
GENERATION_RULE_ID = "TIM_E4_MECHANICAL_NESTED_UNIVERSE_V1"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _coarse(values: Sequence[Any]) -> list[Any]:
    registered = list(values)
    if len(registered) <= 1:
        return registered
    kept = registered[::2]
    if kept[-1] == registered[-1] and len(kept) > 1:
        kept.pop()
    return kept


def _expanded_numeric(values: Sequence[int | float]) -> list[int | float]:
    registered = sorted(set(values))
    positive = [float(value) for value in registered if float(value) > 0.0]
    if not positive:
        return registered
    integral = all(isinstance(value, int) and not isinstance(value, bool) for value in registered)
    if integral:
        lower_gap = positive[1] - positive[0] if len(positive) > 1 else positive[0]
        upper_gap = positive[-1] - positive[-2] if len(positive) > 1 else positive[-1]
        lower: int | float = max(1, int(round(positive[0] - lower_gap)))
        upper = int(round(positive[-1] + upper_gap))
    else:
        ratio_low = positive[1] / positive[0] if len(positive) > 1 else 10.0
        ratio_high = positive[-1] / positive[-2] if len(positive) > 1 else 10.0
        lower = positive[0] / math.sqrt(max(ratio_low, 1.0))
        upper = positive[-1] * math.sqrt(max(ratio_high, 1.0))
    return sorted(set([*registered, lower, upper]))


def _set_path(root: dict[str, Any], path: tuple[str, ...], value: list[Any]) -> None:
    target: dict[str, Any] = root
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


NUMERIC_AXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("K.positive_h_history_multipliers", ("time_profile_grid", "positive_h_history_multipliers")),
    ("K.zero_h_history_in_delta_multipliers", ("time_profile_grid", "zero_h_history_in_delta_multipliers")),
    ("K.candidate_m_tau", ("K_module", "lag_basis", "candidate_m_tau")),
    ("K.candidate_m_x", ("K_module", "amplitude_basis", "candidate_m_x")),
    ("C.ridge_alpha_grid", ("C_module", "joint_basis", "ridge_alpha_grid")),
    ("W.candidate_knot_counts", ("W_module", "candidate_knot_counts")),
    ("W.smoothness_penalties", ("W_module", "smoothness_penalties")),
    ("A.ridge_alpha_grid", ("A_module", "ridge_alpha_grid")),
)


def _get_path(root: Mapping[str, Any], path: tuple[str, ...]) -> list[Any]:
    value: Any = root
    for key in path:
        value = value[key]
    return list(value)


def apply_candidate_universe(
    v211: Mapping[str, Any],
    v21: Mapping[str, Any],
    v2: Mapping[str, Any],
    universe: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if universe not in UNIVERSES:
        raise ValueError(f"unsupported TIM E4 universe: {universe}")
    outputs = [copy.deepcopy(value) for value in (v211, v21, v2)]
    if universe == "standard":
        return tuple(outputs)  # type: ignore[return-value]
    selected = _coarse if universe == "coarse" else _expanded_numeric
    for _, path in NUMERIC_AXES:
        values = _get_path(outputs[2], path)
        _set_path(outputs[2], path, selected(values))

    ranks = list(outputs[2]["K_module"]["rank_candidates"])
    if universe == "coarse":
        outputs[2]["K_module"]["rank_candidates"] = [
            value for value in _coarse(ranks) if value != "FULL"
        ]
        outputs[1]["C"]["C_candidates"] = list(outputs[1]["C"]["C_candidates"][:-1])
        outputs[0]["W"]["candidates"] = list(outputs[0]["W"]["candidates"][:-1])
        outputs[1]["A"]["candidates"] = list(outputs[1]["A"]["candidates"][:-1])
    else:
        numeric_ranks = [int(value) for value in ranks if value != "FULL"]
        outputs[2]["K_module"]["rank_candidates"] = [
            *sorted(set([*numeric_ranks, max(numeric_ranks) + 1])),
            "FULL",
        ]

    for root_index, path in (
        (1, ("W", "monotone_knots")),
        (1, ("W", "natural_cubic_knots")),
        (0, ("W", "soft_overlap_mu")),
        (1, ("A", "soft_overlap_mu")),
    ):
        values = _get_path(outputs[root_index], path)
        _set_path(outputs[root_index], path, selected(values))
    return tuple(outputs)  # type: ignore[return-value]


def maybe_apply_from_environment(
    v211: Mapping[str, Any], v21: Mapping[str, Any], v2: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    universe = os.environ.get(ENVIRONMENT_VARIABLE)
    if universe is None:
        return dict(v211), dict(v21), dict(v2)
    return apply_candidate_universe(v211, v21, v2, universe)


def universe_manifest(
    v211: Mapping[str, Any], v21: Mapping[str, Any], v2: Mapping[str, Any]
) -> dict[str, Any]:
    universes: dict[str, Any] = {}
    for name in UNIVERSES:
        transformed = apply_candidate_universe(v211, v21, v2, name)
        axes = {
            axis: _get_path(transformed[2], path) for axis, path in NUMERIC_AXES
        }
        axes.update(
            {
                "K.rank_candidates": transformed[2]["K_module"]["rank_candidates"],
                "C.families": transformed[1]["C"]["C_candidates"],
                "W.families": transformed[0]["W"]["candidates"],
                "W.monotone_knots": transformed[1]["W"]["monotone_knots"],
                "W.natural_cubic_knots": transformed[1]["W"]["natural_cubic_knots"],
                "W.soft_overlap_mu": transformed[0]["W"]["soft_overlap_mu"],
                "A.families": transformed[1]["A"]["candidates"],
                "A.soft_overlap_mu": transformed[1]["A"]["soft_overlap_mu"],
            }
        )
        universes[name] = {
            "axes": axes,
            "candidate_axis_value_count": sum(len(values) for values in axes.values()),
            "hash": _canonical_hash(axes),
        }
    return {
        "status": "FROZEN_BEFORE_E4_TEST_ACCESS",
        "generation_rule_id": GENERATION_RULE_ID,
        "generation_rule": {
            "coarse": "keep registered indices 0,2,4,... and drop a retained maximum-complexity endpoint; remove the highest registered C/W/A family",
            "standard": "exact deep copy of the formally frozen PRISM v2.1.1 candidate configuration",
            "expanded": "retain every standard value and add one mechanically adjacent lower and upper numeric value; add rank max+1",
            "per_task_manual_changes": False,
        },
        "universes": universes,
        "nesting": "coarse_strict_subset_standard_strict_subset_expanded",
        "test_accessed": False,
    }
