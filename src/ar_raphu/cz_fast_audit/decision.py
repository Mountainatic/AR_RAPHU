"""Frozen FAST-G GO/NO-GO decision mapping and runtime gate."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np


ALLOWED_STATUSES = {
    "GO_FULL_CZ_IDENTIFICATION",
    "GO_PREDICTION_ONLY",
    "GO_PARTIAL_K",
    "NO_GO_FULL_KERNEL",
    "AUDIT_INCOMPLETE",
}


@dataclass(slots=True)
class RuntimeGate:
    """Monotonic hard cap shared by every FAST stage."""

    maximum_seconds: float
    started_at: float

    @classmethod
    def start(cls, maximum_minutes: float) -> "RuntimeGate":
        return cls(
            maximum_seconds=float(maximum_minutes) * 60.0,
            started_at=time.monotonic(),
        )

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return self.maximum_seconds - self.elapsed_seconds

    def check(self, stage: str) -> None:
        if self.remaining_seconds <= 0.0:
            raise TimeoutError(
                f"FAST_AUDIT_RUNTIME_GATE_FAILED:{stage}:"
                f"{self.elapsed_seconds:.3f}s"
            )


def _mean_by_task(
    rows: Iterable[dict[str, object]], field: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["task"]), []).append(float(row[field]))
    return {
        task: float(np.mean(values)) for task, values in sorted(grouped.items())
    }


def decide_go_nogo(
    *,
    conditional_energy_rows: list[dict[str, object]],
    conditional_gram_summary: dict[str, object],
    coarse_rows: list[dict[str, object]],
    q_rows: list[dict[str, object]],
    k_rows: list[dict[str, object]],
    gates: dict[str, object],
    complete: bool = True,
) -> dict[str, object]:
    """Apply only the five predeclared terminal states.

    The numerical interpretations of "moderate coercive dimension" and
    "low-dimensional spectral gap" are explicit configuration fields. They are
    not inferred after looking at the result.
    """

    if not complete:
        return {
            "status": "AUDIT_INCOMPLETE",
            "positive_increment_horizons": [],
            "conditional_energy_summary": {},
            "conditional_gram_summary": {},
            "q_stability_summary": {},
            "k_low_order_summary": {},
            "next_allowed_stage": "NONE_RUNTIME_GATE_OR_AUDIT_FAILURE",
        }

    mean_delta = _mean_by_task(coarse_rows, "delta_X_given_AR_coarse")
    positive_tasks = sorted(
        task for task, value in mean_delta.items() if value > 0.0
    )
    task_horizon = {
        str(row["task"]): int(row["horizon"]) for row in coarse_rows
    }
    positive_horizons = sorted(task_horizon[task] for task in positive_tasks)
    direction_consistent = {
        task: all(
            bool(row["direction_positive"])
            for row in coarse_rows
            if str(row["task"]) == task
        )
        for task in mean_delta
    }

    maximum_energy = max(
        (
            float(row["conditional_energy_ratio"])
            for row in conditional_energy_rows
        ),
        default=0.0,
    )
    energy_threshold = float(gates["minimum_conditional_energy_ratio"])
    has_conditional_energy = maximum_energy >= energy_threshold

    joint_rows: list[dict[str, object]] = []
    for task_rows in conditional_gram_summary.values():
        for fold_rows in task_rows.values():
            joint = fold_rows.get("joint")
            if joint is not None:
                joint_rows.append(joint)
    joint_effective_ranks = [
        float(row["effective_rank"]) for row in joint_rows
    ]
    joint_d_1e3 = [
        int(row["coercive_dimension"][str(1.0e-3)])
        for row in joint_rows
    ]
    median_effective_rank = (
        float(np.median(joint_effective_ranks))
        if joint_effective_ranks
        else 0.0
    )
    median_d_1e3 = (
        float(np.median(joint_d_1e3)) if joint_d_1e3 else 0.0
    )
    moderate_gram = bool(
        median_effective_rank
        >= float(gates["minimum_joint_effective_rank_for_full_go"])
        and median_d_1e3
        >= float(gates["minimum_joint_coercive_dimension_for_full_go"])
    )
    partial_gram = bool(
        median_d_1e3
        >= float(gates["minimum_joint_coercive_dimension_for_partial_k"])
    )

    stable_k_rows = [
        row for row in k_rows if row.get("status") == "K_LOW_ORDER_STABLE"
    ]
    has_stable_k = bool(stable_k_rows)
    q_by_task = {
        str(row["task"]): {
            "correlation": float(row["contribution_correlation"]),
            "status": str(row["status"]),
        }
        for row in q_rows
    }
    medium_long_q = any(
        int(row["horizon"]) in {15, 60}
        and float(row["mean_delta_X_given_AR_coarse"]) > 0.0
        and float(row["contribution_correlation"])
        >= float(gates["q_stability_moderate"])
        for row in q_rows
    )

    minimum_positive = int(gates["minimum_positive_horizons_for_full_go"])
    full_direction = all(
        direction_consistent.get(task, False) for task in positive_tasks
    )
    all_nonpositive = bool(mean_delta) and all(
        value <= 0.0 for value in mean_delta.values()
    )
    weak_energy = maximum_energy < energy_threshold
    weak_gram = not partial_gram

    if (
        len(positive_tasks) >= minimum_positive
        and full_direction
        and has_conditional_energy
        and moderate_gram
        and has_stable_k
    ):
        status = "GO_FULL_CZ_IDENTIFICATION"
        next_stage = "CZ_R2_2_DENSE_BATCHED_RESCUE_THEN_R3"
    elif medium_long_q and (not has_stable_k or not moderate_gram):
        status = "GO_PREDICTION_ONLY"
        next_stage = "FREEZE_Q_LEVEL_THEN_PREDICTIVE_RANK"
    elif has_stable_k and has_conditional_energy and partial_gram:
        status = "GO_PARTIAL_K"
        next_stage = "CERTIFIED_LOW_ORDER_LOCAL_K_ONLY"
    elif all_nonpositive and weak_energy and weak_gram and not has_stable_k:
        status = "NO_GO_FULL_KERNEL"
        next_stage = "STOP_FULL_CZ_K_SEARCH"
    else:
        status = "AUDIT_INCOMPLETE"
        next_stage = "NONE_AMBIGUOUS_FAST_EVIDENCE"

    assert status in ALLOWED_STATUSES
    return {
        "status": status,
        "positive_increment_horizons": positive_horizons,
        "conditional_energy_summary": {
            "maximum_ratio": maximum_energy,
            "threshold": energy_threshold,
            "has_at_least_one_input_at_threshold": has_conditional_energy,
        },
        "conditional_gram_summary": {
            "joint_median_effective_rank": median_effective_rank,
            "joint_median_coercive_dimension_1e-3": median_d_1e3,
            "moderate_for_full_go": moderate_gram,
            "supports_low_dimensional_subspace": partial_gram,
        },
        "q_stability_summary": {
            "by_task": q_by_task,
            "medium_or_long_predictive_gain_with_moderate_stability": (
                medium_long_q
            ),
        },
        "k_low_order_summary": {
            "stable_mode_count": len(stable_k_rows),
            "stable_modes": [
                {
                    "task": row["task"],
                    "input": row["input"],
                    "surface_correlation": row[
                        "leading_surface_mode_correlation"
                    ],
                }
                for row in stable_k_rows
            ],
        },
        "increment_summary": {
            "mean_delta_by_task": mean_delta,
            "two_fold_direction_consistent": direction_consistent,
        },
        "next_allowed_stage": next_stage,
    }
