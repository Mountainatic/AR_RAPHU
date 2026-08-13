from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RateHistoryCondition:
    condition_id: str
    sampling_rate_hz: int
    history_mode: str
    history_steps: int
    history_duration_ms: float
    evaluation_key: str
    alias_of: str | None = None


def registered_conditions(base_history_steps: int = 20, base_rate_hz: int = 100) -> list[RateHistoryCondition]:
    values: list[RateHistoryCondition] = []
    for rate in (100, 200, 400):
        fixed_step = base_history_steps
        fixed_time = base_history_steps * rate // base_rate_hz
        values.append(RateHistoryCondition(
            f"hz{rate}_fixed_step_h{fixed_step}", rate, "FIXED_STEP", fixed_step,
            1000.0 * fixed_step / rate, f"hz{rate}_h{fixed_step}", None,
        ))
        alias = f"hz{rate}_fixed_step_h{fixed_step}" if fixed_time == fixed_step else None
        values.append(RateHistoryCondition(
            f"hz{rate}_fixed_time_h{fixed_time}", rate, "FIXED_TIME", fixed_time,
            1000.0 * fixed_time / rate, f"hz{rate}_h{fixed_time}", alias,
        ))
    return values


def unique_evaluation_conditions() -> list[RateHistoryCondition]:
    seen: set[str] = set()
    result = []
    for value in registered_conditions():
        if value.evaluation_key not in seen:
            seen.add(value.evaluation_key)
            result.append(value)
    return result


def condition_manifest() -> list[dict[str, object]]:
    return [asdict(value) for value in registered_conditions()]


def resync_intervals(rate: int) -> list[int | str]:
    return {
        100: [1, 5, 10, 20, 50, 100, "infinity"],
        200: [1, 2, 5, 10, 20, 40, 100, 200, "infinity"],
        400: [1, 2, 4, 8, 20, 40, 80, 200, 400, "infinity"],
    }[rate]


def primary_horizon_candidates(rate: int) -> list[int]:
    # These are the exact R3 100-Hz grid scaled in physical time.
    factor = rate // 100
    return [factor * value for value in (1, 5, 10, 20, 50, 100)]

