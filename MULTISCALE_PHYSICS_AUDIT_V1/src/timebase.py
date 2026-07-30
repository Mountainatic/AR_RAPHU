"""Physical-time conversions; no model logic is expressed in raw sample counts."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Timebase:
    sample_period_sec: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_period_sec) or self.sample_period_sec <= 0:
            raise ValueError("INVALID_SAMPLE_PERIOD")

    def samples_for_seconds(self, seconds: float, *, minimum: int = 1) -> int:
        value = float(seconds) / self.sample_period_sec
        rounded = int(round(value))
        if abs(value - rounded) > 1.0e-8:
            raise ValueError(
                f"PHYSICAL_TIME_NOT_ALIGNED:{seconds}s/{self.sample_period_sec}s"
            )
        return max(minimum, rounded)

    def samples_for_minutes(self, minutes: float, *, minimum: int = 1) -> int:
        return self.samples_for_seconds(float(minutes) * 60.0, minimum=minimum)

    def cadence_step(self, cadence_sec: float) -> int:
        return self.samples_for_seconds(cadence_sec)
