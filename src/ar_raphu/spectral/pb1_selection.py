"""Frozen H2 history and resolution selectors for PB1 Repair V2."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class H2HistoryScore:
    L_x: int
    L_y: int
    validation_loss: float
    validation_se: float
    status: str = "COMPLETED"


@dataclass(frozen=True, slots=True)
class H2ResolutionScore:
    lag_kind: str
    lag_count: int
    amplitude_count: int
    validation_loss: float
    validation_se: float
    representation_gate_passed: bool
    lepski_stable: bool
    status: str = "COMPLETED"


def history_complexity_key(row: H2HistoryScore) -> tuple[int, int, int, int, int]:
    return (
        row.L_x + row.L_y,
        row.L_x * row.L_y,
        max(row.L_x, row.L_y),
        row.L_x,
        row.L_y,
    )


def select_h2_history_one_se(
    rows: list[H2HistoryScore],
) -> H2HistoryScore:
    completed = [
        row
        for row in rows
        if row.status == "COMPLETED"
        and math.isfinite(row.validation_loss)
        and math.isfinite(row.validation_se)
    ]
    if not completed:
        raise ValueError("H2 history selector has no completed candidates.")
    minimum = min(
        completed,
        key=lambda row: (row.validation_loss, history_complexity_key(row)),
    )
    threshold = minimum.validation_loss + minimum.validation_se
    eligible = [row for row in completed if row.validation_loss <= threshold]
    return min(eligible, key=history_complexity_key)


def resolution_complexity_key(
    row: H2ResolutionScore,
) -> tuple[int, int, int]:
    return (
        row.lag_count * row.amplitude_count,
        row.lag_count,
        row.amplitude_count,
    )


def select_h2_resolution_one_se(
    rows: list[H2ResolutionScore],
) -> H2ResolutionScore:
    gated = [
        row
        for row in rows
        if row.status == "COMPLETED"
        and row.representation_gate_passed
        and row.lepski_stable
        and math.isfinite(row.validation_loss)
        and math.isfinite(row.validation_se)
    ]
    if not gated:
        raise ValueError("H2 resolution selector has no stable gated candidates.")
    minimum = min(
        gated,
        key=lambda row: (row.validation_loss, resolution_complexity_key(row)),
    )
    threshold = minimum.validation_loss + minimum.validation_se
    eligible = [row for row in gated if row.validation_loss <= threshold]
    return min(eligible, key=resolution_complexity_key)
