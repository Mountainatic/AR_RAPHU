"""Frozen CZ split and direct-forecast indexing protocol.

All intervals and slices are zero-based and left-closed/right-open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal


PartitionName = Literal["train", "validation", "test"]

SEQUENCE_LENGTH = 20_103
PREDICTION_HORIZONS = (1, 5, 10, 30, 60)
LEGACY_L_X = 32
LEGACY_L_Y = 32
FORMAL_L_Y_CANDIDATES = (1, 4, 8, 16, 32, 64)
FORMAL_L_X_CANDIDATES = (32, 64, 128, 256)
CONDITIONAL_L_X_EXTENSION = 512


@dataclass(frozen=True, slots=True)
class Interval:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"Invalid interval [{self.start}, {self.stop}).")

    def __contains__(self, index: int) -> bool:
        return self.start <= index < self.stop

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class Fold:
    number: int
    role: Literal["development", "final_lockbox"]
    train: Interval
    validation: Interval
    test: Interval

    def interval(self, partition: PartitionName) -> Interval:
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown partition: {partition!r}.")
        return getattr(self, partition)


@dataclass(frozen=True, slots=True)
class DirectForecastSample:
    """Indices for F_h(X through t, y through t) -> y[t+h]."""

    fold: int
    partition: PartitionName
    horizon: int
    origin: int
    target: int
    x_start: int
    x_stop: int
    y_start: int
    y_stop: int

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("Horizon must be positive.")
        if self.target != self.origin + self.horizon:
            raise ValueError("Target must equal origin + horizon.")
        if self.x_stop != self.origin + 1 or self.y_stop != self.origin + 1:
            raise ValueError("Both input windows must stop immediately after origin t.")
        if self.x_start < 0 or self.y_start < 0:
            raise ValueError("Input windows cannot precede the source sequence.")
        if self.x_start >= self.x_stop or self.y_start >= self.y_stop:
            raise ValueError("Input windows must be non-empty.")
        if max(self.x_stop - 1, self.y_stop - 1) != self.origin:
            raise ValueError("The latest input must be exactly the prediction origin.")
        if self.origin >= self.target:
            raise ValueError("Every input must precede the target.")


FOLDS = (
    Fold(
        1,
        "development",
        Interval(0, 10051),
        Interval(10051, 12061),
        Interval(12061, 14072),
    ),
    Fold(
        2,
        "development",
        Interval(0, 12061),
        Interval(12061, 14072),
        Interval(14072, 16082),
    ),
    Fold(
        3,
        "development",
        Interval(0, 14072),
        Interval(14072, 16082),
        Interval(16082, 18092),
    ),
    Fold(
        4,
        "final_lockbox",
        Interval(0, 16082),
        Interval(16082, 18092),
        Interval(18092, 20103),
    ),
)


class FinalLockboxError(PermissionError):
    """Raised before any Fold-4 test target may be accessed."""


def get_fold(number: int) -> Fold:
    try:
        return FOLDS[number - 1]
    except (IndexError, TypeError):
        raise ValueError(f"Fold number must be in 1..{len(FOLDS)}.") from None


def _validate_window_and_horizon(L_x: int, L_y: int, horizon: int) -> None:
    if not isinstance(L_x, int) or L_x <= 0:
        raise ValueError("L_x must be a positive integer.")
    if not isinstance(L_y, int) or L_y <= 0:
        raise ValueError("L_y must be a positive integer.")
    if horizon not in PREDICTION_HORIZONS:
        raise ValueError(
            f"horizon must be one of {PREDICTION_HORIZONS}, got {horizon!r}."
        )


def iter_direct_forecast_samples(
    fold_number: int,
    partition: PartitionName,
    *,
    L_x: int,
    L_y: int,
    horizon: int,
    protocol_frozen: bool = False,
) -> Iterator[DirectForecastSample]:
    """Yield safe direct-forecast indices, assigning membership by target.

    Historical inputs may precede the target partition's left boundary. Fold 4
    test targets are unavailable until ``protocol_frozen`` is explicitly true.
    """

    _validate_window_and_horizon(L_x, L_y, horizon)
    fold = get_fold(fold_number)
    interval = fold.interval(partition)

    if fold.role == "final_lockbox" and partition == "test" and not protocol_frozen:
        raise FinalLockboxError(
            "Fold 4 test is locked until every model and selection choice is frozen."
        )

    earliest_origin = max(L_x, L_y) - 1
    first_target = max(interval.start, earliest_origin + horizon)

    for target in range(first_target, interval.stop):
        origin = target - horizon
        sample = DirectForecastSample(
            fold=fold_number,
            partition=partition,
            horizon=horizon,
            origin=origin,
            target=target,
            x_start=origin - L_x + 1,
            x_stop=origin + 1,
            y_start=origin - L_y + 1,
            y_stop=origin + 1,
        )
        sample.validate()
        yield sample


def direct_forecast_sample_count(
    fold_number: int,
    partition: PartitionName,
    *,
    L_x: int,
    L_y: int,
    horizon: int,
    protocol_frozen: bool = False,
) -> int:
    """Count samples without reading source values."""

    _validate_window_and_horizon(L_x, L_y, horizon)
    fold = get_fold(fold_number)
    interval = fold.interval(partition)

    if fold.role == "final_lockbox" and partition == "test" and not protocol_frozen:
        raise FinalLockboxError(
            "Fold 4 test is locked until every model and selection choice is frozen."
        )

    earliest_target = max(L_x, L_y) - 1 + horizon
    return max(0, interval.stop - max(interval.start, earliest_target))


def validate_frozen_folds() -> None:
    """Check the pre-registered fold table without reading private values."""

    if len(FOLDS) != 4:
        raise AssertionError("Exactly four folds are required.")
    if FOLDS[-1].test != Interval(18092, SEQUENCE_LENGTH):
        raise AssertionError("Fold 4 lockbox must be the final sequence block.")
    for fold in FOLDS:
        if fold.train.start != 0:
            raise AssertionError("Every expanding training interval must start at zero.")
        if fold.train.stop != fold.validation.start:
            raise AssertionError("Train and validation target intervals must be adjacent.")
        if fold.validation.stop != fold.test.start:
            raise AssertionError("Validation and test target intervals must be adjacent.")
        if fold.test.stop > SEQUENCE_LENGTH:
            raise AssertionError("A fold cannot exceed the source sequence.")


validate_frozen_folds()
