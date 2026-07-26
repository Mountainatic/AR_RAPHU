"""Frozen-mode rank ladders with modal-gain and intercept refits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RankLadderStep:
    rank: int
    coefficients: np.ndarray
    modal_gains: np.ndarray
    intercept: float
    train_prediction: np.ndarray
    validation_prediction: np.ndarray
    normalized_excess_rmse: float


def normalized_excess_rmse(
    target: np.ndarray,
    prediction: np.ndarray,
    full_prediction: np.ndarray,
) -> float:
    y = np.asarray(target, dtype=np.float64)
    rank_error = np.mean((y - np.asarray(prediction)) ** 2)
    full_error = np.mean((y - np.asarray(full_prediction)) ** 2)
    return float(
        np.sqrt(
            max(rank_error - full_error, 0.0)
            / max(float(np.var(y)), np.finfo(np.float64).eps)
        )
    )


def build_rank_ladder(
    *,
    train_design: np.ndarray,
    validation_design: np.ndarray,
    train_target: np.ndarray,
    validation_target: np.ndarray,
    mode_coefficients: list[np.ndarray],
    full_validation_prediction: np.ndarray,
    rank_max: int,
) -> list[RankLadderStep]:
    """Refit only scalar modal gains and the intercept for every rank."""

    train_modes = np.column_stack(
        [train_design @ np.asarray(mode).reshape(-1) for mode in mode_coefficients]
    )
    validation_modes = np.column_stack(
        [
            validation_design @ np.asarray(mode).reshape(-1)
            for mode in mode_coefficients
        ]
    )
    steps = []
    for rank in range(1, rank_max + 1):
        regression = np.column_stack(
            [np.ones(len(train_modes)), train_modes[:, :rank]]
        )
        parameters = np.linalg.lstsq(
            regression, np.asarray(train_target, dtype=np.float64), rcond=None
        )[0]
        intercept = float(parameters[0])
        gains = parameters[1:]
        train_prediction = intercept + train_modes[:, :rank] @ gains
        validation_prediction = (
            intercept + validation_modes[:, :rank] @ gains
        )
        coefficients = sum(
            gain * mode
            for gain, mode in zip(
                gains, mode_coefficients[:rank], strict=True
            )
        )
        steps.append(
            RankLadderStep(
                rank=rank,
                coefficients=coefficients,
                modal_gains=gains,
                intercept=intercept,
                train_prediction=train_prediction,
                validation_prediction=validation_prediction,
                normalized_excess_rmse=normalized_excess_rmse(
                    validation_target,
                    validation_prediction,
                    full_validation_prediction,
                ),
            )
        )
    return steps
