"""Empirical prediction-operator diagnostics."""

from __future__ import annotations

import numpy as np
import scipy.linalg


def empirical_operator_nrmse(
    truth: np.ndarray,
    estimate: np.ndarray,
) -> float:
    target = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(estimate, dtype=np.float64)
    denominator = np.linalg.norm(target - target.mean())
    return float(
        np.linalg.norm(prediction - target)
        / max(float(denominator), np.finfo(np.float64).eps)
    )


def design_spectrum_metrics(matrix: np.ndarray) -> dict[str, float]:
    design = np.asarray(matrix, dtype=np.float64)
    singular_values = scipy.linalg.svdvals(design / np.sqrt(len(design)))
    squared = singular_values**2
    positive = squared[squared > max(float(squared.max()), 1.0) * 1e-12]
    condition = (
        float(np.sqrt(positive.max() / positive.min()))
        if positive.size
        else float("inf")
    )
    probabilities = squared / max(float(squared.sum()), np.finfo(float).eps)
    probabilities = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "design_condition_number": condition,
        "effective_rank": effective_rank,
    }
