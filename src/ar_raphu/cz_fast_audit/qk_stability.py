"""FAST-F cross-fold Q and low-order weighted-K stability."""

from __future__ import annotations

from math import floor

import numpy as np
import scipy.linalg

from ar_raphu.spectral.design import build_spectral_design
from ar_raphu.spectral.spline_basis import clamped_knots, evaluate_basis

from .fast_coarse_xar import CoarseModelRecord
from .residualization import FAST_TASKS, target_indices


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _sqrt_and_inverse(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = scipy.linalg.eigh(
        np.asarray(matrix, dtype=np.float64), check_finite=False
    )
    threshold = max(float(values.max()) * 1.0e-12, np.finfo(np.float64).eps)
    clipped = np.maximum(values, threshold)
    root = vectors @ np.diag(np.sqrt(clipped)) @ vectors.T
    inverse = vectors @ np.diag(1.0 / np.sqrt(clipped)) @ vectors.T
    return root, inverse


def _mode_profile(
    record: CoarseModelRecord,
    *,
    variable: int,
    lag_basis_count: int,
    amplitude_basis_count: int,
) -> dict[str, object]:
    width = lag_basis_count * amplitude_basis_count
    coefficients = record.fit.coefficients[: record.external_width].reshape(
        -1, lag_basis_count, amplitude_basis_count
    )[variable]
    lag_root, lag_inverse = _sqrt_and_inverse(
        record.train_external.lag_gram
    )
    amplitude_root, amplitude_inverse = _sqrt_and_inverse(
        record.train_external.amplitude_grams[variable]
    )
    weighted = lag_root @ coefficients @ amplitude_root
    u, singular_values, vt = scipy.linalg.svd(
        weighted, full_matrices=False, check_finite=False
    )
    lag_coefficients = lag_inverse @ u[:, 0]
    amplitude_coefficients = amplitude_inverse @ vt[0]
    lag_knots = clamped_knots(
        0.0, float(record.task.L_x - 1), lag_basis_count, 3
    )
    lag_nodes = np.linspace(0.0, float(record.task.L_x - 1), 201)
    lag_curve = evaluate_basis(lag_nodes, lag_knots, 3) @ lag_coefficients
    return {
        "singular_values": singular_values,
        "lag_curve": lag_curve,
        "amplitude_coefficients": amplitude_coefficients,
        "amplitude_basis": record.train_external.amplitude_bases[variable],
    }


def qk_stability_audit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    input_names: tuple[str, ...],
    coarse_rows: list[dict[str, object]],
    models: list[CoarseModelRecord],
    lag_basis_count: int,
    amplitude_basis_count: int,
    continuation_scale: float,
    q_high: float,
    q_moderate: float,
    k_mode_correlation_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    mean_delta = {
        task.name: float(
            np.mean(
                [
                    float(row["delta_X_given_AR_coarse"])
                    for row in coarse_rows
                    if row["task"] == task.name
                ]
            )
        )
        for task in FAST_TASKS
    }
    positive_tasks = [
        task.name for task in FAST_TASKS if mean_delta[task.name] > 0.0
    ]
    if not positive_tasks:
        return (
            [],
            [
                {
                    "task": task.name,
                    "input": name,
                    "input_index": variable,
                    "status": "K_NOT_TESTED_DUE_TO_NO_Q_GAIN",
                }
                for task in FAST_TASKS
                for variable, name in enumerate(input_names)
            ],
            {
                "positive_tasks": [],
                "status": "K_NOT_TESTED_DUE_TO_NO_Q_GAIN",
            },
        )
    q_rows: list[dict[str, object]] = []
    k_rows: list[dict[str, object]] = []
    for task in FAST_TASKS:
        if task.name not in positive_tasks:
            for variable, name in enumerate(input_names):
                k_rows.append(
                    {
                        "task": task.name,
                        "input": name,
                        "input_index": variable,
                        "status": "K_NOT_TESTED_DUE_TO_NO_Q_GAIN",
                    }
                )
            continue
        records = sorted(
            [record for record in models if record.task.name == task.name],
            key=lambda record: record.fold.fold,
        )
        if len(records) != 2:
            raise RuntimeError("FAST_F_REQUIRES_TWO_FOLD_MODELS")
        common_targets = target_indices(
            start=floor(len(y) * 0.7),
            stop=floor(len(y) * 0.8),
            task=task,
        )
        contributions = []
        for record in records:
            design = build_spectral_design(
                x,
                target_indices=common_targets,
                train_target_stop=record.fold.effective_train_stop,
                horizon=task.horizon,
                L_x=task.L_x,
                lag_basis_count=lag_basis_count,
                amplitude_basis_count=amplitude_basis_count,
                continuation_scale_factor=continuation_scale,
            )
            contributions.append(
                design.matrix
                @ record.fit.coefficients[: record.external_width]
            )
        q_correlation = _correlation(contributions[0], contributions[1])
        q_status = (
            "HIGH_Q_STABILITY"
            if q_correlation >= q_high
            else (
                "MODERATE_Q_STABILITY"
                if q_correlation >= q_moderate
                else "LOW_Q_STABILITY"
            )
        )
        q_rows.append(
            {
                "task": task.name,
                "horizon": task.horizon,
                "mean_delta_X_given_AR_coarse": mean_delta[task.name],
                "common_target_start": int(common_targets[0]),
                "common_target_stop": int(common_targets[-1]) + 1,
                "contribution_correlation": q_correlation,
                "status": q_status,
            }
        )
        for variable, name in enumerate(input_names):
            first = _mode_profile(
                records[0],
                variable=variable,
                lag_basis_count=lag_basis_count,
                amplitude_basis_count=amplitude_basis_count,
            )
            second = _mode_profile(
                records[1],
                variable=variable,
                lag_basis_count=lag_basis_count,
                amplitude_basis_count=amplitude_basis_count,
            )
            lower = max(
                float(first["amplitude_basis"].lower),
                float(second["amplitude_basis"].lower),
            )
            upper = min(
                float(first["amplitude_basis"].upper),
                float(second["amplitude_basis"].upper),
            )
            if lower >= upper:
                amplitude_correlation = 0.0
                surface_correlation = 0.0
                distance = float("inf")
            else:
                nodes = np.linspace(lower, upper, 201)
                amplitude_first = (
                    first["amplitude_basis"].transform(nodes)
                    @ first["amplitude_coefficients"]
                )
                amplitude_second = (
                    second["amplitude_basis"].transform(nodes)
                    @ second["amplitude_coefficients"]
                )
                amplitude_correlation = abs(
                    _correlation(amplitude_first, amplitude_second)
                )
                surface_first = np.outer(
                    first["lag_curve"], amplitude_first
                )
                surface_second = np.outer(
                    second["lag_curve"], amplitude_second
                )
                raw_correlation = _correlation(
                    surface_first, surface_second
                )
                sign = 1.0 if raw_correlation >= 0.0 else -1.0
                surface_correlation = abs(raw_correlation)
                distance = float(
                    np.linalg.norm(surface_first - sign * surface_second)
                    / max(
                        np.linalg.norm(surface_first),
                        np.linalg.norm(surface_second),
                        np.finfo(np.float64).eps,
                    )
                )
            lag_correlation = abs(
                _correlation(first["lag_curve"], second["lag_curve"])
            )
            principal_angle = float(
                np.degrees(
                    np.arccos(
                        np.clip(surface_correlation, 0.0, 1.0)
                    )
                )
            )
            singular_first = first["singular_values"]
            singular_second = second["singular_values"]
            leading_share_first = float(
                singular_first[0] ** 2
                / max(
                    float(np.sum(singular_first**2)),
                    np.finfo(np.float64).eps,
                )
            )
            leading_share_second = float(
                singular_second[0] ** 2
                / max(
                    float(np.sum(singular_second**2)),
                    np.finfo(np.float64).eps,
                )
            )
            stable = bool(
                lag_correlation >= k_mode_correlation_threshold
                and amplitude_correlation >= k_mode_correlation_threshold
                and surface_correlation >= k_mode_correlation_threshold
            )
            k_rows.append(
                {
                    "task": task.name,
                    "horizon": task.horizon,
                    "input": name,
                    "input_index": variable,
                    "fold1_first_singular_share": leading_share_first,
                    "fold2_first_singular_share": leading_share_second,
                    "fold1_second_singular_share": float(
                        singular_first[1] ** 2
                        / max(
                            float(np.sum(singular_first**2)),
                            np.finfo(np.float64).eps,
                        )
                    ),
                    "fold2_second_singular_share": float(
                        singular_second[1] ** 2
                        / max(
                            float(np.sum(singular_second**2)),
                            np.finfo(np.float64).eps,
                        )
                    ),
                    "leading_lag_mode_correlation": lag_correlation,
                    "leading_amplitude_mode_correlation": amplitude_correlation,
                    "leading_surface_mode_correlation": surface_correlation,
                    "principal_angle_degrees": principal_angle,
                    "sign_aligned_mode_distance": distance,
                    "status": (
                        "K_LOW_ORDER_STABLE"
                        if stable
                        else "K_LOW_ORDER_UNSTABLE"
                    ),
                }
            )
    return q_rows, k_rows, {
        "positive_tasks": positive_tasks,
        "mean_delta_by_task": mean_delta,
    }
