"""Shared feature preparation and Stage-1 profile evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .baselines import (
    local_trend_prediction,
    mean_drift_prediction,
    persistence_prediction,
    raw_history_ar_features,
    scale_matched_ar_features,
)
from .bootstrap import (
    moving_block_improvement,
    residual_correlation_time,
    stratified_two_direction_improvement,
)
from .data_loader import RodData, WorkbookData
from .linear_q import (
    fit_block_ridge,
    fit_ridge,
    regression_metrics,
    relative_improvement,
)
from .multiresolution_lags import (
    LagBlock,
    expand_lag_blocks,
    lag_block_matrix,
    thermal_state_bank,
    thermal_state_features,
)
from .resampling import PCA1Transform, transform_channel
from .segmentation import Segment
from .targets import TargetRows, build_target_rows
from .timebase import Timebase
from .validation import (
    crossfit_predictions,
    rolling_origin_folds,
    select_block_alphas,
    select_ridge_alpha,
)


@dataclass(slots=True)
class RodProfileFeatures:
    sheet: str
    rows: TargetRows
    q: np.ndarray
    ar_scale: np.ndarray
    ar_raw: np.ndarray
    local_trend: np.ndarray
    current_signal: np.ndarray


@dataclass(slots=True)
class DirectionFeatures:
    train_sheet: str
    test_sheet: str
    train: RodProfileFeatures
    test: RodProfileFeatures
    blocks: list[LagBlock]
    pca: PCA1Transform | None
    q_lag_width: int


def profile_variants(
    profile: dict[str, Any], config: dict[str, Any]
) -> tuple[str, ...]:
    return tuple(config["branches"][profile["channel"]]["variants"])


def profile_task_id(profile_id: str, variant: str) -> str:
    return f"{profile_id}__{variant}"


def _slice_rod(rod: RodData, start: int) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(values[start:], dtype=np.float64)
        for name, values in rod.columns.items()
    }


def _rod_features(
    sheet: str,
    columns: dict[str, np.ndarray],
    *,
    profile: dict[str, Any],
    variant: str,
    config: dict[str, Any],
    timebase: Timebase,
    signal: np.ndarray,
    blocks: list[LagBlock],
    resolution_multiplier: float,
) -> RodProfileFeatures:
    branch = config["branches"][profile["channel"]]
    target = columns["晶体直径"]
    segment = Segment(0, len(target), "main")
    rows = build_target_rows(
        target,
        segment,
        timebase=timebase,
        cadence_sec=float(branch["cadence_sec"]),
        horizon_min=float(profile["horizon_min"]),
        target_window_min=float(profile["target_window_min"]),
        history_min=float(profile["history_min"]),
    )
    q = lag_block_matrix(
        signal,
        rows.origins,
        blocks,
        timebase=timebase,
    )
    if profile["channel"] == "heater_power":
        bank = thermal_state_bank(
            signal,
            branch["thermal_tau_min"],
            timebase=timebase,
        )
        q = np.column_stack(
            (q, thermal_state_features(bank, rows.origins))
        )
    ar_scale = scale_matched_ar_features(
        target,
        rows,
        timebase=timebase,
        cadence_sec=float(branch["cadence_sec"]),
        history_min=float(profile["history_min"]),
    )
    ar_raw = raw_history_ar_features(
        target,
        rows,
        timebase=timebase,
        history_min=float(profile["history_min"]),
        block_specification=config["ar"]["raw_lag_blocks_min"],
    )
    trend = local_trend_prediction(target, rows)
    return RodProfileFeatures(
        sheet=sheet,
        rows=rows,
        q=np.ascontiguousarray(q),
        ar_scale=np.ascontiguousarray(ar_scale),
        ar_raw=np.ascontiguousarray(ar_raw),
        local_trend=np.ascontiguousarray(trend),
        current_signal=np.ascontiguousarray(signal[rows.origins]),
    )


def prepare_direction(
    workbook: WorkbookData,
    *,
    starts: dict[str, int],
    profile: dict[str, Any],
    variant: str,
    train_sheet: str,
    test_sheet: str,
    config: dict[str, Any],
    timebase: Timebase,
    resolution_multiplier: float = 1.0,
) -> DirectionFeatures:
    train_columns = _slice_rod(workbook.rods[train_sheet], starts[train_sheet])
    test_columns = _slice_rod(workbook.rods[test_sheet], starts[test_sheet])
    branch = config["branches"][profile["channel"]]
    pca = None
    train_signal, pca = transform_channel(
        profile["channel"],
        variant,
        train_columns,
        sample_period_sec=timebase.sample_period_sec,
        fit_pca_slice=slice(None),
        branch_config=branch,
    )
    test_signal, _ = transform_channel(
        profile["channel"],
        variant,
        test_columns,
        sample_period_sec=timebase.sample_period_sec,
        pca=pca,
        branch_config=branch,
    )
    blocks = expand_lag_blocks(
        branch["lag_blocks_min"],
        history_min=float(profile["history_min"]),
        resolution_multiplier=resolution_multiplier,
    )
    train = _rod_features(
        train_sheet,
        train_columns,
        profile=profile,
        variant=variant,
        config=config,
        timebase=timebase,
        signal=train_signal,
        blocks=blocks,
        resolution_multiplier=resolution_multiplier,
    )
    test = _rod_features(
        test_sheet,
        test_columns,
        profile=profile,
        variant=variant,
        config=config,
        timebase=timebase,
        signal=test_signal,
        blocks=blocks,
        resolution_multiplier=resolution_multiplier,
    )
    return DirectionFeatures(
        train_sheet,
        test_sheet,
        train,
        test,
        blocks,
        pca,
        len(blocks),
    )


def _baseline_selection(
    features: DirectionFeatures,
    folds,
) -> tuple[str, dict[str, float], np.ndarray]:
    target = features.train.rows.target
    candidates = {
        "B0_PERSISTENCE": persistence_prediction(features.train.rows),
        "B2_LOCAL_TREND": features.train.local_trend,
    }
    losses: dict[str, float] = {}
    for name, prediction in candidates.items():
        losses[name] = float(
            np.mean(
                [
                    np.mean(
                        (
                            target[fold.validation_indices]
                            - prediction[fold.validation_indices]
                        )
                        ** 2
                    )
                    for fold in folds
                ]
            )
        )
    mean_losses = []
    for fold in folds:
        prediction = np.full(
            len(fold.validation_indices),
            float(target[fold.train_indices].mean()),
        )
        mean_losses.append(
            float(
                np.mean(
                    (
                        target[fold.validation_indices] - prediction
                    )
                    ** 2
                )
            )
        )
    losses["B1_MEAN_DRIFT"] = float(np.mean(mean_losses))
    order = ("B0_PERSISTENCE", "B1_MEAN_DRIFT", "B2_LOCAL_TREND")
    selected = min(order, key=lambda name: (losses[name], order.index(name)))
    if selected == "B0_PERSISTENCE":
        test_prediction = persistence_prediction(features.test.rows)
    elif selected == "B1_MEAN_DRIFT":
        test_prediction = mean_drift_prediction(
            target, len(features.test.rows.target)
        )
    else:
        test_prediction = features.test.local_trend
    return selected, losses, test_prediction


def _model_entry(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    baseline_loss: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = regression_metrics(target, prediction)
    output: dict[str, Any] = {**metrics}
    if baseline_loss is not None:
        output["relative_improvement"] = relative_improvement(
            baseline_loss, (target - prediction) ** 2
        )
    return output


def evaluate_direction(
    features: DirectionFeatures,
    *,
    profile: dict[str, Any],
    config: dict[str, Any],
    timebase: Timebase,
    stage1_bootstrap_replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train = features.train
    test = features.test
    purge = train.rows.horizon_samples + train.rows.window_samples
    folds = rolling_origin_folds(
        train.rows.origins,
        config["validation"]["inner_validation_fractions"],
        purge_samples=purge,
    )
    grid = config["validation"]["ridge_grid"]
    baseline_name, baseline_cv, baseline_prediction = _baseline_selection(
        features, folds
    )
    test_target = test.rows.target
    baseline_loss = (test_target - baseline_prediction) ** 2

    alpha_q, q_cv = select_ridge_alpha(train.q, train.rows.target, folds, grid)
    alpha_scale, ar_scale_cv = select_ridge_alpha(
        train.ar_scale, train.rows.target, folds, grid
    )
    alpha_raw, ar_raw_cv = select_ridge_alpha(
        train.ar_raw, train.rows.target, folds, grid
    )
    q_fit = fit_ridge(train.q, train.rows.target, alpha=alpha_q)
    ar_scale_fit = fit_ridge(
        train.ar_scale, train.rows.target, alpha=alpha_scale
    )
    ar_raw_fit = fit_ridge(
        train.ar_raw, train.rows.target, alpha=alpha_raw
    )
    q_prediction = q_fit.predict(test.q)
    ar_scale_prediction = ar_scale_fit.predict(test.ar_scale)
    ar_raw_prediction = ar_raw_fit.predict(test.ar_raw)

    oof_indices, oof_target, oof_ar_prediction = crossfit_predictions(
        train.ar_scale,
        train.rows.target,
        folds,
        alpha=alpha_scale,
    )
    oof_residual = oof_target - oof_ar_prediction
    frozen_q_fit = fit_ridge(
        train.q[oof_indices],
        oof_residual,
        alpha=alpha_q,
    )
    frozen_prediction = (
        ar_scale_prediction + frozen_q_fit.predict(test.q)
    )

    (alpha_arx_ar, alpha_arx_q), arx_cv = select_block_alphas(
        train.ar_scale,
        train.q,
        train.rows.target,
        folds,
        grid,
    )
    arx_fit = fit_block_ridge(
        train.ar_scale,
        train.q,
        train.rows.target,
        alpha_ar=alpha_arx_ar,
        alpha_q=alpha_arx_q,
    )
    arx_prediction = arx_fit.predict(test.ar_scale, test.q)

    q_loss = (test_target - q_prediction) ** 2
    ar_scale_loss = (test_target - ar_scale_prediction) ** 2
    frozen_loss = (test_target - frozen_prediction) ** 2
    arx_loss = (test_target - arx_prediction) ** 2
    correlation_time = residual_correlation_time(
        test_target - q_prediction
    )
    cadence = timebase.cadence_step(
        config["branches"][profile["channel"]]["cadence_sec"]
    )
    minimum_block = int(
        np.ceil(
            float(config["validation"]["bootstrap_min_block_min"])
            * 60.0
            / (cadence * timebase.sample_period_sec)
        )
    )
    window_block = int(
        np.ceil(test.rows.window_samples / cadence)
    )
    block_length = max(minimum_block, window_block, correlation_time)
    q_bootstrap = moving_block_improvement(
        baseline_loss,
        q_loss,
        replicates=stage1_bootstrap_replicates,
        block_length=block_length,
        seed=seed,
    )
    conditional_bootstrap = moving_block_improvement(
        ar_scale_loss,
        frozen_loss,
        replicates=stage1_bootstrap_replicates,
        block_length=block_length,
        seed=seed + 1,
    )
    models = {
        "B0_PERSISTENCE": _model_entry(
            test_target, persistence_prediction(test.rows)
        ),
        "B1_MEAN_DRIFT": _model_entry(
            test_target,
            mean_drift_prediction(
                train.rows.target, len(test_target)
            ),
        ),
        "B2_LOCAL_TREND": _model_entry(
            test_target, test.local_trend
        ),
        "B3_AR_SCALE": _model_entry(
            test_target, ar_scale_prediction
        ),
        "B4_AR_RAW": _model_entry(test_target, ar_raw_prediction),
        "Q1_CHANNEL_LINEAR": _model_entry(
            test_target,
            q_prediction,
            baseline_loss=baseline_loss,
        ),
        "Q2_FROZEN_AR_PLUS_Q": _model_entry(
            test_target,
            frozen_prediction,
            baseline_loss=ar_scale_loss,
        ),
        "Q3_JOINT_ARX": _model_entry(
            test_target,
            arx_prediction,
            baseline_loss=ar_scale_loss,
        ),
    }
    kkt_values = {
        "Q1": q_fit.relative_kkt,
        "AR_SCALE": ar_scale_fit.relative_kkt,
        "AR_RAW": ar_raw_fit.relative_kkt,
        "FROZEN_Q": frozen_q_fit.relative_kkt,
        "JOINT_ARX": arx_fit.relative_kkt,
    }
    if any(value > 1.0e-8 for value in kkt_values.values()):
        raise RuntimeError(f"KKT_THRESHOLD_FAILED:{kkt_values}")
    direction = {
        "train_sheet": features.train_sheet,
        "test_sheet": features.test_sheet,
        "n_train": int(len(train.rows.target)),
        "n_test": int(len(test_target)),
        "selected_simple_baseline": baseline_name,
        "baseline_cv_mse": baseline_cv,
        "selected_penalties": {
            "Q1": alpha_q,
            "AR_SCALE": alpha_scale,
            "AR_RAW": alpha_raw,
            "JOINT_ARX_AR": alpha_arx_ar,
            "JOINT_ARX_Q": (
                "Q_ZERO" if alpha_arx_q is None else alpha_arx_q
            ),
        },
        "selection_losses": {
            "Q1": q_cv,
            "AR_SCALE": ar_scale_cv,
            "AR_RAW": ar_raw_cv,
            "JOINT_ARX": arx_cv,
        },
        "models": models,
        "q_bootstrap": q_bootstrap,
        "conditional_bootstrap": conditional_bootstrap,
        "bootstrap_block_length": block_length,
        "residual_correlation_time": correlation_time,
        "kkt": kkt_values,
        "linear_kernel": q_fit.physical_coefficients()[
            : features.q_lag_width
        ].tolist(),
        "thermal_state_coefficients": q_fit.physical_coefficients()[
            features.q_lag_width :
        ].tolist(),
        "lag_midpoints_min": [
            block.midpoint_min for block in features.blocks
        ],
        "pca": (
            None
            if features.pca is None
            else {
                "mean": features.pca.mean.tolist(),
                "scale": features.pca.scale.tolist(),
                "vector": features.pca.vector.tolist(),
                "explained_fraction": features.pca.explained_fraction,
            }
        ),
    }
    arrays = {
        "target": test_target,
        "baseline_prediction": baseline_prediction,
        "q_prediction": q_prediction,
        "ar_scale_prediction": ar_scale_prediction,
        "ar_raw_prediction": ar_raw_prediction,
        "frozen_prediction": frozen_prediction,
        "arx_prediction": arx_prediction,
        "baseline_loss": baseline_loss,
        "q_loss": q_loss,
        "ar_scale_loss": ar_scale_loss,
        "frozen_loss": frozen_loss,
        "arx_loss": arx_loss,
        "current_signal": test.current_signal,
        "q_features": test.q,
    }
    return direction, arrays


def evaluate_profile(
    workbook: WorkbookData,
    *,
    starts: dict[str, int],
    profile: dict[str, Any],
    variant: str,
    config: dict[str, Any],
    timebase: Timebase,
    resolution_multiplier: float = 1.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    directions: list[dict[str, Any]] = []
    all_arrays: dict[str, np.ndarray] = {}
    losses: list[tuple[np.ndarray, np.ndarray]] = []
    conditional_losses: list[tuple[np.ndarray, np.ndarray]] = []
    block_lengths: list[int] = []
    seed = int(config["random_seed"]) + sum(
        ord(character) for character in profile_task_id(profile["id"], variant)
    )
    for direction_index, (train_sheet, test_sheet) in enumerate(
        config["validation"]["outer_directions"]
    ):
        features = prepare_direction(
            workbook,
            starts=starts,
            profile=profile,
            variant=variant,
            train_sheet=train_sheet,
            test_sheet=test_sheet,
            config=config,
            timebase=timebase,
            resolution_multiplier=resolution_multiplier,
        )
        direction, arrays = evaluate_direction(
            features,
            profile=profile,
            config=config,
            timebase=timebase,
            stage1_bootstrap_replicates=int(
                config["validation"]["stage1_bootstrap_replicates"]
            ),
            seed=seed + direction_index * 1000,
        )
        directions.append(direction)
        prefix = f"d{direction_index + 1}_"
        all_arrays.update(
            {prefix + name: value for name, value in arrays.items()}
        )
        losses.append((arrays["baseline_loss"], arrays["q_loss"]))
        conditional_losses.append(
            (arrays["ar_scale_loss"], arrays["frozen_loss"])
        )
        block_lengths.append(int(direction["bootstrap_block_length"]))
    pooled_baseline = sum(float(np.sum(pair[0])) for pair in losses)
    pooled_q = sum(float(np.sum(pair[1])) for pair in losses)
    pooled_ar = sum(float(np.sum(pair[0])) for pair in conditional_losses)
    pooled_frozen = sum(float(np.sum(pair[1])) for pair in conditional_losses)
    pooled_q_improvement = 1.0 - pooled_q / max(
        pooled_baseline, np.finfo(np.float64).eps
    )
    pooled_conditional = 1.0 - pooled_frozen / max(
        pooled_ar, np.finfo(np.float64).eps
    )
    pooled_q_bootstrap = stratified_two_direction_improvement(
        losses,
        replicates=int(
            config["validation"]["stage1_bootstrap_replicates"]
        ),
        block_lengths=block_lengths,
        seed=seed + 20_000,
    )
    pooled_conditional_bootstrap = stratified_two_direction_improvement(
        conditional_losses,
        replicates=int(
            config["validation"]["stage1_bootstrap_replicates"]
        ),
        block_lengths=block_lengths,
        seed=seed + 30_000,
    )
    q_direction_improvements = [
        float(
            direction["models"]["Q1_CHANNEL_LINEAR"][
                "relative_improvement"
            ]
        )
        for direction in directions
    ]
    gates = config["gates"]
    s1_pass = bool(
        bool(profile["confirmatory"])
        and all(
            value > float(gates["s1_min_direction_improvement"])
            for value in q_direction_improvements
        )
        and pooled_q_improvement
        >= float(gates["s1_min_pooled_improvement"])
        and pooled_q_bootstrap["positive_probability"]
        >= float(gates["s1_min_bootstrap_positive_probability"])
    )
    result = {
        "schema": config["schema"],
        "task_id": profile_task_id(profile["id"], variant),
        "profile": profile,
        "variant": variant,
        "resolution_multiplier": float(resolution_multiplier),
        "status": "COMPLETED",
        "confirmatory": bool(profile["confirmatory"]),
        "directions": directions,
        "pooled": {
            "q_improvement": pooled_q_improvement,
            "conditional_improvement": pooled_conditional,
            "q_bootstrap": pooled_q_bootstrap,
            "conditional_bootstrap": pooled_conditional_bootstrap,
        },
        "gates": {
            "S1_candidate": s1_pass,
            "S1_status": (
                "S1_CANDIDATE_PASS"
                if s1_pass
                else (
                    "EXPLORATORY_NOT_ELIGIBLE"
                    if not bool(profile["confirmatory"])
                    else "S1_CANDIDATE_FAIL"
                )
            ),
        },
    }
    return result, all_arrays
