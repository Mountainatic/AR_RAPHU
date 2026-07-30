"""CPU benchmark execution, physics-first residual AR, and result aggregation."""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .common import (
    atomic_json,
    atomic_npz,
    circular_block_bootstrap,
    improvement,
    load_json,
    pooled_two_direction_bootstrap,
    regression_metrics,
    sha256_array,
    write_csv,
)
from .models import model_factory, parameter_count


GROUP_SIMPLE = "SIMPLE_BASELINES"
GROUP_SOFT = "SOFT_SENSOR_CLASSICAL"
GROUP_SYSTEM = "SYSTEM_IDENTIFICATION"
GROUP_PHYSICS = "PHYSICS_FIRST_MODELS"


MODEL_SPECS: dict[str, dict[str, Any]] = {
    # Input-driven soft sensors.
    "Ridge": {"group": GROUP_SOFT, "factory": "Ridge", "status": "STANDARD"},
    "Elastic-Net": {"group": GROUP_SOFT, "factory": "Elastic-Net", "status": "STANDARD"},
    "PCR": {"group": GROUP_SOFT, "factory": "PCR", "status": "STANDARD"},
    "PLS": {"group": GROUP_SOFT, "factory": "PLS", "status": "STANDARD"},
    "Dynamic-PLS": {"group": GROUP_SOFT, "factory": "Dynamic-PLS", "status": "ADAPTED_IMPLEMENTATION"},
    "Kernel-Ridge-RBF": {"group": GROUP_SOFT, "factory": "Kernel-Ridge-RBF", "status": "STANDARD_SUBSAMPLED_FIT"},
    "Nystroem-Ridge": {"group": GROUP_SOFT, "factory": "Nystroem-Ridge", "status": "STANDARD"},
    "SVR-RBF": {"group": GROUP_SOFT, "factory": "SVR-RBF", "status": "STANDARD"},
    "Random-Forest": {"group": GROUP_SOFT, "factory": "Random-Forest", "status": "STANDARD"},
    "Extra-Trees": {"group": GROUP_SOFT, "factory": "Extra-Trees", "status": "STANDARD"},
    "HistGradientBoosting": {"group": GROUP_SOFT, "factory": "HistGradientBoosting", "status": "STANDARD"},
    "XGBoost": {"group": GROUP_SOFT, "factory": "XGBoost", "status": "STANDARD"},
    "MLP-small": {"group": GROUP_SOFT, "factory": "MLP-small", "status": "CPU_BASELINE"},
    "Parallel-Hammerstein": {"group": GROUP_SOFT, "factory": "Parallel-Hammerstein", "status": "ADAPTED_IMPLEMENTATION"},
    "Hammerstein-Wiener-adapted": {"group": GROUP_SOFT, "factory": "Hammerstein-Wiener-adapted", "status": "ADAPTED_IMPLEMENTATION"},
    # Classical system identification.
    "AR": {"group": GROUP_SYSTEM, "factory": "AR", "status": "STANDARD_RIDGE_AR"},
    "Differenced-AR": {"group": GROUP_SYSTEM, "factory": "Differenced-AR", "status": "STANDARD_RIDGE_AR"},
    "ARX": {"group": GROUP_SYSTEM, "factory": "ARX", "status": "STANDARD_RIDGE_ARX"},
    "ARMAX": {"group": GROUP_SYSTEM, "factory": "ARMAX", "status": "ADAPTED_IMPLEMENTATION"},
    "Output-Error": {"group": GROUP_SYSTEM, "factory": "Output-Error", "status": "ADAPTED_IMPLEMENTATION"},
    "Box-Jenkins": {"group": GROUP_SYSTEM, "factory": "Box-Jenkins", "status": "ADAPTED_IMPLEMENTATION"},
    "N4SID-adapted": {"group": GROUP_SYSTEM, "factory": "N4SID-adapted", "status": "ADAPTED_IMPLEMENTATION"},
    "FIR-Ridge": {"group": GROUP_SYSTEM, "factory": "FIR-Ridge", "status": "STANDARD"},
    "TC-kernel": {"group": GROUP_SYSTEM, "factory": "Ridge", "status": "ADAPTED_IMPLEMENTATION"},
    "DC-kernel": {"group": GROUP_SYSTEM, "factory": "Ridge", "status": "ADAPTED_IMPLEMENTATION"},
    "First-order-stable-spline": {"group": GROUP_SYSTEM, "factory": "Ridge", "status": "ADAPTED_IMPLEMENTATION"},
    "Polynomial-NARX": {"group": GROUP_SYSTEM, "factory": "Polynomial-NARX", "status": "ADAPTED_IMPLEMENTATION"},
    "Spline-NARX": {"group": GROUP_SYSTEM, "factory": "Spline-NARX", "status": "ADAPTED_IMPLEMENTATION"},
    "PNLSS-adapted": {"group": GROUP_SYSTEM, "factory": "PNLSS-adapted", "status": "ADAPTED_IMPLEMENTATION"},
}


IMPLEMENTATION_NOTES = {
    "Dynamic-PLS": "PLS on multiresolution block levels plus within-channel block differences.",
    "Kernel-Ridge-RBF": "Exact RBF kernel ridge with deterministic maximum 1500-row training subset per fit.",
    "Parallel-Hammerstein": "Separable degree-2/3 polynomial static maps followed by a shared linear dynamic readout.",
    "Hammerstein-Wiener-adapted": "Separable cubic input map with train-only isotonic output calibration.",
    "ARMAX": "Regularized ARX augmented by mature output-difference proxy terms; not an innovations-MLE reproduction.",
    "Output-Error": "Regularized input-only multiresolution OE surrogate.",
    "Box-Jenkins": "Regularized input, output, and output-difference blocks; not a full polynomial noise-model search.",
    "N4SID-adapted": "PCA latent state extraction from causal U/Y history followed by ridge output map.",
    "TC-kernel": "Finite multiresolution lag basis transformed by a tuned-correlated prior covariance.",
    "DC-kernel": "Finite multiresolution lag basis transformed by a diagonal-correlated prior covariance.",
    "First-order-stable-spline": "Finite multiresolution lag basis transformed by a first-order stable-spline covariance.",
    "Polynomial-NARX": "PCA-reduced causal U/Y history with polynomial regression.",
    "Spline-NARX": "PCA-reduced causal U/Y history with cubic spline regression.",
    "PNLSS-adapted": "PCA latent state surrogate with polynomial state/output map; not a full PNLSS optimizer.",
}


def _configs(name: str, cpu: dict[str, Any]) -> list[dict[str, Any]]:
    alphas = cpu["ridge_alphas"]
    if name in {"Ridge", "FIR-Ridge", "Output-Error"}:
        return [{"alpha": value} for value in alphas]
    if name == "Elastic-Net":
        return [
            {"alpha": alpha, "l1_ratio": ratio}
            for alpha in cpu["elastic_net"]["alpha"]
            for ratio in cpu["elastic_net"]["l1_ratio"]
        ]
    if name in {"PCR", "PLS", "Dynamic-PLS"}:
        return [
            {"components": component, "alpha": 1.0}
            for component in cpu["latent_components"]
        ]
    if name == "Kernel-Ridge-RBF":
        return [
            {"alpha": alpha, "gamma": gamma}
            for alpha in (0.1, 10.0)
            for gamma in (0.01, 0.1)
        ]
    if name == "Nystroem-Ridge":
        return [
            {"alpha": alpha, "gamma": gamma, "components": 128}
            for alpha in (0.1, 10.0)
            for gamma in (0.01, 0.1)
        ]
    if name == "SVR-RBF":
        return [
            {"C": c, "gamma": gamma, "epsilon": 0.01}
            for c in (0.1, 1.0, 10.0)
            for gamma in (0.01, 0.1)
        ]
    if name in {"Random-Forest", "Extra-Trees"}:
        return [
            {"estimators": estimators, "depth": depth}
            for estimators in cpu["tree_estimators"]
            for depth in (6, None)
        ]
    if name == "HistGradientBoosting":
        return [
            {
                "iterations": iterations,
                "depth": depth,
                "learning_rate": 0.05,
                "alpha": 1.0,
            }
            for iterations in (100, 300)
            for depth in (3, 6)
        ]
    if name == "XGBoost":
        return [
            {
                "estimators": estimators,
                "depth": depth,
                "learning_rate": 0.05,
                "alpha": 1.0,
            }
            for estimators in (100, 300)
            for depth in (3, 6)
        ]
    if name == "MLP-small":
        return [
            {"hidden": hidden, "alpha": alpha, "learning_rate": 0.001}
            for hidden in ([16], [32], [32, 16])
            for alpha in (0.001, 0.1)
        ][:6]
    if name == "Parallel-Hammerstein":
        return [
            {"degree": degree, "alpha": alpha}
            for degree in (2, 3)
            for alpha in (0.01, 1.0, 100.0)
        ]
    if name == "Hammerstein-Wiener-adapted":
        return [{"alpha": alpha} for alpha in (0.01, 1.0, 100.0)]
    if name in {"AR", "Differenced-AR"}:
        return [
            {"history_min": history, "alpha": alpha}
            for history in cpu["ar_history_min"]
            for alpha in (0.01, 1.0, 100.0)
        ]
    if name in {"ARX", "ARMAX", "Box-Jenkins"}:
        return [
            {"history_min": history, "alpha": alpha}
            for history in (5.0, 20.0, 40.0)
            for alpha in (0.01, 1.0, 100.0)
        ]
    if name == "N4SID-adapted":
        return [
            {"components": order, "history_min": 20.0, "alpha": 1.0}
            for order in cpu["state_orders"]
        ]
    if name in {"TC-kernel", "First-order-stable-spline"}:
        return [
            {"beta": beta, "alpha": alpha}
            for beta in (0.8, 0.95)
            for alpha in (0.01, 1.0, 100.0)
        ]
    if name == "DC-kernel":
        return [
            {"beta": beta, "rho": rho, "alpha": alpha}
            for beta in (0.8, 0.95)
            for rho in (0.5, 0.9)
            for alpha in (0.1, 10.0)
        ]
    if name in {"Polynomial-NARX", "PNLSS-adapted"}:
        return [
            {"components": component, "degree": degree, "alpha": alpha}
            for component in (4, 8)
            for degree in (2, 3)
            for alpha in (0.1, 10.0)
        ]
    if name == "Spline-NARX":
        return [
            {"components": component, "knots": knots, "alpha": alpha}
            for component in (4, 8)
            for knots in (4, 6)
            for alpha in (0.1, 10.0)
        ]
    raise KeyError(name)


def _load_direction(shared_root: Path, direction: str) -> dict[str, Any]:
    root = shared_root / direction
    with np.load(
        root / "multiresolution_tabular_view" / "train.npz"
    ) as stored:
        train = {name: stored[name] for name in stored.files}
    with np.load(
        root / "multiresolution_tabular_view" / "test.npz"
    ) as stored:
        test = {name: stored[name] for name in stored.files}
    with np.load(root / "sequence_view" / "train.npz") as stored:
        train.update({name: stored[name] for name in stored.files})
    with np.load(root / "sequence_view" / "test.npz") as stored:
        test.update({name: stored[name] for name in stored.files})
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return {"train": train, "test": test, "metadata": metadata}


def _kernel_prior_features(
    matrix: np.ndarray,
    *,
    lag_count: int,
    name: str,
    parameters: dict[str, Any],
) -> np.ndarray:
    transformed = []
    beta = float(parameters["beta"])
    rho = float(parameters.get("rho", 1.0))
    index = np.arange(lag_count, dtype=np.float64)
    for channel in range(4):
        block = matrix[:, channel * lag_count : (channel + 1) * lag_count]
        if name in {"TC-kernel", "First-order-stable-spline"}:
            covariance = beta ** np.maximum.outer(index, index)
        else:
            covariance = (
                beta ** ((index[:, None] + index[None, :]) / 2.0)
                * rho ** np.abs(index[:, None] - index[None, :])
            )
        covariance += 1.0e-10 * np.eye(lag_count)
        values, vectors = np.linalg.eigh(covariance)
        root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
        transformed.append(block @ root)
    return np.column_stack(transformed)


def _feature_matrix(
    name: str,
    values: dict[str, np.ndarray],
    metadata: dict[str, Any],
    parameters: dict[str, Any],
) -> np.ndarray:
    u = np.asarray(values["multiresolution_u"], dtype=np.float64)
    y = np.asarray(values["sequence_y_centered"], dtype=np.float64)
    lag_count = len(metadata["lag_blocks"])
    if name in {
        "Ridge", "Elastic-Net", "PCR", "PLS", "Kernel-Ridge-RBF",
        "Nystroem-Ridge", "SVR-RBF", "Random-Forest", "Extra-Trees",
        "HistGradientBoosting", "XGBoost", "MLP-small",
        "Parallel-Hammerstein", "Hammerstein-Wiener-adapted",
        "FIR-Ridge", "Output-Error",
    }:
        return u
    if name == "Dynamic-PLS":
        reshaped = u.reshape(len(u), 4, lag_count)
        return np.column_stack((u, np.diff(reshaped, axis=2).reshape(len(u), -1)))
    history = int(round(float(parameters.get("history_min", 20.0)) * 6.0))
    history = min(history, y.shape[1])
    y_level = y[:, :history]
    if name == "AR":
        return y_level
    if name == "Differenced-AR":
        return np.diff(y_level, axis=1)
    if name in {"ARX", "Joint-ARX"}:
        return np.column_stack((u, y_level))
    if name == "ARMAX":
        return np.column_stack((u, y_level, np.diff(y_level, axis=1)))
    if name == "Box-Jenkins":
        return np.column_stack((u, y_level, np.diff(y_level, axis=1)))
    if name == "N4SID-adapted":
        return np.column_stack((u, y_level))
    if name in {"TC-kernel", "DC-kernel", "First-order-stable-spline"}:
        return _kernel_prior_features(
            u,
            lag_count=lag_count,
            name=name,
            parameters=parameters,
        )
    if name in {"Polynomial-NARX", "Spline-NARX", "PNLSS-adapted"}:
        return np.column_stack((u, y_level))
    raise KeyError(f"FEATURES_NOT_DEFINED:{name}")


def _complexity(
    name: str, parameters: dict[str, Any], dimension: int
) -> tuple:
    alpha = float(parameters.get("alpha", 0.0))
    if "components" in parameters:
        return (int(parameters["components"]), dimension, -alpha)
    if "history_min" in parameters:
        return (float(parameters["history_min"]), dimension, -alpha)
    if "degree" in parameters:
        return (int(parameters["degree"]), dimension, -alpha)
    if "estimators" in parameters:
        depth = parameters.get("depth")
        return (
            int(parameters["estimators"]),
            999 if depth is None else int(depth),
        )
    if "iterations" in parameters:
        return (
            int(parameters["iterations"]),
            int(parameters["depth"]),
            -alpha,
        )
    return (dimension, -alpha)


def _fit_indices(name: str, indices: np.ndarray) -> np.ndarray:
    if name != "Kernel-Ridge-RBF" or len(indices) <= 1500:
        return indices
    positions = np.linspace(0, len(indices) - 1, 1500).astype(np.int64)
    return indices[positions]


def _folds_for_direction(
    origins: np.ndarray, protocol: dict[str, Any], repo_root: Path
):
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from MULTISCALE_PHYSICS_AUDIT_V1.src.validation import rolling_origin_folds

    purge_raw = int(
        round(
            float(protocol["purge_min"])
            * 60.0
            / float(protocol["sample_period_sec"])
        )
    )
    return rolling_origin_folds(
        origins,
        protocol["inner_folds"],
        purge_samples=purge_raw,
    )


def evaluate_model_task(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        shared_root = Path(payload["shared_root"])
        repo_root = Path(payload["repo_root"])
        direction = payload["direction"]
        name = payload["name"]
        protocol = payload["protocol"]
        cpu = payload["cpu"]
        data = _load_direction(shared_root, direction)
        train, test, metadata = data["train"], data["test"], data["metadata"]
        target = np.asarray(train["target_z"], dtype=np.float64)
        folds = _folds_for_direction(
            train["origin_raw_index"], protocol, repo_root
        )
        configs = _configs(name, cpu)
        if len(configs) > int(cpu["maximum_configs_per_model"]):
            raise RuntimeError(f"CONFIG_BUDGET_EXCEEDED:{name}")
        records = []
        started = time.perf_counter()
        process = psutil.Process(os.getpid())
        peak_rss = process.memory_info().rss
        for config_index, parameters in enumerate(configs):
            matrix = _feature_matrix(name, train, metadata, parameters)
            fold_losses = []
            for fold in folds:
                fit_indices = _fit_indices(name, fold.train_indices)
                model = model_factory(
                    MODEL_SPECS[name]["factory"], parameters
                )
                model.fit(matrix[fit_indices], target[fit_indices])
                prediction = np.asarray(
                    model.predict(matrix[fold.validation_indices])
                ).reshape(-1)
                fold_losses.append(
                    float(
                        np.mean(
                            (
                                target[fold.validation_indices] - prediction
                            )
                            ** 2
                        )
                    )
                )
                peak_rss = max(peak_rss, process.memory_info().rss)
            records.append(
                {
                    "index": config_index,
                    "parameters": parameters,
                    "dimension": int(matrix.shape[1]),
                    "fold_mse": fold_losses,
                    "mean_mse": float(np.mean(fold_losses)),
                    "se": float(np.std(fold_losses, ddof=1) / np.sqrt(len(fold_losses))),
                    "complexity": list(_complexity(name, parameters, matrix.shape[1])),
                }
            )
        minimum = min(records, key=lambda value: (value["mean_mse"], value["index"]))
        threshold = minimum["mean_mse"] + minimum["se"]
        eligible = [value for value in records if value["mean_mse"] <= threshold]
        selected = min(
            eligible,
            key=lambda value: (tuple(value["complexity"]), value["index"]),
        )
        train_matrix = _feature_matrix(
            name, train, metadata, selected["parameters"]
        )
        test_matrix = _feature_matrix(
            name, test, metadata, selected["parameters"]
        )
        full_indices = _fit_indices(name, np.arange(len(target), dtype=np.int64))
        model = model_factory(
            MODEL_SPECS[name]["factory"], selected["parameters"]
        )
        model.fit(train_matrix[full_indices], target[full_indices])
        prediction = np.asarray(model.predict(test_matrix)).reshape(-1).astype(np.float64)
        mask = np.asarray(test["evaluation_mask"], dtype=bool)
        metrics = regression_metrics(test["target_z"][mask], prediction[mask])
        persistence_loss = np.asarray(test["target_z"][mask], dtype=np.float64) ** 2
        model_loss = (
            np.asarray(test["target_z"][mask], dtype=np.float64) - prediction[mask]
        ) ** 2
        elapsed = time.perf_counter() - started
        result = {
            "status": "COMPLETED",
            "name": name,
            "group": MODEL_SPECS[name]["group"],
            "implementation_status": MODEL_SPECS[name]["status"],
            "implementation_note": IMPLEMENTATION_NOTES.get(name, ""),
            "direction": direction,
            "selected_parameters": selected["parameters"],
            "selection": {
                "minimum_mean_mse": minimum["mean_mse"],
                "minimum_se": minimum["se"],
                "one_se_threshold": threshold,
                "selected_complexity": selected["complexity"],
                "records": records,
            },
            "metrics": metrics,
            "relative_persistence": improvement(persistence_loss, model_loss),
            "train_seconds": float(elapsed),
            "inference_seconds": float(0.0),
            "peak_rss_bytes": int(peak_rss),
            "parameter_count": int(parameter_count(model)),
            "prediction_sha256": sha256_array(prediction),
            "sample_id_sha256": sha256_array(test["sample_id"]),
            "fp64_prediction": str(prediction.dtype) == "float64",
            "test_rows": int(np.sum(mask)),
        }
        return {"result": result, "prediction": prediction}
    except Exception as exc:
        return {
            "result": {
                "status": "FAILED",
                "name": payload["name"],
                "group": MODEL_SPECS[payload["name"]]["group"],
                "direction": payload["direction"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            "prediction": None,
        }


def _simple_baselines(direction: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    train, test = data["train"], data["test"]
    target = np.asarray(test["target_z"], dtype=np.float64)
    mask = np.asarray(test["evaluation_mask"], dtype=bool)
    candidates = {
        "Persistence": np.zeros(len(target), dtype=np.float64),
        "Mean-drift": np.full(
            len(target), float(np.mean(train["target_z"])), dtype=np.float64
        ),
        "Local-linear-trend": np.asarray(
            test["local_trend_prediction"], dtype=np.float64
        ),
    }
    rows = []
    for name, prediction in candidates.items():
        metrics = regression_metrics(target[mask], prediction[mask])
        rows.append(
            {
                "status": "COMPLETED",
                "name": name,
                "group": GROUP_SIMPLE,
                "direction": direction,
                "implementation_status": "STANDARD",
                "metrics": metrics,
                "relative_persistence": improvement(
                    target[mask] ** 2,
                    (target[mask] - prediction[mask]) ** 2,
                ),
                "selected_parameters": {},
                "train_seconds": 0.0,
                "peak_rss_bytes": 0,
                "parameter_count": 0 if name == "Persistence" else 1,
                "prediction_sha256": sha256_array(prediction),
                "sample_id_sha256": sha256_array(test["sample_id"]),
                "fp64_prediction": True,
                "test_rows": int(np.sum(mask)),
                "_prediction": prediction,
            }
        )
    # Seasonal drift is enabled only when a mature candidate lag has a clear
    # training autocorrelation peak. It remains explicitly registered otherwise.
    train_target = np.asarray(train["target_z"], dtype=np.float64)
    candidate_lags = [132, 180, 240, 360]
    correlations = {}
    for lag in candidate_lags:
        if lag < len(train_target) // 2:
            correlations[lag] = float(
                np.corrcoef(train_target[:-lag], train_target[lag:])[0, 1]
            )
    if correlations and max(abs(value) for value in correlations.values()) >= 0.5:
        lag = max(correlations, key=lambda value: abs(correlations[value]))
        prediction = np.zeros(len(target), dtype=np.float64)
        prediction[lag:] = target[:-lag]
        seasonal_mask = mask & (np.arange(len(mask)) >= lag)
        metrics = regression_metrics(target[seasonal_mask], prediction[seasonal_mask])
        rows.append(
            {
                "status": "COMPLETED",
                "name": "Seasonal-blocked-drift",
                "group": GROUP_SIMPLE,
                "direction": direction,
                "implementation_status": "TRAIN_PERIODICITY_GATE_PASS",
                "metrics": metrics,
                "relative_persistence": improvement(
                    target[seasonal_mask] ** 2,
                    (target[seasonal_mask] - prediction[seasonal_mask]) ** 2,
                ),
                "selected_parameters": {
                    "lag_rows": lag,
                    "train_autocorrelation": correlations[lag],
                },
                "test_rows": int(np.sum(seasonal_mask)),
                "_prediction": prediction,
            }
        )
    else:
        rows.append(
            {
                "status": "NOT_APPLICABLE",
                "name": "Seasonal-blocked-drift",
                "group": GROUP_SIMPLE,
                "direction": direction,
                "implementation_status": "TRAIN_PERIODICITY_GATE_FAIL",
                "selected_parameters": {"candidate_correlations": correlations},
            }
        )
    return rows


def _custom_ridge_api(repo_root: Path):
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from MULTISCALE_PHYSICS_AUDIT_V1.src.linear_q import fit_ridge

    return fit_ridge


def _select_variable_ridge(
    feature_builder,
    configs: list[dict[str, Any]],
    target: np.ndarray,
    folds: list[Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fit_ridge = _custom_ridge_api(repo_root)
    records = []
    for index, config in enumerate(configs):
        matrix = feature_builder(config)
        losses = []
        for fold in folds:
            fit = fit_ridge(
                matrix[fold.train_indices],
                target[fold.train_indices],
                alpha=float(config["alpha"]),
            )
            prediction = fit.predict(matrix[fold.validation_indices])
            losses.append(
                float(
                    np.mean(
                        (
                            target[fold.validation_indices] - prediction
                        )
                        ** 2
                    )
                )
            )
        dimension = int(matrix.shape[1])
        complexity = (
            float(config.get("history_min", 0.0)),
            dimension,
            -float(config["alpha"]),
        )
        records.append(
            {
                "index": index,
                "parameters": config,
                "dimension": dimension,
                "fold_mse": losses,
                "mean_mse": float(np.mean(losses)),
                "se": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
                "complexity": list(complexity),
            }
        )
    minimum = min(records, key=lambda value: (value["mean_mse"], value["index"]))
    threshold = minimum["mean_mse"] + minimum["se"]
    eligible = [value for value in records if value["mean_mse"] <= threshold]
    selected = min(
        eligible,
        key=lambda value: (tuple(value["complexity"]), value["index"]),
    )
    return selected["parameters"], records


def _oof_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    folds: list[Any],
    *,
    alpha: float,
    repo_root: Path,
) -> np.ndarray:
    fit_ridge = _custom_ridge_api(repo_root)
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    for fold in folds:
        fit = fit_ridge(
            matrix[fold.train_indices],
            target[fold.train_indices],
            alpha=alpha,
        )
        prediction[fold.validation_indices] = fit.predict(
            matrix[fold.validation_indices]
        )
    return prediction


def _residual_history_design(
    residual: np.ndarray,
    *,
    maturity_rows: int,
    maximum_history_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(residual, dtype=np.float64)
    rows = []
    features = []
    first = maturity_rows + maximum_history_rows - 1
    for current in range(first, len(values)):
        end = current - maturity_rows
        history = values[
            end - maximum_history_rows + 1 : end + 1
        ][::-1]
        if len(history) == maximum_history_rows and np.all(np.isfinite(history)):
            rows.append(current)
            features.append(history)
    if not rows:
        raise ValueError("NO_MATURE_RESIDUAL_ROWS")
    return np.asarray(rows, dtype=np.int64), np.asarray(features, dtype=np.float64)


def _select_residual_ar(
    *,
    residual: np.ndarray,
    origins: np.ndarray,
    protocol: dict[str, Any],
    repo_root: Path,
    alphas: list[float],
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    fit_ridge = _custom_ridge_api(repo_root)
    maturity = int(
        round(
            float(protocol["residual_ar"]["maturity_min"])
            * 60.0
            / float(protocol["cadence_sec"])
        )
    )
    maximum = int(
        round(
            max(protocol["residual_ar"]["history_candidates_min"])
            * 60.0
            / float(protocol["cadence_sec"])
        )
    )
    rows, full = _residual_history_design(
        residual,
        maturity_rows=maturity,
        maximum_history_rows=maximum,
    )
    folds = _folds_for_direction(origins[rows], protocol, repo_root)
    target = residual[rows]
    records = []
    zero_losses = [
        float(np.mean(target[fold.validation_indices] ** 2)) for fold in folds
    ]
    records.append(
        {
            "parameters": {"exact_zero": True, "history_min": 0.0, "alpha": None},
            "fold_mse": zero_losses,
            "mean_mse": float(np.mean(zero_losses)),
            "se": float(np.std(zero_losses, ddof=1) / np.sqrt(len(zero_losses))),
            "complexity": [0.0, 0, 0.0],
        }
    )
    for history_min in protocol["residual_ar"]["history_candidates_min"]:
        width = int(round(float(history_min) * 60.0 / protocol["cadence_sec"]))
        matrix = full[:, :width]
        for alpha in alphas:
            losses = []
            for fold in folds:
                fit = fit_ridge(
                    matrix[fold.train_indices],
                    target[fold.train_indices],
                    alpha=float(alpha),
                )
                prediction = fit.predict(matrix[fold.validation_indices])
                losses.append(
                    float(
                        np.mean(
                            (
                                target[fold.validation_indices] - prediction
                            )
                            ** 2
                        )
                    )
                )
            records.append(
                {
                    "parameters": {
                        "exact_zero": False,
                        "history_min": float(history_min),
                        "alpha": float(alpha),
                    },
                    "fold_mse": losses,
                    "mean_mse": float(np.mean(losses)),
                    "se": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
                    "complexity": [float(history_min), width, -float(alpha)],
                }
            )
    minimum = min(records, key=lambda value: value["mean_mse"])
    threshold = minimum["mean_mse"] + minimum["se"]
    eligible = [value for value in records if value["mean_mse"] <= threshold]
    selected = min(eligible, key=lambda value: tuple(value["complexity"]))
    parameters = selected["parameters"]
    if parameters["exact_zero"]:
        fit = None
    else:
        width = int(
            round(
                parameters["history_min"]
                * 60.0
                / float(protocol["cadence_sec"])
            )
        )
        fit = fit_ridge(full[:, :width], target, alpha=parameters["alpha"])
    return parameters, fit, rows, full, records


def _predict_residual_ar(
    fit,
    parameters: dict[str, Any],
    residual: np.ndarray,
    *,
    protocol: dict[str, Any],
) -> np.ndarray:
    prediction = np.zeros(len(residual), dtype=np.float64)
    if parameters["exact_zero"]:
        return prediction
    maturity = int(
        round(
            float(protocol["residual_ar"]["maturity_min"])
            * 60.0
            / float(protocol["cadence_sec"])
        )
    )
    width = int(
        round(
            float(parameters["history_min"])
            * 60.0
            / float(protocol["cadence_sec"])
        )
    )
    for current in range(maturity + width - 1, len(residual)):
        end = current - maturity
        history = residual[end - width + 1 : end + 1][::-1]
        prediction[current] = fit.predict(history[None, :])[0]
    return prediction


def evaluate_physics_direction(
    *,
    shared_root: Path,
    direction: str,
    protocol: dict[str, Any],
    cpu: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    fit_ridge = _custom_ridge_api(repo_root)
    data = _load_direction(shared_root, direction)
    train, test, metadata = data["train"], data["test"], data["metadata"]
    y_train = np.asarray(train["target_z"], dtype=np.float64)
    y_test = np.asarray(test["target_z"], dtype=np.float64)
    mask = np.asarray(test["evaluation_mask"], dtype=bool)
    folds = _folds_for_direction(train["origin_raw_index"], protocol, repo_root)
    lag_count = len(metadata["lag_blocks"])
    lift_train = np.asarray(train["multiresolution_u"][:, :lag_count], dtype=np.float64)
    lift_test = np.asarray(test["multiresolution_u"][:, :lag_count], dtype=np.float64)
    alphas = [float(value) for value in cpu["ridge_alphas"]]
    k_parameters, k_trace = _select_variable_ridge(
        lambda _: lift_train,
        [{"alpha": alpha, "history_min": 40.0} for alpha in alphas],
        y_train,
        folds,
        repo_root=repo_root,
    )
    k_fit = fit_ridge(lift_train, y_train, alpha=k_parameters["alpha"])
    k_train_prediction = k_fit.predict(lift_train)
    k_test_prediction = k_fit.predict(lift_test)
    k_oof = _oof_ridge(
        lift_train,
        y_train,
        folds,
        alpha=k_parameters["alpha"],
        repo_root=repo_root,
    )
    k_oof_residual = y_train - k_oof
    residual_parameters, residual_fit, residual_rows, residual_design, residual_trace = (
        _select_residual_ar(
            residual=k_oof_residual,
            origins=train["origin_raw_index"],
            protocol=protocol,
            repo_root=repo_root,
            alphas=alphas,
        )
    )
    test_k_residual = y_test - k_test_prediction
    residual_test_prediction = _predict_residual_ar(
        residual_fit,
        residual_parameters,
        test_k_residual,
        protocol=protocol,
    )
    k_then_ar = k_test_prediction + residual_test_prediction

    ar_configs = [
        {"history_min": history, "alpha": alpha}
        for history in cpu["ar_history_min"]
        for alpha in (0.01, 1.0, 100.0)
    ]
    ar_parameters, ar_trace = _select_variable_ridge(
        lambda config: np.asarray(
            train["sequence_y_centered"][
                :,
                : int(round(config["history_min"] * 6.0)),
            ],
            dtype=np.float64,
        ),
        ar_configs,
        y_train,
        folds,
        repo_root=repo_root,
    )
    ar_width = int(round(ar_parameters["history_min"] * 6.0))
    ar_train = np.asarray(
        train["sequence_y_centered"][:, :ar_width], dtype=np.float64
    )
    ar_test = np.asarray(
        test["sequence_y_centered"][:, :ar_width], dtype=np.float64
    )
    ar_fit = fit_ridge(ar_train, y_train, alpha=ar_parameters["alpha"])
    ar_test_prediction = ar_fit.predict(ar_test)
    ar_oof = _oof_ridge(
        ar_train,
        y_train,
        folds,
        alpha=ar_parameters["alpha"],
        repo_root=repo_root,
    )
    ar_residual = y_train - ar_oof
    finite_ar = np.flatnonzero(np.isfinite(ar_residual))
    local_folds = _folds_for_direction(
        train["origin_raw_index"][finite_ar], protocol, repo_root
    )
    ar_to_k_parameters, ar_to_k_trace = _select_variable_ridge(
        lambda _: lift_train[finite_ar],
        [{"alpha": alpha, "history_min": 40.0} for alpha in alphas],
        ar_residual[finite_ar],
        local_folds,
        repo_root=repo_root,
    )
    ar_to_k_fit = fit_ridge(
        lift_train[finite_ar],
        ar_residual[finite_ar],
        alpha=ar_to_k_parameters["alpha"],
    )
    ar_then_k = ar_test_prediction + ar_to_k_fit.predict(lift_test)

    u_train = np.asarray(train["multiresolution_u"], dtype=np.float64)
    u_test = np.asarray(test["multiresolution_u"], dtype=np.float64)
    joint_arx_parameters, joint_arx_trace = _select_variable_ridge(
        lambda config: np.column_stack(
            (
                u_train,
                np.asarray(
                    train["sequence_y_centered"][
                        :,
                        : int(round(config["history_min"] * 6.0)),
                    ],
                    dtype=np.float64,
                ),
            )
        ),
        [
            {"history_min": history, "alpha": alpha}
            for history in (5.0, 20.0, 40.0)
            for alpha in (0.01, 1.0, 100.0)
        ],
        y_train,
        folds,
        repo_root=repo_root,
    )
    joint_width = int(round(joint_arx_parameters["history_min"] * 6.0))
    joint_arx_train = np.column_stack(
        (
            u_train,
            np.asarray(train["sequence_y_centered"][:, :joint_width], dtype=np.float64),
        )
    )
    joint_arx_test = np.column_stack(
        (
            u_test,
            np.asarray(test["sequence_y_centered"][:, :joint_width], dtype=np.float64),
        )
    )
    joint_arx_fit = fit_ridge(
        joint_arx_train, y_train, alpha=joint_arx_parameters["alpha"]
    )
    joint_arx_prediction = joint_arx_fit.predict(joint_arx_test)

    joint_k_ar_parameters, joint_k_ar_trace = _select_variable_ridge(
        lambda config: np.column_stack(
            (
                lift_train,
                np.asarray(
                    train["sequence_y_centered"][
                        :,
                        : int(round(config["history_min"] * 6.0)),
                    ],
                    dtype=np.float64,
                ),
            )
        ),
        [
            {"history_min": history, "alpha": alpha}
            for history in (5.0, 20.0, 40.0)
            for alpha in (0.01, 1.0, 100.0)
        ],
        y_train,
        folds,
        repo_root=repo_root,
    )
    joint_k_width = int(round(joint_k_ar_parameters["history_min"] * 6.0))
    joint_k_train = np.column_stack(
        (
            lift_train,
            np.asarray(train["sequence_y_centered"][:, :joint_k_width], dtype=np.float64),
        )
    )
    joint_k_test = np.column_stack(
        (
            lift_test,
            np.asarray(test["sequence_y_centered"][:, :joint_k_width], dtype=np.float64),
        )
    )
    joint_k_fit = fit_ridge(
        joint_k_train, y_train, alpha=joint_k_ar_parameters["alpha"]
    )
    joint_k_prediction = joint_k_fit.predict(joint_k_test)

    # Residual state-space: PCA states from mature residual history.
    state_records = []
    state_target = k_oof_residual[residual_rows]
    residual_origin = train["origin_raw_index"][residual_rows]
    state_folds = _folds_for_direction(residual_origin, protocol, repo_root)
    for order in cpu["state_orders"]:
        for alpha in (0.1, 10.0):
            losses = []
            for fold in state_folds:
                model = model_factory(
                    "PCR", {"components": order, "alpha": alpha}
                )
                model.fit(
                    residual_design[fold.train_indices],
                    state_target[fold.train_indices],
                )
                prediction = np.asarray(
                    model.predict(residual_design[fold.validation_indices])
                ).reshape(-1)
                losses.append(
                    float(
                        np.mean(
                            (
                                state_target[fold.validation_indices] - prediction
                            )
                            ** 2
                        )
                    )
                )
            state_records.append(
                {
                    "parameters": {"components": order, "alpha": alpha},
                    "fold_mse": losses,
                    "mean_mse": float(np.mean(losses)),
                    "se": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
                    "complexity": [order, -alpha],
                }
            )
    state_min = min(state_records, key=lambda value: value["mean_mse"])
    state_threshold = state_min["mean_mse"] + state_min["se"]
    state_selected = min(
        [value for value in state_records if value["mean_mse"] <= state_threshold],
        key=lambda value: tuple(value["complexity"]),
    )
    state_model = model_factory("PCR", state_selected["parameters"])
    state_model.fit(residual_design, state_target)
    maturity = metadata["maturity_rows"]
    maximum = metadata["max_residual_history_rows"]
    test_rows, test_residual_design = _residual_history_design(
        test_k_residual,
        maturity_rows=maturity,
        maximum_history_rows=maximum,
    )
    state_residual_prediction = np.zeros(len(y_test), dtype=np.float64)
    state_residual_prediction[test_rows] = np.asarray(
        state_model.predict(test_residual_design)
    ).reshape(-1)
    k_then_state = k_test_prediction + state_residual_prediction

    predictions = {
        "K-only": k_test_prediction,
        "Residual-AR-only-on-K-residual": residual_test_prediction,
        "K-to-Residual-AR": k_then_ar,
        "AR-only": ar_test_prediction,
        "AR-to-K": ar_then_k,
        "Joint-ARX": joint_arx_prediction,
        "Joint-K+AR": joint_k_prediction,
        "K-to-Residual-linear-state-space": k_then_state,
    }
    parameter_map = {
        "K-only": k_parameters,
        "Residual-AR-only-on-K-residual": residual_parameters,
        "K-to-Residual-AR": {"K": k_parameters, "residual_AR": residual_parameters},
        "AR-only": ar_parameters,
        "AR-to-K": {"AR": ar_parameters, "residual_K": ar_to_k_parameters},
        "Joint-ARX": joint_arx_parameters,
        "Joint-K+AR": joint_k_ar_parameters,
        "K-to-Residual-linear-state-space": {
            "K": k_parameters,
            "state": state_selected["parameters"],
        },
    }
    rows = []
    persistence_loss = y_test[mask] ** 2
    ar_loss = (y_test[mask] - ar_test_prediction[mask]) ** 2
    for name, prediction in predictions.items():
        model_loss = (y_test[mask] - prediction[mask]) ** 2
        rows.append(
            {
                "status": "COMPLETED",
                "name": name,
                "group": GROUP_PHYSICS,
                "direction": direction,
                "implementation_status": (
                    "FORMAL_PRIMARY"
                    if name == "K-to-Residual-AR"
                    else "STRUCTURAL_CONTROL"
                ),
                "selected_parameters": parameter_map[name],
                "metrics": regression_metrics(y_test[mask], prediction[mask]),
                "relative_persistence": improvement(persistence_loss, model_loss),
                "relative_AR": improvement(ar_loss, model_loss),
                "test_rows": int(np.sum(mask)),
                "prediction_sha256": sha256_array(prediction),
                "sample_id_sha256": sha256_array(test["sample_id"]),
                "fp64_prediction": True,
            }
        )
    standardized_lift = k_fit.standardizer.transform(lift_train)
    system = (
        standardized_lift.T @ standardized_lift / len(standardized_lift)
        + float(k_parameters["alpha"]) * np.eye(lift_train.shape[1])
    )
    kernel = {
        "direction": direction,
        "lag_midpoints_min": [
            item["midpoint_min"] for item in metadata["lag_blocks"]
        ],
        "physical_coefficients": k_fit.physical_coefficients().tolist(),
        "alpha": k_parameters["alpha"],
        "relative_kkt": k_fit.relative_kkt,
        "condition_number": float(np.linalg.cond(system)),
        "nonlinear_block": "EXACT_ZERO",
        "residual_ar_parameters": residual_parameters,
        "matured_residual_rows": int(len(residual_rows)),
        "maturity_rows": int(metadata["maturity_rows"]),
    }
    traces = {
        "K": k_trace,
        "residual_AR": residual_trace,
        "AR": ar_trace,
        "AR_to_K": ar_to_k_trace,
        "Joint_ARX": joint_arx_trace,
        "Joint_K_AR": joint_k_ar_trace,
        "state_space": state_records,
    }
    return rows, predictions, {"kernel": kernel, "selection_traces": traces}


def run_cpu_benchmark(
    *,
    repo_root: Path,
    project_root: Path,
    shared_root: Path,
    protocol_path: Path,
    cpu_path: Path,
    results_root: Path,
    n_jobs: int,
) -> dict[str, Any]:
    protocol, protocol_sha = load_json(protocol_path)
    cpu, cpu_sha = load_json(cpu_path)
    shared_protocol = json.loads(
        (shared_root / "BENCHMARK_PROTOCOL.json").read_text(encoding="utf-8")
    )
    if shared_protocol["config_sha256"] != protocol_sha:
        raise RuntimeError("SHARED_PROTOCOL_HASH_MISMATCH")
    directions = [
        f"{train.lower()}_to_{test.lower()}"
        for train, test in protocol["outer_directions"]
    ]
    results_root.mkdir(parents=True, exist_ok=True)
    prediction_root = results_root / "CPU_MODEL_PREDICTIONS"
    kernel_root = results_root / "KERNELS"
    all_rows: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    for direction in directions:
        data = _load_direction(shared_root, direction)
        baseline_rows = _simple_baselines(direction, data)
        for row in baseline_rows:
            prediction = row.pop("_prediction", None)
            if prediction is not None:
                atomic_npz(
                    prediction_root / direction / f"{row['name']}.npz",
                    sample_id=data["test"]["sample_id"],
                    prediction=np.asarray(prediction, dtype=np.float64),
                    target_z=np.asarray(data["test"]["target_z"], dtype=np.float64),
                    evaluation_mask=np.asarray(data["test"]["evaluation_mask"], dtype=bool),
                )
            all_rows.append(row)
    tasks = [
        {
            "shared_root": str(shared_root),
            "repo_root": str(repo_root),
            "direction": direction,
            "name": name,
            "protocol": protocol,
            "cpu": cpu,
        }
        for direction in directions
        for name in MODEL_SPECS
    ]
    task_root = results_root / "tasks"
    pending_tasks = []
    for task in tasks:
        task_file = task_root / f"{task['direction']}__{task['name']}.json"
        prediction_file = (
            prediction_root
            / task["direction"]
            / f"{task['name'].replace('/', '_')}.npz"
        )
        if task_file.is_file() and prediction_file.is_file():
            stored = json.loads(task_file.read_text(encoding="utf-8"))
            if stored.get("status") == "COMPLETED":
                raw_results.append(stored)
                all_rows.append(stored)
                print(
                    f"CPU_MODEL direction={stored['direction']} "
                    f"name={stored['name']} status=SKIPPED_COMPLETE",
                    flush=True,
                )
                continue
        pending_tasks.append(task)
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks))) as executor:
        futures = {
            executor.submit(evaluate_model_task, task): task
            for task in pending_tasks
        }
        for future in as_completed(futures):
            output = future.result()
            result = output["result"]
            raw_results.append(result)
            all_rows.append(result)
            atomic_json(
                task_root
                / f"{result['direction']}__{result['name']}.json",
                result,
            )
            if output["prediction"] is not None:
                direction = result["direction"]
                data = _load_direction(shared_root, direction)
                safe_name = result["name"].replace("/", "_")
                atomic_npz(
                    prediction_root / direction / f"{safe_name}.npz",
                    sample_id=data["test"]["sample_id"],
                    prediction=np.asarray(output["prediction"], dtype=np.float64),
                    target_z=np.asarray(data["test"]["target_z"], dtype=np.float64),
                    evaluation_mask=np.asarray(data["test"]["evaluation_mask"], dtype=bool),
                )
            print(
                f"CPU_MODEL direction={result['direction']} "
                f"name={result['name']} status={result['status']}",
                flush=True,
            )
    physics_details = {}
    for direction in directions:
        rows, predictions, details = evaluate_physics_direction(
            shared_root=shared_root,
            direction=direction,
            protocol=protocol,
            cpu=cpu,
            repo_root=repo_root,
        )
        data = _load_direction(shared_root, direction)
        for row in rows:
            all_rows.append(row)
            prediction = predictions[row["name"]]
            atomic_npz(
                prediction_root / direction / f"{row['name']}.npz",
                sample_id=data["test"]["sample_id"],
                prediction=np.asarray(prediction, dtype=np.float64),
                target_z=np.asarray(data["test"]["target_z"], dtype=np.float64),
                evaluation_mask=np.asarray(data["test"]["evaluation_mask"], dtype=bool),
            )
        physics_details[direction] = details
        atomic_json(kernel_root / f"{direction}_lift_kernel.json", details["kernel"])
        atomic_json(
            results_root / "selection_traces" / f"{direction}_physics.json",
            details["selection_traces"],
        )
        print(f"PHYSICS_MODELS direction={direction} status=COMPLETED", flush=True)
    for result in raw_results:
        atomic_json(
            results_root
            / "selection_traces"
            / f"{result['direction']}__{result['name']}.json",
            result,
        )
    summary = aggregate_results(
        shared_root=shared_root,
        results_root=results_root,
        rows=all_rows,
        physics_details=physics_details,
        protocol=protocol,
        protocol_sha=protocol_sha,
        cpu_sha=cpu_sha,
        directions=directions,
    )
    atomic_json(
        results_root / "checkpoints" / "latest.json",
        {
            "completed_stage": "CPU_BENCHMARK",
            "protocol_sha256": protocol_sha,
            "cpu_config_sha256": cpu_sha,
            "status": summary["status"],
        },
    )
    return summary


def _prediction_payload(
    results_root: Path, direction: str, name: str
) -> dict[str, np.ndarray]:
    path = (
        results_root
        / "CPU_MODEL_PREDICTIONS"
        / direction
        / f"{name.replace('/', '_')}.npz"
    )
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "metrics",
            "selection",
            "selected_parameters",
            "traceback",
        }
    }
    output["selected_parameters"] = json.dumps(
        row.get("selected_parameters", {}), ensure_ascii=False, sort_keys=True
    )
    for key, value in row.get("metrics", {}).items():
        output[key] = value
    return output


def _pooled_model_rows(
    rows: list[dict[str, Any]],
    *,
    results_root: Path,
    directions: list[str],
) -> list[dict[str, Any]]:
    pooled = []
    names = sorted(
        {
            row["name"]
            for row in rows
            if row.get("status") == "COMPLETED"
        }
    )
    for name in names:
        matching = [
            row
            for row in rows
            if row.get("name") == name
            and row.get("direction") in directions
            and row.get("status") == "COMPLETED"
        ]
        if len(matching) != len(directions):
            continue
        total_squared = 0.0
        total_absolute = 0.0
        total_count = 0
        target_parts = []
        prediction_parts = []
        persistence_sum = 0.0
        ar_sum = 0.0
        for direction in directions:
            payload = _prediction_payload(results_root, direction, name)
            mask = payload["evaluation_mask"].astype(bool)
            target = payload["target_z"][mask].astype(np.float64)
            prediction = payload["prediction"][mask].astype(np.float64)
            residual = target - prediction
            total_squared += float(np.sum(residual**2))
            total_absolute += float(np.sum(np.abs(residual)))
            total_count += len(target)
            persistence_sum += float(np.sum(target**2))
            ar_payload = _prediction_payload(results_root, direction, "AR-only")
            ar_prediction = ar_payload["prediction"][mask].astype(np.float64)
            ar_sum += float(np.sum((target - ar_prediction) ** 2))
            target_parts.append(target)
            prediction_parts.append(prediction)
        target_all = np.concatenate(target_parts)
        prediction_all = np.concatenate(prediction_parts)
        mse = total_squared / total_count
        variance = float(np.sum((target_all - target_all.mean()) ** 2))
        first = matching[0]
        pooled.append(
            {
                "status": "COMPLETED",
                "name": name,
                "group": first["group"],
                "direction": "POOLED",
                "implementation_status": first.get("implementation_status", ""),
                "selected_parameters": "DIRECTION_SPECIFIC",
                "MSE": mse,
                "RMSE": float(np.sqrt(mse)),
                "MAE": total_absolute / total_count,
                "R2": float(1.0 - total_squared / max(variance, np.finfo(float).eps)),
                "relative_persistence": float(
                    1.0 - total_squared / max(persistence_sum, np.finfo(float).eps)
                ),
                "relative_AR": float(
                    1.0 - total_squared / max(ar_sum, np.finfo(float).eps)
                ),
                "test_rows": total_count,
            }
        )
    return pooled


def _physics_audit(
    *,
    shared_root: Path,
    results_root: Path,
    physics_details: dict[str, Any],
    protocol: dict[str, Any],
    directions: list[str],
) -> dict[str, Any]:
    direction_results = {}
    pooled_pairs_k = []
    pooled_pairs_ar_given_k = []
    pooled_pairs_total = []
    sums = {
        "baseline": 0.0,
        "k": 0.0,
        "kar": 0.0,
        "ar": 0.0,
    }
    for direction in directions:
        k = _prediction_payload(results_root, direction, "K-only")
        kar = _prediction_payload(results_root, direction, "K-to-Residual-AR")
        ar = _prediction_payload(results_root, direction, "AR-only")
        data = _load_direction(shared_root, direction)
        mask = k["evaluation_mask"].astype(bool)
        target = k["target_z"][mask].astype(np.float64)
        k_prediction = k["prediction"][mask].astype(np.float64)
        kar_prediction = kar["prediction"][mask].astype(np.float64)
        ar_prediction = ar["prediction"][mask].astype(np.float64)
        baseline_loss = target**2
        k_loss = (target - k_prediction) ** 2
        kar_loss = (target - kar_prediction) ** 2
        ar_loss = (target - ar_prediction) ** 2
        g_k = improvement(baseline_loss, k_loss)
        g_ar_k = improvement(k_loss, kar_loss)
        g_total = improvement(baseline_loss, kar_loss)
        rho = g_k / g_total if g_total > 1.0e-12 else "NOT_DEFINED"
        pooled_pairs_k.append((baseline_loss, k_loss))
        pooled_pairs_ar_given_k.append((k_loss, kar_loss))
        pooled_pairs_total.append((baseline_loss, kar_loss))
        sums["baseline"] += float(np.sum(baseline_loss))
        sums["k"] += float(np.sum(k_loss))
        sums["kar"] += float(np.sum(kar_loss))
        sums["ar"] += float(np.sum(ar_loss))
        half = len(target) // 2
        train_signal = data["train"]["multiresolution_u"][:, 0]
        test_signal = data["test"]["multiresolution_u"][:, 0][mask]
        lower, upper = np.quantile(train_signal, [0.01, 0.99])
        common = (test_signal >= lower) & (test_signal <= upper)
        ood = ~common
        placebo = {}
        for shift in (-120, -60, 60, 120):
            if shift > 0:
                shifted_prediction = k_prediction[:-shift]
                shifted_target = target[shift:]
            else:
                shifted_prediction = k_prediction[-shift:]
                shifted_target = target[:shift]
            placebo[str(shift)] = improvement(
                shifted_target**2,
                (shifted_target - shifted_prediction) ** 2,
            )
        direction_results[direction] = {
            "G_K": g_k,
            "G_AR_given_K": g_ar_k,
            "G_total": g_total,
            "rho_phys": rho,
            "relative_AR": improvement(ar_loss, kar_loss),
            "first_half": {
                "G_K": improvement(baseline_loss[:half], k_loss[:half]),
                "G_AR_given_K": improvement(k_loss[:half], kar_loss[:half]),
            },
            "second_half": {
                "G_K": improvement(baseline_loss[half:], k_loss[half:]),
                "G_AR_given_K": improvement(k_loss[half:], kar_loss[half:]),
            },
            "common_support": {
                "bounds": [float(lower), float(upper)],
                "rows": int(np.sum(common)),
                "G_K": improvement(baseline_loss[common], k_loss[common]),
                "G_AR_given_K": improvement(k_loss[common], kar_loss[common]),
            },
            "OOD": {
                "rows": int(np.sum(ood)),
                "G_K": (
                    improvement(baseline_loss[ood], k_loss[ood])
                    if np.any(ood)
                    else "NOT_DEFINED"
                ),
                "G_AR_given_K": (
                    improvement(k_loss[ood], kar_loss[ood])
                    if np.any(ood)
                    else "NOT_DEFINED"
                ),
            },
            "time_shift_placebo_rows": placebo,
        }
    g_k_pooled = 1.0 - sums["k"] / max(sums["baseline"], np.finfo(float).eps)
    g_ar_k_pooled = 1.0 - sums["kar"] / max(sums["k"], np.finfo(float).eps)
    g_total_pooled = 1.0 - sums["kar"] / max(sums["baseline"], np.finfo(float).eps)
    rho_pooled = (
        g_k_pooled / g_total_pooled
        if g_total_pooled > 1.0e-12
        else "NOT_DEFINED"
    )
    block_length = int(
        round(
            max(
                protocol["bootstrap_min_block_min"],
                protocol["target_window_min"],
            )
            * 60.0
            / protocol["cadence_sec"]
        )
    )
    first_kernel = np.asarray(
        physics_details[directions[0]]["kernel"]["physical_coefficients"]
    )
    second_kernel = np.asarray(
        physics_details[directions[1]]["kernel"]["physical_coefficients"]
    )
    threshold_a = 0.2 * max(float(np.max(np.abs(first_kernel))), 1.0e-15)
    threshold_b = 0.2 * max(float(np.max(np.abs(second_kernel))), 1.0e-15)
    support_a = np.abs(first_kernel) >= threshold_a
    support_b = np.abs(second_kernel) >= threshold_b
    union = int(np.sum(support_a | support_b))
    kernel_structure = {
        "correlation": float(np.corrcoef(first_kernel, second_kernel)[0, 1]),
        "support_jaccard": (
            float(np.sum(support_a & support_b) / union) if union else 1.0
        ),
        "peak_indices": [
            int(np.argmax(np.abs(first_kernel))),
            int(np.argmax(np.abs(second_kernel))),
        ],
        "sign_agreement_fraction": float(
            np.mean(np.sign(first_kernel) == np.sign(second_kernel))
        ),
        "nonlinear_blocks": [
            physics_details[direction]["kernel"]["nonlinear_block"]
            for direction in directions
        ],
        "relative_kkt_max": max(
            physics_details[direction]["kernel"]["relative_kkt"]
            for direction in directions
        ),
        "condition_number_max": max(
            physics_details[direction]["kernel"]["condition_number"]
            for direction in directions
        ),
    }
    return {
        "directions": direction_results,
        "pooled": {
            "G_K": g_k_pooled,
            "G_AR_given_K": g_ar_k_pooled,
            "G_total": g_total_pooled,
            "rho_phys": rho_pooled,
            "relative_AR": 1.0 - sums["kar"] / max(sums["ar"], np.finfo(float).eps),
            "bootstrap_K_vs_persistence": pooled_two_direction_bootstrap(
                pooled_pairs_k,
                replicates=int(protocol["bootstrap_replicates"]),
                block_length=block_length,
                seed=int(protocol["random_seed"]) + 100,
            ),
            "bootstrap_AR_given_K": pooled_two_direction_bootstrap(
                pooled_pairs_ar_given_k,
                replicates=int(protocol["bootstrap_replicates"]),
                block_length=block_length,
                seed=int(protocol["random_seed"]) + 200,
            ),
            "bootstrap_total": pooled_two_direction_bootstrap(
                pooled_pairs_total,
                replicates=int(protocol["bootstrap_replicates"]),
                block_length=block_length,
                seed=int(protocol["random_seed"]) + 300,
            ),
        },
        "kernel_structure": kernel_structure,
    }


def aggregate_results(
    *,
    shared_root: Path,
    results_root: Path,
    rows: list[dict[str, Any]],
    physics_details: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha: str,
    cpu_sha: str,
    directions: list[str],
) -> dict[str, Any]:
    pooled = _pooled_model_rows(
        rows, results_root=results_root, directions=directions
    )
    flattened = [_flatten(row) for row in rows] + pooled
    for group, filename in (
        (GROUP_SIMPLE, "SIMPLE_BASELINES.csv"),
        (GROUP_SOFT, "SOFT_SENSOR_CLASSICAL.csv"),
        (GROUP_SYSTEM, "SYSTEM_IDENTIFICATION.csv"),
        (GROUP_PHYSICS, "PHYSICS_FIRST_MODELS.csv"),
    ):
        write_csv(
            results_root / filename,
            [row for row in flattened if row.get("group") == group],
        )
    soft_pooled = [
        row
        for row in pooled
        if row["group"] == GROUP_SOFT and row["status"] == "COMPLETED"
    ]
    dynamic_pooled = [
        row
        for row in pooled
        if row["group"] in {GROUP_SYSTEM, GROUP_PHYSICS}
        and row["status"] == "COMPLETED"
    ]
    soft_pooled.sort(key=lambda value: value["MSE"])
    dynamic_pooled.sort(key=lambda value: value["MSE"])
    write_csv(results_root / "INPUT_DRIVEN_LEADERBOARD.csv", soft_pooled)
    write_csv(results_root / "DYNAMIC_IDENTIFICATION_LEADERBOARD.csv", dynamic_pooled)
    audit = _physics_audit(
        shared_root=shared_root,
        results_root=results_root,
        physics_details=physics_details,
        protocol=protocol,
        directions=directions,
    )
    atomic_json(results_root / "BOOTSTRAP" / "physics_first.json", audit)
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    failed = [row for row in rows if row.get("status") == "FAILED"]
    not_applicable = [row for row in rows if row.get("status") == "NOT_APPLICABLE"]
    primary = next(
        row for row in pooled if row["name"] == "K-to-Residual-AR"
    )
    decision = {
        "schema": protocol["schema"],
        "status": "COMPLETED" if not failed else "COMPLETED_WITH_MODEL_FAILURES",
        "protocol_sha256": protocol_sha,
        "cpu_config_sha256": cpu_sha,
        "completed_model_direction_runs": len(completed),
        "failed_model_direction_runs": len(failed),
        "not_applicable_model_direction_runs": len(not_applicable),
        "input_driven_winner": (
            {"name": soft_pooled[0]["name"], "MSE": soft_pooled[0]["MSE"]}
            if soft_pooled
            else "NOT_AVAILABLE"
        ),
        "dynamic_identification_winner": (
            {"name": dynamic_pooled[0]["name"], "MSE": dynamic_pooled[0]["MSE"]}
            if dynamic_pooled
            else "NOT_AVAILABLE"
        ),
        "formal_primary_model": {
            "name": "K-to-Residual-AR",
            "pooled_MSE": primary["MSE"],
            "pooled_RMSE": primary["RMSE"],
            "pooled_R2": primary["R2"],
            "physics_audit": audit,
        },
        "residual_AR_selected": {
            direction: physics_details[direction]["kernel"][
                "residual_ar_parameters"
            ]
            for direction in directions
        },
        "nonlinear_K": "EXACT_ZERO_BOTH_DIRECTIONS",
        "FP64_certification": {
            "KKT_max": audit["kernel_structure"]["relative_kkt_max"],
            "condition_number_max": audit["kernel_structure"][
                "condition_number_max"
            ],
            "predictions_float64": all(
                bool(row.get("fp64_prediction", True))
                for row in completed
            ),
            "status": (
                "PASS"
                if audit["kernel_structure"]["relative_kkt_max"] <= 1.0e-8
                else "FAIL"
            ),
        },
        "failed_models": [
            {
                "name": row["name"],
                "direction": row["direction"],
                "error": row.get("error"),
            }
            for row in failed
        ],
        "implementation_registry": {
            name: {
                **spec,
                "note": IMPLEMENTATION_NOTES.get(name, ""),
            }
            for name, spec in MODEL_SPECS.items()
        },
    }
    atomic_json(results_root / "CPU_FINAL_DECISION.json", decision)
    return decision
