"""Model factories and validation-only one-SE selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor


@dataclass(slots=True)
class Selection:
    parameters: dict[str, Any]
    losses: list[dict[str, Any]]
    minimum_mean: float
    minimum_se: float
    threshold: float
    selected_complexity: tuple


class SeparablePolynomialRidge:
    def __init__(self, *, degree: int, alpha: float):
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.model = None

    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float64)
        columns = [x]
        for order in range(2, self.degree + 1):
            columns.append(x**order)
        return np.column_stack(columns)

    def fit(self, matrix: np.ndarray, target: np.ndarray):
        self.model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=self.alpha),
        ).fit(self._transform(matrix), target)
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model.predict(self._transform(matrix)), dtype=np.float64
        )


class ReducedPolynomialRegressor:
    def __init__(self, *, components: int, degree: int, alpha: float):
        self.components = int(components)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.model = None

    def fit(self, matrix: np.ndarray, target: np.ndarray):
        components = min(self.components, matrix.shape[1], len(matrix) - 1)
        self.model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, random_state=0),
            PolynomialFeatures(degree=self.degree, include_bias=False),
            StandardScaler(),
            Ridge(alpha=self.alpha),
        ).fit(matrix, target)
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(matrix), dtype=np.float64)


class ReducedSplineRegressor:
    def __init__(self, *, components: int, knots: int, alpha: float):
        self.components = int(components)
        self.knots = int(knots)
        self.alpha = float(alpha)
        self.model = None

    def fit(self, matrix: np.ndarray, target: np.ndarray):
        components = min(self.components, matrix.shape[1], len(matrix) - 1)
        self.model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, random_state=0),
            SplineTransformer(
                n_knots=self.knots,
                degree=3,
                include_bias=False,
            ),
            Ridge(alpha=self.alpha),
        ).fit(matrix, target)
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(matrix), dtype=np.float64)


class HammersteinWienerAdapted:
    """Separable cubic input map plus train-only isotonic output calibration."""

    def __init__(self, *, alpha: float):
        self.alpha = float(alpha)
        self.base = SeparablePolynomialRidge(degree=3, alpha=alpha)
        self.calibrator = None

    def fit(self, matrix: np.ndarray, target: np.ndarray):
        from sklearn.isotonic import IsotonicRegression

        self.base.fit(matrix, target)
        base_prediction = self.base.predict(matrix)
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            base_prediction, target
        )
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.calibrator.predict(self.base.predict(matrix)), dtype=np.float64
        )


def model_factory(name: str, parameters: dict[str, Any]) -> Any:
    alpha = float(parameters.get("alpha", 1.0))
    if name in {
        "Ridge",
        "FIR-Ridge",
        "AR",
        "Differenced-AR",
        "ARX",
        "ARMAX",
        "Output-Error",
        "Box-Jenkins",
        "Joint-ARX",
        "Joint-K+AR",
    }:
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name == "Elastic-Net":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=alpha,
                l1_ratio=float(parameters["l1_ratio"]),
                max_iter=20_000,
                random_state=0,
            ),
        )
    if name in {"PCR", "N4SID-adapted"}:
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=int(parameters["components"]), random_state=0),
            Ridge(alpha=alpha),
        )
    if name in {"PLS", "Dynamic-PLS"}:
        return make_pipeline(
            StandardScaler(),
            PLSRegression(
                n_components=int(parameters["components"]),
                scale=False,
                max_iter=1000,
            ),
        )
    if name == "Kernel-Ridge-RBF":
        return make_pipeline(
            StandardScaler(),
            KernelRidge(
                alpha=alpha,
                kernel="rbf",
                gamma=float(parameters["gamma"]),
            ),
        )
    if name == "Nystroem-Ridge":
        return make_pipeline(
            StandardScaler(),
            Nystroem(
                kernel="rbf",
                gamma=float(parameters["gamma"]),
                n_components=int(parameters.get("components", 128)),
                random_state=0,
            ),
            Ridge(alpha=alpha),
        )
    if name == "SVR-RBF":
        return make_pipeline(
            StandardScaler(),
            SVR(
                C=float(parameters["C"]),
                gamma=float(parameters["gamma"]),
                epsilon=float(parameters.get("epsilon", 0.01)),
                cache_size=2048,
            ),
        )
    if name == "Random-Forest":
        return RandomForestRegressor(
            n_estimators=int(parameters["estimators"]),
            max_depth=parameters["depth"],
            min_samples_leaf=5,
            random_state=0,
            n_jobs=1,
        )
    if name == "Extra-Trees":
        return ExtraTreesRegressor(
            n_estimators=int(parameters["estimators"]),
            max_depth=parameters["depth"],
            min_samples_leaf=5,
            random_state=0,
            n_jobs=1,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=int(parameters["iterations"]),
            max_depth=parameters["depth"],
            learning_rate=float(parameters["learning_rate"]),
            l2_regularization=alpha,
            random_state=0,
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=int(parameters["estimators"]),
            max_depth=int(parameters["depth"]),
            learning_rate=float(parameters["learning_rate"]),
            reg_lambda=alpha,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=0,
            n_jobs=1,
            tree_method="hist",
        )
    if name == "MLP-small":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=tuple(parameters["hidden"]),
                alpha=alpha,
                learning_rate_init=float(parameters["learning_rate"]),
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                max_iter=500,
                random_state=0,
            ),
        )
    if name == "Parallel-Hammerstein":
        return SeparablePolynomialRidge(
            degree=int(parameters["degree"]), alpha=alpha
        )
    if name == "Hammerstein-Wiener-adapted":
        return HammersteinWienerAdapted(alpha=alpha)
    if name in {"Polynomial-NARX", "PNLSS-adapted"}:
        return ReducedPolynomialRegressor(
            components=int(parameters["components"]),
            degree=int(parameters["degree"]),
            alpha=alpha,
        )
    if name == "Spline-NARX":
        return ReducedSplineRegressor(
            components=int(parameters["components"]),
            knots=int(parameters["knots"]),
            alpha=alpha,
        )
    raise KeyError(f"UNKNOWN_MODEL:{name}")


def parameter_count(model: Any) -> int:
    count = 0
    objects = [model]
    if hasattr(model, "steps"):
        objects.extend(item[1] for item in model.steps)
    for item in objects:
        for attribute in ("coef_", "intercept_", "components_", "x_weights_"):
            if hasattr(item, attribute):
                value = np.asarray(getattr(item, attribute))
                count += int(value.size)
    return count


def one_se_select(
    name: str,
    configs: list[dict[str, Any]],
    *,
    matrix: np.ndarray,
    target: np.ndarray,
    folds: list[Any],
    complexity: Callable[[dict[str, Any]], tuple],
) -> Selection:
    if len(configs) > 30:
        raise ValueError(f"CONFIG_BUDGET_EXCEEDED:{name}:{len(configs)}")
    records = []
    for index, parameters in enumerate(configs):
        losses = []
        for fold in folds:
            model = model_factory(name, parameters)
            model.fit(matrix[fold.train_indices], target[fold.train_indices])
            prediction = np.asarray(
                model.predict(matrix[fold.validation_indices])
            ).reshape(-1)
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
                "index": index,
                "parameters": parameters,
                "fold_mse": losses,
                "mean_mse": float(np.mean(losses)),
                "se": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
                "complexity": list(complexity(parameters)),
            }
        )
    minimum = min(records, key=lambda item: (item["mean_mse"], item["index"]))
    threshold = minimum["mean_mse"] + minimum["se"]
    eligible = [item for item in records if item["mean_mse"] <= threshold]
    selected = min(
        eligible,
        key=lambda item: (tuple(item["complexity"]), item["index"]),
    )
    return Selection(
        parameters=selected["parameters"],
        losses=records,
        minimum_mean=minimum["mean_mse"],
        minimum_se=minimum["se"],
        threshold=threshold,
        selected_complexity=tuple(selected["complexity"]),
    )
