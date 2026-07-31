from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, eigvalsh
from scipy.optimize import brentq, minimize_scalar


@dataclass
class EDFFit:
    coefficient: np.ndarray
    intercept: float
    prediction: np.ndarray
    target_df: float
    attained_df: float
    selected_lambda: float
    kkt_residual: float
    condition_number: float


@dataclass
class EDFMap:
    matrix: np.ndarray
    target: np.ndarray
    penalty: np.ndarray
    predict_matrix: np.ndarray
    x_mean: np.ndarray
    y_mean: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_rhs: np.ndarray
    predict_eigen_design: np.ndarray
    n_rows: int
    d0: float = 1.0

    def df_at_lambda(self, regularization: float) -> float:
        lam = float(regularization)
        return self.d0 + float(np.sum(self.eigenvalues / (self.eigenvalues + self.n_rows * lam)))

    def _scale_log10(self) -> float:
        positive = self.eigenvalues[self.eigenvalues > 0]
        if not len(positive):
            return 0.0
        return float(np.log10(np.median(positive) / self.n_rows))

    def lambda_for_df(self, target_df: float, relative_tolerance: float = 1e-8) -> float:
        target = float(target_df)
        rank_upper = self.d0 + float(np.sum(self.eigenvalues > 0))
        if not self.d0 < target < rank_upper + 1e-12:
            raise ValueError(f"EDF_OUTSIDE_INVERTIBLE_RANGE:{target}:{self.d0}:{rank_upper}")
        center = self._scale_log10()
        low = high = center
        while self.df_at_lambda(10.0**low) < target:
            low -= 2.0
            if low < -300:
                raise RuntimeError("EDF_INVERSION_LOWER_BRACKET_FAILED")
        while self.df_at_lambda(10.0**high) > target:
            high += 2.0
            if high > 300:
                raise RuntimeError("EDF_INVERSION_UPPER_BRACKET_FAILED")
        # The public tolerance certifies the attained EDF; it must not also
        # loosen the scalar root solve in log-lambda space.  A wide log bracket
        # can otherwise amplify a nominal 1e-8 xtol into an EDF miss above the
        # frozen acceptance threshold during local profile refinement.
        root = brentq(
            lambda log_value: self.df_at_lambda(10.0**log_value) - target,
            low,
            high,
            xtol=1e-13,
            rtol=1e-13,
            maxiter=300,
        )
        lam = float(10.0**root)
        attained = self.df_at_lambda(lam)
        if abs(attained - target) > relative_tolerance * max(abs(target), 1.0) * 2.0:
            raise RuntimeError(f"EDF_INVERSION_TOLERANCE_FAILED:{target}:{attained}")
        return lam

    def predict_at_df(self, target_df: float, relative_tolerance: float = 1e-8) -> tuple[np.ndarray, float, float]:
        lam = self.lambda_for_df(target_df, relative_tolerance)
        denominator = self.eigenvalues + self.n_rows * lam
        coefficient_eigen = self.projected_rhs / denominator
        prediction = self.y_mean + self.predict_eigen_design @ coefficient_eigen
        return np.asarray(prediction, dtype=np.float64), lam, self.df_at_lambda(lam)

    def fit_at_df(self, target_df: float, relative_tolerance: float = 1e-8) -> EDFFit:
        lam = self.lambda_for_df(target_df, relative_tolerance)
        centered_x = self.matrix - self.x_mean
        centered_y = self.target - self.y_mean
        hessian = centered_x.T @ centered_x + self.n_rows * lam * self.penalty
        rhs = centered_x.T @ centered_y
        factor = cho_factor(hessian, lower=True, check_finite=False)
        coefficient = cho_solve(factor, rhs, check_finite=False)
        intercept = self.y_mean - float(self.x_mean @ coefficient)
        prediction = intercept + self.predict_matrix @ coefficient
        residual = hessian @ coefficient - rhs
        kkt = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30))
        edge = eigvalsh(hessian, subset_by_index=[0, len(hessian) - 1], check_finite=False)
        condition = float(edge[-1] / edge[0])
        return EDFFit(
            coefficient=coefficient,
            intercept=intercept,
            prediction=np.asarray(prediction, dtype=np.float64),
            target_df=float(target_df),
            attained_df=self.df_at_lambda(lam),
            selected_lambda=lam,
            kkt_residual=kkt,
            condition_number=condition,
        )

    def stable_interval(
        self,
        *,
        condition_epsilon_limit: float,
        lower_excess: float,
    ) -> dict[str, float]:
        machine = np.finfo(np.float64).eps
        centered_x = self.matrix - self.x_mean
        xtx = centered_x.T @ centered_x

        def stability(log_lambda: float) -> tuple[bool, float]:
            lam = 10.0**float(log_lambda)
            hessian = xtx + self.n_rows * lam * self.penalty
            try:
                cho_factor(hessian, lower=True, check_finite=False)
                edge = eigvalsh(hessian, subset_by_index=[0, len(hessian) - 1], check_finite=False)
                condition = float(edge[-1] / edge[0])
                finite = bool(np.all(np.isfinite(edge)))
                return finite and condition * machine <= condition_epsilon_limit, condition
            except Exception:
                return False, float("inf")

        stable_log = self._scale_log10()
        ok, _ = stability(stable_log)
        while not ok:
            stable_log += 1.0
            if stable_log > 300:
                raise RuntimeError("NO_COMMON_STABLE_EDF_INTERVAL")
            ok, _ = stability(stable_log)

        unstable_log = stable_log
        while True:
            candidate = unstable_log - 1.0
            ok, _ = stability(candidate)
            if not ok:
                unstable_log = candidate
                break
            unstable_log = candidate
            if self.df_at_lambda(10.0**candidate) >= self.d0 + np.sum(self.eigenvalues > 0) - 1e-8:
                break
            if candidate < -300:
                break

        if stability(unstable_log)[0]:
            upper_df = self.d0 + float(np.sum(self.eigenvalues > 0)) - 1e-8
            upper_lambda = 10.0**unstable_log
            upper_condition = stability(unstable_log)[1]
        else:
            boundary = brentq(
                lambda log_value: np.log10(max(stability(log_value)[1] * machine / condition_epsilon_limit, 1e-300)),
                unstable_log,
                stable_log,
                xtol=1e-8,
                rtol=1e-8,
            )
            inside = boundary + 1e-7
            upper_lambda = float(10.0**inside)
            upper_df = self.df_at_lambda(upper_lambda)
            upper_condition = stability(inside)[1]

        lower_log = self._scale_log10()
        while self.df_at_lambda(10.0**lower_log) - self.d0 > lower_excess:
            lower_log += 2.0
            if lower_log > 300:
                raise RuntimeError("EDF_LOWER_CONTINUATION_FAILED")
        lower_lambda = float(10.0**lower_log)
        lower_df = self.df_at_lambda(lower_lambda)
        if not lower_df < upper_df:
            raise RuntimeError("NO_COMMON_STABLE_EDF_INTERVAL")
        return {
            "d0": self.d0,
            "lower_df": lower_df,
            "upper_df": upper_df,
            "lower_lambda": lower_lambda,
            "upper_lambda": upper_lambda,
            "upper_condition_number": upper_condition,
            "rank": int(np.sum(self.eigenvalues > 0)),
        }


def prepare_edf_map(matrix: np.ndarray, target: np.ndarray, penalty: np.ndarray, predict_matrix: np.ndarray) -> EDFMap:
    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    predict_matrix = np.asarray(predict_matrix, dtype=np.float64)
    x_mean = np.mean(matrix, axis=0)
    y_mean = float(np.mean(target))
    centered_x = matrix - x_mean
    centered_y = target - y_mean
    xtx = centered_x.T @ centered_x
    penalty = (penalty + penalty.T) / 2.0
    eigenvalues, eigenvectors = eigh(xtx, penalty, check_finite=False, driver="gvd")
    tolerance = max(float(np.max(np.abs(eigenvalues))) * 1e-13, 1e-14)
    eigenvalues[np.abs(eigenvalues) < tolerance] = 0.0
    if np.min(eigenvalues) < -tolerance:
        raise RuntimeError(f"GENERALIZED_EIGENVALUE_NEGATIVE:{np.min(eigenvalues)}")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rhs = centered_x.T @ centered_y
    return EDFMap(
        matrix=matrix,
        target=target,
        penalty=penalty,
        predict_matrix=predict_matrix,
        x_mean=x_mean,
        y_mean=y_mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        projected_rhs=eigenvectors.T @ rhs,
        predict_eigen_design=(predict_matrix - x_mean) @ eigenvectors,
        n_rows=len(target),
    )


class ContinuousProfile:
    def __init__(
        self,
        maps: list[EDFMap],
        validation_targets: list[np.ndarray],
        validation_weights: list[int],
        *,
        lower: float,
        upper: float,
        max_evaluations: int,
        d_tolerance: float,
        interpolation_tolerance: float,
        inversion_tolerance: float,
        checkpoint: Callable[[list[dict[str, float]]], None] | None = None,
    ) -> None:
        self.maps = maps
        self.targets = [np.asarray(value, dtype=np.float64) for value in validation_targets]
        self.weights = np.asarray(validation_weights, dtype=np.float64)
        self.lower = float(lower)
        self.upper = float(upper)
        self.max_evaluations = int(max_evaluations)
        self.d_tolerance = float(d_tolerance)
        self.interpolation_tolerance = float(interpolation_tolerance)
        self.inversion_tolerance = float(inversion_tolerance)
        self.checkpoint = checkpoint
        self.cache: dict[float, dict[str, float]] = {}

    def evaluate(self, d: float) -> dict[str, float]:
        key = float(np.float64(d))
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= self.max_evaluations:
            raise RuntimeError("CONTINUOUS_EDF_PROFILE_UNRESOLVED")
        losses: list[float] = []
        lambdas: list[float] = []
        attained: list[float] = []
        for current_map, target in zip(self.maps, self.targets):
            prediction, lam, current_df = current_map.predict_at_df(key, self.inversion_tolerance)
            losses.append(float(np.mean((target - prediction) ** 2)))
            lambdas.append(lam)
            attained.append(current_df)
        mean = float(np.sum(self.weights * losses) / np.sum(self.weights))
        if len(losses) > 1:
            se = float(np.sqrt(np.sum(self.weights * (np.asarray(losses) - mean) ** 2) / ((len(losses) - 1) * np.sum(self.weights))))
        else:
            se = 0.0
        row: dict[str, float] = {"edf": key, "mean_mse": mean, "se_mse": se}
        for index, (loss, lam, current_df) in enumerate(zip(losses, lambdas, attained)):
            row[f"fold_{index}_mse"] = loss
            row[f"fold_{index}_lambda"] = lam
            row[f"fold_{index}_attained_df"] = current_df
        self.cache[key] = row
        if self.checkpoint:
            self.checkpoint(sorted(self.cache.values(), key=lambda item: item["edf"]))
        return row

    def _loss(self, d: float) -> float:
        return float(self.evaluate(float(d))["mean_mse"])

    def resolve(self) -> dict[str, object]:
        midpoint = (self.lower + self.upper) / 2.0
        self.evaluate(self.lower)
        self.evaluate(midpoint)
        self.evaluate(self.upper)
        queue: list[tuple[float, float, float]] = [(self.lower, midpoint, self.upper)]
        interpolation_errors: list[float] = []
        unresolved = False
        while queue:
            queue.sort(key=lambda item: min(self._loss(item[0]), self._loss(item[1]), self._loss(item[2])))
            left, middle, right = queue.pop(0)
            if right - left <= self.d_tolerance:
                continue
            quarter_left = (left + middle) / 2.0
            quarter_right = (middle + right) / 2.0
            if len(self.cache) + 2 > self.max_evaluations - 12:
                unresolved = True
                break
            y = np.array([self._loss(left), self._loss(middle), self._loss(right)])
            coefficients = np.polyfit(np.array([left, middle, right]), y, 2)
            actual = np.array([self._loss(quarter_left), self._loss(quarter_right)])
            predicted = np.polyval(coefficients, np.array([quarter_left, quarter_right]))
            relative = float(np.max(np.abs(actual - predicted)) / max(np.max(np.abs(actual)), np.max(np.abs(y)), 1e-30))
            interpolation_errors.append(relative)
            near_best = float(np.min(actual)) <= min(row["mean_mse"] for row in self.cache.values()) * 1.02
            if relative > self.interpolation_tolerance or (near_best and right - left > 4.0 * self.d_tolerance):
                queue.append((left, quarter_left, middle))
                queue.append((middle, quarter_right, right))

        ordered = sorted(self.cache.values(), key=lambda item: item["edf"])
        brackets: list[tuple[float, float]] = []
        for index in range(1, len(ordered) - 1):
            if ordered[index]["mean_mse"] <= ordered[index - 1]["mean_mse"] and ordered[index]["mean_mse"] <= ordered[index + 1]["mean_mse"]:
                brackets.append((ordered[index - 1]["edf"], ordered[index + 1]["edf"]))
        if not brackets:
            best_index = int(np.argmin([row["mean_mse"] for row in ordered]))
            if best_index in (0, len(ordered) - 1):
                brackets = [(ordered[max(0, best_index - 1)]["edf"], ordered[min(len(ordered) - 1, best_index + 1)]["edf"])]
            else:
                brackets = [(ordered[best_index - 1]["edf"], ordered[best_index + 1]["edf"])]
        minima: list[dict[str, float]] = []
        for left, right in brackets:
            if right - left <= self.d_tolerance:
                candidate = min((row for row in ordered if left <= row["edf"] <= right), key=lambda row: row["mean_mse"])
                minima.append({"edf": candidate["edf"], "mean_mse": candidate["mean_mse"], "bracket_left": left, "bracket_right": right})
                continue
            if len(self.cache) >= self.max_evaluations - 8:
                candidates = [row for row in self.cache.values() if left <= row["edf"] <= right]
                row = min(candidates, key=lambda item: item["mean_mse"])
                minima.append({"edf": row["edf"], "mean_mse": row["mean_mse"], "bracket_left": left, "bracket_right": right})
                unresolved = True
                continue
            try:
                result = minimize_scalar(self._loss, bounds=(left, right), method="bounded", options={"xatol": self.d_tolerance, "maxiter": 100})
                row = self.evaluate(float(result.x))
                minima.append({"edf": row["edf"], "mean_mse": row["mean_mse"], "bracket_left": left, "bracket_right": right})
            except RuntimeError:
                unresolved = True
                candidates = [row for row in self.cache.values() if left <= row["edf"] <= right]
                if candidates:
                    row = min(candidates, key=lambda item: item["mean_mse"])
                    minima.append({"edf": row["edf"], "mean_mse": row["mean_mse"], "bracket_left": left, "bracket_right": right})
        best = min(self.cache.values(), key=lambda row: row["mean_mse"])
        if minima:
            refined = min(minima, key=lambda row: row["mean_mse"])
            best = self.evaluate(refined["edf"])
        upper_hit = best["edf"] >= self.upper - self.d_tolerance
        resolved = not unresolved and not upper_hit
        return {
            "resolved": resolved,
            "upper_bound_hit": upper_hit,
            "d_min": best["edf"],
            "minimum_mse": best["mean_mse"],
            "minimum_se": best["se_mse"],
            "minima": minima,
            "evaluations": sorted(self.cache.values(), key=lambda item: item["edf"]),
            "max_quadratic_interpolation_error": max(interpolation_errors, default=0.0),
            "evaluation_count": len(self.cache),
        }

    def one_se(self, resolution: dict[str, object]) -> dict[str, float | bool]:
        d_min = float(resolution["d_min"])
        minimum = self.evaluate(d_min)
        threshold = float(minimum["mean_mse"] + minimum["se_mse"])
        lower_row = self.evaluate(self.lower)
        if lower_row["mean_mse"] <= threshold:
            selected = self.lower
            hit_lower = True
        else:
            ordered = [row for row in sorted(self.cache.values(), key=lambda row: row["edf"]) if row["edf"] <= d_min]
            bracket = None
            for left, right in zip(ordered[:-1], ordered[1:]):
                if left["mean_mse"] > threshold >= right["mean_mse"]:
                    bracket = (left["edf"], right["edf"])
            if bracket is None:
                raise RuntimeError("CONTINUOUS_ONE_SE_COMPONENT_NOT_BRACKETED")
            selected = float(brentq(lambda value: self._loss(value) - threshold, bracket[0], bracket[1], xtol=self.d_tolerance, rtol=1e-10))
            self.evaluate(selected)
            hit_lower = False
        return {
            "d_min": d_min,
            "d_1se": selected,
            "one_se_threshold": threshold,
            "one_se_hits_lower_complexity_bound": hit_lower,
            "d_min_hits_upper_bound": bool(resolution["upper_bound_hit"]),
        }
