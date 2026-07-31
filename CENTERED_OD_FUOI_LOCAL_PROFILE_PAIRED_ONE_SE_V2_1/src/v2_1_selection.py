from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .edf import EDFMap


def log_excess_coordinate(d: float, lower: float, epsilon: float) -> float:
    return float(np.log(float(d) - float(lower) + float(epsilon)))


def inverse_log_excess(value: float, lower: float, epsilon: float) -> float:
    return float(lower + np.exp(float(value)) - epsilon)


def _hash_order(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class PairedResult:
    d: float
    block_rows: int
    mean_delta: float
    se_delta: float
    g: float
    positive_probability: float


class LocalPairedProfile:
    def __init__(
        self,
        maps: list[EDFMap],
        targets: list[np.ndarray],
        initial_rows: list[dict[str, float]],
        *,
        lower: float,
        upper: float,
        max_evaluations: int,
        d_tolerance: float,
        inversion_tolerance: float,
        bootstrap_replicates: int,
        bootstrap_seed: int,
    ) -> None:
        self.maps = maps
        self.targets = [np.asarray(value, dtype=np.float64) for value in targets]
        self.weights = np.asarray([len(value) for value in targets], dtype=np.float64)
        self.lower = float(lower)
        self.upper = float(upper)
        self.max_evaluations = int(max_evaluations)
        self.d_tolerance = float(d_tolerance)
        self.inversion_tolerance = float(inversion_tolerance)
        self.bootstrap_replicates = int(bootstrap_replicates)
        self.bootstrap_seed = int(bootstrap_seed)
        self.scalar_cache: dict[float, dict[str, float]] = {}
        self.prediction_cache: dict[float, list[dict[str, np.ndarray]]] = {}
        self.paired_cache: dict[tuple[float, float, int], PairedResult] = {}
        self.bootstrap_cache: dict[int, list[np.ndarray]] = {}
        self.imported_count = 0
        self.new_evaluations = 0
        for row in initial_rows:
            key = float(row["edf"])
            self.scalar_cache[key] = {str(name): float(value) for name, value in row.items() if name != "direction" and value not in (None, "")}
        self.imported_count = len(self.scalar_cache)

    def _existing_key(self, d: float) -> float | None:
        if d in self.scalar_cache:
            return d
        for key in self.scalar_cache:
            if abs(key - d) <= 1e-12 * max(abs(key), abs(d), 1.0):
                return key
        return None

    def evaluate(self, d: float, *, need_predictions: bool = False) -> dict[str, float]:
        d = float(d)
        key = self._existing_key(d)
        if key is None:
            if len(self.scalar_cache) >= self.max_evaluations:
                raise RuntimeError("V2_1_PROFILE_EVALUATION_BUDGET_EXHAUSTED")
            key = d
            self.new_evaluations += 1
        predictions: list[dict[str, np.ndarray]] = []
        losses: list[float] = []
        lambdas: list[float] = []
        attained: list[float] = []
        if key not in self.scalar_cache or (need_predictions and key not in self.prediction_cache):
            for current_map, target in zip(self.maps, self.targets):
                prediction, lam, current_df = current_map.predict_at_df(key, self.inversion_tolerance)
                squared = (target - prediction) ** 2
                predictions.append({"prediction": prediction, "squared_error": squared})
                losses.append(float(np.mean(squared)))
                lambdas.append(lam)
                attained.append(current_df)
            if key not in self.scalar_cache:
                mean = float(np.sum(self.weights * losses) / np.sum(self.weights))
                row: dict[str, float] = {"edf": key, "mean_mse": mean}
                for index, (loss, lam, current_df) in enumerate(zip(losses, lambdas, attained)):
                    row[f"fold_{index}_mse"] = loss
                    row[f"fold_{index}_lambda"] = lam
                    row[f"fold_{index}_attained_df"] = current_df
                self.scalar_cache[key] = row
            if need_predictions:
                self.prediction_cache[key] = predictions
        return self.scalar_cache[key]

    def loss(self, d: float) -> float:
        return float(self.evaluate(d)["mean_mse"])

    def _candidate_basins(self) -> list[tuple[float, float, float]]:
        rows = sorted(self.scalar_cache.values(), key=lambda row: row["edf"])
        basins: list[tuple[float, float, float]] = []
        for index in range(1, len(rows) - 1):
            if rows[index]["mean_mse"] <= rows[index - 1]["mean_mse"] and rows[index]["mean_mse"] <= rows[index + 1]["mean_mse"]:
                basins.append((rows[index - 1]["edf"], rows[index]["edf"], rows[index + 1]["edf"]))
        if not basins:
            best = int(np.argmin([row["mean_mse"] for row in rows]))
            if best in (0, len(rows) - 1):
                return []
            basins.append((rows[best - 1]["edf"], rows[best]["edf"], rows[best + 1]["edf"]))
        return basins

    def discover_and_refine(self) -> dict[str, Any]:
        epsilon = 1e-6 * max(1.0, self.upper - self.lower)
        s_left = log_excess_coordinate(self.lower, self.lower, epsilon)
        s_right = log_excess_coordinate(self.upper, self.lower, epsilon)
        queue: list[tuple[float, float, int]] = [(s_left, s_right, 0)]
        tree_rows: list[dict[str, Any]] = []
        while queue and len(self.scalar_cache) < min(self.max_evaluations - 28, self.imported_count + 14):
            left_s, right_s, depth = queue.pop(0)
            middle_s = (left_s + right_s) / 2.0
            left = inverse_log_excess(left_s, self.lower, epsilon)
            middle = inverse_log_excess(middle_s, self.lower, epsilon)
            right = inverse_log_excess(right_s, self.lower, epsilon)
            left_loss, middle_loss, right_loss = self.loss(left), self.loss(middle), self.loss(right)
            tree_rows.append({
                "depth": depth, "d_left": left, "d_mid": middle, "d_right": right,
                "loss_left": left_loss, "loss_mid": middle_loss, "loss_right": right_loss,
            })
            if depth < 6:
                children = [(left_s, middle_s, depth + 1), (middle_s, right_s, depth + 1)]
                children.sort(key=lambda item: min(self.loss(inverse_log_excess(item[0], self.lower, epsilon)), self.loss(inverse_log_excess(item[1], self.lower, epsilon))))
                queue.extend(children)

        basins = self._candidate_basins()
        if not basins:
            rows = sorted(self.scalar_cache.values(), key=lambda row: row["edf"])
            best = min(rows, key=lambda row: row["mean_mse"])
            upper_hit = float(best["edf"]) >= self.upper - self.d_tolerance
            lower_hit = float(best["edf"]) <= self.lower + self.d_tolerance
            return {
                "global_basin_discovery": "UNRESOLVED",
                "global_basin_discovery_pass": False,
                "local_minimum_resolved": False,
                "d_min": float(best["edf"]),
                "minimum_mse": float(best["mean_mse"]),
                "minimum_bracket": [float(best["edf"]), float(best["edf"])],
                "upper_bound_hit": upper_hit,
                "lower_bound_minimum_hit": lower_hit,
                "far_field_interpolation_resolved": False,
                "far_field_required_for_selection": False,
                "max_quadratic_interpolation_error": None,
                "candidate_basins": [],
                "far_field": [{**row, "classification": "BOUNDARY_OR_MONOTONE_UNRESOLVED"} for row in tree_rows],
                "tree": tree_rows,
                "profile_evaluations": len(self.scalar_cache),
                "new_profile_evaluations": self.new_evaluations,
                "log_epsilon": epsilon,
            }
        refined: list[dict[str, Any]] = []
        for basin_index, (left, middle, right) in enumerate(basins):
            if len(self.scalar_cache) >= self.max_evaluations - 12:
                break
            starts: list[tuple[float, float]] = [(left, right)]
            width = right - left
            starts.append((max(self.lower, left - width / 2), min(self.upper, right + width / 2)))
            starts.append((max(self.lower, left - width / 4), min(self.upper, right + width / 4)))
            runs: list[dict[str, float]] = []
            for run_index, bounds in enumerate(starts):
                if len(self.scalar_cache) >= self.max_evaluations - 8:
                    break
                result = minimize_scalar(self.loss, bounds=bounds, method="bounded", options={"xatol": self.d_tolerance / 4.0, "maxiter": 100})
                runs.append({"run": run_index, "edf": float(result.x), "loss": self.loss(float(result.x)), "left": bounds[0], "right": bounds[1]})
            if not runs:
                continue
            best_run = min(runs, key=lambda row: row["loss"])
            center = best_run["edf"]
            bracket_left = max(self.lower, center - self.d_tolerance / 2.0)
            bracket_right = min(self.upper, center + self.d_tolerance / 2.0)
            center_loss = self.loss(center)
            left_loss = self.loss(bracket_left)
            right_loss = self.loss(bracket_right)
            agreement = max(row["edf"] for row in runs) - min(row["edf"] for row in runs) if len(runs) == 3 else float("inf")
            refined.append({
                "basin": basin_index, "d_min": center, "loss": center_loss,
                "bracket_left": bracket_left, "bracket_right": bracket_right,
                "bracket_width": bracket_right - bracket_left,
                "left_loss": left_loss, "right_loss": right_loss,
                "independent_runs": runs, "independent_spread": agreement,
                "local_resolved": bool(len(runs) == 3 and agreement <= self.d_tolerance and left_loss >= center_loss and right_loss >= center_loss),
            })
        if not refined:
            raise RuntimeError("BASIN_DISCOVERY_UNRESOLVED")
        best = min(refined, key=lambda row: row["loss"])
        next_best = min((row["loss"] for row in refined if row is not best), default=float("inf"))
        upper_hit = best["d_min"] >= self.upper - self.d_tolerance

        neighbor_rows = sorted(self.scalar_cache.values(), key=lambda row: abs(row["edf"] - best["d_min"]))
        neighbor = next((row for row in neighbor_rows if abs(row["edf"] - best["d_min"]) > self.d_tolerance), neighbor_rows[0])
        preliminary = self.paired_difference(float(neighbor["edf"]), float(best["d_min"]), 240)
        relevance = best["loss"] + max(preliminary.se_delta, 0.1 * best["loss"])
        far_field: list[dict[str, Any]] = []
        unresolved_relevant = False
        for row in tree_rows:
            observed = min(row["loss_left"], row["loss_mid"], row["loss_right"])
            adjacent = row["d_left"] <= best["d_min"] <= row["d_right"]
            internal_descent = row["loss_mid"] < min(row["loss_left"], row["loss_right"])
            if internal_descent:
                classification = "CANDIDATE_BASIN"
            elif observed > relevance and not adjacent:
                classification = "FAR_FIELD_PRUNED"
            elif adjacent or observed <= relevance:
                classification = "SELECTION_RELEVANT" if not internal_descent else "CANDIDATE_BASIN"
            else:
                classification = "NUMERICALLY_UNRESOLVED"
                unresolved_relevant = True
            far_field.append({**row, "classification": classification, "selection_relevance_limit": relevance})
        global_pass = bool(best["local_resolved"] and next_best > best["loss"] and not upper_hit and not unresolved_relevant)
        return {
            "global_basin_discovery": "PASS" if global_pass else "UNRESOLVED",
            "global_basin_discovery_pass": global_pass,
            "local_minimum_resolved": bool(best["local_resolved"]),
            "d_min": float(best["d_min"]), "minimum_mse": float(best["loss"]),
            "minimum_bracket": [float(best["bracket_left"]), float(best["bracket_right"])],
            "upper_bound_hit": upper_hit,
            "far_field_interpolation_resolved": False,
            "far_field_required_for_selection": False,
            "max_quadratic_interpolation_error": None,
            "candidate_basins": refined,
            "far_field": far_field,
            "tree": tree_rows,
            "profile_evaluations": len(self.scalar_cache),
            "new_profile_evaluations": self.new_evaluations,
            "log_epsilon": epsilon,
        }

    def _bootstrap_indices(self, block_rows: int) -> list[np.ndarray]:
        block_rows = int(block_rows)
        if block_rows in self.bootstrap_cache:
            return self.bootstrap_cache[block_rows]
        rng = np.random.default_rng(self.bootstrap_seed + block_rows)
        per_fold: list[np.ndarray] = []
        for target in self.targets:
            n_rows = len(target)
            draws = np.empty((self.bootstrap_replicates, n_rows), dtype=np.int64)
            for replicate in range(self.bootstrap_replicates):
                starts = rng.integers(0, max(n_rows - block_rows + 1, 1), size=int(np.ceil(n_rows / block_rows)))
                index = np.concatenate([np.arange(start, min(start + block_rows, n_rows)) for start in starts])
                while len(index) < n_rows:
                    start = int(rng.integers(0, max(n_rows - block_rows + 1, 1)))
                    index = np.concatenate((index, np.arange(start, min(start + block_rows, n_rows))))
                draws[replicate] = index[:n_rows]
            per_fold.append(draws)
        self.bootstrap_cache[block_rows] = per_fold
        return per_fold

    def paired_difference(self, candidate_d: float, reference_d: float, block_rows: int) -> PairedResult:
        candidate = self.evaluate(candidate_d, need_predictions=True)
        reference = self.evaluate(reference_d, need_predictions=True)
        candidate_key = self._existing_key(float(candidate["edf"]))
        reference_key = self._existing_key(float(reference["edf"]))
        assert candidate_key is not None and reference_key is not None
        cache_key = (candidate_key, reference_key, int(block_rows))
        if cache_key in self.paired_cache:
            return self.paired_cache[cache_key]
        differences = [
            candidate_item["squared_error"] - reference_item["squared_error"]
            for candidate_item, reference_item in zip(self.prediction_cache[candidate_key], self.prediction_cache[reference_key])
        ]
        observed = float(np.mean(np.concatenate(differences)))
        indices = self._bootstrap_indices(int(block_rows))
        draws = np.empty(self.bootstrap_replicates, dtype=np.float64)
        for replicate in range(self.bootstrap_replicates):
            draws[replicate] = float(np.mean(np.concatenate([difference[fold_indices[replicate]] for difference, fold_indices in zip(differences, indices)])))
        se = float(np.std(draws, ddof=1))
        result = PairedResult(
            d=candidate_key,
            block_rows=int(block_rows),
            mean_delta=observed,
            se_delta=se,
            g=observed - se,
            positive_probability=float(np.mean(draws > 0.0)),
        )
        self.paired_cache[cache_key] = result
        return result

    def paired_one_se(self, d_min: float, *, primary_block_rows: int, sensitivity_block_rows: tuple[int, ...]) -> dict[str, Any]:
        reference = self.paired_difference(d_min, d_min, primary_block_rows)
        if abs(reference.mean_delta) > 1e-15 or abs(reference.se_delta) > 1e-15:
            raise RuntimeError("PAIRED_REFERENCE_NOT_EXACT_ZERO")
        candidates = sorted({self.lower, *[float(row["edf"]) for row in self.scalar_cache.values() if row["edf"] < d_min]}, reverse=True)
        closer_d = d_min
        closer_g = 0.0
        bracket: tuple[float, float] | None = None
        profile_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            if d_min - candidate < self.d_tolerance / 4.0:
                continue
            result = self.paired_difference(candidate, d_min, primary_block_rows)
            profile_rows.append(result.__dict__)
            if result.g > 0.0 and closer_g <= 0.0:
                bracket = (candidate, closer_d)
                break
            closer_d, closer_g = candidate, result.g
        lower_hit = False
        if bracket is None:
            lower_result = self.paired_difference(self.lower, d_min, primary_block_rows)
            profile_rows.append(lower_result.__dict__)
            if lower_result.g <= 0.0:
                selected = self.lower
                lower_hit = True
            else:
                raise RuntimeError("PAIRED_ONE_SE_BOUNDARY_UNRESOLVED")
        else:
            selected = float(brentq(
                lambda value: self.paired_difference(float(value), d_min, primary_block_rows).g,
                bracket[0], bracket[1], xtol=self.d_tolerance, rtol=1e-10, maxiter=100,
            ))
        selected_result = self.paired_difference(selected, d_min, primary_block_rows)
        profile_rows.append(selected_result.__dict__)
        sensitivity: list[dict[str, Any]] = []
        for block in sensitivity_block_rows:
            current = self.paired_difference(selected, d_min, int(block))
            sensitivity.append(current.__dict__)
        return {
            "paired_one_se_boundary_resolved": True,
            "d_paired_1se": float(selected),
            "paired_delta_at_selection": selected_result.mean_delta,
            "paired_se_at_selection": selected_result.se_delta,
            "paired_g_at_selection": selected_result.g,
            "paired_one_se_hits_lower_bound": lower_hit,
            "root_bracket": list(bracket) if bracket else None,
            "profile": profile_rows,
            "sensitivity": sensitivity,
            "fold_order_hashes": [_hash_order(np.arange(len(target), dtype=np.int64)) for target in self.targets],
            "common_resamples_across_d": True,
        }
