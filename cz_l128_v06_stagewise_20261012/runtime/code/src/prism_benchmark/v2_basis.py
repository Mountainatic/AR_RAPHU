from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class AmplitudeBasis:
    kind: str
    mean: float
    scale: float
    knots: tuple[float, ...]
    levels: tuple[float, ...]
    feature_mean: tuple[float, ...]

    @classmethod
    def fit(cls, values: np.ndarray, requested_m_x: int, *, variance_floor: float = 1e-12) -> "AmplitudeBasis":
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            raise ValueError("amplitude basis has no finite training values")
        mean = float(np.mean(finite, dtype=np.float64))
        scale = float(np.std(finite, dtype=np.float64))
        unique = np.unique(finite)
        if scale * scale < variance_floor:
            return cls("constant", mean, 1.0, (), tuple(unique.tolist()), ())
        if len(unique) <= 2:
            raw = (finite == unique[-1]).astype(np.float64)[:, None]
            return cls("binary", mean, scale, (), tuple(unique.tolist()), tuple(raw.mean(axis=0).tolist()))
        if len(unique) <= 8:
            raw = np.column_stack([(finite == level).astype(np.float64) for level in unique[1:]])
            return cls("low_cardinality", mean, scale, (), tuple(unique.tolist()), tuple(raw.mean(axis=0).tolist()))
        standardized = (finite - mean) / scale
        quantiles = np.arange(1, requested_m_x + 1, dtype=np.float64) / (requested_m_x + 1)
        knots = np.quantile(standardized, quantiles, method="linear")
        deduplicated = []
        for knot in knots:
            if not deduplicated or abs(float(knot) - deduplicated[-1]) > 1e-6:
                deduplicated.append(float(knot))
        columns = [standardized]
        if deduplicated:
            last = deduplicated[-1]
            for knot in deduplicated:
                columns.append(np.maximum(standardized - knot, 0.0) ** 3 - np.maximum(standardized - last, 0.0) ** 3)
        raw = np.column_stack(columns)
        return cls("continuous", mean, scale, tuple(deduplicated), (), tuple(raw.mean(axis=0).tolist()))

    @property
    def dimension(self) -> int:
        if self.kind == "constant":
            return 0
        if self.kind == "binary":
            return 1
        if self.kind == "low_cardinality":
            return max(1, len(self.levels) - 1)
        return 1 + len(self.knots)

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        flat = x.reshape(-1)
        if self.kind == "constant":
            result = np.empty((len(flat), 0), dtype=np.float64)
        elif self.kind == "binary":
            result = (flat == self.levels[-1]).astype(np.float64)[:, None]
            result -= np.asarray(self.feature_mean)
        elif self.kind == "low_cardinality":
            indicators = np.column_stack([(flat == level).astype(np.float64) for level in self.levels[1:]])
            result = indicators - np.asarray(self.feature_mean)
        else:
            z = (flat - self.mean) / self.scale
            columns = [z]
            # Centered natural-cubic truncated-power basis with linear tails.
            if self.knots:
                last = self.knots[-1]
                for knot in self.knots:
                    cubic = np.maximum(z - knot, 0.0) ** 3
                    tail = np.maximum(z - last, 0.0) ** 3
                    columns.append(cubic - tail)
            result = np.column_stack(columns)
            result -= np.asarray(self.feature_mean)
        return result.reshape(*x.shape, self.dimension)

    def metadata(self) -> dict[str, Any]:
        return {"kind": self.kind, "mean": self.mean, "scale": self.scale, "knots": list(self.knots), "levels": list(self.levels), "feature_mean": list(self.feature_mean), "dimension": self.dimension}


def causal_geometric_intervals(history_steps: int, m_tau: int, growth: float = 2.0) -> list[tuple[int, int]]:
    if history_steps < 1 or m_tau < 1 or growth <= 1:
        raise ValueError("invalid lag basis arguments")
    blocks = min(history_steps, m_tau)
    weights = growth ** np.arange(blocks, dtype=np.float64)
    raw = np.concatenate(([0.0], np.cumsum(weights / weights.sum() * history_steps)))
    edges = np.rint(raw).astype(np.int64)
    edges[0] = 0
    edges[-1] = history_steps
    for index in range(1, len(edges)):
        edges[index] = max(edges[index], edges[index - 1] + 1)
    edges[-1] = history_steps
    if np.any(np.diff(edges) <= 0):
        edges = np.rint(np.linspace(0, history_steps, blocks + 1)).astype(np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def tensor_design(block_values: np.ndarray, basis: AmplitudeBasis) -> np.ndarray:
    values = np.asarray(block_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("block_values must be two-dimensional")
    return basis.transform(values)


def natural_cubic_columns(latent: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Cubic regression spline columns with natural linear extrapolation."""
    x = np.asarray(latent, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64)
    if len(knots) < 2:
        return x[:, None]
    columns = [np.ones_like(x), x]
    upper = knots[-1]
    lower = knots[-2]
    for knot in knots[:-2]:
        term = (np.maximum(x - knot, 0.0) ** 3 - np.maximum(x - upper, 0.0) ** 3) / (upper - knot)
        tail = (np.maximum(x - lower, 0.0) ** 3 - np.maximum(x - upper, 0.0) ** 3) / (upper - lower)
        columns.append(term - tail)
    return np.column_stack(columns)
