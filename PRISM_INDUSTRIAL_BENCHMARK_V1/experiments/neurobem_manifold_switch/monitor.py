from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ManifoldTemplate:
    mean: np.ndarray
    scale: np.ndarray
    basis: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, rank: int) -> "ManifoldTemplate":
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        z = (features - mean) / scale
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        return cls(mean, scale, vt[:min(rank, vt.shape[0])])

    def projection_score(self, features: np.ndarray) -> np.ndarray:
        z = (features - self.mean) / self.scale
        residual = z - (z @ self.basis.T) @ self.basis
        return np.linalg.norm(residual, axis=1) / np.sqrt(z.shape[1])

    def tangent_score(self, features: np.ndarray, window: int, tangent_rank: int) -> np.ndarray:
        z = (features - self.mean) / self.scale
        out = np.zeros(len(z), dtype=np.float64)
        reference = self.basis[:tangent_rank]
        for end in range(window - 1, len(z)):
            _, _, vt = np.linalg.svd(z[end-window+1:end+1] - z[end-window+1:end+1].mean(axis=0), full_matrices=False)
            singular = np.linalg.svd(reference @ vt[:tangent_rank].T, compute_uv=False)
            out[end] = float(np.sqrt(np.mean(np.square(1.0 - np.clip(singular, 0.0, 1.0)))))
        out[:window-1] = out[window-1] if len(out) >= window else 0.0
        return out


def residual_score(target: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square((target - prediction) / np.maximum(scale, 1e-12)), axis=1))


def persistent_alarm(scores: np.ndarray, threshold: float, persistence: int) -> int | None:
    count = 0
    for i, value in enumerate(scores):
        count = count + 1 if value > threshold else 0
        if count >= persistence:
            return i - persistence + 1
    return None


def calibrate(residual: np.ndarray, geometry: np.ndarray, quantile: float) -> dict[str, float]:
    r = float(np.quantile(residual, quantile))
    g = float(np.quantile(geometry, quantile))
    combined = 0.5 * residual / max(r, 1e-12) + 0.5 * geometry / max(g, 1e-12)
    return {"residual_threshold": r, "geometry_threshold": g, "combined_threshold": float(np.quantile(combined, quantile))}


def component_scores(residual: np.ndarray, projection: np.ndarray, tangent: np.ndarray, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    geometry = 0.7 * projection + 0.3 * tangent
    combined = 0.5 * residual / max(thresholds["residual_threshold"], 1e-12) + 0.5 * geometry / max(thresholds["geometry_threshold"], 1e-12)
    return {"residual": residual, "projection": projection, "tangent": tangent, "geometry": geometry, "combined": combined}

