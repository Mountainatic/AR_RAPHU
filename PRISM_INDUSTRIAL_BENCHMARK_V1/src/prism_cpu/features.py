from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import TaskSpec
from .io import DatasetFrame


@dataclass
class TaskData:
    task_id: str
    dataset: str
    target: str
    cadence_seconds: float | None
    horizon_steps: int
    window_steps: int
    origins: np.ndarray
    y: np.ndarray
    X: np.ndarray
    input_names: list[str]
    split: np.ndarray
    run_id: np.ndarray
    metadata: dict

    @property
    def train(self) -> np.ndarray:
        return self.split == "train"

    @property
    def validation(self) -> np.ndarray:
        return self.split == "validation"

    @property
    def test(self) -> np.ndarray:
        return self.split == "test"


def infer_cadence(ds: DatasetFrame) -> float | None:
    if ds.metadata.get("cadence_seconds") is not None:
        return float(ds.metadata["cadence_seconds"])
    if ds.time_column:
        import pandas as pd
        t = pd.to_datetime(ds.frame[ds.time_column], errors="coerce")
        dt = t.diff().dt.total_seconds().to_numpy()
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt):
            return float(np.median(dt))
    return None


def _safe_steps(seconds: float, cadence: float | None, fallback: int = 1) -> int:
    if cadence is None or not np.isfinite(cadence) or cadence <= 0:
        return max(1, int(round(seconds / max(1.0, seconds))) if seconds else fallback)
    return max(1, int(round(seconds / cadence)))


def build_task_data(ds: DatasetFrame, task: TaskSpec, *, sample_cap: int | None = None, max_history_steps: int = 128) -> TaskData:
    cadence = infer_cadence(ds)
    horizon = _safe_steps(task.horizon_seconds, cadence)
    window = _safe_steps(task.target_window_seconds, cadence) if task.target_window_seconds else 1
    # An unknown physical cadence is represented in sample steps and recorded in metadata.
    if cadence is None:
        horizon = max(1, int(round(task.horizon_seconds / 60.0))) if task.horizon_seconds >= 60 else 1
        window = max(1, int(round(task.target_window_seconds / 60.0))) if task.target_window_seconds else 1
    history = min(max_history_steps, max(8, 2 * horizon, 4 * window))
    frame = ds.frame
    target = task.target
    if target not in frame:
        raise ValueError(f"target {target} not in {ds.name}")
    # Remove all registered target channels from inputs to avoid target-proxy leakage.
    input_names = [c for c in ds.input_columns if c in frame and c not in ds.target_columns]
    if not input_names:
        raise ValueError(f"no input columns for {task.task_id}")
    Xfull = frame[input_names].apply(__import__("pandas").to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    yfull = __import__("pandas").to_numeric(frame[target], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(yfull) & np.isfinite(Xfull).all(axis=1)
    n = len(frame)
    candidates = []
    for t in range(history, n - horizon):
        if not finite[t] or not finite[t + horizon]:
            continue
        run = ds.run_id
        if len(run) and np.any(run[t - history : t + horizon + 1] != run[t]):
            continue
        candidates.append(t)
    origins = np.asarray(candidates, dtype=np.int64)
    if sample_cap and len(origins) > sample_cap:
        # Preserve chronological support and split proportions deterministically.
        selected = []
        for lo, hi in ((0.0, .6), (.6, .8), (.8, 1.0)):
            a, b = int(len(origins) * lo), int(len(origins) * hi)
            part = origins[a:b]
            k = max(1, int(round(sample_cap * (hi - lo))))
            if len(part) > k:
                selected.append(part[np.linspace(0, len(part) - 1, k, dtype=int)])
            else:
                selected.append(part)
        origins = np.concatenate(selected) if selected else origins
    # Split by target index, not by prediction origin, to make the target maturity explicit.
    target_idx = origins + horizon
    cut1, cut2 = int(n * .6), int(n * .8)
    split = np.where(target_idx < cut1, "train", np.where(target_idx < cut2, "validation", "test"))
    y = yfull[target_idx]
    X = Xfull[origins]
    metadata = {
        "target_index_min": int(target_idx.min()) if len(target_idx) else None,
        "target_index_max": int(target_idx.max()) if len(target_idx) else None,
        "history_steps": int(history),
        "target_alignment": "y[t+h]",
        "max_input_index": "t",
        "future_X_used": False,
        "future_y_used": False,
        "split_boundaries": [cut1, cut2],
        "cadence_known": cadence is not None,
    }
    return TaskData(task.task_id, ds.name, target, cadence, horizon, window, origins, y, X, input_names, split, ds.run_id[origins], metadata)


def lag_matrix(values: np.ndarray, origins: np.ndarray, lags: Iterable[int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lags = np.asarray(list(lags), dtype=np.int64)
    idx = origins[:, None] - lags[None, :]
    return values[idx]


def target_lags(data: TaskData, lags: Iterable[int] | None = None) -> tuple[np.ndarray, list[str]]:
    if lags is None:
        lags = range(0, min(32, max(1, data.metadata["history_steps"])))
    # Reconstruct target history using the target column at the original row indices.
    # The target vector is y[t+h], so the original series is recovered only through the
    # data object metadata cache installed by attach_series().
    series = data.metadata.get("target_series")
    if series is None:
        raise RuntimeError("target series not attached")
    lags = list(lags)
    return lag_matrix(np.asarray(series), data.origins, lags), [f"Y_lag_{i}" for i in lags]


def block_lag_matrix(Xfull: np.ndarray, origins: np.ndarray, *, max_history: int, n_blocks: int = 16) -> tuple[np.ndarray, list[str]]:
    """Causal multiresolution block averages, excluding future samples."""
    Xfull = np.asarray(Xfull, dtype=np.float64)
    n, p = Xfull.shape
    edges = np.unique(np.r_[0, np.geomspace(1, max_history, n_blocks + 1).astype(int)])
    edges = np.unique(np.clip(edges, 0, max_history))
    blocks = [(int(edges[i]), int(max(edges[i + 1] - 1, edges[i]))) for i in range(len(edges) - 1) if edges[i] < max_history]
    # The first block includes the value at t, later blocks are strictly older.
    out = np.empty((len(origins), len(blocks) * p), dtype=np.float64)
    names = []
    for j in range(p):
        col = Xfull[:, j]
        for bi, (a, b) in enumerate(blocks):
            vals = []
            for t in origins:
                lo = max(0, int(t - b))
                hi = min(n, int(t - a + 1))
                vals.append(float(np.mean(col[lo:hi])) if hi > lo else float(col[t]))
            out[:, bi * p + j] = vals
            names.append(f"{j}:{a}-{b}")
    return out, names


def attach_series(data: TaskData, ds: DatasetFrame) -> TaskData:
    data.metadata["target_series"] = __import__("pandas").to_numeric(ds.frame[data.target], errors="coerce").to_numpy(dtype=np.float64)
    data.metadata["X_series"] = ds.frame[data.input_names].apply(__import__("pandas").to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return data
