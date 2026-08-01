from __future__ import annotations

import numpy as np


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, *, baseline: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    mse = float(np.mean(err * err))
    out = {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(err))), "n": int(len(y_true))}
    if baseline is not None:
        b = np.asarray(baseline, dtype=np.float64)
        bmse = float(np.mean((y_true - b) ** 2))
        out["relative_improvement_vs_baseline"] = float((bmse - mse) / (bmse + 1e-15))
    return out


def paired_block_bootstrap(y_true: np.ndarray, pred: np.ndarray, baseline: np.ndarray, *, reps: int = 500, block: int = 32, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"reps": 0, "positive_probability": None}
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    gains = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        mse = np.mean((y_true[idx] - pred[idx]) ** 2)
        bmse = np.mean((y_true[idx] - baseline[idx]) ** 2)
        gains[r] = (bmse - mse) / (bmse + 1e-15)
    return {"reps": int(reps), "block": int(block), "mean_gain": float(gains.mean()), "ci_low": float(np.quantile(gains, .025)), "ci_high": float(np.quantile(gains, .975)), "positive_probability": float(np.mean(gains > 0))}

