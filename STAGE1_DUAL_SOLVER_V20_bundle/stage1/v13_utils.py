"""Small, dependency-free primitives used by the v13 audit experiments."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def discrete_stats(q: torch.Tensor):
    """Return actual (not latent Gamma) mean, standard deviation and boundary mass."""
    taus = torch.arange(q.shape[-1], device=q.device, dtype=q.dtype)
    mean = (q * taus).sum(-1)
    std = (q * (taus - mean.unsqueeze(-1)).square()).sum(-1).sqrt()
    boundary = q[..., -3:].sum(-1)
    peak = q.argmax(-1)
    return mean, std, peak, boundary


def discrete_w1(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """One-dimensional discrete W1, i.e. sum of CDF differences."""
    return (p.cumsum(-1) - q.cumsum(-1)).abs().sum(-1)


def active_delay_metrics(q: torch.Tensor, true_h: torch.Tensor, active_mask: torch.Tensor):
    """Delay metrics, deliberately restricted to genuinely active rows."""
    mask = active_mask.bool()
    qm, qs, qp, qb = discrete_stats(q[mask])
    tm, ts, tp, tb = discrete_stats(true_h[mask])
    return {
        "mean_delay_mae": float((qm - tm).abs().mean()),
        "std_delay_mae": float((qs - ts).abs().mean()),
        "w1": float(discrete_w1(q[mask], true_h[mask]).mean()),
        "peak_lag_mae": float((qp.float() - tp.float()).abs().mean()),
        "boundary_mass": float(qb.mean()),
    }


def calibrated_lambda(s0: float, lr: float, median_norm: float, max_step_shrink: float = .05):
    """Convert dimensionless shrink scale to lambda and cap unsafe one-step shrink."""
    if lr <= 0 or median_norm <= 0:
        raise ValueError("lr and median_norm must be positive")
    requested = s0 * median_norm / lr
    capped = min(requested, max_step_shrink * median_norm / lr)
    return capped, bool(capped < requested), min(s0, max_step_shrink)


def continuation_scale(epoch: int, warmup: int = 100, ramp: int = 200) -> float:
    if epoch < warmup:
        return 0.0
    if epoch < warmup + ramp:
        return (epoch - warmup + 1) / ramp
    return 1.0


def one_standard_error_select(records: Iterable[dict]) -> dict:
    """Select using validation RMSE/active count/lambda only; ignores truth keys."""
    rows = list(records)
    if not rows:
        raise ValueError("no candidate records")
    by = {}
    for r in rows:
        by.setdefault(str(r["config_id"]), []).append(float(r["val_rmse"]))
    means = {k: float(np.mean(v)) for k, v in by.items()}
    ses = {k: float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0 for k, v in by.items()}
    best_id = min(means, key=means.get)
    limit = means[best_id] + ses[best_id]
    eligible = []
    for config_id, values in by.items():
        exemplar = next(r for r in rows if str(r["config_id"]) == config_id)
        if means[config_id] <= limit + 1e-12:
            eligible.append((int(exemplar["active_count"]), -float(exemplar["lambda_group"]), config_id))
    chosen_id = min(eligible)[2]
    chosen = next(r.copy() for r in rows if str(r["config_id"]) == chosen_id)
    chosen.update({"mean_val_rmse": means[chosen_id], "one_se_limit": limit,
                   "best_config_id": best_id, "selection_rule": "validation_one_standard_error"})
    return chosen
