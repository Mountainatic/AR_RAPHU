"""Lag order contract for Stage1TargetDelayKAN.

Canonical order (used throughout the project):
    delay[..., 0]  = x_t         (current)
    delay[..., 1]  = x_{t-1}
    delay[..., k]  = x_{t-k}
    delay[..., L-1]= x_{t-L+1}   (deepest past)

External window (from synthetic/process_data):
    X[..., 0]  = oldest = x_{t-L+1}
    X[..., L-1]= newest = x_t

Model internal (after flip):
    x_lag = flip(X, dims=[-1])
    x_lag[..., 0]  = X[..., L-1] = x_t = canonical lag 0
    x_lag[..., L-1]= X[..., 0]   = x_{t-L+1} = canonical lag L-1
"""
import torch
from enum import Enum

class LagOrder(Enum):
    """Explicit lag ordering schemes."""
    CURRENT_TO_PAST = "current_to_past"   # delay[0]=x_t, delay[1]=x_{t-1}
    OLDEST_TO_NEWEST = "oldest_to_newest" # delay[0]=x_{t-L+1}, delay[-1]=x_t

CANONICAL_ORDER = LagOrder.CURRENT_TO_PAST

def convert_lag_order(tensor: torch.Tensor, source: LagOrder, target: LagOrder) -> torch.Tensor:
    """Convert a delay tensor between lag orderings."""
    if source == target:
        return tensor
    # Both conversions are just flip
    return torch.flip(tensor, dims=[-1])

def window_to_canonical(X_window: torch.Tensor) -> torch.Tensor:
    """Convert external window (oldest_to_newest) to canonical current_to_past."""
    return torch.flip(X_window, dims=[-1])

def canonical_to_window(x_canonical: torch.Tensor) -> torch.Tensor:
    """Convert canonical current_to_past to external window order."""
    return torch.flip(x_canonical, dims=[-1])

import numpy as np

def aggregate_known_response(X_window, q_canonical, response_fn, bias=0.0):
    """Pure deterministic aggregation: y = bias + sum_tau q[tau] * f(x_lag[tau]).
    
    Args:
        X_window: [N, L] with oldest at index 0, newest at index -1
        q_canonical: [L] in canonical order (q[0]=weight on x_t)
        response_fn: callable f(x) -> x
        bias: scalar
    Returns:
        scalar prediction
    """
    x_lag = X_window[::-1]  # canonical order: [x_t, x_{t-1}, ..., x_{t-L+1}]
    return float(bias + np.sum(q_canonical * response_fn(x_lag)))
