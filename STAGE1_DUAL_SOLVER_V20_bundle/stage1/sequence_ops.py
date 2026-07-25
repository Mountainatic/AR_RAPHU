"""Exact sequence-first operators for distributed-lag Stage1 models.

The original implementation evaluates ``f_j`` inside every overlapping window.
For a contiguous window tensor this repeats almost every scalar response ``L``
times.  This module reconstructs the unique raw sequence, evaluates every
response once, then applies a depthwise causal convolution.

The static-delay fast path is algebraically identical to

    y_t = b + sum_j sum_tau q[j, tau] f_j(x[j, t-tau]).

No approximation is introduced.  Dynamic sample-dependent lag distributions
fall back to the ordinary window path because their q depends on each window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContiguousWindowBatch:
    """A cached contiguous split.

    Attributes
    ----------
    windows:
        Tensor ``[B, N, L]`` ordered oldest-to-current along the last axis.
    raw_sequence:
        Tensor ``[N, B + L - 1]`` in chronological order.
    target:
        Optional target tensor aligned with the B windows.
    """

    windows: torch.Tensor
    raw_sequence: torch.Tensor
    target: Optional[torch.Tensor] = None

    @classmethod
    def from_windows(
        cls,
        windows: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        *,
        validate: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> "ContiguousWindowBatch":
        raw = windows_to_raw_sequence(
            windows, validate=validate, atol=atol, rtol=rtol
        )
        if target is not None and target.shape[0] != windows.shape[0]:
            raise ValueError("target length must equal number of windows")
        return cls(windows=windows, raw_sequence=raw, target=target)

    def to(self, device=None, dtype=None) -> "ContiguousWindowBatch":
        windows = self.windows.to(device=device, dtype=dtype)
        raw = self.raw_sequence.to(device=device, dtype=dtype)
        target = None if self.target is None else self.target.to(device=device)
        return ContiguousWindowBatch(windows, raw, target)


def validate_contiguous_windows(
    windows: torch.Tensor, *, atol: float = 0.0, rtol: float = 0.0
) -> None:
    if windows.ndim != 3:
        raise ValueError(f"windows must have shape [B,N,L], got {tuple(windows.shape)}")
    if windows.shape[0] < 1 or windows.shape[-1] < 1:
        raise ValueError("windows must be non-empty")
    if windows.shape[0] == 1 or windows.shape[-1] == 1:
        return
    left = windows[:-1, :, 1:]
    right = windows[1:, :, :-1]
    if not torch.allclose(left, right, atol=atol, rtol=rtol):
        max_error = (left - right).abs().max().item()
        raise ValueError(
            "windows are not consecutive overlapping windows; "
            f"maximum overlap mismatch={max_error:.6g}"
        )


def windows_to_raw_sequence(
    windows: torch.Tensor,
    *,
    validate: bool = True,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> torch.Tensor:
    """Reconstruct ``[N,T]`` chronological raw data from ``[B,N,L]`` windows."""
    if validate:
        validate_contiguous_windows(windows, atol=atol, rtol=rtol)
    if windows.ndim != 3:
        raise ValueError("windows must be rank three")
    first = windows[0]  # [N,L]
    if windows.shape[0] == 1:
        return first
    appended = windows[1:, :, -1].transpose(0, 1)  # [N,B-1]
    return torch.cat((first, appended), dim=-1)


def raw_sequence_to_windows(raw_sequence: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Create chronological windows ``[B,N,L]`` from ``[N,T]`` raw data."""
    if raw_sequence.ndim != 2:
        raise ValueError("raw_sequence must have shape [N,T]")
    if not 1 <= max_lag <= raw_sequence.shape[-1]:
        raise ValueError("max_lag must lie in [1,T]")
    # unfold returns [N,B,L], where each window remains oldest-to-current.
    return raw_sequence.unfold(-1, max_lag, 1).permute(1, 0, 2).contiguous()


def lag_windows_from_sequence(sequence_values: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Return current-to-past lag windows ``[B,N,L]`` from ``[N,T]`` values."""
    return torch.flip(raw_sequence_to_windows(sequence_values, max_lag), dims=(-1,))


def static_depthwise_contribution(
    response_sequence: torch.Tensor,
    q: torch.Tensor,
    *,
    centers: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply an exact static distributed-lag operator.

    Parameters
    ----------
    response_sequence:
        ``[N,T]`` chronological values ``f_j(x_j(t))``.
    q:
        ``[N,L]`` current-to-past lag weights.
    centers:
        Optional per-variable empirical centers. Since each q row sums to one,
        subtracting the center before convolution is exactly equivalent to
        subtracting it from the variable contribution afterwards.

    Returns
    -------
    Tensor ``[B,N]`` with one contribution per window and variable.
    """
    if response_sequence.ndim != 2 or q.ndim != 2:
        raise ValueError("response_sequence and q must both be rank two")
    n, t = response_sequence.shape
    if q.shape[0] != n:
        raise ValueError("variable dimension mismatch")
    lag = q.shape[1]
    if t < lag:
        raise ValueError("raw sequence is shorter than lag window")
    x = response_sequence.unsqueeze(0)  # [1,N,T]
    # conv1d performs cross-correlation; reverse q so output index i uses
    # q[tau] * z[i + L - 1 - tau].
    weight = torch.flip(q, dims=(-1,)).unsqueeze(1)  # [N,1,L]
    out = F.conv1d(x, weight, groups=n).squeeze(0).transpose(0, 1)  # [B,N]
    if centers is not None:
        out = out - centers.view(1, n)
    return out


def materialize_static_aux(
    response_sequence: torch.Tensor,
    q: torch.Tensor,
    *,
    centers: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize legacy window-level tensors only when diagnostics need them."""
    lag = q.shape[-1]
    response_raw = lag_windows_from_sequence(response_sequence, lag)
    batch = response_raw.shape[0]
    q_batch = q.unsqueeze(0).expand(batch, -1, -1)
    response = response_raw if centers is None else response_raw - centers.view(1, -1, 1)
    contribution = q_batch * response
    variable_contribution = contribution.sum(-1)
    return q_batch, response, contribution, variable_contribution


def effective_lag_lengths(q: torch.Tensor, tail_tolerance: float = 1e-4) -> torch.Tensor:
    """Smallest retained length whose discarded probability mass is <= tolerance."""
    if q.ndim != 2 or (q < 0).any():
        raise ValueError("q must be a nonnegative [N,L] tensor")
    if not 0 <= tail_tolerance < 1:
        raise ValueError("tail_tolerance must lie in [0,1)")
    tail = torch.flip(torch.cumsum(torch.flip(q, dims=(-1,)), dim=-1), dims=(-1,))
    # tail[:, k] is mass from k through L-1. We retain indices [0,K-1], so
    # discarded mass is tail[:, K]. Find first K satisfying the tolerance.
    n, lag = q.shape
    lengths = torch.full((n,), lag, dtype=torch.long, device=q.device)
    for k in range(1, lag):
        eligible = (lengths == lag) & (tail[:, k] <= tail_tolerance)
        lengths = torch.where(eligible, torch.full_like(lengths, k), lengths)
    return lengths


def truncate_q_with_error_bound(
    q: torch.Tensor,
    tail_tolerance: float = 1e-4,
    *,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero negligible tails and return lengths and discarded-mass error bounds.

    For bounded response ``|z| <= M``, the absolute variable-contribution error
    is at most ``M * discarded_mass``. Renormalization is optional because it
    changes that simple one-sided bound.
    """
    lengths = effective_lag_lengths(q, tail_tolerance)
    truncated = q.clone()
    discarded = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    for j, length in enumerate(lengths.tolist()):
        discarded[j] = truncated[j, length:].sum()
        truncated[j, length:] = 0
    if renormalize:
        truncated = truncated / truncated.sum(-1, keepdim=True).clamp_min(1e-12)
    return truncated, lengths, discarded
