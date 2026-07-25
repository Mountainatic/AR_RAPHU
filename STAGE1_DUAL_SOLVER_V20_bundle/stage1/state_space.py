"""Low-rank exponential state-space approximation for deployment.

A nonnegative lag kernel is approximated as

    q[tau] ~= sum_r a[r] rho[r]**tau,

which admits an online recursion with O(NR) work and O(NR) state instead of an
L-sample history buffer.  This is an optional inference approximation; training
and scientific evaluation continue to use the exact q unless explicitly chosen.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ExponentialKernelApproximation:
    amplitudes: torch.Tensor  # [N,R], may be signed in the stable fit
    decays: torch.Tensor      # [R]
    reconstructed_q: torch.Tensor  # [N,L]
    l1_error: torch.Tensor    # [N]
    max_error: torch.Tensor   # [N]
    signed_amplitudes: bool = False
    condition_number: float = float("nan")


def fit_nonnegative_exponential_sum(
    q: torch.Tensor,
    rank: int = 6,
    *,
    rho_min: float = 0.05,
    rho_max: float = 0.995,
    iterations: int = 3000,
    lr: float = 0.05,
    ridge: float = 1e-8,
) -> ExponentialKernelApproximation:
    """Fit amplitudes on a fixed logarithmic decay grid by projected Adam."""
    if q.ndim != 2 or (q < 0).any():
        raise ValueError("q must be a nonnegative [N,L] tensor")
    if rank < 1:
        raise ValueError("rank must be positive")
    n, lag = q.shape
    # More density near one captures long tails efficiently.
    rates = torch.linspace(
        torch.log(torch.tensor(1.0 - rho_max, device=q.device, dtype=q.dtype)),
        torch.log(torch.tensor(1.0 - rho_min, device=q.device, dtype=q.dtype)),
        rank,
        device=q.device,
        dtype=q.dtype,
    )
    decays = 1.0 - rates.exp()
    tau = torch.arange(lag, device=q.device, dtype=q.dtype)
    dictionary = decays[:, None].pow(tau[None, :])  # [R,L]
    raw = torch.zeros(n, rank, device=q.device, dtype=q.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=lr)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        amplitude = torch.nn.functional.softplus(raw)
        reconstruction = amplitude.matmul(dictionary)
        loss = (reconstruction - q).square().mean() + ridge * amplitude.square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        amplitude = torch.nn.functional.softplus(raw)
        reconstruction = amplitude.matmul(dictionary)
        # Match probability mass; this improves DC accuracy in the recursive filter.
        amplitude = amplitude / reconstruction.sum(-1, keepdim=True).clamp_min(1e-12)
        reconstruction = amplitude.matmul(dictionary)
        error = reconstruction - q
    return ExponentialKernelApproximation(
        amplitudes=amplitude.detach(),
        decays=decays.detach(),
        reconstructed_q=reconstruction.detach(),
        l1_error=error.abs().sum(-1).detach(),
        max_error=error.abs().max(-1).values.detach(),
    )



def fit_stable_exponential_sum(
    q: torch.Tensor,
    rank: int = 16,
    *,
    rho_min: float = 0.03,
    rho_max: float = 0.995,
    ridge: float = 1e-6,
) -> ExponentialKernelApproximation:
    """Fit a stable real-pole state-space kernel by ridge least squares.

    Signed output amplitudes are allowed. This is necessary for peaked Gamma
    kernels: a nonnegative sum of decaying exponentials is monotone and cannot
    represent an interior mode. Stability is retained because every pole rho
    lies strictly inside the unit circle.
    """
    if q.ndim != 2 or (q < 0).any():
        raise ValueError("q must be a nonnegative [N,L] tensor")
    if rank < 1:
        raise ValueError("rank must be positive")
    _, lag = q.shape
    decays = torch.linspace(rho_min, rho_max, rank, device=q.device, dtype=q.dtype)
    tau = torch.arange(lag, device=q.device, dtype=q.dtype)
    dictionary = decays[:, None].pow(tau[None, :]).transpose(0, 1)  # [L,R]
    gram = dictionary.transpose(0, 1).matmul(dictionary)
    eye = torch.eye(rank, device=q.device, dtype=q.dtype)
    amplitudes = torch.linalg.solve(
        gram + float(ridge) * eye,
        dictionary.transpose(0, 1).matmul(q.transpose(0, 1)),
    ).transpose(0, 1)
    reconstruction = amplitudes.matmul(dictionary.transpose(0, 1))
    error = reconstruction - q
    condition = float(torch.linalg.cond(gram + float(ridge) * eye).detach().cpu())
    return ExponentialKernelApproximation(
        amplitudes=amplitudes.detach(),
        decays=decays.detach(),
        reconstructed_q=reconstruction.detach(),
        l1_error=error.abs().sum(-1).detach(),
        max_error=error.abs().max(-1).values.detach(),
        signed_amplitudes=True,
        condition_number=condition,
    )

def exponential_state_space_filter(
    response_sequence: torch.Tensor,
    approximation: ExponentialKernelApproximation,
) -> torch.Tensor:
    """Online-filter chronological response values ``[N,T]``.

    Returns all chronological outputs ``[N,T]``.  The first L-1 values include
    zero-padded prehistory; callers comparing to exact valid convolution should
    discard that prefix.
    """
    if response_sequence.ndim != 2:
        raise ValueError("response_sequence must have shape [N,T]")
    n, time = response_sequence.shape
    amplitude = approximation.amplitudes.to(response_sequence)
    decays = approximation.decays.to(response_sequence)
    if amplitude.shape[0] != n:
        raise ValueError("variable dimension mismatch")
    state = response_sequence.new_zeros(n, decays.numel())
    outputs = []
    for t in range(time):
        state = state * decays[None, :] + response_sequence[:, t, None]
        outputs.append((amplitude * state).sum(-1))
    return torch.stack(outputs, dim=-1)
