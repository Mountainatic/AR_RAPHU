"""M6 free-lag rank-1 initialization from a frozen M5 Gamma checkpoint."""

from __future__ import annotations

import copy

import torch

from .model import ARRAPHURank1


@torch.no_grad()
def initialize_m6_from_m5_gamma(
    m5_model: ARRAPHURank1,
    m6_model: ARRAPHURank1,
    support: list[int],
) -> dict[str, float]:
    """Copy M5 into M6 and represent its Gamma kernels as free logits.

    The conversion is exact up to floating-point roundoff because
    ``softmax(log(q)) == q`` for each normalized positive Gamma kernel.
    """

    if m5_model.track != "XAR" or m6_model.track != "XAR":
        raise ValueError("M6 initialization requires XAR models.")
    if m5_model.external_delay_mode != "static_gamma":
        raise ValueError("The M5 source must use static Gamma kernels.")
    if m6_model.external_delay_mode != "free_static_logits":
        raise ValueError("The M6 target must use free static lag logits.")
    if m5_model.external_branch is None or m6_model.external_branch is None:
        raise ValueError("M5 and M6 must both contain an external branch.")

    support = sorted({int(variable) for variable in support})
    if not support:
        raise ValueError("M6 requires a non-empty frozen M5 support.")
    if support[-1] >= m6_model.external_channels or support[0] < 0:
        raise ValueError("M6 support contains an invalid external variable.")

    gamma_q = m5_model.external_branch._static_q()
    if gamma_q is None:
        raise RuntimeError("M5 did not expose a static Gamma kernel.")
    gamma_q = gamma_q.detach().to(m6_model.bias)
    if not torch.isfinite(gamma_q).all() or (gamma_q <= 0).any():
        raise RuntimeError("M5 Gamma kernels must be finite and strictly positive.")

    source_state = m5_model.state_dict()
    target_state = copy.deepcopy(m6_model.state_dict())
    copied = 0
    for name, target in target_state.items():
        source = source_state.get(name)
        if source is not None and source.shape == target.shape:
            target_state[name] = source.detach().to(
                device=target.device, dtype=target.dtype
            ).clone()
            copied += 1
    m6_model.load_state_dict(target_state)
    m6_model.external_branch.delay_logits.copy_(gamma_q.log())

    support_set = set(support)
    for variable in range(m6_model.external_channels):
        if variable not in support_set:
            m6_model.external_branch.prune_variable(variable)

    free_q = m6_model.external_branch._static_q()
    if free_q is None:
        raise RuntimeError("M6 did not expose a static free-logit kernel.")
    kernel_max_abs_error = float((free_q - gamma_q).abs().max().cpu())
    return {
        "copied_state_entries": float(copied),
        "kernel_max_abs_error": kernel_max_abs_error,
    }
