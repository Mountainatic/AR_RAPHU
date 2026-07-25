"""Allocation-light proximal group lasso for variable selection.

The legacy implementation flattened and concatenated every branch at every
step.  Here norms are computed directly from parameter tensors and shrinkage is
applied with device-resident scalar tensors, avoiding repeated allocations and
per-branch CPU synchronizations.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def apply_group_proximal_step(branches, lr: float, lambda_group: float,
                              eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply group soft-thresholding and return pre-step norms and shrink factors."""
    norms = branches.compute_branch_norms()
    shrink = (1.0 - float(lr) * float(lambda_group) / norms.clamp_min(eps)).clamp_min(0.0)
    # Multiplying by one is cheap and keeps the loop free of tensor-to-Python
    # comparisons, which would synchronize CUDA once per variable.
    for j, branch in enumerate(branches.branches):
        scale = shrink[j]
        for parameter in branch.parameters():
            parameter.mul_(scale)
    return norms, shrink


class ProximalGroupLasso:
    """Stateless helper retaining lambda for training protocols."""
    def __init__(self, lambda_group: float = 0.01):
        self.lambda_group = lambda_group

    def apply(self, branches, lr: float):
        return apply_group_proximal_step(branches, lr, self.lambda_group)
