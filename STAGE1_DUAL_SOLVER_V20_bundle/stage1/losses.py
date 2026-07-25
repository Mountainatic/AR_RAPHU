"""Loss functions for Stage1 (Section 2.8).

Total training objective:
    L = MSE(y_hat, y) + lambda_smooth * L_smooth
Group lasso applied via proximal step (not added to loss).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def smoothness_loss_from_coeffs(spline_coeffs: torch.Tensor) -> torch.Tensor:
    """Second-order difference penalty on spline coefficients."""
    if isinstance(spline_coeffs, list):
        total = torch.tensor(0.0)
        for c in spline_coeffs:
            total = total + _smoothness_single(c)
        return total
    return _smoothness_single(spline_coeffs)


def _smoothness_single(coeffs: torch.Tensor) -> torch.Tensor:
    """Smoothness for a single spline coefficient tensor [out,in,G+k]."""
    diff = coeffs[..., 2:] - 2 * coeffs[..., 1:-1] + coeffs[..., :-2]
    return (diff ** 2).sum()


def compute_smoothness_loss(model) -> torch.Tensor:
    """Extract spline coefficients from all KAN branches and compute smoothness."""
    total = torch.tensor(0.0)
    for branch in model.response_branches.branches:
        for layer in branch:
            if hasattr(layer, 'spline_weight'):
                total = total + _smoothness_single(layer.spline_weight)
    return total


def total_loss(y_hat, y, model, lambda_smooth: float = 0.0) -> tuple:
    """Compute total training loss. Returns (loss, mse, smooth)."""
    mse = F.mse_loss(y_hat, y)
    smooth = torch.tensor(0.0, device=y_hat.device)
    if lambda_smooth > 0:
        smooth = compute_smoothness_loss(model)
    loss = mse + lambda_smooth * smooth
    return loss, mse.detach(), smooth.detach()
