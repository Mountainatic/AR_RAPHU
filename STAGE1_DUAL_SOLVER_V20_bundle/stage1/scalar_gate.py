"""Scalar-gate diagnostic with a single, registered gate parameter."""
import torch
from torch import nn


class ScalarGateModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.gates = nn.Parameter(torch.ones(base_model.num_variables))
        self.register_buffer("response_rms", torch.ones(base_model.num_variables))

    @torch.no_grad()
    def fit_response_rms(self, x_train):
        self.base_model.eval()
        x_lag = self.base_model._flip_input(x_train)
        _, response = self.base_model._compute_q_and_response(x_lag)
        self.response_rms.copy_(response.square().mean((0, 2)).sqrt().clamp_min(1e-6))

    def forward(self, x, return_aux=True):
        _, aux = self.base_model(x, return_aux=True)
        normalized_vc = aux["variable_contribution"] / self.response_rms.unsqueeze(0)
        gated_vc = normalized_vc * self.gates.unsqueeze(0)
        y = self.base_model.bias + gated_vc.sum(-1)
        aux = dict(aux, normalized_variable_contribution=normalized_vc,
                   gated_variable_contribution=gated_vc, gates=self.gates)
        return (y.unsqueeze(-1), aux) if return_aux else y.unsqueeze(-1)

    @torch.no_grad()
    def proximal_gate_step(self, lr: float, lambda_gate: float):
        self.gates.copy_(torch.sign(self.gates) *
                         torch.clamp(self.gates.abs() - lr * lambda_gate, min=0))

    def support(self, threshold=1e-8):
        return torch.where(self.gates.detach().abs().cpu() > threshold)[0].tolist()
