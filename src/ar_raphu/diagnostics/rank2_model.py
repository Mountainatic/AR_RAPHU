"""Exact two-component diagnostic model built from audited rank-1 adapters."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from ar_raphu.model import ARRAPHURank1


QMode = Literal["oracle_fixed", "free_truth_init"]
ACTIVE_SUPPORT = (0, 1, 2)
EXTERNAL_VARIABLES = 10
L_X = 64
L_Y = 32


def _truth_q_tensor(
    values: np.ndarray | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    q = torch.as_tensor(values, dtype=torch.float32, device=device).clone()
    if q.shape != (EXTERNAL_VARIABLES, L_X):
        raise ValueError(f"Truth q must have shape [{EXTERNAL_VARIABLES}, {L_X}].")
    if not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("Truth q must be finite and nonnegative.")
    row_sums = q.sum(dim=1)
    active = torch.tensor(ACTIVE_SUPPORT, device=q.device)
    if not torch.allclose(
        row_sums[active], torch.ones_like(row_sums[active]), atol=1.0e-6
    ):
        raise ValueError("Active truth q rows must sum to one.")
    inactive = sorted(set(range(EXTERNAL_VARIABLES)) - set(ACTIVE_SUPPORT))
    if (row_sums[inactive].abs() > 1.0e-8).any():
        raise ValueError("Inactive truth q rows must be zero.")
    return q


def _v20_fixed_q(q: torch.Tensor) -> torch.Tensor:
    """Satisfy the legacy all-row normalization contract for masked rows."""

    result = q.clone()
    inactive = sorted(set(range(EXTERNAL_VARIABLES)) - set(ACTIVE_SUPPORT))
    result[inactive] = 1.0 / L_X
    return result


class ARRAPHURank2Diagnostic(nn.Module):
    """Frozen-weight rank-2 X model with an optional audited AR branch."""

    def __init__(
        self,
        *,
        horizon: int,
        input_grid_ranges_x: list[tuple[float, float]],
        q_primary: np.ndarray | torch.Tensor,
        q_secondary: np.ndarray | torch.Tensor,
        q_mode: QMode,
        include_ar: bool = False,
        input_grid_range_y: tuple[float, float] | None = None,
        hidden_kan: int = 8,
        grid_size: int = 7,
        response_execution_mode: str = "vectorized",
    ) -> None:
        super().__init__()
        if q_mode not in {"oracle_fixed", "free_truth_init"}:
            raise ValueError("q_mode must be oracle_fixed or free_truth_init.")
        if len(input_grid_ranges_x) != EXTERNAL_VARIABLES:
            raise ValueError("Exactly ten X grid ranges are required.")
        if include_ar and input_grid_range_y is None:
            raise ValueError("include_ar=True requires input_grid_range_y.")

        self.horizon = int(horizon)
        self.q_mode = q_mode
        self.include_ar = bool(include_ar)
        self.register_buffer(
            "component_weights", torch.tensor([0.6, 0.4], dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.tensor(0.0))
        inactive = tuple(
            sorted(set(range(EXTERNAL_VARIABLES)) - set(ACTIVE_SUPPORT))
        )
        delay_mode = (
            "free_static_logits"
            if q_mode == "free_truth_init"
            else "static_gamma"
        )
        component_kwargs = {
            "track": "X",
            "horizon": self.horizon,
            "external_channels": EXTERNAL_VARIABLES,
            "inactive_external_channels": inactive,
            "L_x": L_X,
            "input_grid_ranges_x": input_grid_ranges_x,
            "hidden_kan": hidden_kan,
            "grid_size": grid_size,
            "response_execution_mode": response_execution_mode,
            "external_delay_mode": delay_mode,
        }
        self.component_1 = ARRAPHURank1(**component_kwargs)
        self.component_2 = ARRAPHURank1(**component_kwargs)
        for component in (self.component_1, self.component_2):
            component.bias.detach().zero_()
            component.bias.requires_grad_(False)

        primary = _truth_q_tensor(q_primary)
        secondary = _truth_q_tensor(q_secondary)
        if q_mode == "oracle_fixed":
            self.component_1.external_branch.set_fixed_delays(
                _v20_fixed_q(primary)
            )
            self.component_2.external_branch.set_fixed_delays(
                _v20_fixed_q(secondary)
            )
            for component in (self.component_1, self.component_2):
                for parameter in component.external_branch.delay_prior.parameters():
                    parameter.requires_grad_(False)
        else:
            with torch.no_grad():
                for component, truth in (
                    (self.component_1, primary),
                    (self.component_2, secondary),
                ):
                    logits = torch.zeros_like(
                        component.external_branch.delay_logits
                    )
                    logits[list(ACTIVE_SUPPORT)] = torch.log(
                        truth[list(ACTIVE_SUPPORT)].clamp_min(1.0e-8)
                    )
                    component.external_branch.delay_logits.copy_(logits)

        if include_ar:
            self.ar_component = ARRAPHURank1(
                track="AR",
                horizon=self.horizon,
                L_y=L_Y,
                input_grid_range_y=input_grid_range_y,
                hidden_kan=hidden_kan,
                grid_size=grid_size,
                response_execution_mode=response_execution_mode,
            )
            self.ar_component.bias.detach().zero_()
            self.ar_component.bias.requires_grad_(False)
        else:
            self.ar_component = None

    @property
    def uses_external(self) -> bool:
        return True

    @property
    def uses_ar(self) -> bool:
        return self.ar_component is not None

    def _combine(
        self,
        first_aux: dict,
        second_aux: dict,
        ar_aux: dict | None,
    ) -> tuple[torch.Tensor, dict]:
        first = first_aux["external_contribution"]
        second = second_aux["external_contribution"]
        external = (
            self.component_weights[0] * first
            + self.component_weights[1] * second
        )
        ar = (
            ar_aux["ar_contribution"]
            if ar_aux is not None
            else external.new_zeros((external.shape[0], 1))
        )
        prediction = (self.bias + external.sum(dim=1) + ar.sum(dim=1)).unsqueeze(
            -1
        )
        return prediction, {
            "external_contribution": external,
            "ar_contribution": ar,
            "component_contribution": torch.cat((external, ar), dim=1),
            "rank2_component_1_contribution": first,
            "rank2_component_2_contribution": second,
            "rank2_component_1_aux": first_aux,
            "rank2_component_2_aux": second_aux,
            "ar": ar_aux,
        }

    def forward(
        self,
        x_window: torch.Tensor,
        y_window: torch.Tensor | None = None,
        *,
        return_aux: bool = True,
    ):
        _, first_aux = self.component_1(x_window, return_aux=True)
        _, second_aux = self.component_2(x_window, return_aux=True)
        ar_aux = None
        if self.ar_component is not None:
            if y_window is None:
                raise ValueError("The configured AR component requires y_window.")
            _, ar_aux = self.ar_component(y_window=y_window, return_aux=True)
        elif y_window is not None:
            raise ValueError("An X-only rank-2 model must not receive y_window.")
        prediction, auxiliary = self._combine(first_aux, second_aux, ar_aux)
        return (prediction, auxiliary) if return_aux else prediction

    def forward_contiguous(
        self,
        x_sequence: torch.Tensor,
        y_sequence: torch.Tensor | None = None,
        *,
        return_aux: bool = True,
    ):
        _, first_aux = self.component_1.forward_contiguous(
            x_sequence=x_sequence, return_aux=True
        )
        _, second_aux = self.component_2.forward_contiguous(
            x_sequence=x_sequence, return_aux=True
        )
        ar_aux = None
        if self.ar_component is not None:
            if y_sequence is None:
                raise ValueError("The configured AR component requires y_sequence.")
            _, ar_aux = self.ar_component.forward_contiguous(
                y_sequence=y_sequence, return_aux=True
            )
        elif y_sequence is not None:
            raise ValueError("An X-only rank-2 model must not receive y_sequence.")
        prediction, auxiliary = self._combine(first_aux, second_aux, ar_aux)
        auxiliary["sequence_fast_path"] = True
        return (prediction, auxiliary) if return_aux else prediction

    def response_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for component in (self.component_1, self.component_2):
            parameters.extend(
                parameter
                for parameter in component.external_branch.response_branches.parameters()
                if parameter.requires_grad
            )
        return parameters

    def lag_parameters(self) -> list[nn.Parameter]:
        if self.q_mode != "free_truth_init":
            return []
        return [
            self.component_1.external_branch.delay_logits,
            self.component_2.external_branch.delay_logits,
        ]

    def bias_and_ar_parameters(self) -> list[nn.Parameter]:
        parameters = [self.bias]
        if self.ar_component is not None:
            parameters.extend(
                parameter
                for parameter in self.ar_component.parameters()
                if parameter.requires_grad
            )
        return parameters

    def lag_roughness(self) -> torch.Tensor:
        penalty = self.bias.new_zeros(())
        for logits in self.lag_parameters():
            second = logits[:, 2:] - 2.0 * logits[:, 1:-1] + logits[:, :-2]
            penalty = penalty + second[list(ACTIVE_SUPPORT)].square().sum()
        return penalty

    @torch.no_grad()
    def lag_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.component_1.external_branch._static_q().detach().clone(),
            self.component_2.external_branch._static_q().detach().clone(),
        )
