"""Protocol-safe AR-RAPHU rank-1 model assembled from the original V20 core.

The V20 implementation assumes that every input channel shares one ``max_lag``.
This adapter deliberately keeps the external-process and autoregressive
branches separate so formal experiments can use ``L_x != L_y`` without
changing the audited V20 Gamma or KAN implementations.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from STAGE1_DUAL_SOLVER_V20_bundle.stage1.model import Stage1TargetDelayKAN
from STAGE1_DUAL_SOLVER_V20_bundle.stage1.sequence_ops import (
    ContiguousWindowBatch,
)

from .data_protocol import PREDICTION_HORIZONS


Track = Literal["X", "AR", "XAR"]
ExternalDelayMode = Literal["static_gamma", "free_static_logits"]

EXTERNAL_PROCESS_CHANNELS = 9
ARGON_FLOW_CHANNEL = 6
FORMAL_HIDDEN_KAN_CANDIDATES = (4, 8)
FORMAL_GRID_SIZE_CANDIDATES = (5, 7, 11)
FROZEN_SPLINE_ORDER = 3


def _validate_range(value: tuple[float, float], *, name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly (lower, upper).")
    lower, upper = map(float, value)
    if not torch.isfinite(torch.tensor([lower, upper])).all():
        raise ValueError(f"{name} must be finite.")
    if lower >= upper:
        raise ValueError(f"{name} must satisfy lower < upper.")
    return lower, upper


def _build_v20_branch(
    *,
    num_variables: int,
    max_lag: int,
    input_grid_ranges: list[tuple[float, float]],
    active_mask: torch.Tensor,
    hidden_kan: int,
    grid_size: int,
    response_execution_mode: str,
    delay_mode: str = "static_gamma",
) -> Stage1TargetDelayKAN:
    branch = Stage1TargetDelayKAN(
        num_variables=num_variables,
        max_lag=max_lag,
        hidden_kan=hidden_kan,
        kan_grid_size=grid_size,
        kan_spline_order=FROZEN_SPLINE_ORDER,
        epsilon=0.0,
        use_true_delays=False,
        delta=1e-3,
        prior_parametrization="mean_std",
        delay_mode=delay_mode,
        active_mask=active_mask,
        input_grid_ranges=input_grid_ranges,
        second_layer_grid_range=(-3.0, 3.0),
        response_execution_mode=response_execution_mode,
    )

    # The wrapper owns the sole identifiable intercept. The child intercept is
    # retained only for exact V20 state-dict structure and never enters output.
    branch.bias.requires_grad_(False)
    branch.bias.zero_()
    return branch


class ARRAPHURank1(nn.Module):
    """Rank-1 X-only, AR-only, or X+AR model for one direct horizon.

    Inputs are chronological windows ending at forecast origin ``t``:

    - ``x_window``: ``[batch, 9, L_x]``
    - ``y_window``: ``[batch, L_y]``

    The V20 branches reverse these windows internally to their audited
    current-to-past lag convention. The direct target ``y[t+h]`` is never
    passed to this model.

    First-layer KAN grid ranges are mandatory for every active track. They must
    be fitted outside this class from the corresponding fold's training split
    only.
    """

    def __init__(
        self,
        *,
        track: Track,
        horizon: int,
        external_channels: int = EXTERNAL_PROCESS_CHANNELS,
        inactive_external_channels: tuple[int, ...] = (ARGON_FLOW_CHANNEL,),
        L_x: int | None = None,
        L_y: int | None = None,
        input_grid_ranges_x: list[tuple[float, float]] | None = None,
        input_grid_range_y: tuple[float, float] | None = None,
        hidden_kan: int = 8,
        grid_size: int = 7,
        response_execution_mode: str = "auto",
        external_delay_mode: ExternalDelayMode = "static_gamma",
    ) -> None:
        super().__init__()
        if track not in {"X", "AR", "XAR"}:
            raise ValueError("track must be one of 'X', 'AR', or 'XAR'.")
        if horizon not in PREDICTION_HORIZONS:
            raise ValueError(
                f"horizon must be one of {PREDICTION_HORIZONS}, got {horizon!r}."
            )
        if hidden_kan not in FORMAL_HIDDEN_KAN_CANDIDATES:
            raise ValueError(
                f"hidden_kan must be one of {FORMAL_HIDDEN_KAN_CANDIDATES}."
            )
        if grid_size not in FORMAL_GRID_SIZE_CANDIDATES:
            raise ValueError(
                f"grid_size must be one of {FORMAL_GRID_SIZE_CANDIDATES}."
            )

        self.track = track
        self.horizon = horizon
        if not isinstance(external_channels, int) or external_channels <= 0:
            raise ValueError("external_channels must be a positive integer.")
        invalid_inactive = [
            index
            for index in inactive_external_channels
            if not isinstance(index, int) or not 0 <= index < external_channels
        ]
        if invalid_inactive:
            raise ValueError(
                "inactive_external_channels contains invalid indices: "
                f"{invalid_inactive}."
            )
        if len(set(inactive_external_channels)) != len(
            inactive_external_channels
        ):
            raise ValueError("inactive_external_channels must be unique.")
        self.external_channels = external_channels
        self.inactive_external_channels = tuple(inactive_external_channels)
        self.L_x = L_x
        self.L_y = L_y
        if external_delay_mode not in {"static_gamma", "free_static_logits"}:
            raise ValueError("Unsupported external_delay_mode.")
        self.external_delay_mode = external_delay_mode
        self.bias = nn.Parameter(torch.tensor(0.0))

        if track in {"X", "XAR"}:
            if not isinstance(L_x, int) or L_x <= 0:
                raise ValueError("L_x must be a positive integer for X tracks.")
            if input_grid_ranges_x is None:
                raise ValueError(
                    "input_grid_ranges_x is required and must be fitted on train only."
                )
            if len(input_grid_ranges_x) != external_channels:
                raise ValueError(
                    "input_grid_ranges_x must contain one range for each "
                    "external process channel."
                )
            ranges_x = [
                _validate_range(value, name=f"input_grid_ranges_x[{index}]")
                for index, value in enumerate(input_grid_ranges_x)
            ]
            external_mask = torch.ones(
                external_channels, dtype=torch.bool
            )
            if inactive_external_channels:
                external_mask[list(inactive_external_channels)] = False
            self.external_branch = _build_v20_branch(
                num_variables=external_channels,
                max_lag=L_x,
                input_grid_ranges=ranges_x,
                active_mask=external_mask,
                hidden_kan=hidden_kan,
                grid_size=grid_size,
                response_execution_mode=response_execution_mode,
                delay_mode=external_delay_mode,
            )
            for inactive_channel in inactive_external_channels:
                for parameter in self.external_branch.response_branches.branches[
                    inactive_channel
                ].parameters():
                    parameter.requires_grad_(False)
        else:
            self.external_branch = None

        if track in {"AR", "XAR"}:
            if not isinstance(L_y, int) or L_y <= 0:
                raise ValueError("L_y must be a positive integer for AR tracks.")
            if input_grid_range_y is None:
                raise ValueError(
                    "input_grid_range_y is required and must be fitted on train only."
                )
            range_y = _validate_range(
                input_grid_range_y, name="input_grid_range_y"
            )
            self.ar_branch = _build_v20_branch(
                num_variables=1,
                max_lag=L_y,
                input_grid_ranges=[range_y],
                active_mask=torch.ones(1, dtype=torch.bool),
                hidden_kan=hidden_kan,
                grid_size=grid_size,
                response_execution_mode=response_execution_mode,
                delay_mode="static_gamma",
            )
        else:
            self.ar_branch = None

    @property
    def uses_external(self) -> bool:
        return self.external_branch is not None

    @property
    def uses_ar(self) -> bool:
        return self.ar_branch is not None

    def _validate_inputs(
        self,
        x_window: torch.Tensor | None,
        y_window: torch.Tensor | None,
    ) -> tuple[int, torch.device, torch.dtype]:
        batch_size: int | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None

        if self.uses_external:
            if x_window is None:
                raise ValueError(f"Track-{self.track} requires x_window.")
            if x_window.ndim != 3 or tuple(x_window.shape[1:]) != (
                self.external_channels,
                self.L_x,
            ):
                raise ValueError(
                    f"x_window must have shape [batch, {self.external_channels}, "
                    f"{self.L_x}]."
                )
            batch_size, device, dtype = (
                x_window.shape[0],
                x_window.device,
                x_window.dtype,
            )
        elif x_window is not None:
            raise ValueError("Track-AR must not receive x_window.")

        if self.uses_ar:
            if y_window is None:
                raise ValueError(f"Track-{self.track} requires y_window.")
            if y_window.ndim != 2 or y_window.shape[1] != self.L_y:
                raise ValueError(
                    f"y_window must have shape [batch, {self.L_y}]."
                )
            if batch_size is not None and y_window.shape[0] != batch_size:
                raise ValueError("x_window and y_window batch sizes must match.")
            if device is not None and y_window.device != device:
                raise ValueError("x_window and y_window devices must match.")
            if dtype is not None and y_window.dtype != dtype:
                raise ValueError("x_window and y_window dtypes must match.")
            batch_size, device, dtype = (
                y_window.shape[0],
                y_window.device,
                y_window.dtype,
            )
        elif y_window is not None:
            raise ValueError("Track-X must not receive y_window.")

        if batch_size is None or device is None or dtype is None:
            raise RuntimeError("No active input branch was configured.")
        if not dtype.is_floating_point:
            raise ValueError("Model windows must use a floating-point dtype.")
        return batch_size, device, dtype

    def forward(
        self,
        x_window: torch.Tensor | None = None,
        y_window: torch.Tensor | None = None,
        *,
        return_aux: bool = True,
    ):
        batch_size, device, dtype = self._validate_inputs(x_window, y_window)
        external_contribution = torch.zeros(
            batch_size,
            self.external_channels,
            device=device,
            dtype=dtype,
        )
        ar_contribution = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        external_aux = ar_aux = None

        if self.external_branch is not None:
            _, external_aux = self.external_branch(x_window, return_aux=True)
            external_contribution = external_aux["variable_contribution"]
        if self.ar_branch is not None:
            _, ar_aux = self.ar_branch(
                y_window.unsqueeze(1), return_aux=True
            )
            ar_contribution = ar_aux["variable_contribution"]

        component_contribution = torch.cat(
            (external_contribution, ar_contribution), dim=1
        )
        prediction = self.bias + component_contribution.sum(dim=1)
        prediction = prediction.unsqueeze(-1)
        if not return_aux:
            return prediction

        full_active_mask = torch.zeros(
            self.external_channels + 1, dtype=torch.bool, device=device
        )
        if self.external_branch is not None:
            full_active_mask[: self.external_channels] = (
                self.external_branch.active_mask
            )
        if self.ar_branch is not None:
            full_active_mask[-1] = True

        aux = {
            "track": self.track,
            "horizon": self.horizon,
            "external": external_aux,
            "ar": ar_aux,
            "external_contribution": external_contribution,
            "ar_contribution": ar_contribution,
            "component_contribution": component_contribution,
            "active_mask": full_active_mask,
        }
        return prediction, aux

    def forward_contiguous(
        self,
        x_sequence: torch.Tensor | None = None,
        y_sequence: torch.Tensor | None = None,
        *,
        return_aux: bool = True,
    ):
        """Exact sequence-first forward for consecutive forecast origins.

        ``x_sequence`` has shape ``[external_channels, B + L_x - 1]`` and
        ``y_sequence`` has shape ``[B + L_y - 1]``. Each V20 response is
        evaluated once per unique time point before exact depthwise
        convolution.
        """

        batch_size: int | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        if self.uses_external:
            if x_sequence is None or x_sequence.ndim != 2:
                raise ValueError("An X track requires x_sequence [channels,time].")
            if x_sequence.shape[0] != self.external_channels:
                raise ValueError("x_sequence has the wrong channel count.")
            candidate = x_sequence.shape[1] - int(self.L_x) + 1
            if candidate <= 0:
                raise ValueError("x_sequence is shorter than L_x.")
            batch_size, device, dtype = (
                candidate,
                x_sequence.device,
                x_sequence.dtype,
            )
        elif x_sequence is not None:
            raise ValueError("Track-AR must not receive x_sequence.")

        if self.uses_ar:
            if y_sequence is None or y_sequence.ndim != 1:
                raise ValueError("An AR track requires y_sequence [time].")
            candidate = y_sequence.shape[0] - int(self.L_y) + 1
            if candidate <= 0:
                raise ValueError("y_sequence is shorter than L_y.")
            if batch_size is not None and candidate != batch_size:
                raise ValueError("X and AR sequences imply different batch sizes.")
            if device is not None and y_sequence.device != device:
                raise ValueError("X and AR sequences must use the same device.")
            if dtype is not None and y_sequence.dtype != dtype:
                raise ValueError("X and AR sequences must use the same dtype.")
            batch_size, device, dtype = candidate, y_sequence.device, y_sequence.dtype
        elif y_sequence is not None:
            raise ValueError("Track-X must not receive y_sequence.")

        if batch_size is None or device is None or dtype is None:
            raise RuntimeError("No active sequence branch was configured.")
        external_contribution = torch.zeros(
            batch_size, self.external_channels, device=device, dtype=dtype
        )
        ar_contribution = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        external_aux = ar_aux = None

        if self.external_branch is not None:
            x_windows = x_sequence.unfold(-1, int(self.L_x), 1).permute(1, 0, 2)
            x_batch = ContiguousWindowBatch(
                windows=x_windows,
                raw_sequence=x_sequence,
            )
            _, external_aux = self.external_branch.forward_contiguous(
                x_batch, return_aux=True
            )
            external_contribution = external_aux["variable_contribution"]
        if self.ar_branch is not None:
            ar_raw = y_sequence.unsqueeze(0)
            y_windows = ar_raw.unfold(-1, int(self.L_y), 1).permute(1, 0, 2)
            y_batch = ContiguousWindowBatch(
                windows=y_windows,
                raw_sequence=ar_raw,
            )
            _, ar_aux = self.ar_branch.forward_contiguous(
                y_batch, return_aux=True
            )
            ar_contribution = ar_aux["variable_contribution"]

        component_contribution = torch.cat(
            (external_contribution, ar_contribution), dim=1
        )
        prediction = (
            self.bias + component_contribution.sum(dim=1)
        ).unsqueeze(-1)
        if not return_aux:
            return prediction
        full_active_mask = torch.zeros(
            self.external_channels + 1, dtype=torch.bool, device=device
        )
        if self.external_branch is not None:
            full_active_mask[: self.external_channels] = (
                self.external_branch.active_mask
            )
        if self.ar_branch is not None:
            full_active_mask[-1] = True
        return prediction, {
            "track": self.track,
            "horizon": self.horizon,
            "external": external_aux,
            "ar": ar_aux,
            "external_contribution": external_contribution,
            "ar_contribution": ar_contribution,
            "component_contribution": component_contribution,
            "active_mask": full_active_mask,
            "sequence_fast_path": True,
        }
