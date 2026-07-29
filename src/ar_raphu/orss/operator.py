"""Matrix-free multichannel Urysohn operator for CZ ORSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.interpolate import BSpline

from ar_raphu.spectral.amplitude_domain import AmplitudeDomain
from ar_raphu.spectral.spline_basis import CenteredSplineBasis, clamped_knots


def _torch_bspline_basis(
    values: torch.Tensor,
    knots: np.ndarray,
    degree: int,
) -> torch.Tensor:
    knot = torch.as_tensor(knots, device=values.device, dtype=values.dtype)
    flat = values.reshape(-1)
    lower = knot[degree]
    upper = knot[-degree - 1]
    if bool(torch.any(~torch.isfinite(flat))):
        raise ValueError("Non-finite spline input.")
    if bool(torch.any(flat < lower)) or bool(torch.any(flat > upper)):
        raise ValueError("Spline input outside frozen support.")
    sample = flat[:, None]
    basis = (
        (sample >= knot[:-1][None, :])
        & (sample < knot[1:][None, :])
    ).to(values.dtype)
    tiny = torch.finfo(values.dtype).tiny
    for order in range(1, degree + 1):
        width = len(knot) - order - 1
        left_denominator = knot[order : order + width] - knot[:width]
        right_denominator = (
            knot[order + 1 : order + width + 1] - knot[1 : width + 1]
        )
        left = torch.where(
            left_denominator[None, :] != 0,
            (sample - knot[:width][None, :])
            / left_denominator[None, :].clamp_min(tiny),
            torch.zeros((), device=values.device, dtype=values.dtype),
        )
        right = torch.where(
            right_denominator[None, :] != 0,
            (
                knot[order + 1 : order + width + 1][None, :]
                - sample
            )
            / right_denominator[None, :].clamp_min(tiny),
            torch.zeros((), device=values.device, dtype=values.dtype),
        )
        basis = left * basis[:, :width] + right * basis[:, 1 : width + 1]
    endpoint = flat == upper
    if bool(torch.any(endpoint)):
        basis[endpoint] = 0.0
        basis[endpoint, -1] = 1.0
    return basis.reshape(values.shape + (basis.shape[-1],))


def _bounded_c1_basis(
    basis: CenteredSplineBasis,
    values: torch.Tensor,
    *,
    scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = values.reshape(-1)
    lower = torch.as_tensor(basis.lower, device=values.device, dtype=values.dtype)
    upper = torch.as_tensor(basis.upper, device=values.device, dtype=values.dtype)
    inside = (flat >= lower) & (flat <= upper)
    result = torch.empty(
        (len(flat), len(basis.train_mean)),
        device=values.device,
        dtype=values.dtype,
    )
    mean = torch.as_tensor(
        basis.train_mean, device=values.device, dtype=values.dtype
    )
    if bool(torch.any(inside)):
        result[inside] = (
            _torch_bspline_basis(flat[inside], basis.knots, basis.degree) - mean
        )
    vector_spline = BSpline(
        basis.knots,
        np.eye(len(basis.train_mean), dtype=np.float64),
        basis.degree,
        extrapolate=False,
    )
    left_spacing, right_spacing = basis.boundary_knot_spacings()
    rho_left = float(scale_factor) * left_spacing
    rho_right = float(scale_factor) * right_spacing
    left_value = torch.as_tensor(
        vector_spline(basis.lower) - basis.train_mean,
        device=values.device,
        dtype=values.dtype,
    )
    right_value = torch.as_tensor(
        vector_spline(basis.upper) - basis.train_mean,
        device=values.device,
        dtype=values.dtype,
    )
    left_derivative = torch.as_tensor(
        vector_spline(basis.lower, nu=1),
        device=values.device,
        dtype=values.dtype,
    )
    right_derivative = torch.as_tensor(
        vector_spline(basis.upper, nu=1),
        device=values.device,
        dtype=values.dtype,
    )
    left = flat < lower
    if bool(torch.any(left)):
        normalized = (flat[left] - lower) / rho_left
        result[left] = (
            left_value[None, :]
            + rho_left
            * left_derivative[None, :]
            * torch.tanh(normalized)[:, None]
        )
    right = flat > upper
    if bool(torch.any(right)):
        normalized = (flat[right] - upper) / rho_right
        result[right] = (
            right_value[None, :]
            + rho_right
            * right_derivative[None, :]
            * torch.tanh(normalized)[:, None]
        )
    return result.reshape(values.shape + (len(basis.train_mean),)), inside.reshape(
        values.shape
    )


@dataclass(frozen=True, slots=True)
class OperatorBasisState:
    external_bases: tuple[CenteredSplineBasis, ...]
    ar_basis: CenteredSplineBasis
    external_lag_knots: np.ndarray
    ar_lag_knots: np.ndarray
    degree: int
    lag_basis_count: int
    amplitude_basis_count: int


@dataclass(slots=True)
class BranchCache:
    amplitude: torch.Tensor
    lag_basis: torch.Tensor
    out_of_domain_fraction: float


class UrysohnLinearOperator:
    """Centered design operator represented by branch-local basis caches."""

    def __init__(
        self,
        branches: Sequence[BranchCache],
        *,
        feature_mean: torch.Tensor | None = None,
        chunk_time: int = 2048,
    ) -> None:
        if not branches:
            raise ValueError("At least one branch is required.")
        observations = branches[0].amplitude.shape[0]
        m_tau = branches[0].lag_basis.shape[1]
        m_x = branches[0].amplitude.shape[2]
        for branch in branches:
            if branch.amplitude.shape[0] != observations:
                raise ValueError("Branch observation counts differ.")
            if branch.lag_basis.shape[0] != branch.amplitude.shape[1]:
                raise ValueError("Lag and amplitude cache lengths differ.")
            if branch.lag_basis.shape[1] != m_tau:
                raise ValueError("Branch lag basis counts differ.")
            if branch.amplitude.shape[2] != m_x:
                raise ValueError("Branch amplitude basis counts differ.")
        self.branches = tuple(branches)
        self.observations = observations
        self.channels = len(branches)
        self.m_tau = m_tau
        self.m_x = m_x
        self.dimension = self.channels * m_tau * m_x
        self.device = branches[0].amplitude.device
        self.dtype = branches[0].amplitude.dtype
        self.chunk_time = int(chunk_time)
        if self.chunk_time < 1:
            raise ValueError("chunk_time must be positive.")
        if feature_mean is None:
            ones = torch.ones(
                observations, device=self.device, dtype=self.dtype
            )
            self.feature_mean = self._raw_adjoint(ones) / observations
        else:
            expected = (self.channels, self.m_tau, self.m_x)
            if tuple(feature_mean.shape) != expected:
                raise ValueError("feature_mean has incompatible shape.")
            self.feature_mean = feature_mean.to(
                device=self.device, dtype=self.dtype
            )

    def reshape_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return theta.reshape(self.channels, self.m_tau, self.m_x)

    def _raw_forward(self, theta: torch.Tensor) -> torch.Tensor:
        coefficients = self.reshape_theta(theta)
        output = torch.zeros(
            self.observations, device=self.device, dtype=self.dtype
        )
        for start in range(0, self.observations, self.chunk_time):
            stop = min(start + self.chunk_time, self.observations)
            chunk = torch.zeros(
                stop - start, device=self.device, dtype=self.dtype
            )
            for channel, branch in enumerate(self.branches):
                chunk += torch.einsum(
                    "nlb,la,ab->n",
                    branch.amplitude[start:stop],
                    branch.lag_basis,
                    coefficients[channel],
                )
            output[start:stop] = chunk
        return output

    def _raw_adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (self.channels, self.m_tau, self.m_x),
            device=self.device,
            dtype=self.dtype,
        )
        for start in range(0, self.observations, self.chunk_time):
            stop = min(start + self.chunk_time, self.observations)
            local = residual[start:stop]
            for channel, branch in enumerate(self.branches):
                result[channel] += torch.einsum(
                    "nlb,la,n->ab",
                    branch.amplitude[start:stop],
                    branch.lag_basis,
                    local,
                )
        return result

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        coefficients = self.reshape_theta(theta)
        centered_offset = torch.sum(self.feature_mean * coefficients)
        return self._raw_forward(theta) - centered_offset

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        centered = self._raw_adjoint(residual)
        centered -= self.feature_mean * residual.sum()
        return centered.reshape(-1)

    def normal(self, theta: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(theta)) / self.observations

    def rhs(self, centered_target: torch.Tensor) -> torch.Tensor:
        return self.adjoint(centered_target) / self.observations

    def dense_design(self) -> torch.Tensor:
        blocks = [
            torch.einsum(
                "la,nlb->nab", branch.lag_basis, branch.amplitude
            ).reshape(self.observations, self.m_tau * self.m_x)
            for branch in self.branches
        ]
        raw = torch.cat(blocks, dim=1)
        return raw - self.feature_mean.reshape(-1)[None, :]


def _device_array(
    values: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    host = torch.from_numpy(np.ascontiguousarray(values))
    if device.type == "cuda":
        host = host.pin_memory()
    return host.to(device=device, dtype=dtype, non_blocking=True)


def build_urysohn_operator(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_indices: np.ndarray,
    train_target_stop: int,
    horizon: int,
    L_x: int,
    L_y: int,
    lag_basis_count: int,
    amplitude_basis_count: int,
    continuation_scale_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
    chunk_time: int,
    basis_state: OperatorBasisState | None = None,
    feature_mean: torch.Tensor | None = None,
) -> tuple[UrysohnLinearOperator, OperatorBasisState]:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    targets = np.asarray(target_indices, dtype=np.int64)
    origins = targets - int(horizon)
    if origins.min() - max(L_x, L_y) + 1 < 0:
        raise ValueError("History precedes sequence start.")
    degree = 3
    if basis_state is None:
        train_x = x_array[:train_target_stop]
        external_bases = tuple(
            CenteredSplineBasis.fit(
                train_x[:, variable],
                n_basis=amplitude_basis_count,
                degree=degree,
                domain=AmplitudeDomain.fit(
                    train_x[:, variable],
                    padding_fraction=0.10,
                    core_quantiles=(0.01, 0.99),
                ),
                quantiles=(0.01, 0.99),
            )
            for variable in range(x_array.shape[1])
        )
        train_y = y_array[:train_target_stop]
        ar_basis = CenteredSplineBasis.fit(
            train_y,
            n_basis=amplitude_basis_count,
            degree=degree,
            domain=AmplitudeDomain.fit(train_y, padding_fraction=0.10),
        )
        basis_state = OperatorBasisState(
            external_bases=external_bases,
            ar_basis=ar_basis,
            external_lag_knots=clamped_knots(
                0.0, float(L_x - 1), lag_basis_count, degree
            ),
            ar_lag_knots=clamped_knots(
                0.0, float(L_y - 1), lag_basis_count, degree
            ),
            degree=degree,
            lag_basis_count=lag_basis_count,
            amplitude_basis_count=amplitude_basis_count,
        )
    x_tensor = _device_array(x_array, device=device, dtype=dtype)
    y_tensor = _device_array(y_array, device=device, dtype=dtype)
    target_host = torch.from_numpy(targets)
    if device.type == "cuda":
        target_host = target_host.pin_memory()
    target_tensor = target_host.to(device=device, non_blocking=True)
    origin_tensor = target_tensor - int(horizon)

    external_lag = _torch_bspline_basis(
        torch.arange(L_x, device=device, dtype=dtype),
        basis_state.external_lag_knots,
        degree,
    )
    ar_lag = _torch_bspline_basis(
        torch.arange(L_y, device=device, dtype=dtype),
        basis_state.ar_lag_knots,
        degree,
    )
    external_offsets = torch.arange(L_x, device=device, dtype=torch.int64)
    ar_offsets = torch.arange(L_y, device=device, dtype=torch.int64)
    external_indices = origin_tensor[:, None] - external_offsets[None, :]
    ar_indices = origin_tensor[:, None] - ar_offsets[None, :]
    branches: list[BranchCache] = []
    for variable, basis in enumerate(basis_state.external_bases):
        windows = x_tensor[external_indices, variable]
        amplitude, inside = _bounded_c1_basis(
            basis,
            windows,
            scale_factor=continuation_scale_coefficient,
        )
        branches.append(
            BranchCache(
                amplitude=amplitude,
                lag_basis=external_lag,
                out_of_domain_fraction=float((~inside).float().mean().item()),
            )
        )
    ar_windows = y_tensor[ar_indices]
    ar_amplitude, ar_inside = _bounded_c1_basis(
        basis_state.ar_basis,
        ar_windows,
        scale_factor=continuation_scale_coefficient,
    )
    branches.append(
        BranchCache(
            amplitude=ar_amplitude,
            lag_basis=ar_lag,
            out_of_domain_fraction=float((~ar_inside).float().mean().item()),
        )
    )
    operator = UrysohnLinearOperator(
        branches, feature_mean=feature_mean, chunk_time=chunk_time
    )
    return operator, basis_state

