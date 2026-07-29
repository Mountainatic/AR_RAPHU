"""Parametric reduced-basis sweep for affine penalty families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .operator import UrysohnLinearOperator
from .penalties import PenaltyWeights, SeparablePenalty


@dataclass(frozen=True, slots=True)
class ReducedCandidate:
    weights: PenaltyWeights
    coefficients: torch.Tensor
    reduced_coordinates: torch.Tensor
    relative_residual: float


class ParametricReducedBasis:
    def __init__(
        self,
        operator: UrysohnLinearOperator,
        penalty: SeparablePenalty,
        rhs: torch.Tensor,
        *,
        maximum_dimension: int = 128,
    ) -> None:
        self.operator = operator
        self.penalty = penalty
        self.rhs = rhs
        self.maximum_dimension = maximum_dimension
        self.basis = torch.empty(
            (operator.dimension, 0),
            device=operator.device,
            dtype=torch.float64,
        )
        self.anchor_weights: list[PenaltyWeights] = []
        self._projected: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]

    def add_anchor(
        self, weights: PenaltyWeights, coefficients: torch.Tensor
    ) -> bool:
        vector = coefficients.to(device=self.basis.device, dtype=torch.float64)
        if self.dimension:
            vector = vector - self.basis @ (self.basis.T @ vector)
            vector = vector - self.basis @ (self.basis.T @ vector)
        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) <= 1.0e-12:
            return False
        if self.dimension >= self.maximum_dimension:
            raise RuntimeError("RB_DIMENSION_EXCEEDED")
        self.basis = torch.cat((self.basis, (vector / norm)[:, None]), dim=1)
        self.anchor_weights.append(weights)
        self._projected = None
        return True

    def project_operators(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._projected is not None:
            return self._projected
        if not self.dimension:
            raise RuntimeError("Reduced basis has no anchors.")
        columns = [self.basis[:, index] for index in range(self.dimension)]
        data_columns = torch.stack(
            [
                self.operator.normal(
                    column.to(
                        device=self.operator.device,
                        dtype=self.operator.dtype,
                    )
                ).to(torch.float64)
                for column in columns
            ],
            dim=1,
        )
        lag_columns = torch.stack(
            [
                self.penalty.lag_adjoint(
                    self.penalty.lag_forward(
                        column.to(
                            device=self.penalty.device,
                            dtype=self.penalty.dtype,
                        )
                    )
                ).to(torch.float64)
                for column in columns
            ],
            dim=1,
        )
        amplitude_columns = torch.stack(
            [
                self.penalty.amplitude_adjoint(
                    self.penalty.amplitude_forward(
                        column.to(
                            device=self.penalty.device,
                            dtype=self.penalty.dtype,
                        )
                    )
                ).to(torch.float64)
                for column in columns
            ],
            dim=1,
        )
        h0 = self.basis.T @ data_columns
        lag = self.basis.T @ lag_columns
        amplitude = self.basis.T @ amplitude_columns
        right = self.basis.T @ self.rhs.to(torch.float64)
        self._projected = (
            0.5 * (h0 + h0.T),
            0.5 * (lag + lag.T),
            0.5 * (amplitude + amplitude.T),
            right,
        )
        return self._projected

    def solve(self, weights: PenaltyWeights) -> ReducedCandidate:
        return self.solve_many([weights])[0]

    def solve_many(
        self, weights: list[PenaltyWeights] | tuple[PenaltyWeights, ...]
    ) -> list[ReducedCandidate]:
        if not weights:
            return []
        h0, lag, amplitude, right = self.project_operators()
        lag_weights = torch.as_tensor(
            [row.lag for row in weights], device=h0.device, dtype=h0.dtype
        )
        amplitude_weights = torch.as_tensor(
            [row.amplitude for row in weights],
            device=h0.device,
            dtype=h0.dtype,
        )
        ridge_weights = torch.as_tensor(
            [row.ridge for row in weights], device=h0.device, dtype=h0.dtype
        )
        identity = torch.eye(
            self.dimension, device=h0.device, dtype=h0.dtype
        )
        systems = (
            h0[None, ...]
            + lag_weights[:, None, None] * lag[None, ...]
            + amplitude_weights[:, None, None] * amplitude[None, ...]
            + ridge_weights[:, None, None] * identity[None, ...]
        )
        rights = right[None, :].expand(len(weights), -1)
        coordinates = torch.empty_like(rights)
        exact_zero = torch.as_tensor(
            [row.exact_zero for row in weights],
            device=h0.device,
            dtype=torch.bool,
        )
        regular = ~exact_zero
        if bool(torch.any(regular)):
            solved, info = torch.linalg.solve_ex(
                systems[regular], rights[regular], check_errors=False
            )
            regular_indices = torch.nonzero(regular, as_tuple=False).flatten()
            coordinates[regular] = solved
            failed = torch.nonzero(info != 0, as_tuple=False).flatten()
            for failed_index in failed.detach().cpu().tolist():
                row_index = int(regular_indices[failed_index].item())
                coordinates[row_index] = torch.linalg.lstsq(
                    systems[row_index], rights[row_index]
                ).solution
        for row_index in (
            torch.nonzero(exact_zero, as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        ):
            coordinates[row_index] = torch.linalg.lstsq(
                systems[row_index], rights[row_index]
            ).solution
        coefficients = coordinates @ self.basis.T
        residuals = self.full_residual_many(coefficients, weights)
        denominator = torch.linalg.vector_norm(self.rhs).clamp_min(
            torch.finfo(torch.float64).eps
        )
        relative = (
            torch.linalg.vector_norm(residuals, dim=1) / denominator
        ).detach().cpu().tolist()
        return [
            ReducedCandidate(
                weights=row,
                coefficients=coefficients[index],
                reduced_coordinates=coordinates[index],
                relative_residual=float(relative[index]),
            )
            for index, row in enumerate(weights)
        ]

    def full_residual(
        self, coefficients: torch.Tensor, weights: PenaltyWeights
    ) -> torch.Tensor:
        working = coefficients.to(
            device=self.operator.device, dtype=self.operator.dtype
        )
        applied = self.operator.normal(working) + self.penalty.normal(
            working, weights
        )
        return self.rhs.to(applied) - applied

    def full_residual_many(
        self,
        coefficients: torch.Tensor,
        weights: list[PenaltyWeights] | tuple[PenaltyWeights, ...],
    ) -> torch.Tensor:
        working = coefficients.to(
            device=self.operator.device, dtype=self.operator.dtype
        )
        applied = self.operator.normal_batch(working)
        applied = applied + self.penalty.normal_batch(working, weights)
        return self.rhs.to(applied)[None, :] - applied

    def scan(
        self, candidates: Iterable[PenaltyWeights]
    ) -> list[ReducedCandidate]:
        return self.solve_many(list(candidates))

    def worst_candidate(
        self, candidates: Iterable[PenaltyWeights]
    ) -> ReducedCandidate:
        rows = self.scan(candidates)
        return max(rows, key=lambda row: row.relative_residual)
