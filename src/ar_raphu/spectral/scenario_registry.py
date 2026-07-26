"""Frozen model-class registry for spectral synthetic scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ar_raphu.synthetic import _normalized_gaussian, truth_response


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    name: str
    canonical_name: str
    model_class: str
    eligible_for_2d_capacity: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SCENARIO_REGISTRY = {
    "AR-S1": ScenarioSpec("AR-S1", "AR-S1", "rank1_2d_urysohn", True),
    "AR-S2": ScenarioSpec("AR-S2", "AR-S2", "rank1_2d_urysohn", True),
    "AR-S3": ScenarioSpec("AR-S3", "AR-S3", "rank2_2d_urysohn", True),
    "AR-S4": ScenarioSpec(
        "AR-S4", "AR-S4C", "conditional_urysohn_3d", False
    ),
    "AR-S4U": ScenarioSpec(
        "AR-S4U", "AR-S4U", "amplitude_dependent_2d_urysohn", True
    ),
    "AR-S7": ScenarioSpec("AR-S7", "AR-S7", "closed_loop_unclassified", False),
}


def scenario_spec(name: str) -> ScenarioSpec:
    try:
        return SCENARIO_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unregistered spectral scenario: {name}") from error


def s4c_two_lag_fragment(u0: float, u1: float, *, variable: int = 0) -> float:
    """Two-lag fragment exposing the shared-current-amplitude interaction."""

    center = 8.0 + 12.0 / (1.0 + np.exp(-2.0 * u0))
    weights = _normalized_gaussian(64, center, 2.0)
    return float(
        weights[15] * truth_response(variable, np.array([u0]))[0]
        + weights[16] * truth_response(variable, np.array([u1]))[0]
    )


def s4c_mixed_difference(
    u0: float = 0.3,
    u1: float = -0.4,
    *,
    delta: float = 0.2,
    variable: int = 0,
) -> float:
    function = lambda a, b: s4c_two_lag_fragment(a, b, variable=variable)
    return float(
        function(u0 + delta, u1 + delta)
        - function(u0 + delta, u1)
        - function(u0, u1 + delta)
        + function(u0, u1)
    )
