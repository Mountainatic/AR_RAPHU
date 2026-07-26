"""Frozen predictive, structural, and mother resolution roles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolutionRole:
    name: str
    lag_type: str
    lag_count: int
    amplitude_count: int


FROZEN_RESOLUTION_ROLES = {
    "PREDICTIVE": ResolutionRole("PREDICTIVE", "cubic_bspline", 32, 28),
    "STRUCTURAL": ResolutionRole("STRUCTURAL", "cubic_bspline", 48, 28),
    "MOTHER": ResolutionRole("MOTHER", "discrete_identity", 64, 28),
}


def role_from_config(config: dict[str, object], name: str) -> ResolutionRole:
    """Read a role and reject any drift from the v0.3.3 pre-registration."""

    key = name.upper()
    if key not in FROZEN_RESOLUTION_ROLES:
        raise ValueError(f"Unknown resolution role: {name}.")
    section_name = {
        "PREDICTIVE": "predictive_role",
        "STRUCTURAL": "structural_role",
        "MOTHER": "mother_role",
    }[key]
    section = config["e1b"][section_name]
    observed = ResolutionRole(
        key,
        str(section["lag_type"]),
        int(section["lag_count"]),
        int(section["amplitude_count"]),
    )
    if observed != FROZEN_RESOLUTION_ROLES[key]:
        raise ValueError(f"{key} resolution cannot be overridden.")
    return observed
