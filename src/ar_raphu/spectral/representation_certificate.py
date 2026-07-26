"""Aggregate the frozen v0.3.3 E1B resolution-role certificate."""

from __future__ import annotations

from collections.abc import Iterable


def certify_resolution_roles(
    rows: Iterable[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    materialized = list(rows)
    role_specs = {
        "MOTHER": ("mother_role", "discrete_identity"),
        "STRUCTURAL": ("structural_role", "cubic_bspline"),
        "PREDICTIVE": ("predictive_role", "cubic_bspline"),
    }
    certificates: dict[str, dict[str, object]] = {}
    for role, (section_name, lag_type) in role_specs.items():
        section = config["e1b"][section_name]
        selected = [
            row
            for row in materialized
            if row["lag_basis_type"] == lag_type
            and int(row["lag_basis_count"]) == int(section["lag_count"])
            and int(row["amplitude_basis_count"])
            == int(section["amplitude_count"])
        ]
        if not selected:
            raise ValueError(f"No E1B rows found for {role}.")
        core = [row for row in selected if row["domain"] == "core"]
        fit = [row for row in selected if row["domain"] == "fit"]
        passed = (
            max(float(row["epsilon_joint"]) for row in core)
            <= float(section["max_core_joint_nrmse"])
            and max(float(row["epsilon_joint"]) for row in fit)
            <= float(section["max_fit_joint_nrmse"])
        )
        if role != "MOTHER":
            passed &= (
                max(float(row["epsilon_lag_given_amplitude"]) for row in core)
                <= float(section["max_core_lag_excess"])
                and max(
                    float(row["epsilon_lag_given_amplitude"]) for row in fit
                )
                <= float(section["max_fit_lag_excess"])
            )
        strong_rows = [
            row for row in selected if row["truth_rank_class"] == "strong_rank2"
        ]
        strong_rank_valid = True
        if role == "STRUCTURAL" and strong_rows:
            strong = config["e1b"]["strong_rank2"]
            strong_rank_valid = all(
                float(row["operator_error_over_sigma2"])
                <= float(strong["representation_operator_error_over_sigma2_max"])
                and float(row["operator_error_over_gap"])
                <= float(strong["representation_operator_error_over_gap_max"])
                for row in strong_rows
            )
            passed &= strong_rank_valid
        certificates[role] = {
            "passed": bool(passed),
            "row_count": len(selected),
            "worst_core_joint_nrmse": max(
                float(row["epsilon_joint"]) for row in core
            ),
            "worst_fit_joint_nrmse": max(
                float(row["epsilon_joint"]) for row in fit
            ),
            "strong_rank_resolution_valid": bool(strong_rank_valid),
        }
    if not certificates["MOTHER"]["passed"]:
        status = "E1B_MOTHER_AMPLITUDE_BASIS_FAIL"
    elif not certificates["STRUCTURAL"]["passed"]:
        status = "E1B_STRUCTURAL_RESOLUTION_FAIL"
    elif not certificates["PREDICTIVE"]["passed"]:
        status = "E1B_PREDICTIVE_COMPRESSION_FAIL"
    else:
        status = "E1B_RESOLUTION_ROLES_CERTIFIED"
    return {
        "status": status,
        "roles": certificates,
        "next_allowed_experiment": "E2A0" if status.endswith("CERTIFIED") else "STOP",
    }
