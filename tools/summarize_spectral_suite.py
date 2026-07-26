#!/usr/bin/env python3
"""Generate pre-registered Spectral v0.3 through v0.3.4 decision fields."""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "spectral_v03"


def status(experiment: str) -> str:
    path = RESULT_ROOT / experiment / "summary.json"
    if not path.exists():
        return "NOT_YET_RUN"
    return str(json.loads(path.read_text(encoding="utf-8"))["status"])


def summarize_v03() -> None:
    statuses = {f"E{index}": status(f"E{index}") for index in range(9)}
    e0_pass = statuses["E0"] == "E0_COMPONENT_IDENTITY_PASS"
    e1_pass = statuses["E1"] == "E1_PROJECTION_CAPACITY_PASS"
    if not e0_pass:
        next_stage = "STOP_E0_GENERATOR_REPLAY"
    elif not e1_pass:
        next_stage = "STOP_E1_PROJECTION_CAPACITY"
    else:
        next_stage = "E2_DEVELOPMENT"
    fields = {
        **{f"E{index}_STATUS": statuses[f"E{index}"] for index in range(9)},
        "FULL_KERNEL_CAPACITY": "NOT_YET_RUN",
        "DOUBLE_RESIDUALIZATION_VALID": "NOT_YET_RUN",
        "SUPPORT_RECOVERY_VALID": "NOT_YET_RUN",
        "RANK_ADAPTATION_VALID": "NOT_YET_RUN",
        "ADAPTIVE_WEIGHTING_ADOPTED": "NOT_YET_RUN",
        "PREDICTION_RECOMBINATION_VALID": "NOT_YET_RUN",
        "RECURSIVE_DEPLOYMENT_VALID": "NOT_YET_RUN",
        "NEXT_ALLOWED_STAGE": next_stage,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    decision = "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"
    (RESULT_ROOT / "DEVELOPMENT_DECISION.md").write_text(
        decision, encoding="utf-8"
    )
    with (RESULT_ROOT / "spectral_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        writer.writerows(fields.items())


def summarize_v031() -> None:
    result_root = ROOT / "results" / "spectral_v031"

    def v031_status(experiment: str) -> str:
        path = result_root / experiment / "summary.json"
        if not path.exists():
            return "NOT_YET_RUN"
        return str(json.loads(path.read_text(encoding="utf-8"))["status"])

    e1r = v031_status("E1R")
    e2a = v031_status("E2A")
    e2b = v031_status("E2B")
    e3 = v031_status("E3")
    if e1r != "E1R_REPRESENTATION_CERTIFIED_32x16":
        next_stage = "STOP_REPRESENTATION"
    elif e2a != "E2A_SINGLE_KERNEL_CAPACITY_PASS":
        next_stage = "STOP_SINGLE_KERNEL_CAPACITY"
    elif e2b != "E2B_JOINT_EXTERNAL_CAPACITY_PASS":
        next_stage = "STOP_JOINT_IDENTIFIABILITY"
    elif e3 != "E3_DOUBLE_RESIDUALIZATION_PASS":
        next_stage = "STOP_NUISANCE_ORTHOGONALIZATION"
    else:
        next_stage = "ALLOW_E4_SUPPORT_VALIDATION"
    fields = {
        "E0_STATUS": "REUSED_E0_COMPONENT_IDENTITY_PASS_FROM_V03",
        "OLD_E1_STATUS": "E1_COMPRESSED_LAG_BASIS_UNDERSPECIFIED",
        "E1R_STATUS": e1r,
        "E2A_STATUS": e2a,
        "E2B_STATUS": e2b,
        "E3_STATUS": e3,
        "REPRESENTATION_CERTIFIED": (
            "TRUE"
            if e1r == "E1R_REPRESENTATION_CERTIFIED_32x16"
            else "FALSE"
        ),
        "SINGLE_KERNEL_CAPACITY_VALID": (
            "TRUE" if e2a == "E2A_SINGLE_KERNEL_CAPACITY_PASS" else "FALSE"
        ),
        "JOINT_EXTERNAL_CAPACITY_VALID": (
            "NOT_YET_RUN"
            if e2b == "NOT_YET_RUN"
            else (
                "TRUE"
                if e2b == "E2B_JOINT_EXTERNAL_CAPACITY_PASS"
                else "FALSE"
            )
        ),
        "DOUBLE_RESIDUALIZATION_VALID": (
            "NOT_YET_RUN"
            if e3 == "NOT_YET_RUN"
            else (
                "TRUE"
                if e3 == "E3_DOUBLE_RESIDUALIZATION_PASS"
                else "FALSE"
            )
        ),
        "NEXT_ALLOWED_STAGE": next_stage,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    decision = "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"
    (result_root / "V031_CORE_DECISION.md").write_text(
        decision, encoding="utf-8"
    )
    with (result_root / "spectral_v031_core_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        writer.writerows(fields.items())


def summarize_v032() -> None:
    result_root = ROOT / "results" / "spectral_v032"

    def experiment_status(experiment: str) -> str:
        path = result_root / experiment / "summary.json"
        if not path.exists():
            return "NOT_YET_RUN"
        return str(json.loads(path.read_text(encoding="utf-8"))["status"])

    r1 = experiment_status("R1")
    e1a = experiment_status("E1A")
    e1a_payload = (
        json.loads(
            (result_root / "E1A" / "summary.json").read_text(encoding="utf-8")
        )
        if (result_root / "E1A" / "summary.json").exists()
        else {}
    )
    e0u = str(e1a_payload.get("e0u_status", "NOT_YET_RUN"))
    e2a0 = experiment_status("E2A0")
    natural = experiment_status("E2A_NAT")
    permuted = experiment_status("E2A_PERM")
    space = experiment_status("E2A_SPACE")
    if r1 != "R1_DOMAIN_AND_MODEL_CLASS_AUDIT_PASS":
        primary_limitation = "SCENARIO_OR_DOMAIN_AUDIT"
        next_stage = "STOP_SCENARIO_OR_DOMAIN_AUDIT"
    elif e1a != "E1A_DOMAIN_SAFE_REPRESENTATION_PASS":
        primary_limitation = "REPRESENTATION"
        next_stage = "STOP_REPRESENTATION"
    elif e2a0 != "E2A0_IMPLEMENTATION_CONSISTENCY_PASS":
        primary_limitation = "IMPLEMENTATION_CONSISTENCY"
        next_stage = "STOP_IMPLEMENTATION_CONSISTENCY"
    elif space not in {
        "E2A_SPACE_FULL_SURFACE_CAPACITY_PASS",
        "E2A_SPACE_CAPACITY_PASS",
    }:
        primary_limitation = "FULL_KERNEL_ESTIMATOR_OR_BASIS"
        next_stage = "STOP_CAPACITY"
    else:
        primary_limitation = "NONE"
        next_stage = "ALLOW_E2B"
    fields = {
        "V031_FROZEN_STATUS": "STOP_SINGLE_KERNEL_CAPACITY",
        "R1_STATUS": r1,
        "E0U_STATUS": e0u,
        "E1A_STATUS": e1a,
        "E2A0_STATUS": e2a0,
        "E2A_NAT_STATUS": natural,
        "E2A_PERM_STATUS": permuted,
        "E2A_SPACE_STATUS": space,
        "AMPLITUDE_DOMAIN_VALID": (
            "TRUE"
            if r1 == "R1_DOMAIN_AND_MODEL_CLASS_AUDIT_PASS"
            else "FALSE"
        ),
        "MODEL_CLASS_REGISTRY_VALID": (
            "TRUE"
            if r1 == "R1_DOMAIN_AND_MODEL_CLASS_AUDIT_PASS"
            else "FALSE"
        ),
        "REPRESENTATION_VALID": (
            "TRUE" if e1a == "E1A_DOMAIN_SAFE_REPRESENTATION_PASS" else "FALSE"
        ),
        "IMPLEMENTATION_CONSISTENCY_VALID": (
            "NOT_YET_RUN"
            if e2a0 == "NOT_YET_RUN"
            else (
                "TRUE"
                if e2a0 == "E2A0_IMPLEMENTATION_CONSISTENCY_PASS"
                else "FALSE"
            )
        ),
        "NATURAL_PREDICTIVE_CAPACITY": (
            "NOT_YET_RUN" if natural == "NOT_YET_RUN" else natural
        ),
        "DECORRELATED_CAPACITY": (
            "NOT_YET_RUN" if permuted == "NOT_YET_RUN" else permuted
        ),
        "SPACE_FILLING_SURFACE_CAPACITY": (
            "NOT_YET_RUN" if space == "NOT_YET_RUN" else space
        ),
        "PRIMARY_LIMITATION": primary_limitation,
        "NEXT_ALLOWED_STAGE": next_stage,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "V032_CAPACITY_DECISION.md").write_text(
        "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n",
        encoding="utf-8",
    )
    with (result_root / "spectral_v032_capacity_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        writer.writerows(fields.items())


def summarize_v033() -> None:
    result_root = ROOT / "results" / "spectral_v033"

    def experiment_status(experiment: str) -> str:
        path = result_root / experiment / "summary.json"
        if not path.exists():
            return "NOT_YET_RUN"
        return str(json.loads(path.read_text(encoding="utf-8"))["status"])

    e1b = experiment_status("E1B")
    e2a0 = experiment_status("E2A0")
    mother = experiment_status("E2A_M_SPACE")
    structural = experiment_status("E2A_S_SPACE")
    natural = experiment_status("E2A_P_NAT")
    permuted = experiment_status("E2A_P_PERM")
    if e1b != "E1B_RESOLUTION_ROLES_CERTIFIED":
        primary_limitation = "REPRESENTATION_CERTIFICATE"
        next_stage = "STOP_REPRESENTATION_CERTIFICATE"
    elif e2a0 != "E2A0_IMPLEMENTATION_CLOSURE_PASS":
        primary_limitation = "IMPLEMENTATION_CLOSURE"
        next_stage = "STOP_IMPLEMENTATION_CLOSURE"
    elif mother != "E2A_M_SPACE_CAPACITY_PASS":
        primary_limitation = "MOTHER_SPACE_CAPACITY"
        next_stage = "STOP_MOTHER_SPACE_CAPACITY"
    elif structural != "E2A_S_SPACE_CAPACITY_PASS":
        primary_limitation = "STRUCTURAL_SPACE_CAPACITY"
        next_stage = "STOP_STRUCTURAL_SPACE_CAPACITY"
    elif natural == "E2A_P_NAT_CAPACITY_PASS":
        primary_limitation = "NONE"
        next_stage = "ALLOW_E2B"
    elif permuted == "E2A_P_PERM_CAPACITY_PASS":
        primary_limitation = "NATURAL_LAG_CORRELATION_LIMIT"
        next_stage = "ALLOW_E2B_WITH_NATURAL_CORRELATION_QUALIFIER"
    else:
        primary_limitation = "NATURAL_DISTRIBUTION_COVERAGE_LIMIT"
        next_stage = "ALLOW_E2B_STRUCTURE_ONLY_WITH_DISTRIBUTION_QUALIFIER"
    certificate_path = result_root / "E1B" / "role_certificate.json"
    certificate = (
        json.loads(certificate_path.read_text(encoding="utf-8"))
        if certificate_path.exists()
        else {}
    )
    roles = certificate.get("roles", {})
    fields = {
        "V032_FROZEN_STATUS": "STOP_REPRESENTATION",
        "E1B_STATUS": e1b,
        "E2A0_STATUS": e2a0,
        "E2A_M_SPACE_STATUS": mother,
        "E2A_S_SPACE_STATUS": structural,
        "E2A_P_NAT_STATUS": natural,
        "E2A_P_PERM_STATUS": permuted,
        "MOTHER_REPRESENTATION_VALID": (
            "TRUE" if roles.get("MOTHER", {}).get("passed") else "FALSE"
        ),
        "STRUCTURAL_REPRESENTATION_VALID": (
            "TRUE" if roles.get("STRUCTURAL", {}).get("passed") else "FALSE"
        ),
        "PREDICTIVE_REPRESENTATION_VALID": (
            "TRUE" if roles.get("PREDICTIVE", {}).get("passed") else "FALSE"
        ),
        "STRONG_RANK_RESOLUTION_VALID": (
            "TRUE"
            if roles.get("STRUCTURAL", {}).get(
                "strong_rank_resolution_valid"
            )
            else "FALSE"
        ),
        "IMPLEMENTATION_CLOSURE_VALID": (
            "TRUE" if e2a0 == "E2A0_IMPLEMENTATION_CLOSURE_PASS" else "FALSE"
        ),
        "MOTHER_CAPACITY_VALID": (
            "TRUE" if mother == "E2A_M_SPACE_CAPACITY_PASS" else "FALSE"
        ),
        "STRUCTURAL_CAPACITY_VALID": (
            "TRUE" if structural == "E2A_S_SPACE_CAPACITY_PASS" else "FALSE"
        ),
        "NATURAL_PREDICTIVE_CAPACITY_VALID": (
            "NOT_YET_RUN"
            if natural == "NOT_YET_RUN"
            else (
                "TRUE" if natural == "E2A_P_NAT_CAPACITY_PASS" else "FALSE"
            )
        ),
        "PRIMARY_LIMITATION": primary_limitation,
        "NEXT_ALLOWED_STAGE": next_stage,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "V033_RESOLUTION_CAPACITY_DECISION.md").write_text(
        "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n",
        encoding="utf-8",
    )
    with (
        result_root / "spectral_v033_resolution_capacity_summary.csv"
    ).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        writer.writerows(fields.items())


def summarize_v034() -> None:
    result_root = ROOT / "results" / "spectral_v034"

    def experiment_status(experiment: str) -> str:
        path = result_root / experiment / "summary.json"
        if not path.exists():
            return "NOT_YET_RUN"
        return str(json.loads(path.read_text(encoding="utf-8"))["status"])

    r0 = experiment_status("R0")
    structural = experiment_status("E2A_SR")
    bootstrap = experiment_status("E2A_SRB")
    natural = experiment_status("E2A_P_NAT")
    permuted = experiment_status("E2A_P_PERM")

    r0_pass = r0 == "R0_V033_SCIENTIFIC_REINTERPRETATION_PASS"
    structural_pass = structural == "E2A_SR_RANK_PROFILE_PASS"
    bootstrap_pass = bootstrap == "E2A_SRB_RANK_INTERVAL_PASS"
    natural_pass = natural == "E2A_P_NAT_PREDICTIVE_RANK_PASS"
    permuted_pass = permuted == "E2A_P_PERM_PREDICTIVE_RANK_PASS"

    if not r0_pass:
        next_stage = "STOP_V033_REINTERPRETATION"
        primary_finding = "V033_REINTERPRETATION_NOT_VALIDATED"
    elif not structural_pass:
        next_stage = "STOP_RANK_PROFILE"
        primary_finding = "STRUCTURAL_RANK_PROFILE_NOT_VALIDATED"
    elif not bootstrap_pass:
        next_stage = "STOP_RANK_UNCERTAINTY"
        primary_finding = "STRUCTURAL_RANK_UNCERTAINTY_NOT_VALIDATED"
    elif not natural_pass:
        next_stage = "STOP_PREDICTIVE_CAPACITY"
        primary_finding = "NATURAL_PREDICTIVE_CAPACITY_NOT_VALIDATED"
    else:
        next_stage = "ALLOW_E2B"
        primary_finding = "ADAPTIVE_RANK_PROFILE_VALIDATED"

    profile_value = "TRUE" if structural_pass else "FALSE"
    fields = {
        "V033_FROZEN_STATUS": "STOP_STRUCTURAL_SPACE_CAPACITY",
        "R0_REINTERPRETATION_STATUS": r0,
        "E2A_SR_STATUS": structural,
        "E2A_SRB_STATUS": bootstrap,
        "E2A_P_NAT_STATUS": natural,
        "E2A_P_PERM_STATUS": permuted,
        "FULL_STRUCTURAL_SURFACE_CAPACITY": (
            "PASS" if r0_pass else "NOT_VALIDATED"
        ),
        "NEAR_RANK1_RECOVERY": profile_value,
        "STRONG_RANK2_RECOVERY": profile_value,
        "WEAK_RANK2_PROFILE_RECOVERY": profile_value,
        "HIGHER_RANK_PROFILE_RECOVERY": profile_value,
        "BOOTSTRAP_RANK_STABILITY": (
            "TRUE" if bootstrap_pass else "FALSE"
        ),
        "NATURAL_PREDICTIVE_CAPACITY": (
            "TRUE" if natural_pass else "FALSE"
        ),
        "PREDICTIVE_RANK_PROFILE_VALID": (
            "TRUE" if natural_pass and permuted_pass else "FALSE"
        ),
        "UNIVERSAL_RANK2_HYPOTHESIS": "REJECTED",
        "PRIMARY_FINDING": primary_finding,
        "NEXT_ALLOWED_STAGE": next_stage,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "V034_RANK_PROFILE_DECISION.md").write_text(
        "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n",
        encoding="utf-8",
    )
    with (
        result_root / "spectral_v034_rank_profile_summary.csv"
    ).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        writer.writerows(fields.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "spectral_v03.yaml",
    )
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schema_version = int(config.get("schema_version", 1))
    if schema_version == 5:
        summarize_v034()
    elif schema_version == 4:
        summarize_v033()
    elif schema_version == 3:
        summarize_v032()
    elif schema_version == 2:
        summarize_v031()
    else:
        summarize_v03()


if __name__ == "__main__":
    main()
