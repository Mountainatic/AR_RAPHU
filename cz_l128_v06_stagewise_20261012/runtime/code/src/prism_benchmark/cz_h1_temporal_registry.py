"""Versioned CZ one-step K contract. No data or model fitting is needed.

The frozen raw-2s data/target contract is unchanged. This companion contract
maps its legacy h=0 alias and explicit h=1 to one temporal representation.
Historical protocols remain available for exact registry replay.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL = "cz_h1_k_temporal_registry_v2"
PROTOCOL_ID = "CZ_H1_K_TEMPORAL_REGISTRY_V2"
FROZEN = PROTOCOL_ID + "_FROZEN"
READY = PROTOCOL_ID + "_READY_FOR_CANONICAL_TRAINING"
SEMANTICS = "CZ_ONE_STEP_MULTISCALE_V2"
AUTHORITY_COMMIT = "2ee6273b8f915cbcdff2f46d56bc80047ddae4a7"
CONFIG_ROOT = Path("configs/cz_h1_k_temporal_registry_v2")
REGISTRY_NAME = PROTOCOL_ID + ".json"
SEAL_NAME = PROTOCOL_ID + "_SEAL.json"
DELTAS = (2, 4, 8, 16)
HISTORIES = (8, 16, 32, 64, 128, 256)
M_TAU = (4, 8, 12)
CHANNELS = ("main_heater_power", "joint_lift", "crystal_rotation", "crucible_rotation")
TASKS = ("CZ_DIAM_RAW2S_CURRENT_L256", "CZ_DIAM_RAW2S_CURRENT_L256_H1")
CHECKS = ("registry_exact", "short_resolution_preserved", "causal_intervals",
          "candidate_native_support", "h0_h1_semantic_invariance", "h4_unchanged",
          "no_test_access", "numerical_core_unchanged", "protected_authority_unchanged",
          "both_direction_support", "unit_tests")


def digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path, *, canonical_lf: bool = False) -> str:
    raw = path.read_bytes()
    if canonical_lf:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def task_contract(head) -> dict[str, Any] | None:
    """Recognize only the registered raw-2s one-step target family."""
    if (head.dataset == "cz_czochralski" and head.task_id in TASKS
            and head.target == "crystal_diameter" and head.cadence_seconds == 2.0
            and head.h_steps in (0, 1) and head.w_steps == head.w0_steps == 1):
        return {"effective_forecast_steps": 1, "k_profile_semantics": SEMANTICS}
    return None


def validate_registry(registry: dict[str, Any], v2: dict[str, Any]) -> None:
    exact = {
        "protocol_id": PROTOCOL_ID, "authority_source_commit": AUTHORITY_COMMIT,
        "sampling_seconds": 2, "task_semantics": "explicit_H1_one_step",
        "effective_forecast_steps": 1, "k_profile_semantics": SEMANTICS,
        "delta_steps": list(DELTAS), "history_steps": list(HISTORIES),
        "history_seconds": [2*h for h in HISTORIES], "m_tau": list(M_TAU),
        "lag_basis_family": "AUTHORITY_CAUSAL_GEOMETRIC_GROWTH_2",
        "amplitude_basis": "unchanged", "urysohn_family": "unchanged",
        "native_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        "routing_rule": "unchanged", "test_used_for_design": False,
        "residual_whitening_used_for_candidate_selection": False,
        "change_scope": "K_TEMPORAL_REPRESENTATION_REGISTRY_ONLY",
        "near_field_seconds": 20, "near_field_max_bin_seconds": 4,
        "registration_order": "history_ascending_delta_descending",
        "channels": list(CHANNELS),
    }
    for key, value in exact.items():
        if registry.get(key) != value:
            raise RuntimeError(f"CZ H1 registry changed: {key}")
    if v2["time_profile_grid"]["class_delta_ratio_steps"]["MEDIUM"] != list(DELTAS):
        raise RuntimeError("inherited MEDIUM delta grid changed")


def attach_registry(v2: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    validate_registry(registry, v2)
    config = copy.deepcopy(v2)
    config["cz_h1_temporal_registry"] = copy.deepcopy(registry)
    config["cz_h1_temporal_registry_digest"] = digest(registry)
    return config


def profiles_for_contract(view, channel: str, config: dict[str, Any]):
    registry = config.get("cz_h1_temporal_registry")
    if registry is None:
        return None
    # Detect mutations on every registry access, including in worker processes.
    if digest(registry) != config.get("cz_h1_temporal_registry_digest"):
        raise RuntimeError("CZ H1 registry mutated after loading")
    validate_registry(registry, config)
    if task_contract(view.head) is None:
        return None
    if channel not in CHANNELS:
        raise ValueError(f"unregistered CZ H1 K channel: {channel}")
    return sorted(((d, h) for d in DELTAS for h in HISTORIES if d <= h),
                  key=lambda pair: (pair[1], -pair[0]))


def verify_bindings(project: Path, bindings: dict[str, str]) -> None:
    for relative, expected in bindings.items():
        path = project / relative
        if not path.is_file() or file_sha(path, canonical_lf=True) != expected:
            raise RuntimeError(f"CZ registry bound source changed: {relative}")


def load_frozen_registry(project: Path) -> dict[str, Any]:
    root = project / CONFIG_ROOT
    registry_path = root / REGISTRY_NAME
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    seal = json.loads((root / SEAL_NAME).read_text(encoding="utf-8"))
    if registry.get("status") != FROZEN or seal.get("status") != READY:
        raise RuntimeError("CZ H1 temporal registry has not passed preflight and frozen")
    if file_sha(registry_path) != seal["registry_sha256"]:
        raise RuntimeError("CZ H1 frozen registry SHA256 mismatch")
    preflight_path = root / "CZ_H1_K_REGISTRY_PREFLIGHT.json"
    if file_sha(preflight_path) != seal["preflight_sha256"]:
        raise RuntimeError("CZ H1 preflight proof SHA256 mismatch")
    proof = json.loads(preflight_path.read_text(encoding="utf-8"))
    if any(proof["checks"].get(check) != "PASS" for check in CHECKS):
        raise RuntimeError("CZ H1 preflight has a failed or missing check")
    if proof["registry_sha256"] != seal["registry_sha256"]:
        raise RuntimeError("CZ H1 preflight bound to another registry")
    for relative, sha in seal["artifact_sha256"].items():
        if file_sha(root / relative) != sha:
            raise RuntimeError(f"CZ H1 preflight artifact changed: {relative}")
    verify_bindings(project, seal["runtime_source_bindings"])
    inventory = json.loads((root / "AUTHORITY_UNCHANGED_BINDINGS.json").read_text(encoding="utf-8"))
    verify_bindings(project, inventory["files"])
    return registry


def load_frozen_configs(project: Path):
    """New protocol opt-in; inherited selection, solver and routing unchanged."""
    from .v21_config import load_v21_config
    from .v2_config import load_frozen_config

    registry = load_frozen_registry(project)
    patch = inherited_algorithm_config(project)
    v21 = load_v21_config(project)
    if v21["K_C"]["m_tau"] != list(M_TAU):
        raise RuntimeError("inherited frozen m_tau grid changed")
    v2 = attach_registry(load_frozen_config(project), registry)
    # Only provenance is added to the inherited patch. All algorithm values stay.
    patch["cz_h1_k_temporal_registry_protocol"] = PROTOCOL_ID
    patch["cz_h1_k_temporal_registry_sha256"] = file_sha(project / CONFIG_ROOT / REGISTRY_NAME)
    return patch, v21, v2


def inherited_algorithm_config(project: Path):
    """Same Stage-1 parameter merge, with publication-HEAD source bindings.

The historical descriptor's code hashes predate the publication HEAD. Its
algorithm/config values stay immutable; the new seal verifies the current
authority inventory instead of rewriting the historical descriptor.
"""
    from .v211_metro_config import load_metro_config
    from .v211_representative_stage1_config import (
        CONFIG_RELATIVE_PATH, PROTOCOL_ID as INHERITED_ID, EVIDENCE_CLASS,
        BASE_COMMIT, EXECUTION_BRANCH, ACTIVE_DATASETS, PRIMARY_TASKS,
        PUBLIC_PRIMARY_TASKS, RESERVED_DATASETS,
    )
    inventory = json.loads((project / CONFIG_ROOT / "AUTHORITY_UNCHANGED_BINDINGS.json").read_text(encoding="utf-8"))
    verify_bindings(project, inventory["files"])
    descriptor = json.loads((project / CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))
    inherited = load_metro_config(project)
    merged = dict(inherited)
    merged.update({
        "protocol_id": INHERITED_ID, "evidence_class": EVIDENCE_CLASS,
        "source_commit": BASE_COMMIT, "execution_branch": EXECUTION_BRANCH,
        "support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
        "active_datasets": list(ACTIVE_DATASETS), "primary_tasks": sorted(PRIMARY_TASKS),
        "public_primary_tasks": sorted(PUBLIC_PRIMARY_TASKS), "reserved_datasets": dict(RESERVED_DATASETS),
        "representative_stage1_config_sha256": file_sha(project / CONFIG_RELATIVE_PATH),
    })
    merged["resource"] = {**inherited["resource"],
        "workers": int(descriptor["resource"]["outer_workers"]), "blas_threads": 1,
        "prediction_chunk_rows": int(descriptor["resource"]["prediction_chunk_rows"])}
    return merged
