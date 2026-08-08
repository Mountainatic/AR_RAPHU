from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cpu_data import sha256_file


PROTOCOL_ID = "PRISM_V2_2_METRO_P60_JOINT_PREDICTIVE_STABILITY_V1"
MODEL_VERSION = "PRISM_V2_2"
PARENT_COMMIT = "6ebcac898a75b6c1aa05c920a3a39847db052957"
DEVELOPMENT_ARTIFACT_COMMIT = "b5a4d672f65d0c5a01135f331d193a996e9c8c2d"
PACKAGE_DIRECTORY = "PRISM_V2_2_JOINT_PREDICTIVE_STABILITY_PACKAGE"
CONFIG_NAME = "PRISM_V2_2_JOINT_PREDICTIVE_STABILITY_CONFIG.json"
OUTPUT_DIRECTORY = "results_prism_v2_2_metro_p60_joint_stability"
THEORY_NAME = "PRISM_Theory_v2_2_Joint_Predictive_Stability_Extension_Theory_Only.md"
CHANNEL_COMPRESSED = "CHANNEL_COMPRESSED"
FULL_BASIS = "FULL_BASIS"
K_REPRESENTATIONS = (CHANNEL_COMPRESSED, FULL_BASIS)
ETA_PRED_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def config_path(project: Path) -> Path:
    return project / PACKAGE_DIRECTORY / CONFIG_NAME


def theory_path(project: Path) -> Path:
    return project / PACKAGE_DIRECTORY / "reference" / THEORY_NAME


def load_v22_config(project: Path) -> dict[str, Any]:
    path = config_path(project)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_id": PROTOCOL_ID,
        "model_version": MODEL_VERSION,
        "parent_commit": PARENT_COMMIT,
        "development_artifact_commit": DEVELOPMENT_ARTIFACT_COMMIT,
        "output_root": OUTPUT_DIRECTORY,
        "joint_routes": ["J_K", "J_KW", "J_KA", "J_KWA"],
        "k_representations": list(K_REPRESENTATIONS),
        "predictive_eta_grid": list(ETA_PRED_GRID),
        "predictive_block_ratio": 1.0,
        "reuse_m2_m4": True,
        "run_m2_m4": False,
        "test_access": False,
        "ood_access": False,
        "allow_ar_only": False,
        "allow_k_zero": False,
        "legacy_anchor_selection_eligible": False,
        "blas_threads": 1,
        "outer_view_workers": 2,
        "inner_candidate_workers": 12,
    }
    for key, required in expected.items():
        if value.get(key) != required:
            raise RuntimeError(f"PRISM v2.2 frozen config mismatch: {key}")
    value["config_sha256"] = sha256_file(path)
    return value
