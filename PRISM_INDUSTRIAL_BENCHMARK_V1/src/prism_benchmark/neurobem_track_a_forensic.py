"""Read-only NeuroBEM Track-A forensic contracts.

This module reconstructs the rigid-body target exclusively from measured
physical signals.  Released residual columns are deliberately outside the
API.  It does not expose fitting, selection, or PRISM model code.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


FORENSIC_PROTOCOL_ID = "PRISM_V2_1_1_NEUROBEM_TRACK_A_FORENSIC_CLOSURE_R1"
FROZEN_MASS_KG = 0.772
FROZEN_INERTIA_KG_M2 = (0.0025, 0.0021, 0.0043)
RELATIVE_TOLERANCE = 0.01
ABSOLUTE_ROUNDING_TOLERANCE = 0.0005
AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass(frozen=True)
class GroundTruthContract:
    mass_kg: float = FROZEN_MASS_KG
    inertia_kg_m2: tuple[float, float, float] = FROZEN_INERTIA_KG_M2
    force_frame: str = "BODY_FRONT_LEFT_UP"
    torque_frame: str = "BODY_FRONT_LEFT_UP"
    acceleration_semantics: str = "MEASURED_BODY_LINEAR_ACCELERATION_INCLUDING_GRAVITY"
    angular_acceleration_semantics: str = "MEASURED_BODY_ANGULAR_ACCELERATION"
    quaternion_order: str = "SOURCE_QX_QY_QZ_QW_NOT_USED_BY_GT_FORMULA"
    timestamp_alignment: str = "SAME_ROW_NO_SHIFT"
    filtering: str = "OFFICIAL_PROCESSED_SIGNALS_AS_RELEASED_NO_ADDITIONAL_FILTER"


def reconstruct_force_torque_gt(
    frame: pd.DataFrame,
    contract: GroundTruthContract = GroundTruthContract(),
) -> np.ndarray:
    """Return [force, torque] without reading prediction or residual fields."""
    required = (
        "acc x", "acc y", "acc z", "ang acc x", "ang acc y", "ang acc z",
        "ang vel x", "ang vel y", "ang vel z",
    )
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"TRACK_A_PHYSICAL_SIGNAL_MISSING:{sorted(missing)}")
    mass = float(contract.mass_kg)
    inertia = np.asarray(contract.inertia_kg_m2, dtype=np.float64)
    if mass != FROZEN_MASS_KG or tuple(inertia) != FROZEN_INERTIA_KG_M2:
        raise ValueError("TRACK_A_FROZEN_RIGID_BODY_PARAMETER_MISMATCH")
    force = mass * frame.loc[:, ("acc x", "acc y", "acc z")].to_numpy(dtype=np.float64)
    alpha = frame.loc[:, ("ang acc x", "ang acc y", "ang acc z")].to_numpy(dtype=np.float64)
    omega = frame.loc[:, ("ang vel x", "ang vel y", "ang vel z")].to_numpy(dtype=np.float64)
    torque = alpha * inertia + np.cross(omega, omega * inertia)
    return np.column_stack((force, torque))


def axis_rmse(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    _check_pair(target, prediction)
    values = np.sqrt(np.mean(np.square(prediction - target), axis=0))
    return dict(zip(AXES, map(float, values), strict=True))


def rss21_metric(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """RSS 2021 axis-normalized grouped RMSE contract."""
    component_mse = _component_mse(target, prediction)
    return {
        **dict(zip(AXES, map(float, np.sqrt(component_mse)), strict=True)),
        "Fxy": float(np.sqrt(np.mean(component_mse[:2]))),
        "Fz": float(np.sqrt(component_mse[2])),
        "Mxy": float(np.sqrt(np.mean(component_mse[3:5]))),
        "Mz": float(np.sqrt(component_mse[5])),
        "F": float(np.sqrt(np.mean(component_mse[:3]))),
        "M": float(np.sqrt(np.mean(component_mse[3:]))),
    }


def neuromhe_metric(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """NeuroMHE Table-V vector-error grouped RMSE contract."""
    component_mse = _component_mse(target, prediction)
    return {
        **dict(zip(AXES, map(float, np.sqrt(component_mse)), strict=True)),
        "Fxy": float(np.sqrt(np.sum(component_mse[:2]))),
        "Fz": float(np.sqrt(component_mse[2])),
        "Mxy": float(np.sqrt(np.sum(component_mse[3:5]))),
        "Mz": float(np.sqrt(component_mse[5])),
        "F": float(np.sqrt(np.sum(component_mse[:3]))),
        "M": float(np.sqrt(np.sum(component_mse[3:]))),
    }


def reproduction_pass(recomputed: float, published: float) -> tuple[float, float, bool]:
    absolute = abs(float(recomputed) - float(published))
    relative = absolute / max(abs(float(published)), np.finfo(np.float64).eps)
    passed = relative <= RELATIVE_TOLERANCE or absolute <= ABSOLUTE_ROUNDING_TOLERANCE
    return absolute, relative, bool(passed)


def support_hash(frame: pd.DataFrame) -> str:
    values = frame.loc[:, ["t"]].to_numpy(dtype="<f8", copy=True)
    return sha256(values.tobytes()).hexdigest()


def manifest_identity(stem: str, published_mapping: Mapping[str, str]) -> dict[str, object]:
    label = published_mapping.get(stem)
    return {
        "released_segment": stem,
        "parent_flight": stem.rsplit("_seg_", 1)[0],
        "trajectory_label": label or "UNMAPPED",
        "NeuroMHE_match": label is not None,
        "RSS_match_if_known": False,
        "confidence": "EXACT_FILENAME_AND_TABLE_ORDER" if label else "UNRESOLVED",
        "evidence_source": "RCL-NUS/NeuroMHE Table V and official MATLAB oracle filename",
    }


def _check_pair(target: np.ndarray, prediction: np.ndarray) -> None:
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 6:
        raise ValueError("TRACK_A_FORENSIC_METRIC_SHAPE_MISMATCH")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("TRACK_A_FORENSIC_NONFINITE_VALUE")


def _component_mse(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    _check_pair(target, prediction)
    return np.mean(np.square(prediction - target), axis=0)


def assert_forensic_stage_has_no_training(source_paths: Sequence[Path]) -> None:
    forbidden = {"fit_route_contracts", "run_development", "optimizer", "fit"}
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
        hits = sorted(forbidden.intersection(calls))
        if hits:
            raise RuntimeError(f"FORENSIC_STAGE_TRAINING_FORBIDDEN:{path}:{hits}")
