"""Literature-aligned NeuroBEM force/torque and rollout contracts.

This module is isolated from the completed NeuroBEM runners.  It implements
only the frozen dual-benchmark transformations, metrics, and small FP64 ridge
routes.  Published scores never enter any fit or selection function.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .neurobem_linear import RidgeContract, fit_numerical_ridge, predict_ridge, restricted_cubic_basis
from .neurobem_experiment import ridge_from_json, ridge_to_json


TRACK_A_COLUMNS = (
    "t", "ang acc x", "ang acc y", "ang acc z", "ang vel x", "ang vel y", "ang vel z",
    "quat x", "quat y", "quat z", "quat w", "acc x", "acc y", "acc z",
    "vel x", "vel y", "vel z", "pos x", "pos y", "pos z",
    "mot 1", "mot 2", "mot 3", "mot 4", "dmot 1", "dmot 2", "dmot 3", "dmot 4", "vbat",
)
PREDICTION_EXTRA_COLUMNS = (
    "predicted_fx", "predicted_fy", "predicted_fz", "predicted_tx", "predicted_ty", "predicted_tz",
    "error_residual_fx", "error_residual_fy", "error_residual_fz",
    "error_residual_tx", "error_residual_ty", "error_residual_tz",
)
TRACK_B_STATE_COLUMNS = (
    "vel x", "vel y", "vel z", "quat w", "quat x", "quat y", "quat z",
    "ang vel x", "ang vel y", "ang vel z",
)
MOTOR_COLUMNS = ("mot 1", "mot 2", "mot 3", "mot 4")
CANONICAL_W_CANDIDATES = ("IDENTITY_CORRECTION", "NATURAL_CUBIC_LATENT", "SIGNED_QUADRATIC_LATENT")
FORMAL_ROUTE_IDS = ("PF_KC", "PF_KCW", "J_KC", "J_KCW")
K_CHANNEL_REGISTRY = {
    "motor_actuator": {"columns": list(MOTOR_COLUMNS), "interpretation": "ACTUATOR_PHYSICS_CONSISTENCY"},
    "linear_velocity_context": {"columns": ["vel x", "vel y", "vel z"], "interpretation": "PREDICTIVE_MOTION_CONTEXT"},
    "attitude_context": {"columns": ["quat w", "quat x", "quat y", "quat z"], "interpretation": "PREDICTIVE_MOTION_CONTEXT"},
    "body_rate_context": {"columns": ["ang vel x", "ang vel y", "ang vel z"], "interpretation": "PREDICTIVE_MOTION_CONTEXT"},
}


@dataclass(frozen=True)
class LiteratureTrajectory:
    trajectory_id: str
    partition: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class LatentWContract:
    family: str
    ridge: RidgeContract | None
    knots: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class LiteratureRouteContract:
    route_id: str
    target_kind: str
    history: int
    k_ridge: RidgeContract | None
    c_ridge: RidgeContract | None
    joint_ridge: RidgeContract | None
    w_contract: LatentWContract
    column_slices: Mapping[str, tuple[int, int]]


@dataclass(frozen=True)
class DecoupledLiteratureRouteContract:
    route_id: str
    velocity_contract: LiteratureRouteContract
    attitude_contract: LiteratureRouteContract


def route_contract_to_json(contract: LiteratureRouteContract | DecoupledLiteratureRouteContract) -> dict[str, object]:
    if isinstance(contract, DecoupledLiteratureRouteContract):
        return {
            "contract_type": "DECOUPLED_VELOCITY_ATTITUDE",
            "route_id": contract.route_id,
            "velocity_contract": route_contract_to_json(contract.velocity_contract),
            "attitude_contract": route_contract_to_json(contract.attitude_contract),
            "a_enabled": False,
        }
    return {
        "contract_type": "SINGLE_TARGET_BLOCK",
        "route_id": contract.route_id,
        "target_kind": contract.target_kind,
        "history": contract.history,
        "k_ridge": None if contract.k_ridge is None else ridge_to_json(contract.k_ridge),
        "c_ridge": None if contract.c_ridge is None else ridge_to_json(contract.c_ridge),
        "joint_ridge": None if contract.joint_ridge is None else ridge_to_json(contract.joint_ridge),
        "w_contract": {
            "family": contract.w_contract.family,
            "ridge": None if contract.w_contract.ridge is None else ridge_to_json(contract.w_contract.ridge),
            "knots": [value.tolist() for value in contract.w_contract.knots],
            "input_source": "FROZEN_C_LATENT_ONLY",
        },
        "column_slices": {key: list(value) for key, value in contract.column_slices.items()},
        "a_enabled": False,
    }


def route_contract_from_json(value: Mapping[str, object]) -> LiteratureRouteContract | DecoupledLiteratureRouteContract:
    if value.get("contract_type") == "DECOUPLED_VELOCITY_ATTITUDE":
        velocity = value["velocity_contract"]
        attitude = value["attitude_contract"]
        assert isinstance(velocity, Mapping) and isinstance(attitude, Mapping)
        return DecoupledLiteratureRouteContract(
            route_id=str(value["route_id"]),
            velocity_contract=route_contract_from_json(velocity),  # type: ignore[arg-type]
            attitude_contract=route_contract_from_json(attitude),  # type: ignore[arg-type]
        )
    w = value["w_contract"]
    assert isinstance(w, Mapping)
    return LiteratureRouteContract(
        route_id=str(value["route_id"]),
        target_kind=str(value["target_kind"]),
        history=int(value["history"]),
        k_ridge=None if value["k_ridge"] is None else ridge_from_json(value["k_ridge"]),
        c_ridge=None if value["c_ridge"] is None else ridge_from_json(value["c_ridge"]),
        joint_ridge=None if value["joint_ridge"] is None else ridge_from_json(value["joint_ridge"]),
        w_contract=LatentWContract(
            family=str(w["family"]),
            ridge=None if w["ridge"] is None else ridge_from_json(w["ridge"]),
            knots=tuple(np.asarray(item, dtype=np.float64) for item in w["knots"]),
        ),
        column_slices={str(key): tuple(int(item) for item in values) for key, values in value["column_slices"].items()},
    )


def stable_group_fold(group_id: str, folds: int = 4, salt: str = "NEUROBEM_LITERATURE_R1") -> int:
    if folds < 2:
        raise ValueError("folds must be >=2")
    return int(sha256(f"{salt}|{group_id}".encode()).hexdigest()[:16], 16) % folds


def read_neurobem_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if tuple(frame.columns) != TRACK_A_COLUMNS:
        # Official prediction files intentionally have no header.
        frame = pd.read_csv(path, header=None)
        if frame.shape[1] not in (len(TRACK_A_COLUMNS), len(TRACK_A_COLUMNS) + len(PREDICTION_EXTRA_COLUMNS)):
            raise ValueError(f"NEUROBEM_COLUMN_CONTRACT_MISMATCH:{path}:{frame.shape[1]}")
        columns = TRACK_A_COLUMNS + PREDICTION_EXTRA_COLUMNS[: frame.shape[1] - len(TRACK_A_COLUMNS)]
        frame.columns = columns
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or len(frame) < 21:
        raise ValueError(f"INVALID_NEUROBEM_TRAJECTORY:{path}")
    if not np.all(np.diff(frame["t"].to_numpy(dtype=np.float64)) > 0):
        raise ValueError(f"NONMONOTONE_NEUROBEM_TIME:{path}")
    return frame


def resample_track_b_100hz(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact official hdf5.py time reset and 0.01-second mean resample."""
    data = frame.copy()
    data["t"] = data["t"] - data["t"].iloc[0]
    data["t"] = pd.to_datetime(data["t"], unit="s")
    data = data.set_index("t").resample("0.01s").mean().reset_index()
    required = TRACK_B_STATE_COLUMNS + MOTOR_COLUMNS
    if data.loc[:, required].isna().any().any():
        raise ValueError("TRACK_B_RESAMPLE_PRODUCED_MISSING_REGISTERED_FIELD")
    return data


def track_a_force_torque_target(frame: pd.DataFrame, mass: float = 0.772,
                                inertia: Sequence[float] = (0.0025, 0.0021, 0.0043)) -> np.ndarray:
    """Body force and torque implied by the official rigid-body equations."""
    force = mass * frame.loc[:, ("acc x", "acc y", "acc z")].to_numpy(dtype=np.float64)
    angular_acceleration = frame.loc[:, ("ang acc x", "ang acc y", "ang acc z")].to_numpy(dtype=np.float64)
    angular_velocity = frame.loc[:, ("ang vel x", "ang vel y", "ang vel z")].to_numpy(dtype=np.float64)
    inertia_array = np.asarray(inertia, dtype=np.float64)
    torque = angular_acceleration * inertia_array + np.cross(angular_velocity, angular_velocity * inertia_array)
    return np.column_stack((force, torque))


def official_prediction_force_torque(frame: pd.DataFrame) -> np.ndarray:
    """Return public prediction columns 30--35 (one-based).

    The official dataset README defines columns 30--35 as the predicted total
    body force/torque and columns 36--41 as the paired target-minus-prediction
    residual. They are not base-model and neural-residual components.
    """
    required = PREDICTION_EXTRA_COLUMNS
    if not set(required).issubset(frame.columns):
        raise ValueError("OFFICIAL_PREDICTION_COLUMNS_MISSING")
    return frame.loc[:, required[:6]].to_numpy(dtype=np.float64)


def official_prediction_ground_truth_force_torque(frame: pd.DataFrame) -> np.ndarray:
    """Reconstruct paired ground truth from official prediction + residual."""
    prediction = official_prediction_force_torque(frame)
    residual = frame.loc[:, PREDICTION_EXTRA_COLUMNS[6:]].to_numpy(dtype=np.float64)
    return prediction + residual


def force_torque_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 6:
        raise ValueError("FORCE_TORQUE_METRIC_SHAPE_MISMATCH")
    mse = np.mean(np.square(target - prediction), axis=0)
    return {
        "Fxy": float(np.sqrt(np.mean(mse[:2]))),
        "Fz": float(np.sqrt(mse[2])),
        "Mxy": float(np.sqrt(np.mean(mse[3:5]))),
        "Mz": float(np.sqrt(mse[5])),
        "F": float(np.sqrt(np.mean(mse[:3]))),
        "M": float(np.sqrt(np.mean(mse[3:]))),
    }


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float64).eps):
        raise ValueError("ZERO_NORM_QUATERNION")
    return q / norm


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate((q[..., :1], -q[..., 1:]), axis=-1)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    return np.stack((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ), axis=-1)


def quaternion_log_increment(q_current: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    current = normalize_quaternion(q_current)
    nxt = normalize_quaternion(q_next)
    delta = normalize_quaternion(quaternion_multiply(nxt, quaternion_conjugate(current)))
    delta = np.where(delta[..., :1] < 0.0, -delta, delta)
    vector = delta[..., 1:]
    magnitude = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(magnitude, np.clip(delta[..., :1], 0.0, 1.0))
    scale = np.where(magnitude > 1e-14, angle / magnitude, 2.0)
    return vector * scale


def quaternion_exp_increment(rotation_vector: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation_vector, dtype=np.float64)
    angle = np.linalg.norm(value, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1e-14, np.sin(half) / angle, 0.5)
    return normalize_quaternion(np.concatenate((np.cos(half), value * scale), axis=-1))


def compose_quaternion_increment(q_current: np.ndarray, rotation_vector: np.ndarray) -> np.ndarray:
    return normalize_quaternion(quaternion_multiply(quaternion_exp_increment(rotation_vector), normalize_quaternion(q_current)))


def delta_z_metric(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.shape != prediction.shape or target.shape[-1] != 6:
        raise ValueError("DELTA_Z_METRIC_SHAPE_MISMATCH")
    return float(np.mean(np.sum(np.square(target - prediction), axis=-1)))


def delta_q_metric(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.shape != prediction.shape or target.shape[-1] != 4:
        raise ValueError("DELTA_Q_METRIC_SHAPE_MISMATCH")
    relative = normalize_quaternion(quaternion_multiply(normalize_quaternion(target), quaternion_conjugate(normalize_quaternion(prediction))))
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector_norm = np.linalg.norm(relative[..., 1:], axis=-1)
    # The official evaluator takes the norm of quaternion_log, whose angle is
    # 2*atan2(||q_v||, q_w).  Sign canonicalization keeps q and -q equivalent.
    theta = 2.0 * np.arctan2(vector_norm, np.clip(relative[..., 0], 0.0, None))
    return float(np.mean(theta))


def _history_matrix(values: np.ndarray, history: int, target_offset: int) -> tuple[np.ndarray, np.ndarray]:
    if history < 1 or target_offset < 0:
        raise ValueError("INVALID_HISTORY_CONTRACT")
    last_origin = len(values) - 1 - target_offset
    if last_origin < history - 1:
        return np.empty((0, history * values.shape[1])), np.empty(0, dtype=np.int64)
    origins = np.arange(history - 1, last_origin + 1, dtype=np.int64)
    lags = origins[:, None] - np.arange(history - 1, -1, -1, dtype=np.int64)[None, :]
    return values[lags].reshape(len(origins), -1), origins


def _registered_k_design(motor_history: np.ndarray, state_history: np.ndarray) -> np.ndarray:
    """Registered actuator and predictive-context K columns.

    Motor actuation keeps the published history. Motion context is registered
    before K with current linear/quaternion/angular state columns and frozen
    univariate transforms. It is explicitly predictive context, not a causal
    actuator response.
    """
    current = state_history[:, -10:]
    velocity = current[:, :3]
    quaternion = current[:, 3:7]
    body_rate = current[:, 7:10]
    speed_velocity = np.linalg.norm(velocity, axis=1, keepdims=True) * velocity
    return np.column_stack((
        motor_history,
        velocity,
        velocity * np.abs(velocity),
        speed_velocity,
        quaternion,
        body_rate,
        body_rate * np.abs(body_rate),
    ))


def _context_design(state_history: np.ndarray, k_prediction: np.ndarray) -> np.ndarray:
    """C contains only interactions among already registered K context."""
    current = state_history[:, -10:]
    pairwise = np.column_stack([current[:, i] * current[:, j] for i in range(current.shape[1]) for j in range(i + 1, current.shape[1])])
    interactions = np.einsum("ni,nj->nij", k_prediction, current).reshape(len(current), -1)
    return np.column_stack((pairwise, interactions))


def track_a_design(frame: pd.DataFrame, history: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    motor = np.square(frame.loc[:, MOTOR_COLUMNS].to_numpy(dtype=np.float64))
    context = frame.loc[:, ("vel x", "vel y", "vel z", "quat w", "quat x", "quat y", "quat z", "ang vel x", "ang vel y", "ang vel z")].to_numpy(dtype=np.float64)
    motor_history, origins = _history_matrix(motor, history, 0)
    state_history, state_origins = _history_matrix(context, history, 0)
    if not np.array_equal(origins, state_origins):
        raise RuntimeError("TRACK_A_HISTORY_ALIGNMENT_FAILED")
    y = track_a_force_torque_target(frame)[origins]
    return _registered_k_design(motor_history, state_history), state_history, y, origins


def track_b_design(frame_100hz: pd.DataFrame, history: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    control = frame_100hz.loc[:, MOTOR_COLUMNS].to_numpy(dtype=np.float64) * 0.001
    motor = np.square(control)
    state = frame_100hz.loc[:, TRACK_B_STATE_COLUMNS].to_numpy(dtype=np.float64)
    state[:, 3:7] = normalize_quaternion(state[:, 3:7])
    motor_history, origins = _history_matrix(motor, history, 1)
    state_history, state_origins = _history_matrix(state, history, 1)
    if not np.array_equal(origins, state_origins):
        raise RuntimeError("TRACK_B_HISTORY_ALIGNMENT_FAILED")
    z_current = np.column_stack((state[origins, :3], state[origins, 7:10]))
    z_next = np.column_stack((state[origins + 1, :3], state[origins + 1, 7:10]))
    delta_theta = quaternion_log_increment(state[origins, 3:7], state[origins + 1, 3:7])
    target = np.column_stack((z_next - z_current, delta_theta))
    return _registered_k_design(motor_history, state_history), state_history, target, origins


def latent_w_basis(family: str, latent: np.ndarray, knots: tuple[np.ndarray, ...] = ()) -> np.ndarray:
    if family == "IDENTITY_CORRECTION":
        return np.empty((len(latent), 0), dtype=np.float64)
    if family == "SIGNED_QUADRATIC_LATENT":
        return latent * np.abs(latent)
    if family == "NATURAL_CUBIC_LATENT":
        if len(knots) != latent.shape[1]:
            raise ValueError("LATENT_W_KNOT_DIMENSION_MISMATCH")
        return np.column_stack([restricted_cubic_basis(latent[:, i], knots[i]) for i in range(latent.shape[1])])
    raise ValueError(f"UNKNOWN_CANONICAL_W:{family}")


def fit_latent_w(family: str, latent: np.ndarray, residual: np.ndarray,
                 ridge_grid: Sequence[float], max_condition: float, max_kkt: float) -> LatentWContract:
    if family == "IDENTITY_CORRECTION":
        return LatentWContract(family, None, ())
    knots: tuple[np.ndarray, ...] = ()
    if family == "NATURAL_CUBIC_LATENT":
        knots = tuple(np.unique(np.quantile(latent[:, i], np.linspace(0, 1, 6))) for i in range(latent.shape[1]))
        if any(len(value) < 4 for value in knots):
            raise ValueError("LATENT_W_SUPPORT_DEGENERATE")
    features = latent_w_basis(family, latent, knots)
    return LatentWContract(family, fit_numerical_ridge(features, residual, ridge_grid, max_condition, max_kkt), knots)


def predict_latent_w(contract: LatentWContract, latent: np.ndarray) -> np.ndarray:
    if contract.ridge is None:
        return np.zeros_like(latent)
    return predict_ridge(contract.ridge, latent_w_basis(contract.family, latent, contract.knots))


def _fit_base_blocks(xk: np.ndarray, state_history: np.ndarray, y: np.ndarray,
                     ridge_grid: Sequence[float], max_condition: float, max_kkt: float) -> tuple[RidgeContract, RidgeContract, np.ndarray, np.ndarray]:
    k = fit_numerical_ridge(xk, y, ridge_grid, max_condition, max_kkt)
    k_prediction = predict_ridge(k, xk)
    xc = _context_design(state_history, k_prediction)
    c = fit_numerical_ridge(xc, y - k_prediction, ridge_grid, max_condition, max_kkt)
    c_latent = predict_ridge(c, xc)
    return k, c, k_prediction, c_latent


def fit_route_contracts(xk: np.ndarray, state_history: np.ndarray, y: np.ndarray, w_family: str,
                        ridge_grid: Sequence[float], max_condition: float, max_kkt: float,
                        *, target_kind: str, history: int) -> dict[str, LiteratureRouteContract]:
    if w_family not in CANONICAL_W_CANDIDATES:
        raise ValueError("NONCANONICAL_W_FAMILY")
    k, c, k_prediction, c_latent = _fit_base_blocks(xk, state_history, y, ridge_grid, max_condition, max_kkt)
    xc = _context_design(state_history, k_prediction)
    w = fit_latent_w(w_family, c_latent, y - k_prediction - c_latent, ridge_grid, max_condition, max_kkt)
    xw = latent_w_basis(w.family, c_latent, w.knots)
    joint_kc = np.column_stack((xk, xc))
    joint_kcw = np.column_stack((joint_kc, xw))
    jkc = fit_numerical_ridge(joint_kc, y, ridge_grid, max_condition, max_kkt)
    jkcw = fit_numerical_ridge(joint_kcw, y, ridge_grid, max_condition, max_kkt)
    slices_kc = {"K": (0, xk.shape[1]), "C": (xk.shape[1], joint_kc.shape[1])}
    slices_kcw = {**slices_kc, "W": (joint_kc.shape[1], joint_kcw.shape[1])}
    neutral = LatentWContract("IDENTITY_CORRECTION", None, ())
    return {
        "PF_KC": LiteratureRouteContract("PF_KC", target_kind, history, k, c, None, neutral, slices_kc),
        "PF_KCW": LiteratureRouteContract("PF_KCW", target_kind, history, k, c, None, w, slices_kcw),
        "J_KC": LiteratureRouteContract("J_KC", target_kind, history, None, None, jkc, neutral, slices_kc),
        "J_KCW": LiteratureRouteContract("J_KCW", target_kind, history, None, None, jkcw, w, slices_kcw),
    }


def fit_track_b_route_contracts(
    xk: np.ndarray,
    state_history: np.ndarray,
    y: np.ndarray,
    w_family: str,
    ridge_grid: Sequence[float],
    max_condition: float,
    max_kkt: float,
    *,
    history: int,
) -> dict[str, DecoupledLiteratureRouteContract]:
    """Fit independent published-style velocity and attitude predictors."""
    if y.ndim != 2 or y.shape[1] != 9:
        raise ValueError("TRACK_B_TARGET_MUST_BE_DELTA_Z_PLUS_ROTATION_VECTOR")
    velocity = fit_route_contracts(
        xk, state_history, y[:, :6], w_family, ridge_grid, max_condition, max_kkt,
        target_kind="DELTA_Z_6D", history=history,
    )
    attitude = fit_route_contracts(
        xk, state_history, y[:, 6:], w_family, ridge_grid, max_condition, max_kkt,
        target_kind="ROTATION_VECTOR_3D", history=history,
    )
    return {
        route: DecoupledLiteratureRouteContract(route, velocity[route], attitude[route])
        for route in FORMAL_ROUTE_IDS
    }


def predict_route(contract: LiteratureRouteContract, xk: np.ndarray, state_history: np.ndarray) -> np.ndarray:
    if contract.route_id.startswith("PF_"):
        assert contract.k_ridge is not None and contract.c_ridge is not None
        k_prediction = predict_ridge(contract.k_ridge, xk)
        xc = _context_design(state_history, k_prediction)
        c_latent = predict_ridge(contract.c_ridge, xc)
        return k_prediction + c_latent + predict_latent_w(contract.w_contract, c_latent)
    assert contract.joint_ridge is not None
    # Joint C construction is registered from the frozen PF K/C latent, but
    # all K/C/W coefficients in the formal joint solve are fitted together.
    raise ValueError("JOINT_PREDICTION_REQUIRES_FROZEN_PF_DESIGN")


def predict_joint_route(contract: LiteratureRouteContract, pf_reference: LiteratureRouteContract,
                        xk: np.ndarray, state_history: np.ndarray) -> np.ndarray:
    if not contract.route_id.startswith("J_") or contract.joint_ridge is None:
        raise ValueError("NOT_A_JOINT_ROUTE")
    assert pf_reference.k_ridge is not None and pf_reference.c_ridge is not None
    k_prediction = predict_ridge(pf_reference.k_ridge, xk)
    xc = _context_design(state_history, k_prediction)
    c_latent = predict_ridge(pf_reference.c_ridge, xc)
    blocks = [xk, xc]
    if contract.route_id == "J_KCW":
        blocks.append(latent_w_basis(contract.w_contract.family, c_latent, contract.w_contract.knots))
    return predict_ridge(contract.joint_ridge, np.column_stack(blocks))


def route_prediction(contracts: Mapping[str, LiteratureRouteContract | DecoupledLiteratureRouteContract], route: str,
                     xk: np.ndarray, state_history: np.ndarray) -> np.ndarray:
    contract = contracts[route]
    if isinstance(contract, DecoupledLiteratureRouteContract):
        if route.startswith("PF_"):
            velocity = predict_route(contract.velocity_contract, xk, state_history)
            attitude = predict_route(contract.attitude_contract, xk, state_history)
        else:
            reference = contracts.get("PF_KCW")
            if not isinstance(reference, DecoupledLiteratureRouteContract):
                raise ValueError("DECOUPLED_JOINT_REQUIRES_DECOUPLED_PF_REFERENCE")
            velocity = predict_joint_route(contract.velocity_contract, reference.velocity_contract, xk, state_history)
            attitude = predict_joint_route(contract.attitude_contract, reference.attitude_contract, xk, state_history)
        return np.column_stack((velocity, attitude))
    if route.startswith("PF_"):
        return predict_route(contract, xk, state_history)
    reference = contracts["PF_KCW"]
    if isinstance(reference, DecoupledLiteratureRouteContract):
        raise TypeError("DECOUPLED_REFERENCE_DISPATCH_FAILED")
    return predict_joint_route(contract, reference, xk, state_history)


def candidate_binding_audit(contracts: Mapping[str, LiteratureRouteContract | DecoupledLiteratureRouteContract]) -> dict[str, object]:
    ids = tuple(contracts)
    expected = FORMAL_ROUTE_IDS
    passed = ids == expected and all(contracts[key].route_id == key for key in expected)
    return {
        "passed": bool(passed),
        "candidate_ids": list(ids),
        "expected_candidate_ids": list(expected),
        "a_enabled": False,
        "published_scores_used_for_selection": False,
    }


def normalized_prediction_loss(target: np.ndarray, prediction: np.ndarray, reference_variance: np.ndarray) -> float:
    variance = np.maximum(np.asarray(reference_variance, dtype=np.float64), np.finfo(np.float64).eps)
    return float(np.mean(np.mean(np.square(target - prediction), axis=0) / variance))


def select_w_family(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    evaluation_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    ridge_grid: Sequence[float],
    max_condition: float,
    max_kkt: float,
    *,
    target_kind: str,
    history: int,
    minimum_relative_improvement: float,
) -> tuple[str, dict[str, object]]:
    """Select canonical W from development data only.

    The comparison score is the mean of PF_KCW and J_KCW normalized risks, so
    neither route is privileged. Published baseline values are not accepted as
    arguments and therefore cannot influence this function.
    """
    xk_train, state_train, y_train = train_arrays
    xk_eval, state_eval, y_eval = evaluation_arrays
    variance = np.var(y_train, axis=0)
    losses: dict[str, float] = {}
    route_losses: dict[str, dict[str, float]] = {}
    contracts_by_family: dict[str, dict[str, LiteratureRouteContract]] = {}
    for family in CANONICAL_W_CANDIDATES:
        contracts = fit_route_contracts(
            xk_train, state_train, y_train, family, ridge_grid, max_condition, max_kkt,
            target_kind=target_kind, history=history,
        )
        contracts_by_family[family] = contracts
        current = {}
        for route in ("PF_KCW", "J_KCW"):
            prediction = route_prediction(contracts, route, xk_eval, state_eval)
            current[route] = normalized_prediction_loss(y_eval, prediction, variance)
        route_losses[family] = current
        losses[family] = float(np.mean(list(current.values())))
    neutral = losses["IDENTITY_CORRECTION"]
    best = min(CANONICAL_W_CANDIDATES, key=lambda key: losses[key])
    improvement = (neutral - losses[best]) / max(neutral, np.finfo(float).eps)
    selected = best if best != "IDENTITY_CORRECTION" and improvement >= minimum_relative_improvement else "IDENTITY_CORRECTION"
    return selected, {
        "candidate_losses": losses,
        "route_losses": route_losses,
        "best_unprotected": best,
        "relative_improvement_vs_identity": float(improvement),
        "selected": selected,
        "published_scores_used": False,
    }


def select_track_b_w_family(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    evaluation_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    ridge_grid: Sequence[float],
    max_condition: float,
    max_kkt: float,
    *,
    history: int,
    minimum_relative_improvement: float,
) -> tuple[str, dict[str, object]]:
    """Development-only W selection with decoupled target contracts."""
    xk_train, state_train, y_train = train_arrays
    xk_eval, state_eval, y_eval = evaluation_arrays
    variance = np.var(y_train, axis=0)
    losses: dict[str, float] = {}
    route_losses: dict[str, dict[str, float]] = {}
    for family in CANONICAL_W_CANDIDATES:
        contracts = fit_track_b_route_contracts(
            xk_train, state_train, y_train, family, ridge_grid,
            max_condition, max_kkt, history=history,
        )
        current = {
            route: normalized_prediction_loss(
                y_eval, route_prediction(contracts, route, xk_eval, state_eval), variance,
            )
            for route in ("PF_KCW", "J_KCW")
        }
        route_losses[family] = current
        losses[family] = float(np.mean(list(current.values())))
    neutral = losses["IDENTITY_CORRECTION"]
    best = min(CANONICAL_W_CANDIDATES, key=lambda key: losses[key])
    improvement = (neutral - losses[best]) / max(neutral, np.finfo(float).eps)
    selected = best if best != "IDENTITY_CORRECTION" and improvement >= minimum_relative_improvement else "IDENTITY_CORRECTION"
    return selected, {
        "candidate_losses": losses,
        "route_losses": route_losses,
        "best_unprotected": best,
        "relative_improvement_vs_identity": float(improvement),
        "selected": selected,
        "published_scores_used": False,
        "target_blocks_fitted_independently": True,
    }


def concatenate_track_a_design(trajectories: Iterable[LiteratureTrajectory], history: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [track_a_design(item.frame, history)[:3] for item in trajectories]
    if not arrays:
        raise ValueError("NO_TRACK_A_TRAJECTORIES")
    return tuple(np.concatenate([item[index] for item in arrays], axis=0) for index in range(3))  # type: ignore[return-value]


def concatenate_track_b_design(trajectories: Iterable[LiteratureTrajectory], history: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [track_b_design(item.frame, history)[:3] for item in trajectories]
    if not arrays:
        raise ValueError("NO_TRACK_B_TRAJECTORIES")
    return tuple(np.concatenate([item[index] for item in arrays], axis=0) for index in range(3))  # type: ignore[return-value]


def track_a_route_metrics(contracts: Mapping[str, LiteratureRouteContract], trajectories: Iterable[LiteratureTrajectory], history: int = 20) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    aggregate_y: list[np.ndarray] = []
    aggregate_prediction: dict[str, list[np.ndarray]] = {route: [] for route in FORMAL_ROUTE_IDS}
    per_flight: list[dict[str, object]] = []
    for trajectory in trajectories:
        xk, state, y, _ = track_a_design(trajectory.frame, history)
        aggregate_y.append(y)
        row: dict[str, object] = {"trajectory_id": trajectory.trajectory_id, "rows": len(y)}
        for route in FORMAL_ROUTE_IDS:
            prediction = route_prediction(contracts, route, xk, state)
            aggregate_prediction[route].append(prediction)
            for metric, value in force_torque_metrics(y, prediction).items():
                row[f"{route}_{metric}"] = value
        per_flight.append(row)
    y = np.concatenate(aggregate_y)
    return {
        route: force_torque_metrics(y, np.concatenate(aggregate_prediction[route]))
        for route in FORMAL_ROUTE_IDS
    }, per_flight


def track_b_rollout(
    contracts: Mapping[str, LiteratureRouteContract | DecoupledLiteratureRouteContract],
    route: str,
    frame_100hz: pd.DataFrame,
    *,
    history: int = 20,
    rollout: int = 60,
) -> dict[str, object]:
    """Vectorized strict recursive rollout with future controls only."""
    if route not in FORMAL_ROUTE_IDS:
        raise ValueError("UNKNOWN_FORMAL_ROUTE")
    state = frame_100hz.loc[:, TRACK_B_STATE_COLUMNS].to_numpy(dtype=np.float64)
    state[:, 3:7] = normalize_quaternion(state[:, 3:7])
    control = frame_100hz.loc[:, MOTOR_COLUMNS].to_numpy(dtype=np.float64) * 0.001
    # This matches the official hdf5_trajectories implementation exactly:
    # num_samples = N - history_length - unroll_length (no +1 endpoint).
    window_count = len(state) - history - rollout
    if window_count <= 0:
        raise ValueError("TRACK_B_TRAJECTORY_TOO_SHORT")
    state_windows = np.lib.stride_tricks.sliding_window_view(state, (history, state.shape[1]))[:, 0, :, :]
    control_windows = np.lib.stride_tricks.sliding_window_view(control, (history + rollout, control.shape[1]))[:, 0, :, :]
    state_history = state_windows[:window_count].copy()
    controls = control_windows[:window_count]
    origins = np.arange(history - 1, history - 1 + window_count, dtype=np.int64)
    predicted_z: list[np.ndarray] = []
    predicted_q: list[np.ndarray] = []
    for step in range(rollout):
        current_controls = controls[:, step:step + history]
        motor_history = np.square(current_controls).reshape(window_count, -1)
        state_flat = state_history.reshape(window_count, -1)
        xk = _registered_k_design(motor_history, state_flat)
        increment = route_prediction(contracts, route, xk, state_flat)
        current = state_history[:, -1]
        z = np.column_stack((current[:, :3], current[:, 7:10])) + increment[:, :6]
        q = compose_quaternion_increment(current[:, 3:7], increment[:, 6:9])
        next_state = np.column_stack((z[:, :3], q, z[:, 3:]))
        predicted_z.append(z)
        predicted_q.append(q)
        state_history = np.concatenate((state_history[:, 1:], next_state[:, None, :]), axis=1)
    prediction_z = np.stack(predicted_z, axis=1)
    prediction_q = np.stack(predicted_q, axis=1)
    steps = np.arange(1, rollout + 1, dtype=np.int64)
    target_indices = origins[:, None] + steps[None, :]
    target_z = np.concatenate((state[target_indices, :3], state[target_indices, 7:10]), axis=2)
    target_q = state[target_indices, 3:7]
    assert_no_future_state_access(
        history - 1,
        range(history),
        range(history + rollout),
        rollout,
    )
    per_step_delta_z = [delta_z_metric(target_z[:, step], prediction_z[:, step]) for step in range(rollout)]
    per_step_delta_q = [delta_q_metric(target_q[:, step], prediction_q[:, step]) for step in range(rollout)]
    return {
        "trajectory_rows": len(state),
        "sliding_windows": window_count,
        "delta_z": delta_z_metric(target_z.reshape(-1, 6), prediction_z.reshape(-1, 6)),
        "delta_q": delta_q_metric(target_q.reshape(-1, 4), prediction_q.reshape(-1, 4)),
        "per_step_delta_z": per_step_delta_z,
        "per_step_delta_q": per_step_delta_q,
        "maximum_quaternion_norm_error": float(np.max(np.abs(np.linalg.norm(prediction_q, axis=2) - 1.0))),
        "future_controls_used": True,
        "future_measured_states_used": False,
        "future_target_residual_used": False,
    }


def legacy_aero_w_classification(candidate: str) -> str:
    if candidate in {"SIGNED_QUADRATIC_AERO_CONTEXT", "NATURAL_CUBIC_SPEED_CONTEXT"} or candidate.startswith("NATURAL_CUBIC_SPEED_CONTEXT_K"):
        return "AERODYNAMIC_CONTEXT_W_EXTENSION_DIAGNOSTIC"
    if candidate in CANONICAL_W_CANDIDATES:
        return "CANONICAL_LATENT_W"
    raise ValueError("UNKNOWN_W_CLASSIFICATION")


def metric_reproduction_gate(reproduced: Mapping[str, float], published: Mapping[str, float], tolerance: float = 0.01) -> dict[str, object]:
    keys = ("Fxy", "Fz", "Mxy", "Mz", "F", "M")
    relative = {key: abs(float(reproduced[key]) - float(published[key])) / max(abs(float(published[key])), np.finfo(float).eps) for key in keys}
    passed = all(value <= tolerance for value in relative.values())
    return {
        "passed": bool(passed),
        "relative_difference": relative,
        "tolerance": tolerance,
        "comparison_class": "EXACT_DIRECT_COMPARISON" if passed else "PUBLISHED_AGGREGATE_COMPARISON_ONLY",
    }


def track_b_split_manifest(release_root: Path, official_commit: str) -> dict[str, object]:
    result: dict[str, object] = {
        "source_file": "official data.zip directory membership",
        "official_code_commit": official_commit,
        "paper_counts": {"train": 67, "validation": 17, "test": 12},
    }
    for partition, folder in (("train", "train"), ("validation", "valid"), ("test", "test")):
        names = sorted(path.name for path in (release_root / folder).glob("*.csv"))
        result[f"{partition}_trajectories"] = names
        result[f"{partition}_release_csv_count"] = len(names)
    result["release_counts_match_paper_counts"] = (
        result["train_release_csv_count"] == 67 and result["validation_release_csv_count"] == 17 and result["test_release_csv_count"] == 12
    )
    result["split_identity_status"] = (
        "OFFICIAL_RELEASE_IDENTITIES_RECOVERED_WITH_MANUSCRIPT_COUNT_DISCREPANCY"
        if result["test_release_csv_count"] == 12 else "SPLIT_IDENTITY_UNVERIFIED"
    )
    return result


def assert_no_future_state_access(origin: int, state_indices_read: Iterable[int], control_indices_read: Iterable[int], rollout: int) -> None:
    state_indices = tuple(int(value) for value in state_indices_read)
    control_indices = tuple(int(value) for value in control_indices_read)
    if state_indices and max(state_indices) > origin:
        raise RuntimeError("FUTURE_MEASURED_STATE_ACCESS")
    if control_indices and max(control_indices) > origin + rollout:
        raise RuntimeError("CONTROL_ACCESS_BEYOND_REGISTERED_ROLLOUT")
