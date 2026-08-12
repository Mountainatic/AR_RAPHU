"""Exact published-information training contract for frozen NeuroBEM PRISM.

The official neural evaluator recursively owns one state block and injects the
complementary measured block plus the published future motor command.  This
module applies that same information contract to the existing FP64 PRISM ridge
estimator.  It does not add clipping, projection, stability penalties, or a new
candidate family.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import linalg

from .neurobem_linear import RidgeContract, predict_ridge
from .neurobem_literature import (
    FORMAL_ROUTE_IDS,
    MOTOR_COLUMNS,
    TRACK_B_STATE_COLUMNS,
    DecoupledLiteratureRouteContract,
    LatentWContract,
    LiteratureRouteContract,
    LiteratureTrajectory,
    _context_design,
    _decoupled_branch_prediction,
    _registered_k_design,
    compose_quaternion_increment,
    fit_track_b_route_contracts,
    latent_w_basis,
    normalize_quaternion,
    predict_latent_w,
    published_evaluator_state_updates,
    quaternion_log_increment,
)

PUBLISHED_HISTORY = 20
PUBLISHED_TRAINING_UNROLL = 10
PUBLISHED_TEST_ROLLOUT = 60
PUBLISHED_SAMPLING_HZ = 100


class NonfinitePublishedTrainingError(RuntimeError):
    """Raised instead of masking an unstable registered training rollout."""


@dataclass
class StreamingRidgeMoments:
    """Sufficient statistics for the existing mean-scaled FP64 ridge."""

    feature_count: int
    target_count: int

    def __post_init__(self) -> None:
        self.rows = 0
        self.sum_x = np.zeros(self.feature_count, dtype=np.float64)
        self.sum_y = np.zeros(self.target_count, dtype=np.float64)
        self.sum_xx = np.zeros((self.feature_count, self.feature_count), dtype=np.float64)
        self.sum_xy = np.zeros((self.feature_count, self.target_count), dtype=np.float64)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("INVALID_STREAMING_RIDGE_BATCH")
        if x.shape[1] != self.feature_count or y.shape[1] != self.target_count:
            raise ValueError("STREAMING_RIDGE_DIMENSION_CHANGED")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise NonfinitePublishedTrainingError("NONFINITE_TRAINING_DESIGN_OR_TARGET")
        self.rows += int(x.shape[0])
        self.sum_x += np.sum(x, axis=0, dtype=np.float64)
        self.sum_y += np.sum(y, axis=0, dtype=np.float64)
        try:
            with np.errstate(over="raise", invalid="raise"):
                self.sum_xx += x.T @ x
                self.sum_xy += x.T @ y
        except FloatingPointError as error:
            raise NonfinitePublishedTrainingError("NONFINITE_TRAINING_NORMAL_EQUATIONS") from error

    def merge(self, other: "StreamingRidgeMoments") -> None:
        if (self.feature_count, self.target_count) != (other.feature_count, other.target_count):
            raise ValueError("STREAMING_RIDGE_MERGE_DIMENSION_MISMATCH")
        self.rows += other.rows
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_xx += other.sum_xx
        self.sum_xy += other.sum_xy

    def solve(
        self,
        alpha_grid: Sequence[float],
        maximum_condition_number: float,
        maximum_relative_kkt_residual: float,
    ) -> RidgeContract:
        if self.rows < 2:
            raise ValueError("INSUFFICIENT_STREAMING_RIDGE_ROWS")
        n = float(self.rows)
        feature_mean = self.sum_x / n
        target_mean = self.sum_y / n
        centered_xx = self.sum_xx / n - np.outer(feature_mean, feature_mean)
        centered_xy = self.sum_xy / n - np.outer(feature_mean, target_mean)
        variance = np.maximum(np.diag(centered_xx), 0.0)
        feature_scale = np.sqrt(variance)
        feature_scale = np.where(feature_scale > np.finfo(np.float64).eps, feature_scale, 1.0)
        scale_outer = np.outer(feature_scale, feature_scale)
        gram = centered_xx / scale_outer
        gram = 0.5 * (gram + gram.T)
        rhs = centered_xy / feature_scale[:, None]
        identity = np.eye(self.feature_count, dtype=np.float64)
        failures: list[tuple[float, float, float]] = []
        for alpha in alpha_grid:
            penalized = gram + float(alpha) * identity
            condition = float(np.linalg.cond(penalized))
            try:
                coefficient = linalg.solve(penalized, rhs, assume_a="pos", check_finite=False)
            except linalg.LinAlgError:
                failures.append((float(alpha), condition, float("inf")))
                continue
            residual = penalized @ coefficient - rhs
            relative_kkt = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).eps))
            if condition <= maximum_condition_number and relative_kkt <= maximum_relative_kkt_residual:
                return RidgeContract(
                    alpha=float(alpha),
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    target_mean=target_mean,
                    coefficient_standardized=coefficient,
                    condition_number=condition,
                    relative_kkt_residual=relative_kkt,
                    fit_rows=self.rows,
                )
            failures.append((float(alpha), condition, relative_kkt))
        raise np.linalg.LinAlgError(f"NO_NUMERICALLY_CERTIFIED_STREAMING_RIDGE:{failures}")


@dataclass(frozen=True)
class PublishedTrainingTrace:
    branch: str
    predicted_owned_state_used: bool
    measured_complementary_state_used: bool
    future_control_used: bool
    future_owned_ground_truth_used_as_input: bool
    future_target_residual_used: bool
    quaternion_norm_error: float


def published_training_state_updates(
    predicted_z: np.ndarray,
    predicted_q: np.ndarray,
    measured_next_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared implementation for training and published evaluation updates."""
    return published_evaluator_state_updates(predicted_z, predicted_q, measured_next_state)


def audit_published_training_contract() -> dict[str, object]:
    """Machine-readable access contract used by tests and the development freeze."""
    return {
        "history": PUBLISHED_HISTORY,
        "training_unroll": PUBLISHED_TRAINING_UNROLL,
        "test_rollout": PUBLISHED_TEST_ROLLOUT,
        "sampling_hz": PUBLISHED_SAMPLING_HZ,
        "window_count_formula": "N_MINUS_H_MINUS_U",
        "velocity": {
            "recursive_owned_state": ["linear_velocity", "angular_velocity"],
            "measured_complementary_state": ["attitude"],
            "future_control": True,
            "future_owned_ground_truth": False,
        },
        "attitude": {
            "recursive_owned_state": ["unit_quaternion"],
            "measured_complementary_state": ["linear_velocity", "angular_velocity"],
            "future_control": True,
            "future_owned_ground_truth": False,
        },
        "future_target_residual": False,
        "state_update_implementation_shared_with_evaluator": True,
        "a_enabled": False,
        "stabilization_added": False,
    }


def _branch_contract(
    contracts: Mapping[str, DecoupledLiteratureRouteContract],
    route: str,
    branch: str,
) -> LiteratureRouteContract:
    contract = contracts[route]
    return contract.velocity_contract if branch == "velocity" else contract.attitude_contract


def _replace_branch(
    contracts: Mapping[str, DecoupledLiteratureRouteContract],
    route: str,
    branch: str,
    value: LiteratureRouteContract,
) -> dict[str, DecoupledLiteratureRouteContract]:
    result = dict(contracts)
    prior = result[route]
    result[route] = DecoupledLiteratureRouteContract(
        route,
        value if branch == "velocity" else prior.velocity_contract,
        value if branch == "attitude" else prior.attitude_contract,
    )
    return result


def _zero_ridge(template: RidgeContract) -> RidgeContract:
    """Exact neutral boundary for the block currently being trained."""
    return RidgeContract(
        alpha=template.alpha,
        feature_mean=template.feature_mean,
        feature_scale=template.feature_scale,
        target_mean=np.zeros_like(template.target_mean),
        coefficient_standardized=np.zeros_like(template.coefficient_standardized),
        condition_number=template.condition_number,
        relative_kkt_residual=template.relative_kkt_residual,
        fit_rows=template.fit_rows,
    )


def iter_published_training_rollin(
    contracts: Mapping[str, DecoupledLiteratureRouteContract],
    route: str,
    branch: str,
    trajectory: LiteratureTrajectory,
    *,
    history: int = PUBLISHED_HISTORY,
    unroll: int = PUBLISHED_TRAINING_UNROLL,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (K design, flattened branch history, exact next-state target).

    The target is expressed in the existing PRISM increment coordinates.  The
    roll-in history owns only the state predicted by the selected branch.
    """
    if route not in FORMAL_ROUTE_IDS or branch not in ("velocity", "attitude"):
        raise ValueError("INVALID_PUBLISHED_TRAINING_ROUTE_OR_BRANCH")
    frame = trajectory.frame
    state = frame.loc[:, TRACK_B_STATE_COLUMNS].to_numpy(dtype=np.float64)
    state[:, 3:7] = normalize_quaternion(state[:, 3:7])
    control = frame.loc[:, MOTOR_COLUMNS].to_numpy(dtype=np.float64) * 0.001
    count = len(state) - history - unroll
    if count <= 0:
        return
    state_windows = np.lib.stride_tricks.sliding_window_view(state, (history, state.shape[1]))[:, 0]
    control_windows = np.lib.stride_tricks.sliding_window_view(control, (history + unroll, control.shape[1]))[:, 0]
    branch_history = state_windows[:count].copy()
    controls = control_windows[:count]
    origins = np.arange(history - 1, history - 1 + count, dtype=np.int64)
    for step in range(unroll):
        motor_history = np.square(controls[:, step:step + history]).reshape(count, -1)
        state_flat = branch_history.reshape(count, -1)
        xk = _registered_k_design(motor_history, state_flat)
        if not np.isfinite(state_flat).all() or not np.isfinite(xk).all():
            raise NonfinitePublishedTrainingError(
                f"NONFINITE_TRAINING_FEATURES:route={route}:branch={branch}:trajectory={trajectory.trajectory_id}:step={step + 1}"
            )
        increment = _decoupled_branch_prediction(contracts, route, branch, xk, state_flat)
        if not np.isfinite(increment).all():
            raise NonfinitePublishedTrainingError(
                f"NONFINITE_TRAINING_PREDICTION:route={route}:branch={branch}:trajectory={trajectory.trajectory_id}:step={step + 1}"
            )
        current = branch_history[:, -1]
        measured_next = state[origins + step + 1]
        if branch == "velocity":
            current_z = np.column_stack((current[:, :3], current[:, 7:10]))
            target_z = np.column_stack((measured_next[:, :3], measured_next[:, 7:10]))
            target_increment = target_z - current_z
            predicted_z = current_z + increment
            predicted_q = current[:, 3:7]
        else:
            target_increment = quaternion_log_increment(current[:, 3:7], measured_next[:, 3:7])
            predicted_q = compose_quaternion_increment(current[:, 3:7], increment)
            predicted_z = np.column_stack((current[:, :3], current[:, 7:10]))
        yield xk, state_flat, target_increment
        if step < unroll - 1:
            velocity_row, attitude_row = published_training_state_updates(predicted_z, predicted_q, measured_next)
            next_row = velocity_row if branch == "velocity" else attitude_row
            branch_history = np.concatenate((branch_history[:, 1:], next_row[:, None]), axis=1)


def _collect_moments(
    trajectories: Sequence[LiteratureTrajectory],
    contracts: Mapping[str, DecoupledLiteratureRouteContract],
    route: str,
    branch: str,
    feature_builder,
    target_builder,
    feature_count: int,
    target_count: int,
    workers: int,
) -> StreamingRidgeMoments:
    def collect_one(trajectory: LiteratureTrajectory) -> StreamingRidgeMoments:
        local = StreamingRidgeMoments(feature_count, target_count)
        for xk, state_history, target in iter_published_training_rollin(contracts, route, branch, trajectory):
            features = feature_builder(xk, state_history)
            local.update(features, target_builder(target, xk, state_history))
        return local

    total = StreamingRidgeMoments(feature_count, target_count)
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(workers), len(trajectories))),
        thread_name_prefix=f"exact-{branch}-{route.lower()}",
    ) as executor:
        for local in executor.map(collect_one, trajectories):
            total.merge(local)
    return total


def fit_exact_published_training_contracts(
    trajectories: Sequence[LiteratureTrajectory],
    initial_contracts: Mapping[str, DecoupledLiteratureRouteContract],
    ridge_grid: Sequence[float],
    max_condition: float,
    max_kkt: float,
    *,
    history: int = PUBLISHED_HISTORY,
    unroll: int = PUBLISHED_TRAINING_UNROLL,
    trajectory_workers: int = 16,
) -> tuple[dict[str, DecoupledLiteratureRouteContract], dict[str, object]]:
    """One deterministic recursive roll-in refit of every frozen formal route."""
    if history != PUBLISHED_HISTORY or unroll != PUBLISHED_TRAINING_UNROLL:
        raise ValueError("PUBLISHED_TRAINING_H_OR_U_CHANGED")
    contracts = dict(initial_contracts)
    fit_audit: dict[str, object] = {"branches": {}, "contract": audit_published_training_contract()}
    for branch, target_count in (("velocity", 6), ("attitude", 3)):
        branch_audit: dict[str, object] = {}
        # PF K/C are trained on their own PF_KC recursive state distribution.
        old_pf = _branch_contract(contracts, "PF_KC", branch)
        assert old_pf.k_ridge is not None and old_pf.c_ridge is not None
        neutral = LatentWContract("IDENTITY_CORRECTION", None, ())
        zero_pf = LiteratureRouteContract(
            "PF_KC", old_pf.target_kind, history,
            _zero_ridge(old_pf.k_ridge), _zero_ridge(old_pf.c_ridge), None,
            neutral, dict(old_pf.column_slices),
        )
        contracts = _replace_branch(contracts, "PF_KC", branch, zero_pf)
        k_moments = _collect_moments(
            trajectories, contracts, "PF_KC", branch,
            lambda xk, _state: xk,
            lambda target, _xk, _state: target,
            old_pf.k_ridge.feature_mean.size, target_count,
            trajectory_workers,
        )
        k = k_moments.solve(ridge_grid, max_condition, max_kkt)
        provisional_k = LiteratureRouteContract(
            "PF_KC", old_pf.target_kind, history, k, _zero_ridge(old_pf.c_ridge), None,
            neutral, dict(old_pf.column_slices),
        )
        contracts = _replace_branch(contracts, "PF_KC", branch, provisional_k)
        c_moments = _collect_moments(
            trajectories, contracts, "PF_KC", branch,
            lambda xk, state: _context_design(state, predict_ridge(k, xk)),
            lambda target, xk, _state: target - predict_ridge(k, xk),
            old_pf.c_ridge.feature_mean.size, target_count,
            trajectory_workers,
        )
        c = c_moments.solve(ridge_grid, max_condition, max_kkt)
        slices_kc = dict(old_pf.column_slices)
        pf_kc = LiteratureRouteContract("PF_KC", old_pf.target_kind, history, k, c, None, neutral, slices_kc)
        contracts = _replace_branch(contracts, "PF_KC", branch, pf_kc)

        # PF W keeps the already selected family and train-fitted knots; only
        # its coefficients are refitted on the exact U-step recursive roll-in.
        old_pfw = _branch_contract(contracts, "PF_KCW", branch)
        w_template = old_pfw.w_contract
        zero_w = w_template if w_template.ridge is None else LatentWContract(
            w_template.family, _zero_ridge(w_template.ridge), w_template.knots,
        )
        provisional_w = LiteratureRouteContract(
            "PF_KCW", old_pfw.target_kind, history, k, c, None,
            zero_w, dict(old_pfw.column_slices),
        )
        contracts = _replace_branch(contracts, "PF_KCW", branch, provisional_w)
        if w_template.ridge is None:
            w_new = w_template
            w_rows = 0
        else:
            def c_latent(xk: np.ndarray, state: np.ndarray) -> np.ndarray:
                kp = predict_ridge(k, xk)
                return predict_ridge(c, _context_design(state, kp))
            w_moments = _collect_moments(
                trajectories, contracts, "PF_KCW", branch,
                lambda xk, state: latent_w_basis(w_template.family, c_latent(xk, state), w_template.knots),
                lambda target, xk, state: target - predict_ridge(k, xk) - c_latent(xk, state),
                w_template.ridge.feature_mean.size, target_count,
                trajectory_workers,
            )
            w_new = LatentWContract(w_template.family, w_moments.solve(ridge_grid, max_condition, max_kkt), w_template.knots)
            w_rows = w_moments.rows
        pf_kcw = LiteratureRouteContract(
            "PF_KCW", old_pfw.target_kind, history, k, c, None, w_new, dict(old_pfw.column_slices),
        )
        contracts = _replace_branch(contracts, "PF_KCW", branch, pf_kcw)

        # Joint rows use the same frozen PF construction but solve all included
        # K/C(/W) coefficients together, exactly as in the parent estimator.
        for route in ("J_KC", "J_KCW"):
            old_joint = _branch_contract(contracts, route, branch)
            assert old_joint.joint_ridge is not None
            zero_joint = LiteratureRouteContract(
                route, old_joint.target_kind, history, None, None,
                _zero_ridge(old_joint.joint_ridge),
                w_new if route == "J_KCW" else neutral,
                dict(old_joint.column_slices),
            )
            contracts = _replace_branch(contracts, route, branch, zero_joint)
            def joint_features(xk: np.ndarray, state: np.ndarray, *, route_id: str = route) -> np.ndarray:
                kp = predict_ridge(k, xk)
                xc = _context_design(state, kp)
                blocks = [xk, xc]
                if route_id == "J_KCW":
                    cl = predict_ridge(c, xc)
                    blocks.append(latent_w_basis(w_new.family, cl, w_new.knots))
                return np.column_stack(blocks)
            moments = _collect_moments(
                trajectories, contracts, route, branch,
                joint_features,
                lambda target, _xk, _state: target,
                old_joint.joint_ridge.feature_mean.size, target_count,
                trajectory_workers,
            )
            joint = moments.solve(ridge_grid, max_condition, max_kkt)
            updated = LiteratureRouteContract(
                route, old_joint.target_kind, history, None, None, joint,
                w_new if route == "J_KCW" else neutral,
                dict(old_joint.column_slices),
            )
            contracts = _replace_branch(contracts, route, branch, updated)
            branch_audit[f"{route}_fit_rows"] = moments.rows
        branch_audit.update({
            "K_fit_rows": k_moments.rows,
            "C_fit_rows": c_moments.rows,
            "W_fit_rows": w_rows,
            "recursive_rollin": unroll,
            "owned_state_teacher_forced": False,
            "complementary_state_measured": True,
            "future_controls_used": True,
            "future_target_residual_used": False,
        })
        fit_audit["branches"][branch] = branch_audit
    return contracts, fit_audit


def initial_exact_training_contracts(
    trajectories: Sequence[LiteratureTrajectory],
    w_family: str,
    ridge_grid: Sequence[float],
    max_condition: float,
    max_kkt: float,
) -> dict[str, DecoupledLiteratureRouteContract]:
    """Train-only one-step initialization; no validation or test row enters."""
    from .neurobem_literature import concatenate_track_b_design

    return fit_track_b_route_contracts(
        *concatenate_track_b_design(trajectories, PUBLISHED_HISTORY),
        w_family,
        ridge_grid,
        max_condition,
        max_kkt,
        history=PUBLISHED_HISTORY,
    )
