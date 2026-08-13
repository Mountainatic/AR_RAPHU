from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from prism_benchmark.neurobem_literature import (
    MOTOR_COLUMNS, TRACK_B_STATE_COLUMNS, DecoupledLiteratureRouteContract,
    _registered_k_design, compose_quaternion_increment, fit_track_b_route_contracts,
    normalize_quaternion, route_contract_from_json, route_contract_to_json,
    route_prediction, track_b_design,
)


def geometry_features(frame: pd.DataFrame) -> np.ndarray:
    state = frame.loc[:, TRACK_B_STATE_COLUMNS].to_numpy(dtype=np.float64)
    state[:, 3:7] = normalize_quaternion(state[:, 3:7])
    motor = frame.loc[:, MOTOR_COLUMNS].to_numpy(dtype=np.float64) * 0.001
    return np.column_stack((state, motor, np.square(motor), np.linalg.norm(state[:, :3], axis=1), np.linalg.norm(state[:, 7:10], axis=1)))


class FrozenPrismAdapter:
    def __init__(self, contracts: Mapping[str, DecoupledLiteratureRouteContract], model_id: str, history: int = 20):
        self.contracts = dict(contracts)
        self.model_id = model_id
        self.history = history

    @classmethod
    def load(cls, path: Path, model_id: str = "GLOBAL_FROZEN_PRISM") -> "FrozenPrismAdapter":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: route_contract_from_json(v) for k, v in raw.items()}, model_id)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: route_contract_to_json(v) for k, v in self.contracts.items()}, sort_keys=True) + "\n", encoding="utf-8")

    def increment(self, route: str, state_history: np.ndarray, raw_control_history: np.ndarray) -> np.ndarray:
        if state_history.shape != (self.history, 10) or raw_control_history.shape != (self.history, 4):
            raise ValueError("PRISM_ADAPTER_HISTORY_SHAPE_MISMATCH")
        state = np.asarray(state_history, dtype=np.float64).copy()
        state[:, 3:7] = normalize_quaternion(state[:, 3:7])
        motor = np.square(np.asarray(raw_control_history, dtype=np.float64) * 0.001).reshape(1, -1)
        flat = state.reshape(1, -1)
        return route_prediction(self.contracts, route, _registered_k_design(motor, flat), flat)[0]

    def next_state(self, route: str, state_history: np.ndarray, raw_control_history: np.ndarray) -> np.ndarray:
        delta = self.increment(route, state_history, raw_control_history)
        current = state_history[-1]
        z = np.r_[current[:3], current[7:10]] + delta[:6]
        q = compose_quaternion_increment(current[3:7][None], delta[6:9][None])[0]
        return np.r_[z[:3], q, z[3:]]

    def one_step(self, route: str, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xk, state_history, target, origins = track_b_design(frame, self.history)
        return target, route_prediction(self.contracts, route, xk, state_history), origins


def fit_local_adapter(
    trajectories: Sequence, model_id: str, w_family: str, row_cap_per_trajectory: int,
    ridge_grid: Sequence[float], max_condition: float, max_kkt: float, history: int = 20,
) -> FrozenPrismAdapter:
    xs, states, ys = [], [], []
    for trajectory in trajectories:
        x, state, y, _ = track_b_design(trajectory.frame, history)
        if len(y) > row_cap_per_trajectory:
            index = np.linspace(0, len(y) - 1, row_cap_per_trajectory, dtype=np.int64)
            x, state, y = x[index], state[index], y[index]
        xs.append(x); states.append(state); ys.append(y)
    contracts = fit_track_b_route_contracts(
        np.concatenate(xs), np.concatenate(states), np.concatenate(ys), w_family,
        ridge_grid, max_condition, max_kkt, history=history,
    )
    return FrozenPrismAdapter(contracts, model_id, history)

