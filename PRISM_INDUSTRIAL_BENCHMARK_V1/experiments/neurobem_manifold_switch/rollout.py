from __future__ import annotations

import time
import numpy as np
import pandas as pd

from prism_benchmark.neurobem_literature import MOTOR_COLUMNS, TRACK_B_STATE_COLUMNS, normalize_quaternion
from .metrics import divergence_time, state_errors
from .model_bank import ModelBank
from .monitor import ManifoldTemplate, component_scores, persistent_alarm, residual_score
from .online_reid import causal_reidentify
from .prism_adapter import FrozenPrismAdapter, geometry_features


def evaluate(
    frame: pd.DataFrame, route: str, global_adapter: FrozenPrismAdapter, global_template: ManifoldTemplate,
    bank: ModelBank, calibration: dict[str, object], config: dict, ablation: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    history = int(config["history"])
    y, one_step_prediction, origins = global_adapter.one_step(route, frame)
    residual = residual_score(y, one_step_prediction, np.asarray(calibration["residual_scale"]))
    features_all = geometry_features(frame)
    features = features_all[origins]
    projection = global_template.projection_score(features)
    tangent = global_template.tangent_score(features, int(config["tangent_window"]), int(config["tangent_rank"]))
    scores = component_scores(residual, projection, tangent, calibration["thresholds"])
    signal = {"residual_only": "residual", "geometry_only": "geometry"}.get(ablation, "combined")
    threshold = {"residual": calibration["thresholds"]["residual_threshold"], "geometry": calibration["thresholds"]["geometry_threshold"], "combined": calibration["thresholds"]["combined_threshold"]}[signal]
    state = frame.loc[:, TRACK_B_STATE_COLUMNS].to_numpy(dtype=np.float64)
    state[:, 3:7] = normalize_quaternion(state[:, 3:7])
    control = frame.loc[:, MOTOR_COLUMNS].to_numpy(dtype=np.float64)
    steps = min(int(config["max_rollout_steps"]), len(state) - history)
    alarm = None if ablation == "static" else persistent_alarm(scores[signal][:steps], threshold, int(config["alarm_persistence"]))
    predicted_history = state[:history].copy()
    predictions = np.full((steps, 10), np.nan, dtype=np.float64)
    model_ids: list[str] = []
    active = global_adapter
    switch_time = None
    candidate_scores: dict[str, float] = {}
    unknown = False
    promotion_time = None
    promotion_gain = None
    reid_attempted = False
    start_clock = time.perf_counter()
    for step in range(steps):
        # The residual that triggers alarm a becomes available only after the
        # a-th next observation arrives.  Switching therefore starts at a+1;
        # it never rewrites the prediction that produced the innovation.
        if alarm is not None and step == alarm + 1 and ablation in {"combined_switch", "combined_switch_reid"}:
            chosen, candidate_scores = bank.choose(features_all[history + alarm])
            if chosen is not None:
                active = chosen.adapter
                switch_time = step
                predicted_history = state[step:step + history].copy()
            else:
                unknown = True
        if unknown and ablation == "combined_switch_reid" and promotion_time is None and not reid_attempted and step >= alarm + int(config["reid_fit_samples"]) + int(config["reid_validation_samples"]):
            reid_attempted = True
            # At rollout step s the newest observable row is history+s-1;
            # row history+s is the target about to be predicted.
            decision = causal_reidentify(frame, history + step - 1, history + alarm, route, active, config)
            if decision.promoted and decision.entry is not None:
                bank.add(decision.entry)
                active = decision.entry.adapter
                promotion_time = step
                promotion_gain = decision.gain
                predicted_history = state[step:step + history].copy()
        try:
            nxt = active.next_state(route, predicted_history, control[step:step + history])
        except Exception:
            nxt = np.full(10, np.nan)
        predictions[step] = nxt
        model_ids.append(active.model_id)
        if np.isfinite(nxt).all():
            predicted_history = np.vstack((predicted_history[1:], nxt))
        else:
            predictions[step:] = np.nan
            model_ids.extend([active.model_id] * (steps - step - 1))
            break
    runtime = time.perf_counter() - start_clock
    target_state = state[history:history + steps]
    errors = state_errors(target_state, predictions)
    finite = np.isfinite(predictions).all(axis=1)
    divergence = divergence_time(errors, finite, config["divergence_thresholds"], int(config["divergence_persistence"]))
    sensitivity = {}
    for multiplier in config["divergence_sensitivity_multipliers"]:
        adjusted = {key: float(value) * float(multiplier) for key, value in config["divergence_thresholds"].items()}
        sensitivity[str(multiplier)] = divergence_time(errors, finite, adjusted, int(config["divergence_persistence"]))
    log = pd.DataFrame({
        "step": np.arange(steps), "residual_score": scores["residual"][:steps], "geometry_score": scores["geometry"][:steps],
        "combined_score": scores["combined"][:steps], "detector_threshold": threshold,
        "velocity_error": errors["velocity"], "attitude_error": errors["attitude"], "body_rate_error": errors["body_rate"],
        "finite": finite, "active_model_id": model_ids[:steps],
    })
    return {
        "ablation": ablation, "route": route, "t_alarm": alarm, "t_switch": switch_time, "t_diverge": divergence,
        "lead_time": None if alarm is None or divergence is None else divergence - alarm,
        "one_step_error": float(np.mean(np.square(y - one_step_prediction))),
        "rollout_velocity_error": float(np.nanmean(errors["velocity"])), "rollout_attitude_error": float(np.nanmean(errors["attitude"])),
        "rollout_body_rate_error": float(np.nanmean(errors["body_rate"])), "diverged": divergence is not None,
        "num_switches": int(switch_time is not None) + int(promotion_time is not None), "false_alarm_count": int(alarm is not None and divergence is None),
        "false_switch_count": int(switch_time is not None and divergence is None), "unknown_regime": unknown,
        "new_models_created": int(promotion_time is not None), "samples_to_new_model_promotion": None if promotion_time is None else promotion_time - alarm,
        "promotion_gain": promotion_gain, "candidate_model_scores": candidate_scores, "monitor_reid_runtime": runtime,
        "reid_attempted": reid_attempted,
        "divergence_threshold_sensitivity": sensitivity,
        "causal_state_resynchronization_on_switch": switch_time is not None or promotion_time is not None,
    }, log
