from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from hashlib import sha256
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from experiments.neurobem_manifold_switch.data import NeuroBEMSource
from experiments.neurobem_manifold_switch.prism_adapter import FrozenPrismAdapter
from .core import (
    ABLATION_BLOCKS, direction_vector, effective_delta, frame_arrays,
    history_at_anchor, newest_state_jacobian, paired_growth_with_channels, rollout,
)


_CONTEXT: dict[str, object] = {}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _safe(value):
    if isinstance(value, dict): return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.generic): return value.item()
    return value


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def calibration_scales(trajectories) -> dict[str, float]:
    states = np.concatenate([frame_arrays(item.frame)[0] for item in trajectories])
    return {
        "velocity": float(np.median(np.std(states[:, :3], axis=0))),
        "attitude": 1.0,
        "body_rate": float(np.median(np.std(states[:, 7:10], axis=0))),
    }


def effective_scale(direction: np.ndarray, scales: dict[str, float], fraction: float) -> float:
    weights = np.r_[np.full(3, scales["velocity"]), np.full(3, scales["attitude"]), np.full(3, scales["body_rate"])]
    return max(float(np.linalg.norm(direction * weights)) * fraction, 1e-10)


def one_step_bounds(adapter, route, trajectories, cfg):
    blocks = {"velocity": [], "attitude": [], "body_rate": []}
    from experiments.neurobem_manifold_switch.metrics import state_errors
    from prism_benchmark.neurobem_literature import compose_quaternion_increment, TRACK_B_STATE_COLUMNS, normalize_quaternion
    for item in trajectories:
        state, _ = frame_arrays(item.frame)
        target, pred, _ = adapter.one_step(route, item.frame)
        # Route targets are increments: reconstruct next state from observed origins.
        origin = state[adapter.history - 1:-1]
        linear = np.c_[origin[:, :3] + pred[:, :3], np.zeros((len(pred), 4)), origin[:, 7:10] + pred[:, 3:6]]
        linear[:, 3:7] = compose_quaternion_increment(origin[:, 3:7], pred[:, 6:9])
        error = state_errors(state[adapter.history:], linear)
        for key in blocks: blocks[key].extend(error[key].tolist())
    q = float(cfg["reliability_error_quantile"]); m = float(cfg["reliability_error_multiplier"])
    return {key: float(np.quantile(value, q) * m) for key, value in blocks.items()}


def _task(task: tuple[int, str]):
    index, route = task; c = _CONTEXT; item = c["trajectories"][index]
    adapter = c["adapter"]; cfg = c["cfg"]; state, control = frame_arrays(item.frame)
    steps = min(int(cfg["max_rollout_steps"]), len(state) - adapter.history)
    baseline = rollout(adapter, route, state, control, steps, cfg["divergence_thresholds"], int(cfg["divergence_persistence"]))
    base_row = {"trajectory_id": item.trajectory_id, "route": route, "baseline_t_diverge": baseline.t_diverge,
        "baseline_diverged": baseline.t_diverge is not None, "one_step_mse": float(np.mean(np.square(np.subtract(*adapter.one_step(route, item.frame)[:2])))),
        "velocity_error": float(np.nanmean(baseline.errors["velocity"])), "attitude_error": float(np.nanmean(baseline.errors["attitude"])),
        "body_rate_error": float(np.nanmean(baseline.errors["body_rate"])), "position_error": None}
    resync = []
    for raw in cfg["resync_intervals"]:
        interval = None if raw == "infinity" else int(raw)
        value = baseline if interval is None else rollout(adapter, route, state, control, steps, cfg["divergence_thresholds"], int(cfg["divergence_persistence"]), resync_interval=interval)
        within = value.finite & (value.errors["velocity"] <= c["reliability_bounds"][route]["velocity"]) & (value.errors["attitude"] <= c["reliability_bounds"][route]["attitude"]) & (value.errors["body_rate"] <= c["reliability_bounds"][route]["body_rate"])
        resync.append({"trajectory_id": item.trajectory_id, "route": route, "resync_interval": raw,
            "diverged": value.t_diverge is not None, "t_diverge": value.t_diverge,
            "velocity_error": float(np.nanmean(value.errors["velocity"])), "attitude_error": float(np.nanmean(value.errors["attitude"])),
            "body_rate_error": float(np.nanmean(value.errors["body_rate"])), "maximum_velocity_error": float(np.nanmax(value.errors["velocity"])),
            "within_reliable_envelope_fraction": float(within.mean()), "runtime_seconds": value.elapsed})
    channels = []
    for name, blocks in ABLATION_BLOCKS.items():
        value = baseline if name == "full_recursive" else rollout(adapter, route, state, control, steps, cfg["divergence_thresholds"], int(cfg["divergence_persistence"]), recurse_blocks=blocks)
        channels.append({"trajectory_id": item.trajectory_id, "route": route, "channel_ablation": name,
            "diverged": value.t_diverge is not None, "t_diverge": value.t_diverge,
            "velocity_error": float(np.nanmean(value.errors["velocity"])), "attitude_error": float(np.nanmean(value.errors["attitude"])),
            "body_rate_error": float(np.nanmean(value.errors["body_rate"]))})
    perturbations, jacobians = [], []
    last_anchor = steps if baseline.t_diverge is None else min(steps, baseline.t_diverge + 1)
    anchors = list(range(0, last_anchor, int(cfg["anchor_stride"])))
    for anchor in anchors:
        history = history_at_anchor(adapter, route, state, control, anchor)
        if history is None: break
        remaining = min(int(cfg["perturbation_horizon"]), steps - anchor)
        if remaining <= 0: continue
        controls = control[anchor:anchor + adapter.history + remaining - 1]
        for direction_name in cfg["perturbation_directions"]:
            direction = direction_vector(direction_name, int(cfg["seed"]) + index * 97)
            for fraction in cfg["perturbation_scale_fractions"]:
                epsilon = effective_scale(direction, c["state_scales"], float(fraction))
                growth = paired_growth_with_channels(adapter, route, history, controls, direction, epsilon, remaining)
                perturbations.append({"trajectory_id": item.trajectory_id, "route": route, "anchor_step": anchor,
                    "direction": direction_name, "epsilon_fraction": fraction, "epsilon": epsilon,
                    "max_growth": float(np.nanmax(growth["total"])), "terminal_growth": float(growth["total"][-1]),
                    "max_velocity_growth": float(np.nanmax(growth["velocity"])), "max_attitude_growth": float(np.nanmax(growth["attitude"])),
                    "max_body_rate_growth": float(np.nanmax(growth["body_rate"])),
                    "finite_time_growth_rate": float(np.log(max(float(growth["total"][-1]), 1e-300)) / len(growth["total"])),
                    "growth_curve": json.dumps(_safe(growth["total"].tolist())), "velocity_growth_curve": json.dumps(_safe(growth["velocity"].tolist())),
                    "attitude_growth_curve": json.dumps(_safe(growth["attitude"].tolist())), "body_rate_growth_curve": json.dumps(_safe(growth["body_rate"].tolist())),
                    "baseline_t_diverge": baseline.t_diverge})
        # Jacobian scale validation uses two registered calibration-derived steps.
        for scale_index in cfg["jacobian_scale_indices"]:
            fraction = float(cfg["perturbation_scale_fractions"][scale_index])
            epsilon = max(min(c["state_scales"].values()) * fraction, 1e-10)
            matrix = newest_state_jacobian(adapter, route, history, controls, epsilon)
            jacobians.append({"trajectory_id": item.trajectory_id, "route": route, "anchor_step": anchor,
                "epsilon_fraction": fraction, "sigma_max_J": float(np.linalg.svd(matrix, compute_uv=False)[0]),
                "jacobian_frobenius": float(np.linalg.norm(matrix)), "baseline_t_diverge": baseline.t_diverge})
    return base_row, resync, channels, perturbations, jacobians


def _baseline_task(task: tuple[int, str]):
    index, route = task; c = _CONTEXT; item = c["trajectories"][index]
    adapter = c["adapter"]; cfg = c["cfg"]; state, control = frame_arrays(item.frame)
    steps = min(int(cfg["max_rollout_steps"]), len(state) - adapter.history)
    value = rollout(adapter, route, state, control, steps, cfg["divergence_thresholds"], int(cfg["divergence_persistence"]))
    return {"trajectory_id": item.trajectory_id, "route": route, "diverged": value.t_diverge is not None, "t_diverge": value.t_diverge,
        "velocity_error": float(np.nanmean(value.errors["velocity"])), "attitude_error": float(np.nanmean(value.errors["attitude"])),
        "body_rate_error": float(np.nanmean(value.errors["body_rate"]))}


def _component_task(task: tuple[int, str]):
    index, route = task; c = _CONTEXT; item = c["trajectories"][index]
    adapter = c["adapter"]; cfg = c["cfg"]; state, control = frame_arrays(item.frame)
    steps = min(int(cfg["max_rollout_steps"]), len(state) - adapter.history)
    value = rollout(adapter, route, state, control, steps, cfg["divergence_thresholds"], int(cfg["divergence_persistence"]))
    return {"trajectory_id": item.trajectory_id, "route": route, "diverged": value.t_diverge is not None,
        "t_diverge": value.t_diverge, "velocity_error": float(np.nanmean(value.errors["velocity"])),
        "attitude_error": float(np.nanmean(value.errors["attitude"])), "body_rate_error": float(np.nanmean(value.errors["body_rate"]))}


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=("calibration", "test"), required=True)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--release-root", type=Path, required=True); p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--frozen-adapter", type=Path, required=True); p.add_argument("--calibration-freeze", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args(argv); cfg = json.loads(args.config.read_text()); source = NeuroBEMSource(args.data_root, args.split_manifest, args.release_root)
    fit_names, calibration_names, split_audit = source.train_parent_holdout(0.75)
    adapter = FrozenPrismAdapter.load(args.frozen_adapter, "GLOBAL_PRISM_R2_TRAIN_ONLY")
    calibration_items = source.load("train", calibration_names)
    state_scales = calibration_scales(calibration_items)
    if args.stage == "calibration":
        reliability_bounds = {route: one_step_bounds(adapter, route, calibration_items, cfg) for route in cfg["routes"]}
        trajectories = calibration_items
    else:
        if args.calibration_freeze is None: raise ValueError("TEST_REQUIRES_CALIBRATION_FREEZE")
        freeze = json.loads(args.calibration_freeze.read_text()); reliability_bounds = freeze["reliability_bounds"]
        if freeze["config_sha256"] != _sha(args.config): raise RuntimeError("R3_CONFIG_HASH_MISMATCH")
        if freeze["adapter_sha256"] != _sha(args.frozen_adapter): raise RuntimeError("R3_ADAPTER_HASH_MISMATCH")
        state_scales = freeze["state_scales"]; trajectories = source.load("test")
    run = args.output_root / f"{cfg['protocol_id']}_{args.stage}_{time.strftime('%Y%m%dT%H%M%S')}"; run.mkdir(parents=True)
    metadata = {"stage": args.stage, "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        "config_sha256": _sha(args.config), "adapter_sha256": _sha(args.frozen_adapter), "split_manifest_sha256": _sha(args.split_manifest),
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(), "hostname": socket.gethostname(),
        "r2_data_contract": split_audit, "prism_predictor_modified": False, "test_accessed": args.stage == "test", "test_used_for_tuning": False,
        "position_channel_status": cfg["position_channel_status"], "jacobian_scope": "NEWEST_STATE_9X9_BLOCK_OF_HISTORY_DEPENDENT_MAP"}
    atomic_json(run / "metadata.json", metadata); atomic_json(run / "config.json", cfg)
    global _CONTEXT; _CONTEXT = {"trajectories": trajectories, "adapter": adapter, "cfg": cfg, "state_scales": state_scales, "reliability_bounds": reliability_bounds}
    tasks = [(i, route) for i in range(len(trajectories)) for route in cfg["routes"]]; workers = min(int(cfg["trajectory_workers"]), len(tasks))
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool: baseline_preflight = list(pool.imap(_baseline_task, tasks, chunksize=1))
    else: baseline_preflight = list(map(_baseline_task, tasks))
    baseline_preflight = pd.DataFrame(baseline_preflight); baseline_preflight.to_csv(run / "R2_BASELINE_REPRODUCTION.csv", index=False)
    reference = cfg["r2_baseline_reference"][args.stage]; preflight_mismatches = []
    for route, expected in reference.items():
        part = baseline_preflight[baseline_preflight.route == route]
        if int(part.diverged.sum()) != expected["divergence_count"]: preflight_mismatches.append(f"{route}:divergence_count")
        if abs(float(part.t_diverge.median()) - expected["median_t_diverge"]) > float(cfg["r2_baseline_median_tolerance_steps"]): preflight_mismatches.append(f"{route}:median_t_diverge")
    if preflight_mismatches:
        blocked = {"status": "BLOCKED_R2_BASELINE_REPRODUCTION_MISMATCH", "r2_baseline_reproduced": False,
            "r2_baseline_mismatches": preflight_mismatches, "test_accessed": args.stage == "test", "new_tracks_executed": False}
        atomic_json(run / "summary.json", blocked); print(json.dumps(blocked, indent=2)); return
    component_tasks = [(i, route) for i in range(len(trajectories)) for route in cfg["component_routes"]]
    if workers > 1:
        with mp.get_context("fork").Pool(min(workers, len(component_tasks))) as pool: component_rows = list(pool.imap(_component_task, component_tasks, chunksize=1))
    else: component_rows = list(map(_component_task, component_tasks))
    pd.DataFrame(component_rows).to_csv(run / "component_route_attribution.csv", index=False)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool: values = list(pool.imap(_task, tasks, chunksize=1))
    else: values = list(map(_task, tasks))
    baseline = pd.DataFrame([v[0] for v in values]); resync = pd.DataFrame(sum((v[1] for v in values), [])); channels = pd.DataFrame(sum((v[2] for v in values), []))
    perturb = pd.DataFrame(sum((v[3] for v in values), [])); jacobian = pd.DataFrame(sum((v[4] for v in values), []))
    if args.stage == "calibration":
        thresholds = {}
        for route, part in perturb.groupby("route"):
            # Teacher-observed scales are frozen; threshold is a robust upper tail
            # over the registered calibration perturbation audit.
            anchor_max = part.groupby(["trajectory_id", "anchor_step"]).max_growth.max().replace(np.inf, np.nan).dropna()
            thresholds[route] = float(np.quantile(anchor_max, float(cfg["expansion_quantile"])))
        freeze = {"status": "R3_CALIBRATION_FROZEN", "config_sha256": _sha(args.config), "adapter_sha256": _sha(args.frozen_adapter),
            "state_scales": state_scales, "reliability_bounds": reliability_bounds, "expansion_thresholds": thresholds,
            "calibration_trajectory_count": len(trajectories), "test_accessed": False, "test_used_for_tuning": False}
        atomic_json(run / "R3_CALIBRATION_FREEZE.json", freeze)
    else:
        thresholds = freeze["expansion_thresholds"]
    # Expansion event is earliest persistent anchor whose maximum registered
    # perturbation growth exceeds the route threshold.
    anchor = perturb.groupby(["trajectory_id", "route", "anchor_step"], as_index=False).max_growth.max()
    events = []
    for (trajectory_id, route), part in anchor.groupby(["trajectory_id", "route"]):
        part = part.sort_values("anchor_step"); run_length = 0; event = None
        for row in part.itertuples():
            run_length = run_length + 1 if row.max_growth > thresholds[route] else 0
            if run_length >= int(cfg["expansion_persistence"]): event = int(row.anchor_step - (int(cfg["expansion_persistence"]) - 1) * int(cfg["anchor_stride"])); break
        div = baseline[(baseline.trajectory_id == trajectory_id) & (baseline.route == route)].iloc[0].baseline_t_diverge
        events.append({"trajectory_id": trajectory_id, "route": route, "t_expansion": event, "baseline_t_diverge": div,
            "expansion_lead_time": None if event is None or pd.isna(div) else int(div - event), "positive_expansion_lead": event is not None and not pd.isna(div) and event < div})
    events = pd.DataFrame(events)
    for name, frame in (("baseline.csv", baseline), ("resynchronization.csv", resync), ("channel_attribution.csv", channels), ("perturbation_growth.csv", perturb), ("jacobian.csv", jacobian), ("expansion_events.csv", events)):
        frame.to_csv(run / name, index=False)
    reliable = {}
    for route, part in resync.groupby("route"):
        # First classify each trajectory using its within-trajectory probability,
        # then require the registered probability across held-out trajectories.
        per_trajectory = part.assign(
            trajectory_reliable=part.within_reliable_envelope_fraction >= float(cfg["reliability_probability_minimum"])
        ).groupby("resync_interval").trajectory_reliable.mean().to_dict()
        allowed = [n for n in cfg["resync_intervals"] if n != "infinity" and per_trajectory.get(str(n), per_trajectory.get(n, 0)) >= float(cfg["reliability_probability_minimum"])]
        reliable[route] = max(allowed) if allowed else None
    summary = {"status": "COMPLETED", "stage": args.stage, "routes": {}, "reliable_open_loop_horizon_steps": reliable,
        "expansion_thresholds": thresholds, "reliability_bounds": reliability_bounds, "parallelism": "LINUX_FORK_COW", "workers": workers,
        "test_accessed": args.stage == "test", "test_used_for_tuning": False, "formal_lyapunov_claim": False}
    for route, part in baseline.groupby("route"):
        event_part = events[events.route == route]
        summary["routes"][route] = {"divergence_count": int(part.baseline_diverged.sum()), "trajectory_count": len(part),
            "median_t_diverge": None if part.baseline_t_diverge.dropna().empty else float(part.baseline_t_diverge.median()),
            "positive_expansion_lead_fraction": float(event_part.positive_expansion_lead.mean()),
            "median_expansion_lead_time": None if event_part.expansion_lead_time.dropna().empty else float(event_part.expansion_lead_time.median())}
    reference = cfg["r2_baseline_reference"][args.stage]
    mismatches = []
    for route, expected in reference.items():
        actual = summary["routes"][route]
        if actual["divergence_count"] != expected["divergence_count"]:
            mismatches.append(f"{route}:divergence_count")
        if abs(actual["median_t_diverge"] - expected["median_t_diverge"]) > float(cfg["r2_baseline_median_tolerance_steps"]):
            mismatches.append(f"{route}:median_t_diverge")
    summary["r2_baseline_reproduced"] = not mismatches
    summary["r2_baseline_mismatches"] = mismatches
    if mismatches:
        summary["status"] = "BLOCKED_R2_BASELINE_REPRODUCTION_MISMATCH"
    atomic_json(run / "summary.json", summary); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
