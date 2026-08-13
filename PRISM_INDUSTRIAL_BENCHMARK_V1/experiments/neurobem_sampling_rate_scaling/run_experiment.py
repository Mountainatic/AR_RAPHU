from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from experiments.neurobem_manifold_switch.data import NeuroBEMSource
from experiments.neurobem_manifold_switch.metrics import state_errors
from experiments.neurobem_manifold_switch.prism_adapter import FrozenPrismAdapter, fit_local_adapter
from experiments.neurobem_recursive_stability.core import ABLATION_BLOCKS, frame_arrays, rollout
from prism_benchmark.neurobem_literature import (
    LiteratureTrajectory,
    TRACK_B_STATE_COLUMNS,
    compose_quaternion_increment,
    read_neurobem_csv,
    route_contract_from_json,
)

from .history_scaling import (
    condition_manifest,
    primary_horizon_candidates,
    registered_conditions,
    resync_intervals,
    unique_evaluation_conditions,
)
from .resampling import audit_resampling, resample_track_b


_CONTEXT: dict[str, object] = {}


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def split_train_without_test_reads(source: NeuroBEMSource, cfg: dict) -> tuple[list[str], list[str], dict[str, object]]:
    """Reproduce R2 parent holdout without opening any test trajectory.

    R2 found one released train/test content duplicate before this experiment.
    Its frozen train-side identity is now explicitly excluded by name.
    """
    names = source.names("train")
    parents = sorted({name.rsplit("_seg_", 1)[0] for name in names})
    cut = int(math.ceil(len(parents) * float(cfg["train_parent_fit_fraction"])))
    fit_parents, calibration_parents = set(parents[:cut]), set(parents[cut:])
    duplicate_excluded = set(cfg["excluded_train_test_duplicate_names"])
    non400_excluded = set(cfg["excluded_non400hz_train_names"])
    excluded = duplicate_excluded | non400_excluded
    clean = [name for name in names if name not in excluded]
    if len(clean) + len(excluded) != len(names) or not excluded.issubset(names):
        raise RuntimeError("FROZEN_TRAIN_DUPLICATE_EXCLUSION_MISMATCH")
    fit = [name for name in clean if name.rsplit("_seg_", 1)[0] in fit_parents]
    calibration = [name for name in clean if name.rsplit("_seg_", 1)[0] in calibration_parents]
    audit = {
        "contract": "CHRONOLOGICAL_PARENT_FLIGHT_TRAIN_FIT_CALIBRATION_R2_NO_TEST_READ",
        "fit_fraction": float(cfg["train_parent_fit_fraction"]),
        "parent_count": len(parents),
        "fit_parent_count": len(fit_parents),
        "calibration_parent_count": len(calibration_parents),
        "fit_segments": len(fit),
        "calibration_segments": len(calibration),
        "excluded_train_test_duplicate_names": sorted(duplicate_excluded),
        "excluded_non400hz_train_names": sorted(non400_excluded),
        "sampling_scaling_fit_support": cfg["sampling_scaling_fit_support"],
        "test_trajectory_opened": False,
    }
    if (len(fit), len(calibration)) != (171, 60):
        raise RuntimeError("R2_TRAIN_CALIBRATION_SUPPORT_COUNT_MISMATCH")
    return fit, calibration, audit


def load_rate_trajectories(source: NeuroBEMSource, partition: str, names: list[str], rate: int) -> tuple[list[LiteratureTrajectory], list[dict[str, object]]]:
    values, audits = [], []
    for name in names:
        # NeuroBEMSource.load is intentionally the historical 100-Hz loader.
        # Scaling must instead start from each isolated native 400-Hz CSV.
        native_frame = read_neurobem_csv(source.path(name))
        sampled = resample_track_b(native_frame, rate)
        values.append(LiteratureTrajectory(name, partition, sampled))
        audits.append(asdict(audit_resampling(native_frame, sampled, rate)))
    return values, audits


def load_adapter(path: Path, history: int, model_id: str) -> FrozenPrismAdapter:
    raw = json.loads(path.read_text())
    contracts = {key: route_contract_from_json(value) for key, value in raw.items()}
    for route in contracts.values():
        if route.velocity_contract.history != history or route.attitude_contract.history != history:
            raise RuntimeError("RATE_ADAPTER_HISTORY_CONTRACT_MISMATCH")
    return FrozenPrismAdapter(contracts, model_id, history)


def _fit_condition(payload: tuple[dict, str, str, str, str, str]) -> dict[str, object]:
    cfg, condition_json, data_root, split_manifest, release_root, output_dir = payload
    condition = json.loads(condition_json)
    source = NeuroBEMSource(Path(data_root), Path(split_manifest), Path(release_root))
    fit_names, _, split_audit = split_train_without_test_reads(source, cfg)
    trajectories, audits = load_rate_trajectories(source, "train", fit_names, int(condition["sampling_rate_hz"]))
    adapter = fit_local_adapter(
        trajectories,
        f"RATE_{condition['sampling_rate_hz']}HZ_HISTORY_{condition['history_steps']}",
        cfg["w_family"],
        int(cfg["row_cap_per_trajectory"]),
        cfg["ridge_grid"],
        float(cfg["max_condition"]),
        float(cfg["max_kkt"]),
        int(condition["history_steps"]),
    )
    path = Path(output_dir) / f"{condition['evaluation_key']}.json"
    adapter.save(path)
    return {
        "evaluation_key": condition["evaluation_key"],
        "sampling_rate_hz": condition["sampling_rate_hz"],
        "history_steps": condition["history_steps"],
        "adapter_path": str(path),
        "adapter_sha256": _sha(path),
        "fit_segments": split_audit["fit_segments"],
        "resampling_operator": audits[0]["operator"],
        "test_accessed": False,
    }


def fit_rate_adapters(args, cfg: dict, run: Path) -> dict[str, dict[str, object]]:
    adapter_dir = run / "rate_adapters"
    adapter_dir.mkdir(exist_ok=True)
    conditions = unique_evaluation_conditions()
    output: dict[str, dict[str, object]] = {}
    pending = conditions
    payloads = [
        (
            cfg,
            json.dumps(asdict(value)),
            str(args.data_root),
            str(args.split_manifest),
            str(args.release_root),
            str(adapter_dir),
        )
        for value in pending
    ]
    workers = min(int(cfg["model_fit_workers"]), len(payloads))
    if workers > 1:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("RATE_MODEL_PARALLELISM_REQUIRES_LINUX_FORK")
        with mp.get_context("fork").Pool(workers) as pool:
            rows = list(pool.imap(_fit_condition, payloads, chunksize=1))
    else:
        rows = list(map(_fit_condition, payloads))
    output.update({str(row["evaluation_key"]): row for row in rows})
    return output


def one_step_bounds(adapter: FrozenPrismAdapter, route: str, trajectories: list[LiteratureTrajectory], cfg: dict) -> dict[str, float]:
    blocks = {"velocity": [], "attitude": [], "body_rate": []}
    for item in trajectories:
        state, _ = frame_arrays(item.frame)
        target, prediction, _ = adapter.one_step(route, item.frame)
        origin = state[adapter.history - 1:-1]
        reconstructed = np.c_[
            origin[:, :3] + prediction[:, :3],
            np.zeros((len(prediction), 4)),
            origin[:, 7:10] + prediction[:, 3:6],
        ]
        reconstructed[:, 3:7] = compose_quaternion_increment(origin[:, 3:7], prediction[:, 6:9])
        error = state_errors(state[adapter.history:], reconstructed)
        for key in blocks:
            blocks[key].extend(error[key].tolist())
    quantile = float(cfg["reliability_error_quantile"])
    multiplier = float(cfg["reliability_error_multiplier"])
    return {key: float(np.quantile(value, quantile) * multiplier) for key, value in blocks.items()}


def _condition_task(task: tuple[str, int, str]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    key, trajectory_index, route = task
    context = _CONTEXT
    condition = context["conditions"][key]
    item = context["datasets"][key][trajectory_index]
    adapter = context["adapters"][key]
    cfg = context["cfg"]
    bounds = context["reliability_bounds"][key][route]
    rate = int(condition["sampling_rate_hz"])
    history = int(condition["history_steps"])
    state, control = frame_arrays(item.frame)
    steps = min(int(round(float(cfg["physical_rollout_duration_seconds"]) * rate)), len(state) - history)
    baseline = rollout(adapter, route, state, control, steps, cfg["divergence_thresholds"], int(cfg["divergence_persistence"]))
    target, prediction, _ = adapter.one_step(route, item.frame)
    baseline_row = {
        "evaluation_key": key,
        "sampling_rate_hz": rate,
        "dt_seconds": 1.0 / rate,
        "history_steps": history,
        "history_duration_ms": 1000.0 * history / rate,
        "trajectory_id": item.trajectory_id,
        "route": route,
        "one_step_error": float(np.mean(np.square(target - prediction))),
        "diverged": baseline.t_diverge is not None,
        "t_diverge_steps": baseline.t_diverge,
        "t_diverge_ms": None if baseline.t_diverge is None else 1000.0 * baseline.t_diverge / rate,
        "velocity_error": float(np.nanmean(baseline.errors["velocity"])),
        "attitude_error": float(np.nanmean(baseline.errors["attitude"])),
        "body_rate_error": float(np.nanmean(baseline.errors["body_rate"])),
    }
    resync_rows = []
    for raw_interval in resync_intervals(rate):
        interval = None if raw_interval == "infinity" else int(raw_interval)
        result = baseline if interval is None else rollout(
            adapter,
            route,
            state,
            control,
            steps,
            cfg["divergence_thresholds"],
            int(cfg["divergence_persistence"]),
            resync_interval=interval,
        )
        within = (
            result.finite
            & (result.errors["velocity"] <= bounds["velocity"])
            & (result.errors["attitude"] <= bounds["attitude"])
            & (result.errors["body_rate"] <= bounds["body_rate"])
        )
        resync_rows.append({
            "evaluation_key": key,
            "sampling_rate_hz": rate,
            "dt_seconds": 1.0 / rate,
            "history_steps": history,
            "history_duration_ms": 1000.0 * history / rate,
            "trajectory_id": item.trajectory_id,
            "route": route,
            "resync_interval_steps": raw_interval,
            "resync_interval_ms": None if interval is None else 1000.0 * interval / rate,
            "diverged": result.t_diverge is not None,
            "t_diverge_steps": result.t_diverge,
            "t_diverge_ms": None if result.t_diverge is None else 1000.0 * result.t_diverge / rate,
            "velocity_error": float(np.nanmean(result.errors["velocity"])),
            "attitude_error": float(np.nanmean(result.errors["attitude"])),
            "body_rate_error": float(np.nanmean(result.errors["body_rate"])),
            "within_reliable_envelope_fraction": float(within.mean()),
            "trajectory_reliable": bool(within.mean() >= float(cfg["reliability_probability_minimum"])),
            "runtime_seconds": result.elapsed,
        })
    channel_rows = []
    for name, blocks in ABLATION_BLOCKS.items():
        result = baseline if name == "full_recursive" else rollout(
            adapter,
            route,
            state,
            control,
            steps,
            cfg["divergence_thresholds"],
            int(cfg["divergence_persistence"]),
            recurse_blocks=blocks,
        )
        channel_rows.append({
            "evaluation_key": key,
            "sampling_rate_hz": rate,
            "history_steps": history,
            "trajectory_id": item.trajectory_id,
            "route": route,
            "channel_ablation": name,
            "diverged": result.t_diverge is not None,
            "t_diverge_steps": result.t_diverge,
            "t_diverge_ms": None if result.t_diverge is None else 1000.0 * result.t_diverge / rate,
            "velocity_error": float(np.nanmean(result.errors["velocity"])),
            "attitude_error": float(np.nanmean(result.errors["attitude"])),
            "body_rate_error": float(np.nanmean(result.errors["body_rate"])),
        })
    return baseline_row, resync_rows, channel_rows


def run_conditions(keys: list[str], datasets, adapters, conditions, bounds, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global _CONTEXT
    _CONTEXT = {
        "datasets": datasets,
        "adapters": adapters,
        "conditions": conditions,
        "reliability_bounds": bounds,
        "cfg": cfg,
    }
    tasks = [(key, index, route) for key in keys for index in range(len(datasets[key])) for route in cfg["routes"]]
    workers = min(int(cfg["trajectory_workers"]), len(tasks))
    if workers > 1:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("RATE_EVALUATION_PARALLELISM_REQUIRES_LINUX_FORK")
        with mp.get_context("fork").Pool(workers) as pool:
            values = list(pool.imap(_condition_task, tasks, chunksize=1))
    else:
        values = list(map(_condition_task, tasks))
    _CONTEXT = {}
    baseline = pd.DataFrame([value[0] for value in values])
    resync = pd.DataFrame(sum((value[1] for value in values), []))
    channels = pd.DataFrame(sum((value[2] for value in values), []))
    return baseline, resync, channels


def reliable_horizons(resync: pd.DataFrame, cfg: dict) -> dict[str, dict[str, int | None]]:
    output: dict[str, dict[str, int | None]] = {}
    for (key, route), part in resync.groupby(["evaluation_key", "route"]):
        rate = int(part.sampling_rate_hz.iloc[0])
        probability = part.groupby("resync_interval_steps").trajectory_reliable.mean().to_dict()
        allowed = [
            candidate
            for candidate in primary_horizon_candidates(rate)
            if probability.get(str(candidate), probability.get(candidate, 0.0)) >= float(cfg["reliability_probability_minimum"])
        ]
        output.setdefault(key, {})[route] = max(allowed) if allowed else None
    return output


def expand_history_modes(frame: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for condition in registered_conditions():
        part = frame[frame.evaluation_key == condition.evaluation_key].copy()
        part["condition_id"] = condition.condition_id
        part["history_mode"] = condition.history_mode
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def aggregate_results(baseline: pd.DataFrame, resync: pd.DataFrame, channels: pd.DataFrame, reliable: dict, cfg: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for (key, route), part in baseline.groupby(["evaluation_key", "route"]):
        rate = int(part.sampling_rate_hz.iloc[0])
        horizon = reliable[key][route]
        rows.append({
            "evaluation_key": key,
            "sampling_rate_hz": rate,
            "history_steps": int(part.history_steps.iloc[0]),
            "history_duration_ms": float(part.history_duration_ms.iloc[0]),
            "route": route,
            "trajectory_count": len(part),
            "one_step_error_mean": float(part.one_step_error.mean()),
            "divergence_rate": float(part.diverged.mean()),
            "median_t_diverge_steps": None if part.t_diverge_steps.dropna().empty else float(part.t_diverge_steps.median()),
            "median_t_diverge_ms": None if part.t_diverge_ms.dropna().empty else float(part.t_diverge_ms.median()),
            "reliable_horizon_steps": horizon,
            "reliable_horizon_ms": None if horizon is None else 1000.0 * horizon / rate,
        })
    aggregate = pd.DataFrame(rows)
    base = aggregate[aggregate.evaluation_key == "hz100_h20"].set_index("route")
    scaling = []
    for row in aggregate.itertuples():
        reference = base.loc[row.route]
        scaling.append({
            "evaluation_key": row.evaluation_key,
            "route": row.route,
            "sampling_rate_hz": row.sampling_rate_hz,
            "R_step": None if row.reliable_horizon_steps is None else row.reliable_horizon_steps / reference.reliable_horizon_steps,
            "R_time": None if row.reliable_horizon_ms is None else row.reliable_horizon_ms / reference.reliable_horizon_ms,
        })
    summary = {
        "status": "COMPLETED",
        "aggregate": rows,
        "scaling_diagnostics": scaling,
        "test_used_for_tuning": False,
        "stabilization_added": False,
        "formal_stability_claim": False,
    }
    return aggregate, summary


def r3_reproduction_check(stage: str, baseline: pd.DataFrame, reliable: dict, cfg: dict) -> list[str]:
    mismatches = []
    reference = cfg["r3_100hz_reference"][stage]
    for route, expected in reference.items():
        part = baseline[(baseline.evaluation_key == "hz100_h20") & (baseline.route == route)]
        if int(part.diverged.sum()) != int(expected["divergence_count"]):
            mismatches.append(f"{route}:divergence_count")
        actual_median = float(part.t_diverge_steps.median())
        if abs(actual_median - float(expected["median_t_diverge"])) > float(cfg["r3_baseline_median_tolerance_steps"]):
            mismatches.append(f"{route}:median_t_diverge")
        if reliable["hz100_h20"][route] != int(expected["reliable_horizon_steps"]):
            mismatches.append(f"{route}:reliable_horizon")
    return mismatches


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibration", "test"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--frozen-100hz-adapter", type=Path, required=True)
    parser.add_argument("--calibration-freeze", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    source = NeuroBEMSource(args.data_root, args.split_manifest, args.release_root)
    fit_names, calibration_names, split_audit = split_train_without_test_reads(source, cfg)
    run = args.output_root / f"{cfg['protocol_id']}_{args.stage}_{time.strftime('%Y%m%dT%H%M%S')}"
    run.mkdir(parents=True)
    if args.stage == "calibration":
        adapter_manifest = fit_rate_adapters(args, cfg, run)
    else:
        if args.calibration_freeze is None:
            raise ValueError("TEST_REQUIRES_RATE_CALIBRATION_FREEZE")
        freeze = json.loads(args.calibration_freeze.read_text())
        if freeze["config_sha256"] != _sha(args.config):
            raise RuntimeError("RATE_CONFIG_HASH_MISMATCH")
        if freeze["track0_adapter_sha256"] != _sha(args.frozen_100hz_adapter):
            raise RuntimeError("TRACK0_ADAPTER_HASH_MISMATCH")
        adapter_manifest = freeze["rate_adapters"]
        for value in adapter_manifest.values():
            if _sha(Path(value["adapter_path"])) != value["adapter_sha256"]:
                raise RuntimeError("RATE_ADAPTER_HASH_MISMATCH")
    condition_values = {value.evaluation_key: asdict(value) for value in unique_evaluation_conditions()}
    adapters = {
        key: load_adapter(Path(adapter_manifest[key]["adapter_path"]), int(condition["history_steps"]), key)
        for key, condition in condition_values.items()
    }
    frozen_track0_adapter = FrozenPrismAdapter.load(args.frozen_100hz_adapter, "FROZEN_R3_TRACK0")
    if args.stage == "calibration":
        partition, names = "train", calibration_names
    else:
        # This is the only point at which formal test identities are requested.
        partition, names = "test", source.names("test")
    datasets, resampling_audits = {}, {}
    for key, condition in condition_values.items():
        datasets[key], audits = load_rate_trajectories(source, partition, names, int(condition["sampling_rate_hz"]))
        resampling_audits[key] = {
            "operator": audits[0]["operator"],
            "trajectory_count": len(audits),
            "maximum_dt_error_seconds": max(value["maximum_dt_error_seconds"] for value in audits),
            "missing_registered_fields": any(value["missing_registered_fields"] for value in audits),
        }
    if args.stage == "calibration":
        reliability_bounds = {
            key: {route: one_step_bounds(adapters[key], route, datasets[key], cfg) for route in cfg["routes"]}
            for key in condition_values
        }
        track0_reliability_bounds = {
            route: one_step_bounds(frozen_track0_adapter, route, datasets["hz100_h20"], cfg)
            for route in cfg["routes"]
        }
    else:
        reliability_bounds = freeze["reliability_bounds"]
        track0_reliability_bounds = freeze["track0_reliability_bounds"]
    metadata = {
        "stage": args.stage,
        "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        "config_sha256": _sha(args.config),
        "split_manifest_sha256": _sha(args.split_manifest),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "split_audit": split_audit,
        "condition_manifest": condition_manifest(),
        "resampling_audit": resampling_audits,
        "trajectory_workers": int(cfg["trajectory_workers"]),
        "model_fit_workers": int(cfg["model_fit_workers"]),
        "blas_threads_per_worker": 1,
        "test_accessed": args.stage == "test",
        "test_used_for_tuning": False,
        "prism_core_modified": False,
    }
    atomic_json(run / "metadata.json", metadata)
    atomic_json(run / "config.json", cfg)
    atomic_json(run / "rate_adapter_manifest.json", adapter_manifest)
    # Frozen-adapter Track 0 executes first. No newly fitted scaling condition
    # begins unless the R3 baseline and reliability horizon reproduce exactly.
    baseline_track0, resync_track0, _ = run_conditions(
        ["hz100_h20"], datasets, {"hz100_h20": frozen_track0_adapter}, condition_values,
        {"hz100_h20": track0_reliability_bounds}, cfg,
    )
    reliable_track0 = reliable_horizons(resync_track0, cfg)
    mismatches = r3_reproduction_check(args.stage, baseline_track0, reliable_track0, cfg)
    baseline_track0.to_csv(run / "R3_100HZ_REPRODUCTION.csv", index=False)
    resync_track0.to_csv(run / "R3_100HZ_RESYNCHRONIZATION_REPRODUCTION.csv", index=False)
    if mismatches:
        blocked = {
            "status": "BLOCKED_R3_100HZ_REPRODUCTION_MISMATCH",
            "mismatches": mismatches,
            "higher_rates_executed": False,
            "test_accessed": args.stage == "test",
        }
        atomic_json(run / "summary.json", blocked)
        print(json.dumps(blocked, indent=2))
        return
    all_keys = list(condition_values)
    baseline, resync, channels = run_conditions(
        all_keys, datasets, adapters, condition_values, reliability_bounds, cfg,
    )
    reliable = reliable_horizons(resync, cfg)
    aggregate, summary = aggregate_results(baseline, resync, channels, reliable, cfg)
    summary.update({
        "stage": args.stage,
        "r3_100hz_reproduced": True,
        "test_accessed": args.stage == "test",
        "rate_adapters": adapter_manifest,
        "reliability_bounds": reliability_bounds,
    })
    baseline_modes = expand_history_modes(baseline)
    resync_modes = expand_history_modes(resync)
    channels_modes = expand_history_modes(channels)
    aggregate_modes = expand_history_modes(aggregate)
    baseline_modes.to_csv(run / "SAMPLING_RATE_BASELINE.csv", index=False)
    resync_modes.to_csv(run / "SAMPLING_RATE_RESYNCHRONIZATION.csv", index=False)
    channels_modes.to_csv(run / "SAMPLING_RATE_CHANNEL_ATTRIBUTION.csv", index=False)
    aggregate_modes.to_csv(run / "SAMPLING_RATE_AGGREGATE.csv", index=False)
    atomic_json(run / "summary.json", summary)
    if args.stage == "calibration":
        freeze = {
            "status": "SAMPLING_RATE_CALIBRATION_FROZEN",
            "config_sha256": _sha(args.config),
            "rate_adapters": adapter_manifest,
            "track0_adapter_sha256": _sha(args.frozen_100hz_adapter),
            "condition_manifest": condition_manifest(),
            "reliability_bounds": reliability_bounds,
            "track0_reliability_bounds": track0_reliability_bounds,
            "calibration_aggregate": aggregate_modes.to_dict("records"),
            "r3_100hz_reproduced": True,
            "test_accessed": False,
            "test_used_for_tuning": False,
        }
        atomic_json(run / "SAMPLING_RATE_CALIBRATION_FREEZE.json", freeze)
    print(json.dumps(_safe(summary), indent=2))


if __name__ == "__main__":
    main()
