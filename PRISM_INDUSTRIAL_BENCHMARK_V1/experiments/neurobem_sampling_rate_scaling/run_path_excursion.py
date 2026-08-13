from __future__ import annotations

import argparse
from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from experiments.neurobem_manifold_switch.data import NeuroBEMSource
from experiments.neurobem_recursive_stability.core import frame_arrays
from prism_benchmark.neurobem_literature import read_neurobem_csv

from .composition_consistency import RATE_KEYS, RATES, ROUTES, exact_time_index, registered_anchor_times
from .path_excursion import CHANNELS, excursion_metrics, replay_path, transient_order_class
from .resampling import resample_track_b
from .run_composition_audit import _load_frozen_adapters, audit_frozen_r3_r4_reproduction_gate
from .run_experiment import atomic_json


_CONTEXT: dict[str, object] = {}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _task(task: tuple[str, str]) -> list[dict]:
    name, route = task
    source: NeuroBEMSource = _CONTEXT["source"]
    adapters = _CONTEXT["adapters"]
    cfg = _CONTEXT["cfg"]
    bounds = _CONTEXT["bounds"]
    native = read_neurobem_csv(source.path(name))
    native_state, _ = frame_arrays(native)
    frames = {rate: resample_track_b(native, rate) for rate in RATES}
    anchors = registered_anchor_times(
        native, history_ms=int(cfg["history_duration_ms"]), maximum_horizon_ms=max(cfg["physical_horizons_ms"]),
        spacing_ms=int(cfg["anchor_spacing_ms"]), maximum_anchors=int(cfg["maximum_anchors_per_trajectory"]),
    )
    rows = []
    for anchor in anchors:
        initial = native_state[exact_time_index(native, anchor, 400)]
        for rate in RATES:
            path = replay_path(adapters[rate], route, frames[rate], native, anchor, max(cfg["physical_horizons_ms"]), initial, rate)
            route_bounds = bounds[RATE_KEYS[rate]][route]
            for horizon in cfg["physical_horizons_ms"]:
                steps = int(round(horizon * rate / 1000.0))
                for channel in CHANNELS:
                    rows.append({"trajectory_id": name, "route": route, "sampling_rate_hz": rate,
                                 "anchor_time_seconds": anchor, "horizon_ms": horizon, "channel": channel,
                                 "frozen_velocity_bound": route_bounds["velocity"],
                                 "frozen_attitude_bound": route_bounds["attitude"],
                                 "frozen_body_rate_bound": route_bounds["body_rate"],
                                 **excursion_metrics(path, route_bounds, channel, steps, rate)})
    return rows


def summarize(detailed: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    grouped = detailed.groupby(["route", "channel", "horizon_ms", "sampling_rate_hz"])
    summary = grouped.agg(
        terminal_median=("terminal_error", "median"), max_path_median=("max_path_error", "median"),
        rms_path_median=("rms_path_error", "median"), q90_max_path=("max_path_error", lambda x: x.quantile(.9)),
        first_cross_median_ms=("first_cross_ms", "median"), fraction_ever_crossing=("ever_crossing", "mean"),
        fraction_inside_median=("fraction_inside_envelope", "median"), crossing_count_median=("envelope_crossing_count", "median"),
        reentry_count_median=("reentry_count", "median"), max_over_terminal_median=("max_over_terminal", "median"),
    ).reset_index()
    evidence = []
    for route, channel in (("PF_KCW", "velocity"), ("J_KCW", "attitude"), ("J_KCW", "body_rate")):
        for horizon in (100, 200):
            part = summary[(summary.route == route) & (summary.channel == channel) & (summary.horizon_ms == horizon)].set_index("sampling_rate_hz")
            maxima = part.max_path_median.to_dict()
            evidence.append({"route": route, "channel": channel, "horizon_ms": horizon,
                             "max_path_class": transient_order_class(maxima),
                             "max_path": maxima,
                             "emax_400_over_100": float(maxima[400] / maxima[100]),
                             "rms_400_over_100": float(part.loc[400, "rms_path_median"] / part.loc[100, "rms_path_median"]),
                             "crossing_rate_difference_400_minus_100": float(part.loc[400, "fraction_ever_crossing"] - part.loc[100, "fraction_ever_crossing"])})
    classes = {row["max_path_class"] for row in evidence}
    if classes == {"HIGHER_RATE_TRANSIENT_PATH_EXCURSION_SUPPORTED"}:
        overall = "HIGHER_RATE_TRANSIENT_PATH_EXCURSION_SUPPORTED"
    elif classes == {"HIGHER_RATE_TRANSIENT_PATH_EXCURSION_NOT_SUPPORTED"}:
        overall = "HIGHER_RATE_TRANSIENT_PATH_EXCURSION_NOT_SUPPORTED"
    else:
        overall = "HIGHER_RATE_TRANSIENT_PATH_EXCURSION_MIXED"
    return summary, {"overall_decision": overall, "registered_evidence": evidence}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--sampling-freeze", type=Path, required=True)
    p.add_argument("--composition-summary", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--release-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text()); freeze = json.loads(args.sampling_freeze.read_text())
    composition = json.loads(args.composition_summary.read_text())
    if composition.get("status") != "COMPLETED" or not composition.get("test_accessed"):
        raise RuntimeError("PATH_REPLAY_REQUIRES_COMPLETED_COMPOSITION_TEST")
    gate = audit_frozen_r3_r4_reproduction_gate(args.sampling_freeze, freeze)
    adapters = _load_frozen_adapters(freeze)
    source = NeuroBEMSource(args.data_root, args.split_manifest, args.release_root)
    names = source.names("test")
    run = args.output_root / f"{cfg['protocol_id']}_{time.strftime('%Y%m%dT%H%M%S')}"; run.mkdir(parents=True)
    global _CONTEXT
    _CONTEXT = {"source": source, "adapters": adapters, "cfg": cfg, "bounds": freeze["reliability_bounds"]}
    tasks = [(name, route) for name in names for route in ROUTES]
    with mp.get_context("fork").Pool(min(int(cfg["workers"]), len(tasks))) as pool:
        detailed = pd.DataFrame(sum(pool.imap(_task, tasks, chunksize=1), []))
    _CONTEXT = {}
    summary, decision = summarize(detailed)
    detailed.to_csv(run / "PATH_EXCURSION_DETAILED.csv", index=False)
    summary.to_csv(run / "PATH_EXCURSION_AGGREGATE.csv", index=False)
    metadata = {"status": "COMPLETED_DETERMINISTIC_REPRODUCTION_REPLAY", "git_commit": subprocess.check_output(("git","rev-parse","HEAD"),text=True).strip(),
                "config_sha256": _sha(args.config), "sampling_freeze_sha256": _sha(args.sampling_freeze),
                "composition_summary_sha256": _sha(args.composition_summary), "trajectory_count": len(names), "workers": min(int(cfg["workers"]), len(tasks)),
                "existing_logs_sufficient": False, "deterministic_replay_performed": True,
                "new_test_decision_access": False, "test_used_for_tuning": False, "model_retrained": False,
                "threshold_changed": False, "prism_core_modified": False, "stabilization_added": False,
                "r3_r4_reproduction_gate": gate, **decision}
    atomic_json(run / "PATH_EXCURSION_SUMMARY.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
