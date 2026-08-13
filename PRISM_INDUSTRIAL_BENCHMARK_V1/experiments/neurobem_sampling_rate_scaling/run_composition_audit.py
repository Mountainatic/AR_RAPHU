from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from experiments.neurobem_manifold_switch.data import NeuroBEMSource
from experiments.neurobem_manifold_switch.prism_adapter import FrozenPrismAdapter
from experiments.neurobem_recursive_stability.core import frame_arrays
from prism_benchmark.neurobem_literature import read_neurobem_csv

from .composition_consistency import (
    HORIZONS_MS, RATE_KEYS, RATES, ROUTES, composition_endpoints,
    composition_failure_class, exact_time_index, one_step_order_supported,
    registered_anchor_times, state_norms,
)
from .resampling import resample_track_b
from .run_experiment import _safe, atomic_json, load_adapter, split_train_without_test_reads


_CONTEXT: dict[str, object] = {}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_frozen_adapters(freeze: dict) -> dict[int, FrozenPrismAdapter]:
    output = {}
    for rate, key in RATE_KEYS.items():
        item = freeze["rate_adapters"][key]
        path = Path(item["adapter_path"])
        if _sha(path) != item["adapter_sha256"]:
            raise RuntimeError("COMPOSITION_ADAPTER_HASH_MISMATCH")
        output[rate] = load_adapter(path, int(item["history_steps"]), f"COMPOSITION_{rate}HZ")
    return output


def audit_frozen_r3_r4_reproduction_gate(sampling_freeze_path: Path, sampling_freeze: dict) -> dict:
    if not sampling_freeze.get("r3_100hz_reproduced"):
        raise RuntimeError("COMPOSITION_R3_R4_REPRODUCTION_GATE_FAILED")
    reproduction = sampling_freeze_path.parent / "R3_100HZ_REPRODUCTION.csv"
    resynchronization = sampling_freeze_path.parent / "R3_100HZ_RESYNCHRONIZATION_REPRODUCTION.csv"
    if not reproduction.is_file() or not resynchronization.is_file():
        raise RuntimeError("COMPOSITION_R3_R4_REPRODUCTION_ARTIFACT_MISSING")
    frame = pd.read_csv(reproduction)
    expected = {"PF_KCW": (56, 284.5), "J_KCW": (51, 417.0)}
    actual = {}
    for route, (count, median) in expected.items():
        part = frame[frame.route == route]
        value = (int(part.diverged.sum()), float(part.t_diverge_steps.median()))
        if value != (count, median):
            raise RuntimeError("COMPOSITION_R3_R4_REPRODUCTION_VALUE_MISMATCH")
        actual[route] = {"divergence_count": value[0], "median_t_diverge_steps": value[1]}
    return {
        "status": "PASS",
        "sampling_freeze_r3_flag": True,
        "reproduction_sha256": _sha(reproduction),
        "resynchronization_sha256": _sha(resynchronization),
        "routes": actual,
    }


def _one_step_mse(adapter: FrozenPrismAdapter, route: str, frame: pd.DataFrame) -> float:
    target, prediction, _ = adapter.one_step(route, frame)
    return float(np.mean(np.square(target - prediction)))


def _trajectory_task(task: tuple[str, str]) -> tuple[list[dict], list[dict]]:
    name, route = task
    source: NeuroBEMSource = _CONTEXT["source"]
    adapters: dict[int, FrozenPrismAdapter] = _CONTEXT["adapters"]
    cfg: dict = _CONTEXT["cfg"]
    native = read_neurobem_csv(source.path(name))
    frames = {rate: resample_track_b(native, rate) for rate in RATES}
    one_step = [{"trajectory_id": name, "route": route, "sampling_rate_hz": rate,
                 "one_step_mse": _one_step_mse(adapters[rate], route, frames[rate])} for rate in RATES]
    anchors = registered_anchor_times(
        native, history_ms=int(cfg["history_duration_ms"]),
        maximum_horizon_ms=max(cfg["physical_horizons_ms"]),
        spacing_ms=int(cfg["anchor_spacing_ms"]),
        maximum_anchors=int(cfg["maximum_anchors_per_trajectory"]),
    )
    native_state, _ = frame_arrays(native)
    rows = []
    for anchor in anchors:
        initial = native_state[exact_time_index(native, anchor, 400)]
        endpoints = {
            rate: composition_endpoints(
                adapters[rate], route, frames[rate], anchor,
                tuple(cfg["physical_horizons_ms"]), initial, rate,
            ) for rate in RATES
        }
        for horizon in cfg["physical_horizons_ms"]:
            target_time = anchor + horizon / 1000.0
            target = native_state[exact_time_index(native, target_time, 400)]
            reference = endpoints[100][horizon].prediction
            for rate in RATES:
                endpoint = endpoints[rate][horizon]
                gt = state_norms(target, endpoint.prediction)
                defect = state_norms(reference, endpoint.prediction) if rate != 100 else {
                    "full_state": 0.0, "velocity": 0.0, "attitude": 0.0, "body_rate": 0.0,
                }
                rows.append({
                    "trajectory_id": name, "route": route,
                    "anchor_time_seconds": anchor, "horizon_ms": horizon,
                    "sampling_rate_hz": rate, "steps": endpoint.steps,
                    "finite": endpoint.finite,
                    **{f"gt_{key}_error": value for key, value in gt.items()},
                    **{f"defect_vs_100_{key}": value for key, value in defect.items()},
                })
    return rows, one_step


def aggregate(rows: pd.DataFrame, one_step: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics = [
        "gt_full_state_error", "gt_velocity_error", "gt_attitude_error", "gt_body_rate_error",
        "defect_vs_100_full_state", "defect_vs_100_velocity", "defect_vs_100_attitude", "defect_vs_100_body_rate",
    ]
    grouped = rows.groupby(["route", "horizon_ms", "sampling_rate_hz"])
    values = grouped[metrics].median().add_suffix("_median")
    q90 = grouped[metrics].quantile(.9).add_suffix("_q90")
    finite = grouped.finite.mean().rename("finite_fraction")
    summary = pd.concat((values, q90, finite), axis=1).reset_index()
    one = one_step.groupby(["route", "sampling_rate_hz"], as_index=False).one_step_mse.mean()
    decisions = {}
    for route in ROUTES:
        rates = one[one.route == route].set_index("sampling_rate_hz").one_step_mse.to_dict()
        h100 = summary[(summary.route == route) & (summary.horizon_ms == 100)].set_index("sampling_rate_hz").gt_full_state_error_median.to_dict()
        decisions[route] = {
            "one_step_order_supported": one_step_order_supported(rates),
            "composition_100ms_class": composition_failure_class(h100),
            "one_step_mse": rates,
            "composition_100ms_gt_full_state_error": h100,
        }
    classes = {value["composition_100ms_class"] for value in decisions.values()}
    if classes == {"COMPOSITION_CONSISTENCY_FAILURE_SUPPORTED"}:
        overall = "COMPOSITION_CONSISTENCY_FAILURE_SUPPORTED"
    elif classes == {"COMPOSITION_CONSISTENCY_FAILURE_NOT_SUPPORTED"}:
        overall = "COMPOSITION_CONSISTENCY_FAILURE_NOT_SUPPORTED"
    else:
        overall = "COMPOSITION_CONSISTENCY_MIXED"
    return summary, one, {"overall_decision": overall, "routes": decisions}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("calibration", "test"), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--sampling-freeze", type=Path, required=True)
    p.add_argument("--composition-freeze", type=Path)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--release-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    sampling_freeze = json.loads(args.sampling_freeze.read_text())
    reproduction_gate = audit_frozen_r3_r4_reproduction_gate(args.sampling_freeze, sampling_freeze)
    adapters = _load_frozen_adapters(sampling_freeze)
    source = NeuroBEMSource(args.data_root, args.split_manifest, args.release_root)
    _, calibration_names, split_audit = split_train_without_test_reads(source, json.loads((args.config.parent / "calibration.yaml").read_text()))
    if args.stage == "calibration":
        names = calibration_names
    else:
        if args.composition_freeze is None:
            raise RuntimeError("COMPOSITION_TEST_REQUIRES_FREEZE")
        frozen = json.loads(args.composition_freeze.read_text())
        if frozen["config_sha256"] != _sha(args.config) or frozen["sampling_freeze_sha256"] != _sha(args.sampling_freeze):
            raise RuntimeError("COMPOSITION_FREEZE_HASH_MISMATCH")
        names = source.names("test")
    run = args.output_root / f"{cfg['protocol_id']}_{args.stage}_{time.strftime('%Y%m%dT%H%M%S')}"
    run.mkdir(parents=True)
    global _CONTEXT
    _CONTEXT = {"source": source, "adapters": adapters, "cfg": cfg}
    tasks = [(name, route) for name in names for route in ROUTES]
    workers = min(int(cfg["workers"]), len(tasks))
    with mp.get_context("fork").Pool(workers) as pool:
        output = list(pool.imap(_trajectory_task, tasks, chunksize=1))
    _CONTEXT = {}
    detailed = pd.DataFrame(sum((value[0] for value in output), []))
    one_step = pd.DataFrame(sum((value[1] for value in output), []))
    summary, one_aggregate, decision = aggregate(detailed, one_step)
    detailed.to_csv(run / "COMPOSITION_ENDPOINTS.csv", index=False)
    summary.to_csv(run / "COMPOSITION_AGGREGATE.csv", index=False)
    one_aggregate.to_csv(run / "ONE_STEP_AGGREGATE.csv", index=False)
    metadata = {
        "status": "COMPLETED", "stage": args.stage,
        "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        "config_sha256": _sha(args.config), "sampling_freeze_sha256": _sha(args.sampling_freeze),
        "trajectory_count": len(names), "anchor_count": int(detailed[["trajectory_id", "anchor_time_seconds"]].drop_duplicates().shape[0]),
        "workers": workers, "blas_threads_per_worker": 1,
        "test_accessed": args.stage == "test", "test_used_for_tuning": False,
        "model_retrained": False, "prism_core_modified": False, "stabilization_added": False,
        "split_audit": split_audit, "r3_r4_reproduction_gate": reproduction_gate, **decision,
    }
    atomic_json(run / "COMPOSITION_SUMMARY.json", metadata)
    if args.stage == "calibration":
        atomic_json(run / "COMPOSITION_CALIBRATION_FREEZE.json", {
            "status": "COMPOSITION_CALIBRATION_FROZEN", "config_sha256": _sha(args.config),
            "sampling_freeze_sha256": _sha(args.sampling_freeze), "physical_horizons_ms": cfg["physical_horizons_ms"],
            "adapter_sha256": {str(rate): sampling_freeze["rate_adapters"][RATE_KEYS[rate]]["adapter_sha256"] for rate in RATES},
            "calibration_decision": decision, "test_accessed": False,
        })
    print(json.dumps(_safe(metadata), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
