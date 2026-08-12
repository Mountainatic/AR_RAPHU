"""Frozen runner for the prospective NeuroBEM multi-horizon PRISM audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .neurobem_data import sha256_file, write_json
from .neurobem_experiment import load_partition, records_from_json
from .neurobem_multihorizon import (
    ARM_ROUTE_NAMES,
    W0,
    W1,
    W2,
    bootstrap_contrast,
    collect_prediction_records,
    common_horizon_support_start,
    fit_final_horizon_contracts,
    grouped_oof_records,
    integrated_k_topology,
    materialize_horizon,
    metrics_for_records,
    run_a_arms,
    run_baselines,
    run_k_horizon,
    run_w_arms,
    runtime_memory_gib,
)


AXES = ("ROLL", "PITCH", "YAW", "BODY_Z")
ALL_ROUTES = (
    "K__W0__NO_A",
    "K__W1_GENERIC__NO_A",
    "K__W2_AERO__NO_A",
    "K__W0__A",
    "K__W1_GENERIC__A",
    "K__W2_AERO__A",
    "PERSISTENCE",
    "TARGET_ONLY_VAR_AR",
    "LINEAR_NARX",
)


def _load_json(path: Path) -> dict[str, object] | list[object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _runtime(repo_root: Path, started: float | None = None) -> dict[str, object]:
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": _git(repo_root, "status", "--porcelain"),
        "blas_thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "peak_ram_gib": runtime_memory_gib(),
        "wall_seconds": None if started is None else time.monotonic() - started,
    }


def _config_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "PRISM_INDUSTRIAL_BENCHMARK_V1"
        / "PRISM_V2_1_1_NEUROBEM_MULTI_HORIZON_W_PRIOR_AUDIT_PACKAGE"
        / "MULTIHORIZON_CONFIG_FROZEN.json"
    )


def _theory_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "PRISM_INDUSTRIAL_BENCHMARK_V1"
        / "PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_PACKAGE"
        / "reference"
        / "PRISM_Theory_v2_1_1_Implementation_Safe_Stagewise_Routed_Modular_Assembly_Theory_Only.md"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def _metric_rows(
    phase: str,
    support: str,
    horizon: int,
    horizon_ms: float,
    records: Sequence[dict[str, object]],
    routes: Iterable[str] = ALL_ROUTES,
    *,
    minimum_target_row: int | None = None,
    speed_range: tuple[float, float | None] | None = None,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for route in routes:
        metric = metrics_for_records(records, route, minimum_target_row=minimum_target_row, speed_range=speed_range)
        if metric["status"] != "COMPLETED":
            output.append({"phase": phase, "support": support, "horizon": horizon, "horizon_ms": horizon_ms, "route": route, "axis": "POOLED", "status": metric["status"], "rows": metric["rows"]})
            continue
        for index, axis in enumerate(AXES):
            output.append({
                "phase": phase,
                "support": support,
                "horizon": horizon,
                "horizon_ms": horizon_ms,
                "route": route,
                "axis": axis,
                "status": "COMPLETED",
                "rows": metric["rows"],
                "mse": metric["mse_by_axis"][index],
                "rmse": metric["rmse_by_axis"][index],
                "mae": metric["mae_by_axis"][index],
                "r2": metric["r2_by_axis"][index],
                "nrmse": metric["nrmse_by_axis"][index],
                "normalized_mse": metric["normalized_mse_by_axis"][index],
                "pooled_normalized_mse": metric["pooled_normalized_mse"],
            })
        output.append({
            "phase": phase,
            "support": support,
            "horizon": horizon,
            "horizon_ms": horizon_ms,
            "route": route,
            "axis": "POOLED",
            "status": "COMPLETED",
            "rows": metric["rows"],
            "pooled_normalized_mse": metric["pooled_normalized_mse"],
        })
    return output


def _pooled(rows: Sequence[Mapping[str, object]], phase: str, route: str, horizon: int, support: str) -> float:
    matches = [row for row in rows if row.get("phase") == phase and row.get("route") == route and int(row.get("horizon", -1)) == horizon and row.get("support") == support and row.get("axis") == "POOLED"]
    if len(matches) != 1:
        raise RuntimeError(f"POOLED_METRIC_BINDING_FAILURE:{phase}:{support}:h{horizon}:{route}")
    return float(matches[0]["pooled_normalized_mse"])


def _gain_rows(metric_rows: Sequence[Mapping[str, object]], phase: str, support: str, horizons: Sequence[int], milliseconds: Sequence[float]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    w_rows: list[dict[str, object]] = []
    a_rows: list[dict[str, object]] = []
    for horizon, ms in zip(horizons, milliseconds, strict=True):
        k = _pooled(metric_rows, phase, "K__W0__NO_A", horizon, support)
        generic = _pooled(metric_rows, phase, "K__W1_GENERIC__NO_A", horizon, support)
        aero = _pooled(metric_rows, phase, "K__W2_AERO__NO_A", horizon, support)
        w_rows.extend([
            {"phase": phase, "support": support, "horizon": horizon, "horizon_ms": ms, "gain": "G_GENERIC", "value": (k - generic) / k},
            {"phase": phase, "support": support, "horizon": horizon, "horizon_ms": ms, "gain": "G_AERO_PRIOR", "value": (generic - aero) / generic},
        ])
        for arm, no_a, with_a in (
            (W0, "K__W0__NO_A", "K__W0__A"),
            (W1, "K__W1_GENERIC__NO_A", "K__W1_GENERIC__A"),
            (W2, "K__W2_AERO__NO_A", "K__W2_AERO__A"),
        ):
            base = _pooled(metric_rows, phase, no_a, horizon, support)
            candidate = _pooled(metric_rows, phase, with_a, horizon, support)
            a_rows.append({"phase": phase, "support": support, "horizon": horizon, "horizon_ms": ms, "W_arm": arm, "gain": (base - candidate) / base})
    return w_rows, a_rows


def _support_audit(records: Sequence[dict[str, object]], horizon: int, phase: str) -> dict[str, object]:
    route_hashes: dict[str, str] = {}
    expected_rows = 0
    for route in ALL_ROUTES:
        digest = __import__("hashlib").sha256()
        rows = 0
        for item in records:
            if route not in item["routes"]:
                raise RuntimeError(f"MISSING_ROUTE:{route}")
            if len(item["routes"][route]) != len(item["sample_ids"]):
                raise RuntimeError(f"ROUTE_SAMPLE_COUNT_MISMATCH:{route}")
            for value in item["sample_ids"]:
                digest.update(bytes.fromhex(value))
            rows += len(item["sample_ids"])
        route_hashes[route] = digest.hexdigest()
        expected_rows = rows
    if len(set(route_hashes.values())) != 1:
        raise RuntimeError(f"ROUTE_SAMPLE_SUPPORT_MISMATCH:h{horizon}:{phase}")
    return {"phase": phase, "horizon": horizon, "rows": expected_rows, "support_sha256": next(iter(route_hashes.values())), "all_route_support_hashes_equal": True}


def stage0(repo_root: Path, data_root: Path, output_root: Path, prior_root: Path) -> None:
    started = time.monotonic()
    config_path = _config_path(repo_root)
    config = _load_json(config_path)
    prior_n0 = _load_json(prior_root / "N0" / "RESULT.json")
    prior_registry = _load_json(prior_root / "N1" / "SEGMENT_REGISTRY.json")
    if prior_n0["sources"]["processed_data.zip"]["sha256"] != config["source_and_split"]["source_archive_sha256"]:
        raise RuntimeError("R1_SOURCE_ARCHIVE_PROVENANCE_MISMATCH")
    records = records_from_json(prior_registry)
    counts = {partition: len({item.flight_id for item in records if item.partition == partition}) for partition in ("train", "validation", "test", "excluded_cadence")}
    expected = {"train": 62, "validation": 19, "test": 13, "excluded_cadence": 1}
    if counts != expected:
        raise RuntimeError(f"R1_SPLIT_REUSE_MISMATCH:{counts}")
    train = load_partition(records, data_root / "extracted", "train")
    validation = load_partition(records, data_root / "extracted", "validation")
    write_json(output_root / "N0" / "RESULT.json", {
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "evidence_class": config["evidence_class"],
        "prior_R1_test_access_disclosed": True,
        "historical_test_access_exists": True,
        "historical_test_access_experiment": config["historical_test_access_experiment"],
        "source_archive_sha256": config["source_and_split"]["source_archive_sha256"],
        "config_sha256": sha256_file(config_path),
        "canonical_theory_sha256": sha256_file(_theory_path(repo_root)),
        "split_parent_flights": counts,
        "development_segments_loaded": len(train) + len(validation),
        "test_numeric_values_accessed": False,
        "ood_accessed": False,
        "runtime": _runtime(repo_root, started),
    })
    write_json(output_root / "N1" / "SEGMENT_REGISTRY.json", prior_registry)
    write_json(output_root / "N1" / "SPLIT_MANIFEST.json", {
        "status": "PASS_REUSED_FROM_R1",
        "history_entity": "CONTIGUOUS_PROCESSED_SEGMENT_ID",
        "parent_group_disjoint": True,
        "cross_segment_history": False,
        "test_numeric_values_accessed": False,
    })
    write_json(output_root / "RUN_STATUS.json", {"status": "STAGE0_PASS", "test_accessed": False, "ood_accessed": False})


def development(repo_root: Path, data_root: Path, output_root: Path) -> None:
    started = time.monotonic()
    config_path = _config_path(repo_root)
    config = _load_json(config_path)
    n0 = _load_json(output_root / "N0" / "RESULT.json")
    if n0["status"] != "PASS" or n0["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("STAGE0_OR_CONFIG_PROVENANCE_FAILURE")
    records = records_from_json(_load_json(output_root / "N1" / "SEGMENT_REGISTRY.json"))
    train = load_partition(records, data_root / "extracted", "train")
    validation = load_partition(records, data_root / "extracted", "validation")
    horizons = [int(value) for value in config["targets"]["forecast_horizons_samples"]]
    milliseconds = [float(value) for value in config["targets"]["forecast_horizons_milliseconds"]]
    all_metric_rows: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    support_rows: list[dict[str, object]] = []
    k_rows: list[dict[str, object]] = []
    a_audit: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    final_contracts: dict[str, object] = {}
    common_start = common_horizon_support_start(config)
    for horizon, ms in zip(horizons, milliseconds, strict=True):
        k_result, route_folds, validation_frames, validation_k = run_k_horizon(train, validation, horizon, config)
        if k_result["status"] != "PASS":
            raise RuntimeError(f"K_PROTOCOL_OR_NUMERICAL_FAILURE:h{horizon}")
        w_result, arm_folds, validation_by_arm, _ = run_w_arms(route_folds, validation_frames, config)
        a_result, oof_by_arm, validation_output, _, _ = run_a_arms(arm_folds, validation_by_arm, horizon, config)
        baseline_result, val_baselines, oof_baselines, _, _ = run_baselines(route_folds, validation_frames, horizon, int(k_result["selected_history"]), config)
        oof_records = grouped_oof_records(oof_by_arm, oof_baselines, horizon)
        val_records = collect_prediction_records(validation_output, val_baselines, horizon)
        support_rows.extend([_support_audit(oof_records, horizon, "DEVELOPMENT_OOF"), _support_audit(val_records, horizon, "VALIDATION")])
        for phase, current in (("DEVELOPMENT_OOF", oof_records), ("VALIDATION", val_records)):
            all_metric_rows.extend(_metric_rows(phase, "NATIVE_HORIZON_SUPPORT", horizon, ms, current))
            all_metric_rows.extend(_metric_rows(phase, "COMMON_HORIZON_SUPPORT", horizon, ms, current, minimum_target_row=common_start))
        selections[str(horizon)] = {"K": k_result, "W": w_result, "A": a_result, "BASELINES": baseline_result}
        horizon_dir = output_root / "DEVELOPMENT" / f"HORIZON_{horizon}"
        write_json(horizon_dir / "RESULT.json", selections[str(horizon)])
        write_json(horizon_dir / "SUMMARY.json", {
            "status": "PASS",
            "horizon": horizon,
            "horizon_ms": ms,
            "selected_K_history": k_result["selected_history"],
            "selected_generic_W": w_result[W1]["final_selected_candidate"],
            "selected_aero_W": w_result[W2]["final_selected_candidate"],
            "selected_A_by_arm": {arm: a_result[arm]["final_selected_candidate"] for arm in (W0, W1, W2)},
            "K_input_gate_status": k_result["K_input_gate_status"],
            "physics_consistency_supported": k_result["physics_consistency_supported"],
            "test_accessed": False,
        })
        k_rows.append({"horizon": horizon, "horizon_ms": ms, "selected_history": k_result["selected_history"], "history_status": k_result["history_status"], "K_input_gate_status": k_result["K_input_gate_status"], "physics_consistency_supported": k_result["physics_consistency_supported"], "relative_improvement_vs_zero": k_result["relative_improvement_vs_zero"], "positive_fold_fraction": k_result["positive_fold_fraction"]})
        for arm in (W0, W1, W2):
            a_audit.append({"horizon": horizon, "horizon_ms": ms, "W_arm": arm, "selected_A": a_result[arm]["final_selected_candidate"], "selected_mature_ages": a_result[arm]["selected_mature_ages"], "actual_target_lags": a_result[arm]["selected_actual_target_lags"], "maturity_pass": a_result[arm]["maturity_pass"], "raw_input_used": a_result[arm]["raw_input_used"]})
        topology = integrated_k_topology(validation_k, int(k_result["selected_history"]))
        for motor in range(4):
            for axis in range(4):
                topology_rows.append({"phase": "DEVELOPMENT", "horizon": horizon, "horizon_ms": ms, "motor": motor + 1, "axis": AXES[axis], "integrated_coefficient": topology[motor, axis], "sign": int(np.sign(topology[motor, axis]))})
        final_contracts[str(horizon)] = fit_final_horizon_contracts(list(train) + list(validation), horizon, selections[str(horizon)], config)
        write_json(horizon_dir / "FINAL_REFIT_CONTRACTS.json", final_contracts[str(horizon)])
    _write_csv(output_root / "HORIZON_NATIVE_SUPPORT_METRICS.csv", [row for row in all_metric_rows if row["support"] == "NATIVE_HORIZON_SUPPORT"])
    _write_csv(output_root / "HORIZON_COMMON_SUPPORT_METRICS.csv", [row for row in all_metric_rows if row["support"] == "COMMON_HORIZON_SUPPORT"])
    _write_csv(output_root / "K_HISTORY_SELECTION.csv", k_rows)
    _write_csv(output_root / "A_MATURITY_AUDIT.csv", a_audit)
    _write_csv(output_root / "SAMPLE_SUPPORT_AUDIT.csv", support_rows)
    _write_csv(output_root / "K_TOPOLOGY_STABILITY.csv", topology_rows)
    freeze = {
        "status": "PASS_GLOBAL_DEVELOPMENT_FREEZE",
        "development_frozen": True,
        "protocol_id": config["protocol_id"],
        "formal_horizons": horizons,
        "formal_routes": list(config["formal_routes"]),
        "baseline_routes": list(config["baselines"]["routes"]),
        "horizon_contracts": final_contracts,
        "config_sha256": sha256_file(config_path),
        "canonical_theory_sha256": sha256_file(_theory_path(repo_root)),
        "source_archive_sha256": n0["source_archive_sha256"],
        "generating_commit": _git(repo_root, "rev-parse", "HEAD"),
        "historical_test_access_exists": True,
        "historical_test_access_experiment": config["historical_test_access_experiment"],
        "current_extension_type": "POST_LOCKBOX_PROSPECTIVE_EXTENSION",
        "current_extension_new_test_scores_seen_before_freeze": False,
        "current_extension_test_access_count": 1,
        "test_driven_reselection": False,
        "test_accessed": False,
        "ood_accessed": False,
        "test_access_authorized": True,
        "common_horizon_support_start": common_start,
        "runtime": _runtime(repo_root, started),
    }
    write_json(output_root / "GLOBAL_DEVELOPMENT_FREEZE.json", freeze)
    write_json(output_root / "RUN_STATUS.json", {"status": "GLOBAL_DEVELOPMENT_FROZEN", "stage": "GLOBAL_FREEZE", "test_accessed": False, "ood_accessed": False})


def _speed_rows(records_by_horizon: Mapping[int, Sequence[dict[str, object]]], config: Mapping[str, object], support_start: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    speed_rows: list[dict[str, object]] = []
    high_rows: list[dict[str, object]] = []
    ms_map = dict(zip(config["targets"]["forecast_horizons_samples"], config["targets"]["forecast_horizons_milliseconds"], strict=True))
    for horizon, records in records_by_horizon.items():
        ms = float(ms_map[horizon])
        for low, high in config["evaluation"]["speed_bins_m_per_s"]:
            label = f"[{low},{'inf' if high is None else high})"
            for row in _metric_rows("LOCKED_TEST", "SPEED_BIN", horizon, ms, records, config["formal_routes"], speed_range=(float(low), None if high is None else float(high))):
                row["speed_bin"] = label
                speed_rows.append(row)
        threshold = float(config["evaluation"]["high_speed_threshold_m_per_s"])
        for row in _metric_rows("LOCKED_TEST", "LOCKED_HIGH_SPEED_CHALLENGE_SUBSET", horizon, ms, records, config["formal_routes"], speed_range=(threshold, None)):
            row["speed_threshold_m_per_s"] = threshold
            row["ood_interpretation"] = False
            high_rows.append(row)
    return speed_rows, high_rows


def _bootstrap_rows(records_by_horizon: Mapping[int, Sequence[dict[str, object]]], config: Mapping[str, object], cluster_key: str) -> list[dict[str, object]]:
    primary = cluster_key == "flight_id"
    replicates = int(config["statistics"]["primary_bootstrap_replicates"] if primary else config["statistics"]["secondary_bootstrap_replicates"])
    seed = int(config["statistics"]["primary_seed"] if primary else config["statistics"]["secondary_seed"])
    output: list[dict[str, object]] = []
    for horizon, records in records_by_horizon.items():
        for index, (candidate, baseline) in enumerate(config["statistics"]["contrasts"]):
            result = bootstrap_contrast(records, candidate, baseline, cluster_key=cluster_key, replicates=replicates, seed=seed + horizon * 100 + index)
            result["horizon"] = horizon
            output.append(result)
    return output


def _figures(output_root: Path, metrics: Sequence[Mapping[str, object]], w_gain: Sequence[Mapping[str, object]], a_gain: Sequence[Mapping[str, object]], speed: Sequence[Mapping[str, object]], topology: Sequence[Mapping[str, object]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_root / "FIGURES"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figure_dir / f"{name}.png", dpi=180)
        plt.savefig(figure_dir / f"{name}.pdf")
        plt.close()

    for name, routes in (
        ("FIGURE_1_NO_A_HORIZON_NMSE", ("K__W0__NO_A", "K__W1_GENERIC__NO_A", "K__W2_AERO__NO_A")),
        ("FIGURE_2_WITH_A_HORIZON_NMSE", ("K__W0__A", "K__W1_GENERIC__A", "K__W2_AERO__A")),
    ):
        for route in routes:
            rows = sorted([row for row in metrics if row["phase"] == "LOCKED_TEST" and row["support"] == "COMMON_HORIZON_SUPPORT" and row["axis"] == "POOLED" and row["route"] == route], key=lambda row: row["horizon"])
            plt.plot([row["horizon_ms"] for row in rows], [row["pooled_normalized_mse"] for row in rows], marker="o", label=route)
        plt.xlabel("Forecast horizon (ms)"); plt.ylabel("Pooled normalized MSE"); plt.legend(fontsize=7); save(name)
    for gain in ("G_GENERIC", "G_AERO_PRIOR"):
        rows = sorted([row for row in w_gain if row["phase"] == "LOCKED_TEST" and row["support"] == "COMMON_HORIZON_SUPPORT" and row["gain"] == gain], key=lambda row: row["horizon"])
        plt.plot([row["horizon_ms"] for row in rows], [row["value"] for row in rows], marker="o", label=gain)
    for arm in (W0, W1, W2):
        rows = sorted([row for row in a_gain if row["phase"] == "LOCKED_TEST" and row["support"] == "COMMON_HORIZON_SUPPORT" and row["W_arm"] == arm], key=lambda row: row["horizon"])
        plt.plot([row["horizon_ms"] for row in rows], [row["gain"] for row in rows], marker="o", label=f"G_A:{arm}")
    plt.axhline(0, color="black", lw=.7); plt.xlabel("Forecast horizon (ms)"); plt.ylabel("Relative gain"); plt.legend(fontsize=7); save("FIGURE_3_GAIN_CURVES")
    for axis in AXES:
        rows = sorted([row for row in metrics if row["phase"] == "LOCKED_TEST" and row["support"] == "COMMON_HORIZON_SUPPORT" and row["axis"] == axis and row["route"] == "K__W2_AERO__A"], key=lambda row: row["horizon"])
        plt.plot([row["horizon_ms"] for row in rows], [row["rmse"] for row in rows], marker="o", label=axis)
    plt.xlabel("Forecast horizon (ms)"); plt.ylabel("RMSE (K+W2+A)"); plt.legend(); save("FIGURE_4_PER_AXIS_RMSE")
    pooled_speed = [row for row in speed if row["axis"] == "POOLED"]
    for horizon in sorted({int(row["horizon"]) for row in pooled_speed}):
        rows = [row for row in pooled_speed if int(row["horizon"]) == horizon]
        by_bin = {row["speed_bin"]: row for row in rows if row["route"] == "K__W2_AERO__NO_A"}
        generic = {row["speed_bin"]: row for row in rows if row["route"] == "K__W1_GENERIC__NO_A"}
        bins = sorted(set(by_bin).intersection(generic))
        values = [(generic[key]["pooled_normalized_mse"] - by_bin[key]["pooled_normalized_mse"]) / generic[key]["pooled_normalized_mse"] for key in bins]
        plt.plot(bins, values, marker="o", label=f"h={horizon}")
    plt.xlabel("Speed bin (m/s)"); plt.ylabel("Aero-prior incremental gain"); plt.xticks(rotation=20); plt.legend(fontsize=7); save("FIGURE_5_W_GAIN_BY_SPEED_BIN")
    dev = [row for row in topology if row["phase"] == "DEVELOPMENT"]
    horizons = sorted({int(row["horizon"]) for row in dev})
    matrix = np.array([[next(float(row["integrated_coefficient"]) for row in dev if int(row["horizon"]) == h and int(row["motor"]) == motor and row["axis"] == axis) for motor in range(1, 5) for axis in AXES] for h in horizons])
    scale = np.maximum(np.max(np.abs(matrix), axis=1, keepdims=True), np.finfo(float).eps)
    plt.imshow(matrix / scale, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.yticks(range(len(horizons)), [f"h={h}" for h in horizons]); plt.xticks(range(16), [f"M{m}-{a}" for m in range(1, 5) for a in AXES], rotation=90, fontsize=6); plt.colorbar(label="Row-normalized integrated coefficient"); save("FIGURE_6_K_TOPOLOGY")


def _report(output_root: Path, config: Mapping[str, object], metrics: Sequence[Mapping[str, object]], w_gain: Sequence[Mapping[str, object]], a_gain: Sequence[Mapping[str, object]], selections: Mapping[str, object]) -> dict[str, object]:
    horizons = [int(value) for value in config["targets"]["forecast_horizons_samples"]]
    table: dict[str, object] = {}
    for horizon in horizons:
        table[str(horizon)] = {route: _pooled(metrics, "LOCKED_TEST", route, horizon, "COMMON_HORIZON_SUPPORT") for route in ALL_ROUTES}
    summary = {
        "status": "COMPLETED",
        "evidence_class": "POST_LOCKBOX_PROSPECTIVE_EXTENSION",
        "historical_R1_test_access_disclosed": True,
        "horizons_completed": horizons,
        "selected_K_history_by_horizon": {key: value["K"]["selected_history"] for key, value in selections.items()},
        "selected_generic_W_by_horizon": {key: value["W"][W1]["final_selected_candidate"] for key, value in selections.items()},
        "selected_aero_W_by_horizon": {key: value["W"][W2]["final_selected_candidate"] for key, value in selections.items()},
        "selected_A_by_horizon_and_W_arm": {key: {arm: value["A"][arm]["final_selected_candidate"] for arm in (W0, W1, W2)} for key, value in selections.items()},
        "test_common_support_pooled_normalized_mse": table,
        "generic_and_aero_gain": list(w_gain),
        "A_gain": list(a_gain),
        "retained_failures": ["No route, horizon, or negative ablation was removed after locked-test access."],
        "interpretation_limits": ["Predictive evidence is not identification of true airflow, drag, wind, vortex state, or physical poles.", "W2 is an aerodynamic-context extension and is not canonical PRISM v2.1.1 W."],
        "test_accessed": True,
        "ood_accessed": False,
    }
    write_json(output_root / "NEUROBEM_MULTI_HORIZON_RESULT_SUMMARY.json", summary)
    lines = [
        "# NeuroBEM Multi-Horizon × Wiener-Prior Final Report",
        "",
        "Status: **COMPLETED**. This is a post-lockbox prospective extension; the prior one-step R1 test access is explicitly disclosed.",
        "",
        "All six horizons, three Wiener arms, three A comparisons, and all baselines were frozen before the single extension test access. The locked high-speed challenge is not labeled OOD.",
        "",
        "## Direct answers",
        "",
        "1. A-gain persistence or decay is reported without post-test model changes in `A_HORIZON_GAIN.csv`.",
        "2. Canonical latent-only W evidence is isolated by W1 versus W0 in `W_HORIZON_GAIN.csv`.",
        "3. The additional registered aerodynamic-context evidence is isolated by W2 versus W1.",
        "4. Its speed localization is reported in `SPEED_BIN_METRICS.csv` and `HIGH_SPEED_CHALLENGE_METRICS.csv`.",
        "5. K rotor-topology stability is based only on frozen development contracts and is reported in `K_TOPOLOGY_STABILITY.csv`.",
        "6. Boundary K selections are retained in `K_HISTORY_SELECTION.csv`; the grid was not expanded.",
        "7. No-A route capability is shown in Figure 1 and the metrics tables.",
        "8. With-A route capability is shown in Figure 2 and the metrics tables.",
        "9. Persistence, target-only VAR/AR, and linear NARX remain visible at every horizon.",
        "10. Test performance, W gains, and A gains are predictive evidence only.",
        "11. Pre-registered K integrated-sign agreement is physics-consistency evidence, not full physical identification.",
        "12. None of the results identifies true airflow, drag, wind, vortex state, or physical poles.",
        "",
        "## Common-support pooled normalized MSE",
        "",
        "| h (samples) | " + " | ".join(ALL_ROUTES) + " |",
        "|---:" + "|---:" * len(ALL_ROUTES) + "|",
    ]
    for horizon in horizons:
        lines.append(f"| {horizon} | " + " | ".join(f"{table[str(horizon)][route]:.6g}" for route in ALL_ROUTES) + " |")
    lines.extend(["", "## Interpretation boundary", "", "W1 uses only frozen K latent predictions. W2 is explicitly an aerodynamic-context extension. A uses only mature residual states with actual target lag h+age. Negative and unstable results are retained."])
    (output_root / "NEUROBEM_MULTI_HORIZON_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def final(repo_root: Path, data_root: Path, output_root: Path) -> None:
    started = time.monotonic()
    config_path = _config_path(repo_root)
    config = _load_json(config_path)
    freeze_path = output_root / "GLOBAL_DEVELOPMENT_FREEZE.json"
    freeze = _load_json(freeze_path)
    if freeze["status"] != "PASS_GLOBAL_DEVELOPMENT_FREEZE" or not freeze["test_access_authorized"]:
        raise RuntimeError("TEST_ACCESS_NOT_AUTHORIZED")
    audit_path = output_root / "TEST_ACCESS_AUDIT.json"
    if audit_path.exists():
        raise RuntimeError("CURRENT_EXTENSION_LOCKED_TEST_ALREADY_ACCESSED")
    current_commit = _git(repo_root, "rev-parse", "HEAD")
    if freeze["generating_commit"] != current_commit:
        raise RuntimeError("CODE_COMMIT_CHANGED_AFTER_GLOBAL_FREEZE")
    if _git(repo_root, "status", "--porcelain"):
        raise RuntimeError("DIRTY_CODE_WORKTREE_BEFORE_EXTENSION_LOCKBOX")
    if freeze["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("CONFIG_CHANGED_AFTER_GLOBAL_FREEZE")
    if _load_json(output_root / "RUN_STATUS.json") != {"status": "GLOBAL_DEVELOPMENT_FROZEN", "stage": "GLOBAL_FREEZE", "test_accessed": False, "ood_accessed": False}:
        raise RuntimeError("PRELOCKBOX_RUN_STATUS_MISMATCH")
    write_json(audit_path, {
        "status": "ACCESS_STARTED",
        "historical_test_access_exists": True,
        "historical_test_access_experiment": config["historical_test_access_experiment"],
        "current_extension_type": "POST_LOCKBOX_PROSPECTIVE_EXTENSION",
        "current_extension_new_test_scores_seen_before_freeze": False,
        "current_extension_test_access_count": 1,
        "test_driven_reselection": False,
        "development_freeze_sha256": sha256_file(freeze_path),
        "code_commit": current_commit,
        "config_sha256": sha256_file(config_path),
    })
    records = records_from_json(_load_json(output_root / "N1" / "SEGMENT_REGISTRY.json"))
    test = load_partition(records, data_root / "extracted", "test", allow_locked_test=True)
    horizons = [int(value) for value in config["targets"]["forecast_horizons_samples"]]
    milliseconds = [float(value) for value in config["targets"]["forecast_horizons_milliseconds"]]
    records_by_horizon: dict[int, Sequence[dict[str, object]]] = {}
    metric_rows: list[dict[str, object]] = []
    support_rows = pd.read_csv(output_root / "SAMPLE_SUPPORT_AUDIT.csv").to_dict("records")
    common_start = int(freeze["common_horizon_support_start"])
    for horizon, ms in zip(horizons, milliseconds, strict=True):
        current = materialize_horizon(test, freeze["horizon_contracts"][str(horizon)], config)
        records_by_horizon[horizon] = current
        support_rows.append(_support_audit(current, horizon, "LOCKED_TEST"))
        metric_rows.extend(_metric_rows("LOCKED_TEST", "NATIVE_HORIZON_SUPPORT", horizon, ms, current))
        metric_rows.extend(_metric_rows("LOCKED_TEST", "COMMON_HORIZON_SUPPORT", horizon, ms, current, minimum_target_row=common_start))
    development_metrics = pd.concat([pd.read_csv(output_root / "HORIZON_NATIVE_SUPPORT_METRICS.csv"), pd.read_csv(output_root / "HORIZON_COMMON_SUPPORT_METRICS.csv")], ignore_index=True).to_dict("records")
    all_metrics = development_metrics + metric_rows
    _write_csv(output_root / "HORIZON_FINAL_METRICS.csv", all_metrics)
    _write_csv(output_root / "HORIZON_NATIVE_SUPPORT_METRICS.csv", [row for row in all_metrics if row["support"] == "NATIVE_HORIZON_SUPPORT"])
    _write_csv(output_root / "HORIZON_COMMON_SUPPORT_METRICS.csv", [row for row in all_metrics if row["support"] == "COMMON_HORIZON_SUPPORT"])
    _write_csv(output_root / "SAMPLE_SUPPORT_AUDIT.csv", support_rows)
    speed_rows, high_rows = _speed_rows(records_by_horizon, config, common_start)
    _write_csv(output_root / "SPEED_BIN_METRICS.csv", speed_rows)
    _write_csv(output_root / "HIGH_SPEED_CHALLENGE_METRICS.csv", high_rows)
    w_gain: list[dict[str, object]] = []
    a_gain: list[dict[str, object]] = []
    for phase in ("DEVELOPMENT_OOF", "VALIDATION", "LOCKED_TEST"):
        for support in ("NATIVE_HORIZON_SUPPORT", "COMMON_HORIZON_SUPPORT"):
            w, a = _gain_rows(all_metrics, phase, support, horizons, milliseconds)
            w_gain.extend(w); a_gain.extend(a)
    _write_csv(output_root / "W_HORIZON_GAIN.csv", w_gain)
    _write_csv(output_root / "A_HORIZON_GAIN.csv", a_gain)
    _write_csv(output_root / "W_PRIOR_ABLATION.csv", [row for row in w_gain if row["phase"] == "LOCKED_TEST"])
    _write_csv(output_root / "BASELINE_METRICS.csv", [row for row in all_metrics if row["route"] in config["baselines"]["routes"]])
    parent_bootstrap = _bootstrap_rows(records_by_horizon, config, "flight_id")
    segment_bootstrap = _bootstrap_rows(records_by_horizon, config, "segment_id")
    _write_csv(output_root / "BOOTSTRAP_PARENT_FLIGHT.csv", parent_bootstrap)
    _write_csv(output_root / "BOOTSTRAP_SEGMENT_SENSITIVITY.csv", segment_bootstrap)
    selections = {str(h): _load_json(output_root / "DEVELOPMENT" / f"HORIZON_{h}" / "RESULT.json") for h in horizons}
    topology_rows = pd.read_csv(output_root / "K_TOPOLOGY_STABILITY.csv").to_dict("records")
    expected = {"ROLL": [-1, -1, 1, 1], "PITCH": [1, -1, 1, -1], "BODY_Z": [1, 1, 1, 1]}
    for row in topology_rows:
        if row["axis"] in expected:
            row["expected_sign"] = expected[row["axis"]][int(row["motor"]) - 1]
            row["expected_sign_agreement"] = int(row["sign"]) == int(row["expected_sign"])
        else:
            row["expected_sign"] = "DESCRIPTIVE_ONLY"
            row["expected_sign_agreement"] = "NOT_APPLICABLE"
    _write_csv(output_root / "K_TOPOLOGY_STABILITY.csv", topology_rows)
    _figures(output_root, all_metrics, w_gain, a_gain, speed_rows, topology_rows)
    summary = _report(output_root, config, all_metrics, w_gain, a_gain, selections)
    write_json(output_root / "FINAL" / "RESULT.json", {**summary, "runtime": _runtime(repo_root, started)})
    write_json(audit_path, {
        "status": "ACCESS_COMPLETED_ONCE",
        "historical_test_access_exists": True,
        "historical_test_access_experiment": config["historical_test_access_experiment"],
        "current_extension_type": "POST_LOCKBOX_PROSPECTIVE_EXTENSION",
        "current_extension_new_test_scores_seen_before_freeze": False,
        "current_extension_test_access_count": 1,
        "test_driven_reselection": False,
        "development_freeze_sha256": sha256_file(freeze_path),
        "code_commit": current_commit,
        "config_sha256": sha256_file(config_path),
        "test_accessed": True,
        "ood_accessed": False,
        "locked_high_speed_challenge_is_ood": False,
        "model_selection_after_access": False,
    })
    write_json(output_root / "RUN_STATUS.json", {"status": "COMPLETED", "stage": "FINAL", "test_accessed": True, "ood_accessed": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stage0", "development", "final"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path)
    args = parser.parse_args()
    if args.stage == "stage0":
        if args.prior_root is None:
            parser.error("--prior-root is required for stage0")
        stage0(args.repo_root, args.data_root, args.output_root, args.prior_root)
    elif args.stage == "development":
        development(args.repo_root, args.data_root, args.output_root)
    else:
        final(args.repo_root, args.data_root, args.output_root)


if __name__ == "__main__":
    main()
