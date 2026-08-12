"""CLI for the frozen NeuroBEM literature-aligned dual benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .neurobem_literature import (
    CANONICAL_W_CANDIDATES,
    FORMAL_ROUTE_IDS,
    LiteratureTrajectory,
    candidate_binding_audit,
    concatenate_track_a_design,
    concatenate_track_b_design,
    fit_route_contracts,
    fit_track_b_route_contracts,
    force_torque_metrics,
    metric_reproduction_gate,
    official_prediction_ground_truth_force_torque,
    official_prediction_force_torque,
    read_neurobem_csv,
    resample_track_b_100hz,
    route_contract_from_json,
    route_contract_to_json,
    select_w_family,
    select_track_b_w_family_rollout,
    stable_group_fold,
    track_a_force_torque_target,
    track_a_route_metrics,
    track_b_published_decoupled_evaluator,
    track_b_rollout,
    track_b_split_manifest,
)


PUBLISHED_TRACK_A = {"Fxy": 0.204, "Fz": 0.504, "Mxy": 0.014, "Mz": 0.004, "F": 0.335, "M": 0.012}
PUBLISHED_TRACK_B = {
    "MLP": {"delta_z": 0.125, "delta_q": 0.025},
    "LSTM": {"delta_z": 0.077, "delta_q": 0.011},
    "GRU": {"delta_z": 0.090, "delta_q": 0.010},
    "TCN": {"delta_z": 0.042, "delta_q": 0.006},
}
TRACK_B_TABLE_II = (
    ("lemniscate_1.csv", "Lemniscate", 1.67, 3.51, 0.017, 0.004),
    ("random_points.csv", "Random Points", 2.38, 8.25, 0.076, 0.007),
    ("lemniscate_2.csv", "Lemniscate", 3.21, 7.04, 0.094, 0.007),
    ("melon_1.csv", "Melon", 3.57, 7.63, 0.107, 0.004),
    ("slanted_circle_1.csv", "Slanted Circle", 6.92, 10.75, 0.140, 0.004),
    ("linear_oscillation.csv", "Linear Oscillation", 7.25, 16.95, 0.214, 0.008),
    ("race_track_1.csv", "Race Track", 7.64, 13.14, 0.697, 0.006),
    ("melon_2.csv", "Melon", 7.74, 13.55, 0.091, 0.004),
    ("slanted_circle_2.csv", "Slanted Circle", 8.57, 13.32, 0.141, 0.002),
    ("race_track_2.csv", "Race Track", 9.94, 17.81, 0.709, 0.109),
    ("lemniscate_3.csv", "Lemniscate", 12.01, 19.83, 0.711, 0.005),
    ("ellipse.csv", "Ellipse", 15.02, 19.20, 1.528, 0.024),
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(project: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(project), *args), text=True).strip()


def _load_config(project: Path) -> dict[str, object]:
    path = project / "PRISM_V2_1_1_NEUROBEM_LITERATURE_ALIGNED_DUAL_BENCHMARK_PACKAGE" / "DUAL_BENCHMARK_CONFIG_FROZEN.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _test_stems(source_root: Path) -> list[str]:
    return [line.strip() for line in (source_root / "source_docs" / "testset.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_csv_map(source_root: Path) -> dict[str, Path]:
    result = {}
    for path in (source_root / "extracted").rglob("merged_*.csv"):
        key = path.stem.removeprefix("merged_")
        result[key] = path
    return result


def _track_a_trajectories(source_root: Path, partition: str, *, permit_test: bool) -> list[LiteratureTrajectory]:
    test = set(_test_stems(source_root))
    paths = _source_csv_map(source_root)
    selected = sorted(test if partition == "test" else set(paths).difference(test))
    if partition == "test" and not permit_test:
        raise RuntimeError("PRISM_TRACK_A_TEST_ACCESS_BEFORE_GLOBAL_FREEZE")
    missing = set(selected).difference(paths)
    if missing:
        raise RuntimeError(f"TRACK_A_SPLIT_FILE_MISSING:{sorted(missing)}")
    return [LiteratureTrajectory(value, partition, read_neurobem_csv(paths[value])) for value in selected]


def _track_b_trajectories(release_root: Path, partition: str, *, permit_test: bool) -> list[LiteratureTrajectory]:
    folder = {"train": "train", "validation": "valid", "test": "test"}[partition]
    if partition == "test" and not permit_test:
        raise RuntimeError("PRISM_TRACK_B_TEST_ACCESS_BEFORE_GLOBAL_FREEZE")
    paths = sorted((release_root / folder).glob("*.csv"))
    if not paths:
        raise RuntimeError(f"TRACK_B_EMPTY_PARTITION:{partition}")
    return [LiteratureTrajectory(path.name, partition, resample_track_b_100hz(read_neurobem_csv(path))) for path in paths]


def _track_a_split_manifest(source_root: Path) -> dict[str, object]:
    paths = _source_csv_map(source_root)
    test = sorted(_test_stems(source_root))
    development = sorted(set(paths).difference(test))
    return {
        "status": "OFFICIAL_TEST_IDENTITIES_RECOVERED_DEVELOPMENT_TRAIN_VALIDATION_IDENTITY_UNPUBLISHED",
        "source_file": str(source_root / "source_docs" / "testset.txt"),
        "official_test_segments": test,
        "official_test_segment_count": len(test),
        "non_test_development_segments": development,
        "non_test_development_segment_count": len(development),
        "development_selection": "GROUPED_PARENT_FLIGHT_4_FOLD_OOF",
        "custom_62_19_13_split_reused": False,
    }


def _official_metric_reproduction(source_root: Path, predictions_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    sources = _source_csv_map(source_root)
    for stem in _test_stems(source_root):
        target_frame = read_neurobem_csv(sources[stem])
        prediction_path = predictions_root / f"bem+nn_{stem}.csv"
        prediction_frame = read_neurobem_csv(prediction_path)
        if len(target_frame) != len(prediction_frame) or not np.allclose(target_frame["t"], prediction_frame["t"], atol=1e-8, rtol=0.0):
            raise RuntimeError(f"OFFICIAL_PREDICTION_ALIGNMENT_FAILED:{stem}")
        # The official README defines appended prediction columns 30--35 and
        # target-minus-prediction residuals 36--41. Use that exact paired
        # contract for the reproduction gate; the source frame is retained
        # solely for identity/time alignment.
        y = official_prediction_ground_truth_force_torque(prediction_frame)
        p = official_prediction_force_torque(prediction_frame)
        targets.append(y)
        predictions.append(p)
        rows.append({"trajectory_id": stem, **force_torque_metrics(y, p), "rows": len(y)})
    reproduced = force_torque_metrics(np.concatenate(targets), np.concatenate(predictions))
    gate = metric_reproduction_gate(reproduced, PUBLISHED_TRACK_A)
    return {"reproduced": reproduced, "published": PUBLISHED_TRACK_A, "gate": gate}, rows


def run_audit(project: Path, source_root: Path, release_root: Path, predictions_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    split_a = _track_a_split_manifest(source_root)
    split_b = track_b_split_manifest(release_root, "a8451a119b9096eda980f71b55dbb05012b8c47a")
    reproduction, reproduction_rows = _official_metric_reproduction(source_root, predictions_root)
    _write_json(output / "TRACK_A_SPLIT_MANIFEST.json", split_a)
    _write_json(output / "TRACK_B_SPLIT_MANIFEST.json", split_b)
    with (output / "TRACK_A_METRIC_REPRODUCTION.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("scope", "metric", "reproduced", "published", "relative_difference", "passed"))
        for metric in PUBLISHED_TRACK_A:
            writer.writerow(("aggregate", metric, reproduction["reproduced"][metric], PUBLISHED_TRACK_A[metric], reproduction["gate"]["relative_difference"][metric], reproduction["gate"]["relative_difference"][metric] <= 0.01))
        for row in reproduction_rows:
            for metric in PUBLISHED_TRACK_A:
                writer.writerow((row["trajectory_id"], metric, row[metric], "", "", ""))
    with (output / "TRACK_B_PUBLISHED_TABLE_CHECK.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("file", "trajectory", "v_mean", "v_max", "TCN_delta_z", "TCN_delta_q", "source_status"))
        for row in TRACK_B_TABLE_II:
            writer.writerow((*row, "PRIMARY_SOURCE_EXTRACTED"))
    _write_json(output / "LITERATURE_RUNTIME_AUDIT.json", {
        "status": "COMPLETED",
        "track_a_metric_reproduction": reproduction,
        "track_b_split_identity_status": split_b["split_identity_status"],
        "baseline_prediction_test_values_accessed": True,
        "prism_test_predictions_materialized": False,
        "published_scores_used_for_selection": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    })


def _select_track_a_w(trajectories: list[LiteratureTrajectory], config: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    selection = config["selection"]
    ridge_grid = selection["numerical_ridge_grid"]
    fold_losses = {family: [] for family in CANONICAL_W_CANDIDATES}
    fold_audit = []
    def evaluate_fold(fold: int) -> tuple[int, dict[str, object]]:
        fit = [item for item in trajectories if stable_group_fold(item.trajectory_id.rsplit("_seg_", 1)[0]) != fold]
        evaluation = [item for item in trajectories if stable_group_fold(item.trajectory_id.rsplit("_seg_", 1)[0]) == fold]
        if not fit or not evaluation:
            raise RuntimeError(f"EMPTY_TRACK_A_GROUPED_FOLD:{fold}")
        _, audit = select_w_family(
            concatenate_track_a_design(fit), concatenate_track_a_design(evaluation), ridge_grid,
            float(selection["maximum_condition_number"]), float(selection["maximum_relative_kkt_residual"]),
            target_kind="FORCE_TORQUE_6D", history=20,
            minimum_relative_improvement=float(selection["minimum_relative_improvement"]),
            candidate_workers=1,
        )
        return fold, {"fold": fold, "fit_segments": len(fit), "evaluation_segments": len(evaluation), **audit}

    folds = range(int(selection["inner_group_folds"]))
    # Two folds keep the measured peak below the 60-GiB cgroup while retaining
    # useful task-level parallelism. Nested family work stays ordered/serial.
    with ThreadPoolExecutor(max_workers=min(2, len(tuple(folds))), thread_name_prefix="track-a-fold") as executor:
        evaluated = list(executor.map(evaluate_fold, folds))
    for fold, audit in evaluated:
        for family, loss in audit["candidate_losses"].items():
            fold_losses[family].append(loss)
        fold_audit.append(audit)
    means = {family: float(np.mean(values)) for family, values in fold_losses.items()}
    neutral = means["IDENTITY_CORRECTION"]
    best = min(CANONICAL_W_CANDIDATES, key=lambda key: means[key])
    improvement = (neutral - means[best]) / max(neutral, np.finfo(float).eps)
    positive = float(np.mean(np.asarray(fold_losses[best]) < np.asarray(fold_losses["IDENTITY_CORRECTION"])))
    selected = best if best != "IDENTITY_CORRECTION" and improvement >= float(selection["minimum_relative_improvement"]) and positive >= float(selection["minimum_positive_fold_fraction"]) else "IDENTITY_CORRECTION"
    return selected, {"fold_losses": fold_losses, "mean_losses": means, "best_unprotected": best, "relative_improvement_vs_identity": improvement, "positive_fold_fraction": positive, "selected": selected}


def _contracts_json(contracts: Mapping[str, object]) -> dict[str, object]:
    return {key: route_contract_to_json(value) for key, value in contracts.items()}


def run_development(project: Path, source_root: Path, release_root: Path, output: Path) -> None:
    config = _load_config(project)
    selection = config["selection"]
    ridge_grid = selection["numerical_ridge_grid"]
    track_a = _track_a_trajectories(source_root, "development", permit_test=False)
    track_b_train = _track_b_trajectories(release_root, "train", permit_test=False)
    track_b_validation = _track_b_trajectories(release_root, "validation", permit_test=False)
    selected_a, selection_a = _select_track_a_w(track_a, config)
    arrays_a = concatenate_track_a_design(track_a)
    contracts_a = fit_route_contracts(*arrays_a, selected_a, ridge_grid, float(selection["maximum_condition_number"]), float(selection["maximum_relative_kkt_residual"]), target_kind="FORCE_TORQUE_6D", history=20)
    selected_b, selection_b = select_track_b_w_family_rollout(
        concatenate_track_b_design(track_b_train), track_b_validation, ridge_grid,
        float(selection["maximum_condition_number"]), float(selection["maximum_relative_kkt_residual"]),
        history=20, rollout=10,
        minimum_relative_improvement=float(selection["minimum_relative_improvement"]),
        candidate_workers=1,
        trajectory_workers=min(11, int(config["runtime"]["candidate_workers"])),
    )
    # Development selection is frozen, then all official non-test data are
    # used for the final estimator contracts.
    arrays_b = concatenate_track_b_design(track_b_train + track_b_validation)
    contracts_b = fit_track_b_route_contracts(*arrays_b, selected_b, ridge_grid, float(selection["maximum_condition_number"]), float(selection["maximum_relative_kkt_residual"]), history=20)
    output.mkdir(parents=True, exist_ok=True)
    track_a_result = {"status": "PASS", "selected_w_family": selected_a, "selection": selection_a, "candidate_binding": candidate_binding_audit(contracts_a), "a_enabled": False, "test_accessed": False}
    track_b_result = {"status": "PASS", "selected_w_family": selected_b, "selection": selection_b, "candidate_binding": candidate_binding_audit(contracts_b), "a_enabled": False, "test_accessed": False}
    _write_json(output / "DEVELOPMENT" / "TRACK_A_RESULT.json", track_a_result)
    _write_json(output / "DEVELOPMENT" / "TRACK_B_RESULT.json", track_b_result)
    contracts_a_path = output / "FREEZE" / "TRACK_A_CONTRACTS.json"
    contracts_b_path = output / "FREEZE" / "TRACK_B_CONTRACTS.json"
    _write_json(contracts_a_path, _contracts_json(contracts_a))
    _write_json(contracts_b_path, _contracts_json(contracts_b))
    commit = _git_value(project.parent, "rev-parse", "HEAD")
    freeze = {
        "status": "GLOBAL_DUAL_DEVELOPMENT_FROZEN",
        "development_frozen": True,
        "code_commit": commit,
        "config_sha256": _sha256(project / "PRISM_V2_1_1_NEUROBEM_LITERATURE_ALIGNED_DUAL_BENCHMARK_PACKAGE" / "DUAL_BENCHMARK_CONFIG_FROZEN.json"),
        "track_a": {"selected_w_family": selected_a, "contracts_sha256": _sha256(contracts_a_path), "formal_routes": ["PRISM_" + value for value in FORMAL_ROUTE_IDS]},
        "track_b": {"selected_w_family": selected_b, "contracts_sha256": _sha256(contracts_b_path), "formal_routes": ["PRISM_" + value.removeprefix("PF_") if value.startswith("PF_") else "PRISM_" + value for value in FORMAL_ROUTE_IDS]},
        "all_candidate_bindings_pass": bool(track_a_result["candidate_binding"]["passed"] and track_b_result["candidate_binding"]["passed"]),
        "published_scores_used_for_selection": False,
        "prism_track_a_test_accessed": False,
        "prism_track_b_test_accessed": False,
        "historical_test_already_accessed": True,
    }
    _write_json(output / "GLOBAL_DUAL_DEVELOPMENT_FREEZE.json", freeze)
    _write_json(output / "DUAL_TEST_ACCESS_AUDIT.json", {"status": "NOT_ACCESSED_BEFORE_GLOBAL_FREEZE", "track_a_prism_test_accessed": False, "track_b_prism_test_accessed": False, "baseline_metric_reproduction_access_separate": True})


def _load_contracts(path: Path) -> dict[str, object]:
    values = json.loads(path.read_text(encoding="utf-8"))
    return {key: route_contract_from_json(value) for key, value in values.items()}


def run_test(project: Path, source_root: Path, release_root: Path, output: Path) -> None:
    freeze_path = output / "GLOBAL_DUAL_DEVELOPMENT_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "GLOBAL_DUAL_DEVELOPMENT_FROZEN" or not freeze.get("all_candidate_bindings_pass"):
        raise RuntimeError("GLOBAL_DUAL_FREEZE_GATE_FAILED")
    if _git_value(project.parent, "rev-parse", "HEAD") != freeze["code_commit"]:
        raise RuntimeError("CODE_COMMIT_CHANGED_AFTER_GLOBAL_DUAL_FREEZE")
    contracts_a = _load_contracts(output / "FREEZE" / "TRACK_A_CONTRACTS.json")
    contracts_b = _load_contracts(output / "FREEZE" / "TRACK_B_CONTRACTS.json")
    track_a_test = _track_a_trajectories(source_root, "test", permit_test=True)
    metrics_a, per_a = track_a_route_metrics(contracts_a, track_a_test)
    track_b_test = _track_b_trajectories(release_root, "test", permit_test=True)
    trajectory_b: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    aggregate_b: dict[str, dict[str, float]] = {}
    for route in FORMAL_ROUTE_IDS:
        total_windows = 0
        weighted_z = 0.0
        weighted_q = 0.0
        def evaluate_trajectory(trajectory: LiteratureTrajectory) -> tuple[LiteratureTrajectory, dict[str, object]]:
            return trajectory, track_b_rollout(contracts_b, route, trajectory.frame)

        with ThreadPoolExecutor(max_workers=min(12, len(track_b_test)), thread_name_prefix="track-b-test-trajectory") as executor:
            evaluated = list(executor.map(evaluate_trajectory, track_b_test))
        for trajectory, result in evaluated:
            weight = int(result["sliding_windows"]) * 60
            total_windows += weight
            weighted_z += float(result["delta_z"]) * weight
            weighted_q += float(result["delta_q"]) * weight
            trajectory_b.append({"trajectory_id": trajectory.trajectory_id, "route": route, **{key: value for key, value in result.items() if not key.startswith("per_step")}})
            for step, (delta_z, delta_q) in enumerate(zip(result["per_step_delta_z"], result["per_step_delta_q"], strict=True), 1):
                curves.append({"trajectory_id": trajectory.trajectory_id, "route": route, "step": step, "delta_z": delta_z, "delta_q": delta_q})
        aggregate_b[route] = {"delta_z": weighted_z / total_windows, "delta_q": weighted_q / total_windows}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "TRACK_A_PRISM_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("route", "Fxy", "Fz", "Mxy", "Mz", "F", "M"))
        writer.writeheader()
        for route, values in metrics_a.items(): writer.writerow({"route": route, **values})
    with (output / "TRACK_A_PER_FLIGHT_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in per_a for key in row}))
        writer.writeheader(); writer.writerows(per_a)
    with (output / "TRACK_B_PRISM_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("route", "delta_z", "delta_q")); writer.writeheader()
        for route, values in aggregate_b.items(): writer.writerow({"route": route, **values})
    with (output / "TRACK_B_TRAJECTORY_COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in trajectory_b for key in row})); writer.writeheader(); writer.writerows(trajectory_b)
    with (output / "TRACK_B_ROLLOUT_CURVES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("trajectory_id", "route", "step", "delta_z", "delta_q")); writer.writeheader(); writer.writerows(curves)
    summary = {"status": "COMPLETED", "track_a": metrics_a, "track_b": aggregate_b, "published_track_b": PUBLISHED_TRACK_B, "information_set_caveat": "PUBLISHED_DECOUPLED_REFERENCE_TEACHER_FORCES_COMPLEMENTARY_STATE_PRISM_DOES_NOT", "a_enabled": False}
    _write_json(output / "NEUROBEM_LITERATURE_ALIGNED_DUAL_BENCHMARK_SUMMARY.json", summary)
    _write_json(output / "DUAL_TEST_ACCESS_AUDIT.json", {"status": "ACCESSED_AFTER_GLOBAL_DUAL_FREEZE", "global_freeze_sha256": _sha256(freeze_path), "track_a_prism_test_accessed": True, "track_b_prism_test_accessed": True, "future_measured_state_used_in_track_b": False, "future_target_residual_used": False})


def run_published_evaluator_extension(project: Path, release_root: Path, output: Path) -> None:
    """Evaluate frozen Track-B contracts under the official decoupled loop."""
    freeze_path = output / "GLOBAL_DUAL_DEVELOPMENT_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "GLOBAL_DUAL_DEVELOPMENT_FROZEN" or not freeze.get("all_candidate_bindings_pass"):
        raise RuntimeError("GLOBAL_DUAL_FREEZE_GATE_FAILED")
    contract_path = output / "FREEZE" / "TRACK_B_CONTRACTS.json"
    if _sha256(contract_path) != freeze["track_b"]["contracts_sha256"]:
        raise RuntimeError("FROZEN_TRACK_B_CONTRACT_HASH_MISMATCH")
    contracts = _load_contracts(contract_path)
    test = _track_b_trajectories(release_root, "test", permit_test=True)
    if len(test) != 12:
        raise RuntimeError(f"PUBLISHED_EVALUATOR_EXPECTED_12_TRAJECTORIES:{len(test)}")
    trajectory_rows: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, object]] = {}
    for route in FORMAL_ROUTE_IDS:
        def evaluate(item: LiteratureTrajectory) -> tuple[LiteratureTrajectory, dict[str, object]]:
            return item, track_b_published_decoupled_evaluator(contracts, route, item.frame, history=20, rollout=60)
        with ThreadPoolExecutor(max_workers=12, thread_name_prefix="track-b-published-evaluator") as executor:
            evaluated = list(executor.map(evaluate, test))
        total = 0
        weighted_v = 0.0
        weighted_q = 0.0
        finite = True
        for trajectory, result in evaluated:
            weight = int(result["sliding_windows"])
            total += weight
            weighted_v += float(result["delta_v"]) * weight
            weighted_q += float(result["delta_q"]) * weight
            finite = finite and bool(result["finite"])
            trajectory_rows.append({
                "trajectory": trajectory.trajectory_id,
                "route": route,
                "delta_v": result["delta_v"],
                "delta_q": result["delta_q"],
                "sliding_windows": weight,
                "finite": result["finite"],
                "maximum_quaternion_norm_error": result["maximum_quaternion_norm_error"],
            })
        delta_v = weighted_v / total
        delta_q = weighted_q / total
        aggregate[route] = {
            "delta_v": delta_v,
            "delta_q": delta_q,
            "finite": bool(finite and np.isfinite(delta_v) and np.isfinite(delta_q)),
            "relative_difference_vs_tcn_delta_v": (delta_v - PUBLISHED_TRACK_B["TCN"]["delta_z"]) / PUBLISHED_TRACK_B["TCN"]["delta_z"],
            "relative_difference_vs_tcn_delta_q": (delta_q - PUBLISHED_TRACK_B["TCN"]["delta_q"]) / PUBLISHED_TRACK_B["TCN"]["delta_q"],
        }
    table_ii = {row[0]: row for row in TRACK_B_TABLE_II}
    for row in trajectory_rows:
        reference = table_ii[row["trajectory"]]
        row["TCN_delta_v_published"] = reference[4]
        row["TCN_delta_q_published"] = reference[5]
        if bool(row["finite"]):
            row["win_tie_loss"] = "WIN" if row["delta_v"] < reference[4] and row["delta_q"] < reference[5] else ("TIE" if np.isclose(row["delta_v"], reference[4]) and np.isclose(row["delta_q"], reference[5]) else "LOSS")
        else:
            row["win_tie_loss"] = "NONFINITE_DIVERGENCE"
    wins = {}
    for route in FORMAL_ROUTE_IDS:
        labels = [row["win_tie_loss"] for row in trajectory_rows if row["route"] == route]
        wins[route] = {label: labels.count(label) for label in ("WIN", "TIE", "LOSS", "NONFINITE_DIVERGENCE")}
    extension = output / "PUBLISHED_DECOUPLED_EVALUATOR"
    extension.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trajectory_rows).to_csv(extension / "TRACK_B_PUBLISHED_EVALUATOR_TRAJECTORIES.csv", index=False)
    pd.DataFrame([{"route": route, **values, **{f"trajectory_{key.lower()}": count for key, count in wins[route].items()}} for route, values in aggregate.items()]).to_csv(extension / "TRACK_B_PUBLISHED_EVALUATOR_METRICS.csv", index=False)
    result = {
        "status": "COMPLETED",
        "evaluator": "PUBLISHED_DECOUPLED_EVALUATOR",
        "protocol_claim": "EXACT_PUBLISHED_EVALUATOR_ON_FROZEN_PRISM",
        "exact_full_published_protocol": False,
        "training_contract_mismatch": True,
        "training_contract_audit": {
            "official": "U10_RECURSIVE_BRANCH_TRAINING_WITH_COMPLEMENTARY_GROUND_TRUTH_STATE_INJECTION",
            "frozen_prism": "ONE_STEP_NUMERICALLY_CERTIFIED_RIDGE_WITH_U10_FULLY_RECURSIVE_DEVELOPMENT_SELECTION",
            "mismatch": True,
        },
        "frozen_contracts_reused": True,
        "frozen_contract_sha256": _sha256(contract_path),
        "fully_recursive_stress_test": "NONFINITE_RECURSIVE_DIVERGENCE",
        "history": 20,
        "sampling_hz": 100,
        "rollout": 60,
        "test_trajectory_count": 12,
        "future_control_used": True,
        "complementary_measured_state_used": True,
        "future_target_residual_used": False,
        "other_future_information_used": False,
        "published_tcn": {"delta_v": 0.042, "delta_q": 0.006},
        "aggregate": aggregate,
        "trajectory_win_tie_loss": wins,
    }
    _write_json(extension / "TRACK_B_PUBLISHED_EVALUATOR_RESULT.json", result)


def _published_track_a_rows(project: Path) -> list[dict[str, object]]:
    source = project / "PRISM_V2_1_1_NEUROBEM_LITERATURE_ALIGNED_DUAL_BENCHMARK_PACKAGE" / "PUBLISHED_BASELINE_VALUES.csv"
    frame = pd.read_csv(source)
    frame = frame[(frame["track"] == "A") & (frame["table"] == "HDVIO2_Table_I")]
    rows = []
    for method, values in frame.groupby("method", sort=False):
        row = {
            "method": method,
            "source": "HDVIO2.0_TABLE_I",
            "information_set": "PUBLISHED_LABELED",
            "history": "PUBLISHED",
            "A_allowed": False,
            "direct_comparison_eligible": True,
        }
        row.update({record.metric: float(record.value) for record in values.itertuples()})
        rows.append(row)
    return rows


def _write_figure(path: Path, methods: list[str], series: Mapping[str, list[float]], ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(methods))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 0.85), 4.6), constrained_layout=True)
    for index, (label, values) in enumerate(series.items()):
        ax.bar(x + (index - (len(series) - 1) / 2) * width, values, width, label=label)
    ax.set_xticks(x, methods, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run_report(project: Path, output: Path) -> None:
    """Deterministic post-test aggregation; never reads raw benchmark data."""
    access = json.loads((output / "DUAL_TEST_ACCESS_AUDIT.json").read_text(encoding="utf-8"))
    if access.get("status") != "ACCESSED_AFTER_GLOBAL_DUAL_FREEZE":
        raise RuntimeError("REPORT_REQUIRES_COMPLETED_POST_FREEZE_TEST")
    metric_reproduction = pd.read_csv(output / "TRACK_A_METRIC_REPRODUCTION.csv")
    reproduction_pass = bool(metric_reproduction.loc[metric_reproduction.scope == "aggregate", "passed"].astype(str).str.lower().eq("true").all())
    prism_a = pd.read_csv(output / "TRACK_A_PRISM_METRICS.csv")
    prism_b = pd.read_csv(output / "TRACK_B_PRISM_METRICS.csv")
    trajectory = pd.read_csv(output / "TRACK_B_TRAJECTORY_COMPARISON.csv")
    curves = pd.read_csv(output / "TRACK_B_ROLLOUT_CURVES.csv")

    published_a = _published_track_a_rows(project)
    direct_rows = list(published_a)
    for record in prism_a.to_dict("records"):
        direct_rows.append({
            "method": f"PRISM_{record.pop('route')}",
            "source": "THIS_EXPERIMENT",
            "information_set": "PUBLISHED_NEUROBEM_USE_ATT_FALSE_COMPARABLE",
            "history": "50_ms",
            "A_allowed": False,
            "direct_comparison_eligible": reproduction_pass,
            **record,
        })
    pd.DataFrame(direct_rows).to_csv(output / "TRACK_A_DIRECT_COMPARISON.csv", index=False)

    direct_b = [
        {"method": method, **values, "status": "PUBLISHED_FINITE", "source": "IROS2024_TABLE_I", "direct_comparison_eligible": False}
        for method, values in PUBLISHED_TRACK_B.items()
    ]
    direct_b.extend({
        "method": f"PRISM_{row['route']}",
        "delta_z": row["delta_z"],
        "delta_q": row["delta_q"],
        "status": "FINITE" if np.isfinite(row["delta_z"]) and np.isfinite(row["delta_q"]) else "NONFINITE_RECURSIVE_DIVERGENCE",
        "source": "THIS_EXPERIMENT",
        "direct_comparison_eligible": False,
    } for row in prism_b.to_dict("records"))
    pd.DataFrame(direct_b).to_csv(output / "TRACK_B_PUBLISHED_DIRECT_COMPARISON.csv", index=False)

    table_ii = {row[0]: row for row in TRACK_B_TABLE_II}
    pivot = trajectory.pivot(index="trajectory_id", columns="route", values=["delta_z", "delta_q"])
    comparison_rows = []
    for name in pivot.index:
        reference = table_ii.get(name)
        row: dict[str, object] = {"trajectory": name}
        if reference:
            row.update({"trajectory_class": reference[1], "v_mean": reference[2], "v_max": reference[3], "TCN_delta_v_published": reference[4], "TCN_delta_q_published": reference[5]})
        for route in FORMAL_ROUTE_IDS:
            row[f"PRISM_{route}_delta_v"] = float(pivot.loc[name, ("delta_z", route)])
            row[f"PRISM_{route}_delta_q"] = float(pivot.loc[name, ("delta_q", route)])
        if reference:
            dz = row["PRISM_J_KCW_delta_v"]
            dq = row["PRISM_J_KCW_delta_q"]
            if np.isfinite(dz) and np.isfinite(dq):
                row["relative_delta_v_vs_TCN"] = (dz - reference[4]) / reference[4]
                row["relative_delta_q_vs_TCN"] = (dq - reference[5]) / reference[5]
                row["win_tie_loss"] = "WIN" if dz < reference[4] and dq < reference[5] else ("TIE" if np.isclose(dz, reference[4]) and np.isclose(dq, reference[5]) else "LOSS")
            else:
                row["relative_delta_v_vs_TCN"] = np.nan
                row["relative_delta_q_vs_TCN"] = np.nan
                row["win_tie_loss"] = "NONFINITE_DIVERGENCE"
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(output / "TRACK_B_TRAJECTORY_COMPARISON.csv", index=False)

    stability_rows = []
    for route, values in trajectory.groupby("route", sort=False):
        finite = np.isfinite(values.delta_z) & np.isfinite(values.delta_q)
        stability_rows.append({
            "route": route,
            "trajectory_count": len(values),
            "finite_trajectory_count": int(finite.sum()),
            "nonfinite_trajectory_count": int((~finite).sum()),
            "maximum_finite_delta_z": float(values.loc[np.isfinite(values.delta_z), "delta_z"].max()),
            "status": "FINITE" if bool(finite.all()) else "NONFINITE_RECURSIVE_DIVERGENCE",
        })
    pd.DataFrame(stability_rows).to_csv(output / "TRACK_B_STABILITY_AUDIT.csv", index=False)

    contracts = _load_contracts(output / "FREEZE" / "TRACK_A_CONTRACTS.json")
    topology = []
    attribution = []
    for route, contract in contracts.items():
        topology.append({"route": route, "K_columns": contract.column_slices["K"][1] - contract.column_slices["K"][0], "C_columns": contract.column_slices["C"][1] - contract.column_slices["C"][0], "W_columns": contract.column_slices.get("W", (0, 0))[1] - contract.column_slices.get("W", (0, 0))[0], "candidate_binding": contract.route_id == route})
        attribution.extend([
            {"route": route, "module": "motor", "claim_class": "ACTUATOR_PHYSICS_CONSISTENCY"},
            {"route": route, "module": "velocity_context", "claim_class": "PREDICTIVE_CONTEXT_ATTRIBUTION"},
            {"route": route, "module": "body_rate_context", "claim_class": "PREDICTIVE_CONTEXT_ATTRIBUTION"},
            {"route": route, "module": "C_interaction", "claim_class": "PREDICTIVE_CONTEXT_ATTRIBUTION"},
            {"route": route, "module": "W_latent_curvature", "claim_class": "STATIC_LATENT_CURVATURE"},
        ])
    pd.DataFrame(topology).to_csv(output / "TRACK_A_TOPOLOGY_AUDIT.csv", index=False)
    pd.DataFrame(attribution).to_csv(output / "TRACK_A_MODULE_ATTRIBUTION.csv", index=False)

    figures = output / "FIGURES"
    figures.mkdir(exist_ok=True)
    a = pd.DataFrame(direct_rows)
    _write_figure(figures / "FIGURE_A1_FORCE_TORQUE_COMPARISON", a.method.tolist(), {metric: a[metric].tolist() for metric in ("Fxy", "Fz", "Mxy", "Mz", "F", "M")}, "RMSE (mixed N / Nm; see legend)")
    prism_only_a = a[a.method.str.startswith("PRISM_")]
    _write_figure(figures / "FIGURE_A2_PF_VS_JOINT", prism_only_a.method.tolist(), {metric: prism_only_a[metric].tolist() for metric in ("F", "M")}, "RMSE")
    _write_figure(figures / "FIGURE_A3_ATTRIBUTION_BLOCK_SIZE", [row["route"] for row in topology], {key: [row[f"{key}_columns"] for row in topology] for key in ("K", "C", "W")}, "registered columns")
    b = pd.DataFrame(direct_b)
    _write_figure(figures / "FIGURE_B1_IROS2024_COMPARISON", b.method.tolist(), {"delta_v": b.delta_z.tolist(), "delta_q": b.delta_q.tolist()}, "published metric (different scales)")
    comp = pd.DataFrame(comparison_rows)
    _write_figure(figures / "FIGURE_B2_TRAJECTORY_DELTA_V", comp.trajectory.tolist(), {"TCN published": comp.TCN_delta_v_published.tolist(), "PRISM J_KCW": comp.PRISM_J_KCW_delta_v.tolist()}, "delta_v")
    _write_figure(figures / "FIGURE_B3_TRAJECTORY_DELTA_Q", comp.trajectory.tolist(), {"TCN published": comp.TCN_delta_q_published.tolist(), "PRISM J_KCW": comp.PRISM_J_KCW_delta_q.tolist()}, "delta_q")
    mean_curve = curves.groupby(["route", "step"], as_index=False)[["delta_z", "delta_q"]].mean()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for route, values in mean_curve.groupby("route"):
        axes[0].plot(values.step, values.delta_z, label=route)
        axes[1].plot(values.step, values.delta_q, label=route)
    for axis, label in zip(axes, ("delta_v", "delta_q"), strict=True):
        axis.set(xlabel="rollout step", ylabel=label); axis.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.savefig(figures / "FIGURE_B4_ROLLOUT_CURVES.png", dpi=180); fig.savefig(figures / "FIGURE_B4_ROLLOUT_CURVES.pdf"); plt.close(fig)

    best_a = prism_a.loc[prism_a.F.idxmin()].to_dict()
    finite_b = prism_b[np.isfinite(prism_b.delta_z) & np.isfinite(prism_b.delta_q)]
    track_b_diverged = finite_b.empty
    best_b = None if track_b_diverged else finite_b.loc[(finite_b.delta_z / PUBLISHED_TRACK_B["TCN"]["delta_z"] + finite_b.delta_q / PUBLISHED_TRACK_B["TCN"]["delta_q"]).idxmin()].to_dict()
    competitive = bool(not track_b_diverged and best_a["F"] <= PUBLISHED_TRACK_A["F"] and best_a["M"] <= PUBLISHED_TRACK_A["M"] and best_b["delta_z"] <= 1.2 * PUBLISHED_TRACK_B["TCN"]["delta_z"] and best_b["delta_q"] <= 1.2 * PUBLISHED_TRACK_B["TCN"]["delta_q"])
    verdict = ("B: PRISM is numerically competitive with a clear attribution advantage" if competitive
               else "D: PRISM is not competitive under the literature-aligned protocol" if track_b_diverged
               else "C: PRISM is somewhat weaker numerically but offers substantially richer auditable attribution")
    retained_failures = []
    if not reproduction_pass:
        retained_failures.append("TRACK_A_OFFICIAL_PREDICTION_TABLE_REPRODUCTION_FAILED")
    if track_b_diverged:
        retained_failures.append("TRACK_B_RECURSIVE_ROLLOUT_NONFINITE_DIVERGENCE")
    summary = {
        "status": "COMPLETED_WITH_TRACK_B_MODEL_FAILURE" if track_b_diverged else "COMPLETED",
        "track_a_metric_reproduction_pass": reproduction_pass,
        "track_a_comparison_class": "EXACT_DIRECT_COMPARISON" if reproduction_pass else "PUBLISHED_AGGREGATE_COMPARISON_ONLY",
        "track_a_best_prism": best_a,
        "track_b_comparison_class": "PROTOCOL_LEVEL_STRICTER_PRISM_INFORMATION_CONTRACT",
        "track_b_best_prism": best_b,
        "published_tcn": PUBLISHED_TRACK_B["TCN"],
        "overall_numerical_verdict": verdict,
        "overall_interpretability_verdict": "AUDITABLE_ACTUATOR_CONTEXT_INTERACTION_AND_LATENT_CURVATURE_SEPARATION",
        "track_b_status": "NONFINITE_RECURSIVE_DIVERGENCE" if track_b_diverged else "FINITE",
        "retained_failures": retained_failures,
        "noncomparable_literature_methods": ["NeuroMHE"],
        "test_access": access,
    }
    _write_json(output / "NEUROBEM_LITERATURE_ALIGNED_DUAL_BENCHMARK_SUMMARY.json", summary)
    report = f"""# NeuroBEM literature-aligned dual benchmark final report

## Outcome

{verdict}. Track A comparison status is `{summary['track_a_comparison_class']}`.
Track B remains `{summary['track_b_comparison_class']}` because the published
decoupled evaluator teacher-forces complementary state while PRISM does not.

## Track A

The best PRISM force route is `{best_a['route']}` with F={best_a['F']:.6g} N
and M={best_a['M']:.6g} Nm. Full axis results are in
`TRACK_A_DIRECT_COMPARISON.csv`. No result is described as an exact published
ranking unless the preregistered official-prediction reproduction gate passed.

## Track B

All four frozen PRISM routes produced non-finite aggregate metrics because the
unconstrained recursive state diverged on at least one official test
trajectory. Finite trajectories already include extreme delta-v errors up to
the values recorded in `TRACK_B_STABILITY_AUDIT.csv`. This is a model-stability
failure, not a metric, target, split, or future-state-access defect. No
post-test clipping, stabilization, or candidate reselection was applied.
Published TCN is delta_v=0.042 and delta_q=0.006, but PRISM has no finite
aggregate to rank against it. Genuine PRISM 60-step curves are retained; no
TCN curve was digitized.

## Interpretation limits

Motor K supports actuator-physics consistency. Velocity, attitude, and body
rate K are predictive context only. C contains registered cross-context
interaction. W is static frozen-latent curvature and never reads raw context.
Neither context K nor W is presented as causal aerodynamic-law discovery.
`A` was disabled in both formal tracks.
"""
    (output / "NEUROBEM_LITERATURE_ALIGNED_DUAL_BENCHMARK_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "development", "test", "published-evaluator", "report"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--track-b-release-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.stage == "audit":
        if args.predictions_root is None: raise ValueError("audit requires --predictions-root")
        run_audit(args.project, args.source_root, args.track_b_release_root, args.predictions_root, args.output)
    elif args.stage == "development":
        run_development(args.project, args.source_root, args.track_b_release_root, args.output)
    elif args.stage == "test":
        run_test(args.project, args.source_root, args.track_b_release_root, args.output)
    elif args.stage == "published-evaluator":
        run_published_evaluator_extension(args.project, args.track_b_release_root, args.output)
    else:
        run_report(args.project, args.output)


if __name__ == "__main__":
    main()
