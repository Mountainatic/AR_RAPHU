"""Run the frozen NeuroBEM Track-B exact published-training experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from prism_benchmark.neurobem_exact_training import (
    PUBLISHED_HISTORY,
    PUBLISHED_TEST_ROLLOUT,
    PUBLISHED_TRAINING_UNROLL,
    NonfinitePublishedTrainingError,
    audit_published_training_contract,
    fit_exact_published_training_contracts,
    initial_exact_training_contracts,
)
from prism_benchmark.neurobem_literature import (
    CANONICAL_W_CANDIDATES,
    FORMAL_ROUTE_IDS,
    candidate_binding_audit,
    concatenate_track_b_design,
    route_contract_from_json,
    route_contract_to_json,
    track_b_published_decoupled_evaluator,
    track_b_split_manifest,
)
from prism_benchmark.neurobem_literature_runner import (
    PUBLISHED_TRACK_B,
    TRACK_B_TABLE_II,
    _track_b_trajectories,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _config(project: Path) -> dict[str, object]:
    path = project / "PRISM_V2_1_1_NEUROBEM_EXACT_PUBLISHED_TRAINING_PACKAGE" / "TRACK_B_EXACT_TRAINING_CONFIG_FROZEN.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _save_contracts(path: Path, contracts: dict[str, object]) -> None:
    _write_json(path, {key: route_contract_to_json(value) for key, value in contracts.items()})


def _load_contracts(path: Path) -> dict[str, object]:
    values = json.loads(path.read_text(encoding="utf-8"))
    return {key: route_contract_from_json(value) for key, value in values.items()}


def _aggregate_development(contracts, trajectories, *, workers: int = 11) -> dict[str, object]:
    route_result: dict[str, object] = {}
    for route in FORMAL_ROUTE_IDS:
        with ThreadPoolExecutor(max_workers=min(workers, len(trajectories)), thread_name_prefix="exact-valid") as executor:
            results = list(executor.map(
                lambda item: track_b_published_decoupled_evaluator(
                    contracts, route, item.frame, history=PUBLISHED_HISTORY, rollout=PUBLISHED_TRAINING_UNROLL,
                ),
                trajectories,
            ))
        weights = np.asarray([int(value["sliding_windows"]) for value in results], dtype=np.float64)
        dv = np.asarray([float(value["delta_v"]) for value in results], dtype=np.float64)
        dq = np.asarray([float(value["delta_q"]) for value in results], dtype=np.float64)
        route_result[route] = {
            "delta_v": float(np.average(dv, weights=weights)),
            "delta_q": float(np.average(dq, weights=weights)),
            "finite": bool(np.isfinite(dv).all() and np.isfinite(dq).all()),
            "validation_segments": len(results),
            "validation_windows": int(weights.sum()),
        }
    return route_result


def run_development(project: Path, release_root: Path, output: Path) -> None:
    cfg = _config(project)
    if cfg["status"] != "FROZEN_BEFORE_FORMAL_TEST_ACCESS":
        raise RuntimeError("EXACT_TRAINING_CONFIG_NOT_FROZEN")
    train = _track_b_trajectories(release_root, "train", permit_test=False)
    validation = _track_b_trajectories(release_root, "validation", permit_test=False)
    split = track_b_split_manifest(release_root, str(cfg["official_code_commit"]))
    ridge = cfg["estimator"]
    ridge_grid = ridge["numerical_ridge_grid"]
    max_condition = float(ridge["maximum_condition_number"])
    max_kkt = float(ridge["maximum_relative_kkt_residual"])
    train_target = concatenate_track_b_design(train, PUBLISHED_HISTORY)[2]
    velocity_scale = max(float(np.mean(np.sum(np.square(train_target[:, :6]), axis=1))), np.finfo(float).eps)
    attitude_scale = max(float(np.mean(np.sum(np.square(train_target[:, 6:]), axis=1))), np.finfo(float).eps)
    def evaluate_family(family: str):
        try:
            initial = initial_exact_training_contracts(train, family, ridge_grid, max_condition, max_kkt)
            candidate, audit = fit_exact_published_training_contracts(
                train, initial, ridge_grid, max_condition, max_kkt,
                history=PUBLISHED_HISTORY, unroll=PUBLISHED_TRAINING_UNROLL,
                trajectory_workers=8,
            )
            values = _aggregate_development(candidate, validation)
            nonfinite_routes = [route for route, value in values.items() if not bool(value["finite"])]
            if nonfinite_routes:
                raise NonfinitePublishedTrainingError(
                    f"NONFINITE_DEVELOPMENT_VALIDATION:family={family}:routes={','.join(nonfinite_routes)}"
                )
            risk = float(np.mean([
                float(values[route]["delta_v"]) / velocity_scale
                + np.square(float(values[route]["delta_q"])) / attitude_scale
                for route in ("PF_KCW", "J_KCW")
            ]))
            return family, values, risk, candidate, audit, None
        except NonfinitePublishedTrainingError as error:
            return family, None, float("inf"), None, None, str(error)

    candidate_risk: dict[str, float] = {}
    candidate_validation: dict[str, object] = {}
    candidate_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(CANONICAL_W_CANDIDATES), thread_name_prefix="exact-w-family") as executor:
        evaluated_families = list(executor.map(evaluate_family, CANONICAL_W_CANDIDATES))
    for family, values, risk, _candidate, _audit, error in evaluated_families:
        candidate_validation[family] = values
        candidate_risk[family] = risk
        if error is not None:
            candidate_errors[family] = error
    if not any(np.isfinite(value) for value in candidate_risk.values()):
        failure = {
            "status": "DEVELOPMENT_TRAINING_NONFINITE_DIVERGENCE",
            "generating_commit": _git(project.parent, "rev-parse", "HEAD"),
            "development_frozen": False,
            "training_contract_matched": True,
            "evaluator_contract_matched": True,
            "formal_test_executed": False,
            "test_accessed": False,
            "candidate_errors": candidate_errors,
            "candidate_risk": candidate_risk,
            "published_scores_used_for_model_selection": False,
            "previous_test_used_for_stabilization": False,
            "stabilization_added": False,
            "decision": "STOP_BEFORE_TEST_DEVELOPMENT_COULD_NOT_FREEZE",
            "formal_route_status": {
                route: "NOT_EVALUATED_DEVELOPMENT_NOT_FROZEN" for route in FORMAL_ROUTE_IDS
            },
        }
        _write_json(output / "TRACK_B_EXACT_TRAINING_CONFIG_FROZEN.json", cfg)
        _write_json(output / "TRACK_B_SPLIT_MANIFEST.json", split)
        _write_json(output / "TRACK_B_EXACT_TRAINING_DEVELOPMENT_RESULT.json", failure)
        _write_json(output / "TRACK_B_EXACT_TRAINING_DEVELOPMENT_FREEZE.json", failure)
        _write_json(output / "TRACK_B_EXACT_TRAINING_TEST_ACCESS_AUDIT.json", {
            "status": "NOT_ACCESSED_DEVELOPMENT_NOT_FROZEN",
            "formal_test_executed": False,
            "test_accessed": False,
            "future_test_state_or_control_read": False,
        })
        pd.DataFrame([
            {"route": route, "finite": False, "delta_v": np.nan, "delta_q": np.nan,
             "status": "NOT_EVALUATED_DEVELOPMENT_NOT_FROZEN"}
            for route in FORMAL_ROUTE_IDS
        ]).to_csv(output / "TRACK_B_EXACT_TRAINING_METRICS.csv", index=False)
        pd.DataFrame(columns=("trajectory", "route", "finite", "delta_v", "delta_q", "win_tie_loss")).to_csv(
            output / "TRACK_B_EXACT_TRAINING_PER_TRAJECTORY.csv", index=False,
        )
        pd.DataFrame(columns=("trajectory", "route", "step", "delta_v", "delta_q")).to_csv(
            output / "TRACK_B_EXACT_TRAINING_ROLLOUT_CURVES.csv", index=False,
        )
        pd.DataFrame([
            {"route": route, "status": "DEVELOPMENT_TRAINING_NONFINITE_DIVERGENCE", "formal_test_executed": False}
            for route in FORMAL_ROUTE_IDS
        ]).to_csv(output / "TRACK_B_EXACT_TRAINING_STABILITY_AUDIT.csv", index=False)
        _write_json(output / "TRACK_B_EXACT_TRAINING_SUMMARY.json", {
            **failure,
            "overall_track_b_verdict": "EXACT_TRAINING_DIVERGED_BEFORE_FORMAL_TEST",
            "published_tcn": {"delta_v": 0.042, "delta_q": 0.006},
            "retained_failures": [
                "FULLY_RECURSIVE_STRESS_TEST=NONFINITE_RECURSIVE_DIVERGENCE",
                "EXACT_PUBLISHED_EVALUATOR_ON_FROZEN_PRISM",
            ],
        })
        (output / "TRACK_B_EXACT_TRAINING_FINAL_REPORT.md").write_text(
            "# Track B exact published training final report\n\n"
            "The exact U=10 published-information training roll-in became non-finite on development data. "
            "No clipping, projection, stability constraint, optimizer substitution, or test-driven retuning was added. "
            "Development was not frozen, so the formal 12-trajectory test was not accessed.\n",
            encoding="utf-8",
        )
        return
    neutral_risk = candidate_risk["IDENTITY_CORRECTION"]
    best = min(CANONICAL_W_CANDIDATES, key=candidate_risk.get)
    improvement = (neutral_risk - candidate_risk[best]) / max(neutral_risk, np.finfo(float).eps)
    selected_w = best if best != "IDENTITY_CORRECTION" and improvement >= float(cfg["w_minimum_relative_improvement"]) else "IDENTITY_CORRECTION"
    # After selection, use all official non-test rows for the final estimator.
    final_fit = train + validation
    initial = initial_exact_training_contracts(final_fit, selected_w, ridge_grid, max_condition, max_kkt)
    contracts, fit_audit = fit_exact_published_training_contracts(
        final_fit, initial, ridge_grid, max_condition, max_kkt,
        history=PUBLISHED_HISTORY, unroll=PUBLISHED_TRAINING_UNROLL,
        trajectory_workers=16,
    )
    binding = candidate_binding_audit(contracts)
    validation_result = _aggregate_development(contracts, validation)
    contract_path = output / "FREEZE" / "TRACK_B_EXACT_TRAINING_CONTRACTS.json"
    _save_contracts(contract_path, contracts)
    generating_commit = _git(project.parent, "rev-parse", "HEAD")
    config_path = project / "PRISM_V2_1_1_NEUROBEM_EXACT_PUBLISHED_TRAINING_PACKAGE" / "TRACK_B_EXACT_TRAINING_CONFIG_FROZEN.json"
    freeze = {
        "status": "TRACK_B_EXACT_TRAINING_DEVELOPMENT_FROZEN",
        "development_frozen": True,
        "generating_commit": generating_commit,
        "config_sha256": _sha256(config_path),
        "contracts_sha256": _sha256(contract_path),
        "formal_routes": list(FORMAL_ROUTE_IDS),
        "selected_w_family": selected_w,
        "w_selection": {
            "candidate_risk": candidate_risk,
            "candidate_validation": candidate_validation,
            "best_unprotected": best,
            "relative_improvement_vs_identity": float(improvement),
            "selected": selected_w,
            "published_scores_used": False,
            "training_unroll": PUBLISHED_TRAINING_UNROLL,
        },
        "candidate_binding": binding,
        "training_contract_matched": True,
        "evaluator_contract_matched": True,
        "fit_audit": fit_audit,
        "validation": validation_result,
        "release_split_identity": split,
        "published_scores_known_before_training": True,
        "published_scores_used_for_model_selection": False,
        "previous_test_divergence_known": True,
        "previous_test_used_for_stabilization": False,
        "stabilization_added": False,
        "test_accessed": False,
    }
    _write_json(output / "TRACK_B_EXACT_TRAINING_CONFIG_FROZEN.json", cfg)
    _write_json(output / "TRACK_B_SPLIT_MANIFEST.json", split)
    _write_json(output / "TRACK_B_EXACT_TRAINING_DEVELOPMENT_FREEZE.json", freeze)
    _write_json(output / "TRACK_B_EXACT_TRAINING_DEVELOPMENT_RESULT.json", {
        "status": "PASS" if binding["passed"] and all(value["finite"] for value in validation_result.values()) else "FAILED",
        "training_contract": audit_published_training_contract(),
        "validation": validation_result,
        "test_accessed": False,
    })


def _first_nonfinite_step(result: dict[str, object]) -> int | None:
    for index, (dv, dq) in enumerate(zip(result["per_step_delta_v"], result["per_step_delta_q"], strict=True), 1):
        if not (np.isfinite(float(dv)) and np.isfinite(float(dq))):
            return index
    return None


def run_test(project: Path, release_root: Path, output: Path) -> None:
    freeze_path = output / "TRACK_B_EXACT_TRAINING_DEVELOPMENT_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "TRACK_B_EXACT_TRAINING_DEVELOPMENT_FROZEN":
        raise RuntimeError("EXACT_TRAINING_DEVELOPMENT_NOT_FROZEN")
    if _git(project.parent, "rev-parse", "HEAD") != freeze["generating_commit"]:
        raise RuntimeError("GENERATING_COMMIT_CHANGED_BEFORE_FORMAL_TEST")
    if _git(project.parent, "status", "--porcelain"):
        raise RuntimeError("WORKTREE_NOT_CLEAN_BEFORE_FORMAL_TEST")
    contract_path = output / "FREEZE" / "TRACK_B_EXACT_TRAINING_CONTRACTS.json"
    if _sha256(contract_path) != freeze["contracts_sha256"]:
        raise RuntimeError("EXACT_TRAINING_CONTRACT_HASH_CHANGED")
    contracts = _load_contracts(contract_path)
    test = _track_b_trajectories(release_root, "test", permit_test=True)
    if len(test) != 12:
        raise RuntimeError(f"EXPECTED_12_PUBLISHED_TEST_TRAJECTORIES:{len(test)}")
    table = {row[0]: row for row in TRACK_B_TABLE_II}
    trajectory_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    aggregate: dict[str, object] = {}
    for route in FORMAL_ROUTE_IDS:
        with ThreadPoolExecutor(max_workers=12, thread_name_prefix="exact-formal-test") as executor:
            evaluated = list(executor.map(
                lambda item: (item, track_b_published_decoupled_evaluator(
                    contracts, route, item.frame, history=PUBLISHED_HISTORY, rollout=PUBLISHED_TEST_ROLLOUT,
                )), test,
            ))
        weights, dvs, dqs, labels = [], [], [], []
        for trajectory, result in evaluated:
            weight = int(result["sliding_windows"])
            dv, dq = float(result["delta_v"]), float(result["delta_q"])
            ref = table[trajectory.trajectory_id]
            finite = bool(result["finite"] and np.isfinite(dv) and np.isfinite(dq))
            label = "NONFINITE" if not finite else ("WIN" if dv < ref[4] and dq < ref[5] else ("TIE" if np.isclose(dv, ref[4]) and np.isclose(dq, ref[5]) else "LOSS"))
            labels.append(label); weights.append(weight); dvs.append(dv); dqs.append(dq)
            trajectory_rows.append({
                "trajectory": trajectory.trajectory_id, "route": route, "finite": finite,
                "delta_v": dv, "delta_q": dq, "TCN_delta_v": ref[4], "TCN_delta_q": ref[5],
                "relative_gap_delta_v": (dv-ref[4])/ref[4], "relative_gap_delta_q": (dq-ref[5])/ref[5],
                "win_tie_loss": label, "first_divergence_step": _first_nonfinite_step(result),
                "sliding_windows": weight, "maximum_quaternion_norm_error": result["maximum_quaternion_norm_error"],
            })
            for step, (step_v, step_q) in enumerate(zip(result["per_step_delta_v"], result["per_step_delta_q"], strict=True), 1):
                curve_rows.append({"trajectory": trajectory.trajectory_id, "route": route, "step": step, "delta_v": step_v, "delta_q": step_q})
        weight_array = np.asarray(weights, dtype=np.float64)
        dv = float(np.average(np.asarray(dvs), weights=weight_array))
        dq = float(np.average(np.asarray(dqs), weights=weight_array))
        aggregate[route] = {
            "finite": bool(np.isfinite(dv) and np.isfinite(dq) and "NONFINITE" not in labels),
            "delta_v": dv, "delta_q": dq,
            "relative_gap_vs_tcn_delta_v": (dv-0.042)/0.042,
            "relative_gap_vs_tcn_delta_q": (dq-0.006)/0.006,
            "win": labels.count("WIN"), "tie": labels.count("TIE"),
            "loss": labels.count("LOSS"), "nonfinite": labels.count("NONFINITE"),
            "first_divergence_step": min((row["first_divergence_step"] for row in trajectory_rows if row["route"] == route and row["first_divergence_step"] is not None), default=None),
        }
    pd.DataFrame(trajectory_rows).to_csv(output / "TRACK_B_EXACT_TRAINING_PER_TRAJECTORY.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "TRACK_B_EXACT_TRAINING_ROLLOUT_CURVES.csv", index=False)
    pd.DataFrame([{"route": route, **value} for route, value in aggregate.items()]).to_csv(output / "TRACK_B_EXACT_TRAINING_METRICS.csv", index=False)
    pd.DataFrame([{"route": route, "status": "FINITE" if value["finite"] else "NONFINITE_RECURSIVE_DIVERGENCE", "first_divergence_step": value["first_divergence_step"]} for route, value in aggregate.items()]).to_csv(output / "TRACK_B_EXACT_TRAINING_STABILITY_AUDIT.csv", index=False)
    _write_json(output / "TRACK_B_EXACT_TRAINING_TEST_RESULT.json", {
        "status": "COMPLETED", "protocol_claim": "EXACT_FULL_PUBLISHED_PROTOCOL",
        "training_contract_matched": True, "evaluator_contract_matched": True,
        "stabilization_added": False, "aggregate": aggregate,
        "published_tcn": {"delta_v": 0.042, "delta_q": 0.006},
        "test_trajectory_count": 12, "test_accessed": True,
    })
    _write_json(output / "TRACK_B_EXACT_TRAINING_TEST_ACCESS_AUDIT.json", {
        "status": "ACCESSED_ONCE_AFTER_DEVELOPMENT_FREEZE", "development_freeze_sha256": _sha256(freeze_path),
        "future_control_used": True, "complementary_measured_state_used": True,
        "future_owned_ground_truth_used": False, "future_target_residual_used": False,
    })


def run_report(project: Path, output: Path) -> None:
    result = json.loads((output / "TRACK_B_EXACT_TRAINING_TEST_RESULT.json").read_text(encoding="utf-8"))
    aggregate = result["aggregate"]
    selected_pf, selected_joint = aggregate["PF_KCW"], aggregate["J_KCW"]
    finite = bool(selected_pf["finite"] and selected_joint["finite"])
    better = finite and selected_pf["delta_v"] <= 0.042 and selected_pf["delta_q"] <= 0.006 and selected_joint["delta_v"] <= 0.042 and selected_joint["delta_q"] <= 0.006
    verdict = "D_FINITE_AND_REACHES_OR_EXCEEDS_TCN" if better else ("C_FINITE_BUT_WEAKER_THAN_TCN" if finite else "B_ALIGNED_TRAINING_STILL_DIVERGES")
    summary = {
        "status": "COMPLETED", "overall_track_b_verdict": verdict,
        "training_contract_matched": True, "evaluator_contract_matched": True,
        "stabilization_added": False, "selected_pf_route": "PF_KCW", "selected_joint_route": "J_KCW",
        "aggregate": aggregate, "published_tcn": {"delta_v": 0.042, "delta_q": 0.006},
        "retained_failures": ["FULLY_RECURSIVE_STRESS_TEST=NONFINITE_RECURSIVE_DIVERGENCE", "EXACT_PUBLISHED_EVALUATOR_ON_FROZEN_PRISM"],
    }
    _write_json(output / "TRACK_B_EXACT_TRAINING_SUMMARY.json", summary)
    report = f"""# Track B exact published training final report

Status: `{verdict}`

The training and evaluator information contracts both match the published
decoupled protocol. No stabilization, clipping, projection, spectral penalty,
or test-driven retuning was added. The historical fully-recursive stress-test
divergence and frozen-PRISM evaluator result remain unchanged.

| Route | finite | delta_v | delta_q | relative gap v vs TCN | relative gap q vs TCN | W/T/L/NF |
|---|---:|---:|---:|---:|---:|---:|
"""
    for route in FORMAL_ROUTE_IDS:
        value = aggregate[route]
        report += f"| {route} | {value['finite']} | {value['delta_v']} | {value['delta_q']} | {value['relative_gap_vs_tcn_delta_v']} | {value['relative_gap_vs_tcn_delta_q']} | {value['win']}/{value['tie']}/{value['loss']}/{value['nonfinite']} |\n"
    (output / "TRACK_B_EXACT_TRAINING_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development", "test", "report"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--track-b-release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.stage == "development": run_development(args.project, args.track_b_release_root, args.output)
    elif args.stage == "test": run_test(args.project, args.track_b_release_root, args.output)
    else: run_report(args.project, args.output)


if __name__ == "__main__":
    main()
