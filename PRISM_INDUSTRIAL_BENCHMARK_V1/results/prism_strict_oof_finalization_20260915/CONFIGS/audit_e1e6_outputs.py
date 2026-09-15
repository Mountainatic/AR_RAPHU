from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_STATUSES = {"COMPLETED", "PARTIAL", "NOT_RUN", "PROTOCOL_BLOCKED", "INVALID"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[str] = []
    checks: dict[str, Any] = {}

    required = {
        "DIAGNOSTICS/SRU_SO2_ETA_COUNTERFACTUAL.csv": 2,
        "DIAGNOSTICS/TEP_JOINT_VS_A_COUNTERFACTUAL.csv": 114,
        "E1_STAGEWISE/table_stagewise.csv": 9,
        "E1_STAGEWISE/admission_margin_distribution.csv": 96,
        "E2_SYNTHETIC/identifiable_formal_30seed.csv": 120,
        "E2_SYNTHETIC/null_calibration.csv": 4,
        "E3_MULTISCALE/equal_budget_multiscale.csv": 40,
        "E3_MULTISCALE/paired_multiscale_gain.csv": 20,
        "E4A_GRID_SENSITIVITY/grid_density_runs.csv": 120,
        "E4B_FAMILY_ABLATION/family_ablation_runs.csv": 200,
        "E6_MEASUREMENT_ROBUSTNESS/n1_frozen_model.csv": 750,
        "E6_MEASUREMENT_ROBUSTNESS/n2_reidentification.csv": 3800,
        "E6_MEASUREMENT_ROBUSTNESS/n2_robustness_summary.csv": 95,
    }
    frames: dict[str, pd.DataFrame] = {}
    for relative, expected_rows in required.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing_or_empty:{relative}")
            continue
        frame = pd.read_csv(path)
        frames[relative] = frame
        if len(frame) != expected_rows:
            failures.append(f"row_count:{relative}:{len(frame)}!={expected_rows}")
    checks["required_csv_row_counts"] = {
        name: len(frame) for name, frame in frames.items()
    }

    status_rows = []
    for path in sorted(root.rglob("*STATUS*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        status = value.get("status")
        status_rows.append({"file": str(path.relative_to(root)), "status": status})
        if status not in ALLOWED_STATUSES:
            failures.append(f"invalid_status:{path.relative_to(root)}:{status}")
    checks["statuses"] = status_rows

    access_true = []
    for path in sorted(root.rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        if '"test_accessed": true' in text or '"ood_accessed": true' in text:
            access_true.append(str(path.relative_to(root)))
    for relative, frame in frames.items():
        for field in ("test_accessed", "ood_accessed"):
            if field in frame and frame[field].astype(bool).any():
                access_true.append(f"{relative}:{field}")
    checks["test_or_ood_access_true"] = access_true
    if access_true:
        failures.append("test_or_ood_access_detected")

    unique_keys = {
        "DIAGNOSTICS/SRU_SO2_ETA_COUNTERFACTUAL.csv": ["eta"],
        "DIAGNOSTICS/TEP_JOINT_VS_A_COUNTERFACTUAL.csv": ["comparison_set", "scope", "candidate_id"],
        "E1_STAGEWISE/table_stagewise.csv": ["task", "view"],
        "E2_SYNTHETIC/identifiable_formal_30seed.csv": ["regime", "seed"],
        "E3_MULTISCALE/equal_budget_multiscale.csv": ["task", "seed", "arm"],
        "E4A_GRID_SENSITIVITY/grid_density_runs.csv": ["condition", "regime", "seed"],
        "E4B_FAMILY_ABLATION/family_ablation_runs.csv": ["condition", "regime", "seed"],
        "E6_MEASUREMENT_ROBUSTNESS/n1_frozen_model.csv": ["view", "mode", "condition", "magnitude", "seed"],
        "E6_MEASUREMENT_ROBUSTNESS/n2_reidentification.csv": ["view", "mode", "condition", "magnitude", "seed", "outer_fold"],
    }
    duplicate_counts = {}
    for relative, keys in unique_keys.items():
        if relative not in frames:
            continue
        missing = [key for key in keys if key not in frames[relative]]
        if missing:
            failures.append(f"missing_key_columns:{relative}:{','.join(missing)}")
            continue
        count = int(frames[relative].duplicated(keys).sum())
        duplicate_counts[relative] = count
        if count:
            failures.append(f"duplicate_keys:{relative}:{count}")
    checks["duplicate_primary_keys"] = duplicate_counts

    core_metrics = {
        "DIAGNOSTICS/SRU_SO2_ETA_COUNTERFACTUAL.csv": ["outer_oof_risk", "Delta_RMSE", "Delta_R2", "Level_R2"],
        "DIAGNOSTICS/TEP_JOINT_VS_A_COUNTERFACTUAL.csv": ["evaluation_mse", "Delta_RMSE", "Delta_R2", "Level_R2"],
        "E1_STAGEWISE/table_stagewise.csv": ["K_RMSE", "KC_RMSE", "KCW_RMSE", "KCWA_RMSE", "J_RMSE"],
        "E2_SYNTHETIC/identifiable_formal_30seed.csv": ["rmse", "C_margin", "W_margin", "A_margin"],
        "E3_MULTISCALE/equal_budget_multiscale.csv": ["Delta_RMSE", "Delta_R2", "Level_R2", "wall_time_seconds"],
        "E4A_GRID_SENSITIVITY/grid_density_runs.csv": ["rmse", "parameter_count"],
        "E4B_FAMILY_ABLATION/family_ablation_runs.csv": ["rmse", "parameter_count"],
        "E6_MEASUREMENT_ROBUSTNESS/n1_frozen_model.csv": ["Delta_RMSE", "Delta_R2", "Level_R2", "persistence_skill"],
        "E6_MEASUREMENT_ROBUSTNESS/n2_reidentification.csv": ["Delta_RMSE", "Delta_R2", "Level_R2", "persistence_skill", "C_margin", "W_margin", "A_margin"],
    }
    finite_checks = {}
    for relative, columns in core_metrics.items():
        if relative not in frames:
            continue
        missing = [column for column in columns if column not in frames[relative]]
        if missing:
            failures.append(f"missing_metric_columns:{relative}:{','.join(missing)}")
            continue
        values = frames[relative][columns].to_numpy(dtype=np.float64)
        count = int((~np.isfinite(values)).sum())
        finite_checks[relative] = count
        if count:
            failures.append(f"nonfinite_core_metrics:{relative}:{count}")
    checks["nonfinite_core_metric_cells"] = finite_checks

    e3 = frames.get("E3_MULTISCALE/equal_budget_multiscale.csv")
    if e3 is not None:
        paired = e3.groupby(["task", "seed"])
        e3_cert = {
            "same_budget": bool((paired["candidate_fits"].nunique() == 1).all()),
            "same_support": bool((paired["support_hash"].nunique() == 1).all()),
            "positive_wall_time": bool((e3["wall_time_seconds"] > 0).all()),
        }
        checks["e3_certificates"] = e3_cert
        if not all(e3_cert.values()):
            failures.append("e3_pairing_certificate_failed")

    e6 = frames.get("E6_MEASUREMENT_ROBUSTNESS/n2_reidentification.csv")
    if e6 is not None:
        clean = e6[(e6["condition"] == "GAUSSIAN") & (e6["magnitude"] == 0.0)]
        clean_groups = clean.groupby(["view", "mode", "outer_fold"])
        bundle_sizes = e6.groupby(["view", "mode", "seed", "outer_fold"]).size()
        e6_cert = {
            "outer_folds": sorted(int(value) for value in e6["outer_fold"].unique()),
            "seeds": sorted(int(value) for value in e6["seed"].unique()),
            "conditions_per_view_mode_seed_fold_min": int(bundle_sizes.min()),
            "conditions_per_view_mode_seed_fold_max": int(bundle_sizes.max()),
            "alpha_zero_prediction_invariant": bool((clean_groups["prediction_hash"].nunique() == 1).all()),
            "alpha_zero_route_invariant": bool((clean_groups["stage_vector"].nunique() == 1).all()),
            "support_constant": bool(
                (e6.groupby(["view", "mode", "outer_fold"])["support_hash"].nunique() == 1).all()
            ),
            "process_sigma_outer_fold_constant": bool(
                (e6.groupby(["view", "mode", "outer_fold"])["outer_train_process_sigma_hash"].nunique() == 1).all()
            ),
            "target_sigma_outer_fold_constant": bool(
                (e6.groupby(["view", "mode", "outer_fold"])["outer_train_target_sigma"].nunique() == 1).all()
            ),
        }
        checks["e6_certificates"] = e6_cert
        expected = {
            "outer_folds": [0, 1, 2, 3],
            "seeds": list(range(10)),
            "conditions_per_view_mode_seed_fold_min": 19,
            "conditions_per_view_mode_seed_fold_max": 19,
            "alpha_zero_prediction_invariant": True,
            "alpha_zero_route_invariant": True,
            "support_constant": True,
            "process_sigma_outer_fold_constant": True,
            "target_sigma_outer_fold_constant": True,
        }
        if e6_cert != expected:
            failures.append("e6_certificate_failed")

    freeze = root / "PROVENANCE/FINAL_ABLATION_PROTOCOL_FREEZE.json"
    recorded_hash = root / "PROVENANCE/PROTOCOL_FREEZE_SHA256.json"
    if freeze.is_file() and recorded_hash.is_file():
        expected_hash = json.loads(recorded_hash.read_text(encoding="utf-8"))["sha256"]
        checks["protocol_freeze_sha256"] = sha256(freeze)
        if checks["protocol_freeze_sha256"] != expected_hash:
            failures.append("protocol_freeze_hash_mismatch")
    git_state = (root / "PROVENANCE/GIT_STATE.txt").read_text(encoding="utf-8")
    checks["expected_commit"] = args.expected_commit
    checks["git_state_commit_match"] = f"commit={args.expected_commit}" in git_state
    checks["git_state_clean"] = "status=CLEAN" in git_state and "diff=CLEAN" in git_state
    if not checks["git_state_commit_match"]:
        failures.append("git_commit_mismatch")
    if not checks["git_state_clean"]:
        failures.append("git_state_not_clean")

    result = {
        "status": "COMPLETED" if not failures else "INVALID",
        "root": str(root),
        "failures": failures,
        "checks": checks,
    }
    output = root / "PROVENANCE/CODE_AND_RESULT_AUDIT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
