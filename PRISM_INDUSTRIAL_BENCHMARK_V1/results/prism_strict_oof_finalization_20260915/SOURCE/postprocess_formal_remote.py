"""Mechanically audit and summarize the completed frozen formal run."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
    "Debutanizer": {"K", "KC", "KCW", "KCWA", "J"},
    "SRU H2S": {"K", "KC", "KCW", "KCWA", "J"},
    "SRU SO2": {"K", "KC", "KCW", "KCWA", "J"},
    "PMSM proxy-excluded": {"K", "KC", "KCW", "KCWA", "J"},
    "MetroPT P60": {"K", "KC", "KCW", "KCWA", "J"},
    "MetroPT Oil20": {"K", "KC", "KCW", "KCWA", "J"},
    "TEP input-only": {"K", "KC", "KCW", "KCWA"},
    "TEP record-time": {"K", "KC", "KCW", "KCWA", "J"},
    "TEP maturity-5": {"K", "KC", "KCW", "KCWA"},
}
PREDICTION_COLUMNS = {"K": "K", "KC": "KC", "KCW": "KCW", "KCWA": "KCWA", "J": "J"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def prediction_hash(ids: pd.Series, values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode())
        digest.update(b"\0")
    digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def delta_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float | None]:
    error = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    denominator = float(np.sum((y - np.mean(y)) ** 2, dtype=np.float64))
    return (
        float(np.sqrt(np.mean(error * error, dtype=np.float64))),
        float(np.mean(np.abs(error), dtype=np.float64)),
        float(1.0 - np.sum(error * error, dtype=np.float64) / denominator) if denominator else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    args = parser.parse_args()
    formal = args.formal
    status = json.loads((formal / "RUN_STATUS.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED" or status.get("fit_calls") != 0 or status.get("selection_calls") != 0:
        raise RuntimeError("formal run is not completed cleanly")

    table = pd.read_csv(formal / "TABLE_STAGEWISE.csv")
    development = pd.read_csv(args.development)
    dev = development.set_index("task", drop=False)
    audit_rows: list[dict] = []
    baselines: list[dict] = []
    normalized: list[dict] = []

    for task, expected_models in EXPECTED.items():
        task_rows = table.loc[table["task"] == task].copy()
        present = set(task_rows["model"].astype(str))
        if present != expected_models or task_rows["model"].duplicated().any():
            raise RuntimeError(f"formal model inventory mismatch: {task}: {present}")
        prediction_path = formal / "PREDICTIONS" / f"{task.replace(' ', '_')}.parquet"
        predictions = pd.read_parquet(prediction_path)
        if len(predictions) != int(task_rows["rows"].iloc[0]):
            raise RuntimeError(f"prediction row mismatch: {task}")
        y = predictions["y_true"].to_numpy(dtype=np.float64)
        if not np.isfinite(y).all() or predictions["base_origin_id"].duplicated().any():
            raise RuntimeError(f"invalid formal prediction population: {task}")
        for _, source in task_rows.iterrows():
            model = str(source["model"])
            values = predictions[PREDICTION_COLUMNS[model]].to_numpy(dtype=np.float64)
            observed_hash = prediction_hash(predictions["base_origin_id"], values)
            hash_ok = observed_hash == source["prediction_hash"]
            finite = bool(np.isfinite(values).all())
            if not hash_ok or not finite:
                raise RuntimeError(f"formal prediction audit failed: {task}/{model}")
            record = source.to_dict()
            if pd.isna(record.get("formal_membership_rows")):
                record["formal_membership_rows"] = int(source["rows"])
            prefix_field = f"{model}_prefix_params"
            record["parameter_count"] = int(dev.loc[task, prefix_field]) if prefix_field in dev.columns else int(source["parameter_count"])
            record["parameter_count_semantics"] = "PREFIX_CUMULATIVE" if model != "J" else "JOINT_TOTAL"
            normalized.append(record)
            audit_rows.append({
                "task": task,
                "model": model,
                "rows": len(values),
                "finite": finite,
                "prediction_hash_expected": source["prediction_hash"],
                "prediction_hash_observed": observed_hash,
                "prediction_hash_match": hash_ok,
                "status": "PASS",
            })
        rmse, mae, r2 = delta_metrics(y, np.zeros_like(y))
        first = task_rows.iloc[0]
        baselines.append({
            "task": task,
            "view": first["view"],
            "baseline": "Persistence",
            "status": "PASS_FROZEN_DEFINITION",
            "rows": len(y),
            "support_hash": first["support_hash"],
            "rmse": rmse,
            "mae": mae,
            "delta_r2": r2,
            "level_rmse": rmse,
            "level_mae": mae,
            "level_r2": float(first["persistence_r2"]),
        })
        for baseline in ("AR", "ARX", "Linear-NARX"):
            baselines.append({
                "task": task,
                "view": first["view"],
                "baseline": baseline,
                "status": "NOT_APPLICABLE_MISSING_PRE_FROZEN_DEVELOPMENT_CONTRACT",
                "rows": 0,
                "support_hash": "",
                "rmse": np.nan,
                "mae": np.nan,
                "delta_r2": np.nan,
                "level_rmse": np.nan,
                "level_mae": np.nan,
                "level_r2": np.nan,
            })

    normalized_frame = pd.DataFrame(normalized)
    normalized_frame.to_csv(formal / "TABLE_STAGEWISE.csv", index=False)
    pd.DataFrame(baselines).to_csv(formal / "TABLE_BASELINES.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(formal / "FORMAL_PREDICTION_AUDIT.csv", index=False)

    full = normalized_frame.loc[normalized_frame["model"].isin(["KCWA", "J"])].copy()
    full["route"] = np.where(full["model"] == "J", "ACCURACY_ORIENTED_JOINT", "PHYSICS_FIRST_FINAL")
    full.to_csv(formal / "TABLE_FULL_MODEL.csv", index=False)

    effects: list[dict] = []
    for task in EXPECTED:
        current = normalized_frame.loc[normalized_frame["task"] == task].set_index("model")
        for stage, parent, child in (("C", "K", "KC"), ("W", "KC", "KCW"), ("A", "KCW", "KCWA")):
            parent_rmse = float(current.loc[parent, "rmse"])
            child_rmse = float(current.loc[child, "rmse"])
            effects.append({
                "task": task,
                "view": current.loc[child, "view"],
                "stage": stage,
                "development_route": dev.loc[task, f"{stage}_route"],
                "development_admission_margin": float(dev.loc[task, f"{stage}_margin"]),
                "formal_parent_rmse": parent_rmse,
                "formal_child_rmse": child_rmse,
                "formal_relative_rmse_gain": (parent_rmse - child_rmse) / parent_rmse if parent_rmse else np.nan,
                "formal_absolute_mse_gain": parent_rmse**2 - child_rmse**2,
                "exact_zero_required": dev.loc[task, f"{stage}_route"] == "ZERO_IDENTITY",
                "prediction_hash_equal": current.loc[parent, "prediction_hash"] == current.loc[child, "prediction_hash"],
            })
    effect_frame = pd.DataFrame(effects)
    zero_failures = effect_frame.loc[effect_frame["exact_zero_required"] & ~effect_frame["prediction_hash_equal"]]
    if not zero_failures.empty:
        raise RuntimeError(f"formal exact-zero replay failed: {zero_failures[['task', 'stage']].to_dict('records')}")
    effect_frame.to_csv(formal / "TABLE_ADMISSION_AND_EFFECT.csv", index=False)

    physics = normalized_frame.loc[normalized_frame["model"] == "KCWA", ["task", "delta_r2", "persistence_skill", "rmse"]]
    active = effect_frame.loc[effect_frame["development_route"] == "ACTIVE"]
    generalized = active.loc[active["formal_relative_rmse_gain"] > 0]
    boundary = active.loc[active["development_admission_margin"].abs() < 1e-3]
    claims = {
        "A_strict_nested_oof_rejects_unnecessary_stages": "SUPPORTED" if effect_frame["exact_zero_required"].any() and zero_failures.empty else "NOT_SUPPORTED",
        "B_stage_utility_is_task_dependent": "SUPPORTED" if set(effect_frame["development_route"]) == {"ACTIVE", "ZERO_IDENTITY"} else "PARTIALLY_SUPPORTED",
        "C_margin_distinguishes_boundary_activity": "SUPPORTED" if len(boundary) else "PARTIALLY_SUPPORTED",
        "D_heterogeneous_formal_generalization": (
            "SUPPORTED" if (physics["persistence_skill"] > 0).all()
            else "PARTIALLY_SUPPORTED" if (physics["persistence_skill"] > 0).mean() >= 0.75
            else "NOT_SUPPORTED"
        ),
        "E_multiscale_beyond_search_budget": "PARTIALLY_SUPPORTED",
        "F_unique_structural_interpretation": "NOT_SUPPORTED",
    }
    write_json(formal / "FORMAL_SUMMARY.json", {
        "status": "COMPLETED_AND_AUDITED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_test_opened": True,
        "formal_views": len(EXPECTED),
        "formal_model_rows": len(normalized_frame),
        "prediction_audit_failures": 0,
        "exact_zero_failures": 0,
        "active_stage_count": len(active),
        "active_stage_positive_formal_gain_count": len(generalized),
        "claims": claims,
        "joint_exception": "TEP maturity-5: NOT_APPLICABLE_MISSING_FROZEN_FEATURE_CONTRACT",
        "baseline_exception": "AR/ARX/Linear-NARX: NOT_APPLICABLE_MISSING_PRE_FROZEN_DEVELOPMENT_CONTRACT",
    })

    best = physics.sort_values("persistence_skill", ascending=False).head(3)
    worst = physics.sort_values("persistence_skill", ascending=True).head(3)
    report = f"""# Formal Test Master Report

## Status

The one-shot formal test is complete and audited for {len(EXPECTED)} frozen views. No fit or selection call was made after opening formal test. All {len(normalized_frame)} saved model rows passed finite-value and prediction-hash verification. Frozen ZERO routes replayed exactly.

PMSM formal membership required a mechanical common-support restriction using the already frozen maximum input history. This restriction used only sample metadata (`origin`, `latest_available_target_index`, and `causal_history_floor`) and did not inspect target values, predictions, errors, or difficulty.

## Formal Findings

- Physics-first final models with positive persistence skill: {int((physics['persistence_skill'] > 0).sum())}/{len(physics)} views.
- Development-active stages with positive formal incremental RMSE gain: {len(generalized)}/{len(active)}.
- Boundary-active development stages (`|margin| < 1e-3`): {len(boundary)}.
- TEP maturity-5 Joint: `NOT_APPLICABLE_MISSING_FROZEN_FEATURE_CONTRACT`.
- AR/ARX/Linear-NARX formal baselines: `NOT_APPLICABLE_MISSING_PRE_FROZEN_DEVELOPMENT_CONTRACT`.

Highest persistence skill views: {', '.join(best['task'])}.

Lowest persistence skill views: {', '.join(worst['task'])}.

For TEP, delta metrics and level/persistence metrics must be read together. Strong level scores can be largely inherited from the persistence anchor when delta R2 is near zero.

## Interpretation

Formal evidence supports task-dependent stage utility and exact rejection of unnecessary stages. It does not support unique physical-structure identification. The existing E3 result supports multiscale value on Debutanizer under equal search budget but not universally, so the cross-task multiscale claim remains partial.
"""
    (formal / "FORMAL_TEST_MASTER_REPORT.md").write_text(report, encoding="utf-8")
    claim_lines = ["# Final Claim Recommendation", ""]
    for key, value in claims.items():
        claim_lines.append(f"- `{key}`: **{value}**")
    claim_lines += ["", "Use: PRISM identifies admissible predictive structures, not necessarily unique physical structures.", ""]
    (formal / "FINAL_CLAIM_RECOMMENDATION.md").write_text("\n".join(claim_lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(normalized_frame), "claims": claims}, indent=2))


if __name__ == "__main__":
    main()
