"""Finalize reporting-only audits after the frozen NeuroBEM test access.

This tool never loads NeuroBEM samples or predictions.  It reads development
contracts and already-aggregated result tables only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism_benchmark.neurobem_experiment import ridge_from_json
from prism_benchmark.neurobem_multihorizon import integrated_k_topology


AXES = ("ROLL", "PITCH", "YAW", "BODY_Z")
EXPECTED = {"ROLL": (-1, -1, 1, 1), "PITCH": (1, -1, 1, -1), "BODY_Z": (1, 1, 1, 1)}
HORIZONS = (1, 4, 8, 20, 40, 80)
MILLISECONDS = {1: 2.5, 4: 10.0, 8: 20.0, 20: 50.0, 40: 100.0, 80: 200.0}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def topology(root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        result = load(root / "DEVELOPMENT" / f"HORIZON_{horizon}" / "RESULT.json")
        history = int(result["K"]["selected_history"])
        contracts = {"FULL_DEVELOPMENT": result["K"]["train_to_validation_contract"]}
        contracts.update({f"FOLD_{key}": value for key, value in result["K"]["selected_fold_contracts"].items()})
        for source, value in contracts.items():
            matrix = integrated_k_topology(ridge_from_json(value), history)
            for motor in range(4):
                for axis_index, axis in enumerate(AXES):
                    sign = int(np.sign(matrix[motor, axis_index]))
                    expected = None if axis == "YAW" else EXPECTED[axis][motor]
                    rows.append({
                        "phase": "DEVELOPMENT",
                        "contract_source": source,
                        "horizon": horizon,
                        "horizon_ms": MILLISECONDS[horizon],
                        "motor": motor + 1,
                        "axis": axis,
                        "integrated_coefficient": matrix[motor, axis_index],
                        "sign": sign,
                        "expected_sign": "DESCRIPTIVE_ONLY" if expected is None else expected,
                        "expected_sign_agreement": "NOT_APPLICABLE" if expected is None else sign == expected,
                        "test_target_used_for_sign_decision": False,
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "K_TOPOLOGY_STABILITY.csv", index=False)
    formal = frame[(frame.contract_source == "FULL_DEVELOPMENT") & (frame.axis != "YAW")].copy()
    formal["agreement"] = formal.expected_sign_agreement.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    by_horizon = {str(int(key)): float(value) for key, value in formal.groupby("horizon").agreement.mean().items()}
    fold = frame[(frame.contract_source.str.startswith("FOLD_")) & (frame.axis != "YAW")].copy()
    fold["agreement"] = fold.expected_sign_agreement.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    fold_agreement = {str(int(key)): float(value) for key, value in fold.groupby("horizon").agreement.mean().items()}
    sign_stability: dict[str, float] = {}
    for horizon, group in fold.groupby("horizon"):
        stable = []
        for _, cell in group.groupby(["motor", "axis"]):
            stable.append(len(set(cell.sign)) == 1)
        sign_stability[str(int(horizon))] = float(np.mean(stable))
    result = {
        "status": "COMPLETED_REPORTING_ONLY_FROM_DEVELOPMENT_CONTRACTS",
        "full_development_expected_sign_agreement_by_horizon": by_horizon,
        "fold_expected_sign_agreement_by_horizon": fold_agreement,
        "fold_sign_stability_fraction_by_horizon": sign_stability,
        "yaw": "DESCRIPTIVE_ONLY",
        "test_target_used_for_sign_decision": False,
    }
    (root / "K_TOPOLOGY_SUMMARY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def report(root: Path, topology_summary: dict[str, object]) -> None:
    metrics = pd.read_csv(root / "HORIZON_COMMON_SUPPORT_METRICS.csv")
    pooled = metrics[(metrics.phase == "LOCKED_TEST") & (metrics.axis == "POOLED")]
    table = pooled.pivot(index="horizon", columns="route", values="pooled_normalized_mse")
    w_gain = pd.read_csv(root / "W_HORIZON_GAIN.csv")
    w_gain = w_gain[(w_gain.phase == "LOCKED_TEST") & (w_gain.support == "COMMON_HORIZON_SUPPORT")]
    a_gain = pd.read_csv(root / "A_HORIZON_GAIN.csv")
    a_gain = a_gain[(a_gain.phase == "LOCKED_TEST") & (a_gain.support == "COMMON_HORIZON_SUPPORT")]
    k = pd.read_csv(root / "K_HISTORY_SELECTION.csv")
    summary = load(root / "NEUROBEM_MULTI_HORIZON_RESULT_SUMMARY.json")
    summary["K_topology"] = topology_summary
    summary["official_R1_h1_comparison_status"] = "NOT_APPLICABLE_SAMPLE_SUPPORT_AND_ROUTE_CONTRACT_NOT_IDENTICAL"
    summary["retained_failures"] = [
        "h=20 K Roll MSE-preservation diagnostic failed; the registered horizon was retained.",
        "h=80 K Pitch MSE-preservation diagnostic failed; the registered horizon was retained.",
        "Generic W selected identity at h=20 and h=40.",
        "A selected exact zero for all W arms at h=80.",
        "No route, horizon, or negative ablation was removed after locked-test access.",
    ]
    (root / "NEUROBEM_MULTI_HORIZON_RESULT_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    a_w0 = a_gain[a_gain.W_arm == "W0_IDENTITY"].set_index("horizon").gain
    generic = w_gain[w_gain.gain == "G_GENERIC"].set_index("horizon").value
    aero = w_gain[w_gain.gain == "G_AERO_PRIOR"].set_index("horizon").value
    routes = ("K__W0__NO_A", "K__W1_GENERIC__NO_A", "K__W2_AERO__NO_A", "K__W0__A", "K__W1_GENERIC__A", "K__W2_AERO__A", "PERSISTENCE", "TARGET_ONLY_VAR_AR", "LINEAR_NARX")
    lines = [
        "# NeuroBEM Multi-Horizon × Wiener-Prior Final Report",
        "",
        "Status: **COMPLETED** under `POST_LOCKBOX_PROSPECTIVE_EXTENSION`. The official test set had already been accessed by the historical R1 experiment; this extension froze all new horizons, arms, maturity rules, metrics, and statistics before its one formal multi-horizon access.",
        "",
        "## Direct conclusions",
        "",
        f"1. The W0 A gain decayed from {100*a_w0[1]:.1f}% at 2.5 ms to {100*a_w0[4]:.1f}% at 10 ms, {100*a_w0[8]:.1f}% at 20 ms, {100*a_w0[20]:.1f}% at 50 ms, {100*a_w0[40]:.1f}% at 100 ms, and {100*a_w0[80]:.1f}% at 200 ms. The old one-step gain is therefore dominated by very-short-range mature residual state and does not persist to 200 ms.",
        f"2. Canonical latent-only W produced no-A gains of {', '.join(f'{100*generic[h]:.1f}%' for h in HORIZONS)} across the registered horizons. It was modest at short horizons, selected identity at 50/100 ms, and was not uniformly stable.",
        f"3. The additional W2 aerodynamic-context candidate pool produced incremental no-A gains of {', '.join(f'{100*aero[h]:.1f}%' for h in HORIZONS)}. This is predictive ablation evidence, not causal identification.",
        "4. The high-speed table shows larger W2-versus-W1 gains for several horizons, but the subset is a locked challenge, not OOD and not proof of airflow or drag recovery.",
        "5. K topology agreement and fold stability are reported from development contracts only in `K_TOPOLOGY_STABILITY.csv` and `K_TOPOLOGY_SUMMARY.json`; no test target was used for sign decisions.",
        "6. K selected the 64-sample boundary at h=1,4,8,40 and selected 32 at h=20,80. The grid was not expanded.",
        "7. No-A W2 pooled nMSE changed from {:.4f} at 2.5 ms to {:.4f} at 200 ms.".format(table.loc[1, "K__W2_AERO__NO_A"], table.loc[80, "K__W2_AERO__NO_A"]),
        "8. With-A routes dominated at short horizons but converged to their no-A counterparts when A froze to exact zero at 200 ms.",
        "9. Linear NARX/target-only AR were stronger than PF at the shortest horizons. The registered W2 PF route became competitive or better on pooled nMSE at 50/100/200 ms; all baselines remain reported.",
        "10. Losses, gains, speed localization, and baseline comparisons are predictive evidence only.",
        "11. Only the pre-registered integrated K sign audit is physics-consistency evidence, and h=20 Roll plus h=80 Pitch failed their K MSE-preservation diagnostics.",
        "12. No result identifies true airflow, drag, wind, vortex state, or physical poles. W2 is an aerodynamic-context extension, not canonical PRISM v2.1.1 W.",
        "",
        "## Common-support locked-test pooled normalized MSE",
        "",
        "| h (samples / ms) | " + " | ".join(routes) + " |",
        "|---:" + "|---:" * len(routes) + "|",
    ]
    for horizon in HORIZONS:
        lines.append(f"| {horizon} / {MILLISECONDS[horizon]:g} | " + " | ".join(f"{table.loc[horizon, route]:.6g}" for route in routes) + " |")
    lines += [
        "",
        "## Retained failures and limitations",
        "",
        "- h=20 K failed only the Roll MSE-preservation diagnostic; h=80 K failed only the Pitch counterpart. Both horizons were retained exactly as registered.",
        "- Generic W selected identity at h=20 and h=40. A selected exact zero in every W arm at h=80.",
        "- Direct comparison with historical official R1 h=1 is `NOT_APPLICABLE_SAMPLE_SUPPORT_AND_ROUTE_CONTRACT_NOT_IDENTICAL`; the extension uses a different mature-route support contract.",
        "- No test-driven reselection, candidate deletion, threshold change, or horizon deletion occurred.",
    ]
    (root / "NEUROBEM_MULTI_HORIZON_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def integrity(root: Path) -> None:
    required = [
        "GLOBAL_DEVELOPMENT_FREEZE.json", "TEST_ACCESS_AUDIT.json", "HORIZON_FINAL_METRICS.csv",
        "HORIZON_NATIVE_SUPPORT_METRICS.csv", "HORIZON_COMMON_SUPPORT_METRICS.csv", "W_PRIOR_ABLATION.csv",
        "A_HORIZON_GAIN.csv", "W_HORIZON_GAIN.csv", "SPEED_BIN_METRICS.csv", "HIGH_SPEED_CHALLENGE_METRICS.csv",
        "K_HISTORY_SELECTION.csv", "K_TOPOLOGY_STABILITY.csv", "A_MATURITY_AUDIT.csv", "SAMPLE_SUPPORT_AUDIT.csv",
        "BASELINE_METRICS.csv", "BOOTSTRAP_PARENT_FLIGHT.csv", "BOOTSTRAP_SEGMENT_SENSITIVITY.csv",
        "NEUROBEM_MULTI_HORIZON_FINAL_REPORT.md", "NEUROBEM_MULTI_HORIZON_RESULT_SUMMARY.json", "PYTEST_OUTPUT.txt",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    sample = pd.read_csv(root / "SAMPLE_SUPPORT_AUDIT.csv")
    k = pd.read_csv(root / "K_HISTORY_SELECTION.csv")
    audit = load(root / "TEST_ACCESS_AUDIT.json")
    forbidden = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".parquet", ".npy", ".npz"}]
    result = {
        "status": "PASS" if not missing and not forbidden else "FAILED",
        "required_files_missing": missing,
        "forbidden_large_prediction_artifacts": forbidden,
        "all_route_support_hashes_equal": bool(sample.all_route_support_hashes_equal.astype(bool).all()),
        "horizons_present": sorted(int(value) for value in k.horizon.unique()),
        "test_access_status": audit["status"],
        "test_access_count": audit["current_extension_test_access_count"],
        "test_driven_reselection": audit["test_driven_reselection"],
        "ood_accessed": audit["ood_accessed"],
        "reporting_only": True,
    }
    (root / "RESULT_INTEGRITY_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS" or not result["all_route_support_hashes_equal"]:
        raise RuntimeError(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    summary = topology(args.result_root)
    report(args.result_root, summary)
    integrity(args.result_root)


if __name__ == "__main__":
    main()
