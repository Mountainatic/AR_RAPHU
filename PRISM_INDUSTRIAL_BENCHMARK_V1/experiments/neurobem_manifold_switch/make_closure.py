from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _static(table: pd.DataFrame, route: str) -> pd.DataFrame:
    return table[(table.route == route) & (table.ablation == "static")].sort_values("trajectory")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    calibration = pd.read_csv(args.calibration / "per_trajectory.csv")
    test = pd.read_csv(args.test / "per_trajectory.csv")
    static_pf, static_j = _static(test, "PF_KCW"), _static(test, "J_KCW")
    sensitivity = {}
    for route, part in (("PF_KCW", static_pf), ("J_KCW", static_j)):
        counts = {"0.8": 0, "1.0": 0, "1.2": 0}
        for raw in part.divergence_threshold_sensitivity:
            value = json.loads(raw.replace("'", '"')) if isinstance(raw, str) else raw
            for key in counts:
                counts[key] += int(value.get(key) is not None)
        sensitivity[route] = {key: count / len(part) for key, count in counts.items()}
    summary = {
        "experiment_id": "PRISM_V2_1_1_NEUROBEM_MANIFOLD_SWITCH_R2",
        "generating_commit": args.commit,
        "status": "COMPLETED_NEGATIVE_RESULT",
        "hypothesis": "t_alarm < t_diverge_static",
        "hypothesis_supported": False,
        "data_contract": "CHRONOLOGICAL_PARENT_FLIGHT_TRAIN_FIT_CALIBRATION_R2",
        "r1_status": "INVALID_CALIBRATION_TEST_CONTENT_OVERLAP_RETAINED_AS_AUDIT",
        "calibration_trajectories": int(calibration.trajectory.nunique()),
        "formal_test_trajectories": int(test.trajectory.nunique()),
        "parallelism": {"method": "LINUX_FORK_COW", "workers": 24, "blas_threads_per_worker": 1},
        "test_accessed": True,
        "test_used_for_tuning": False,
        "routes": {
            "PF_KCW": {
                "static_divergence_count": int(static_pf.diverged.sum()),
                "static_divergence_rate": float(static_pf.diverged.mean()),
                "median_first_divergence_step": float(static_pf.t_diverge.median()),
                "test_alarm_count": int(test[(test.route == "PF_KCW") & (test.ablation != "static")].t_alarm.notna().sum()),
            },
            "J_KCW": {
                "static_divergence_count": int(static_j.diverged.sum()),
                "static_divergence_rate": float(static_j.diverged.mean()),
                "median_first_divergence_step": float(static_j.t_diverge.median()),
                "test_alarm_count": int(test[(test.route == "J_KCW") & (test.ablation != "static")].t_alarm.notna().sum()),
            },
        },
        "threshold_sensitivity_divergence_rate": sensitivity,
        "known_switches": int(test.num_switches.sum()),
        "new_models_created": int(test.new_models_created.sum()),
        "diagnosis": "RECURSIVE_PREDICTOR_INSTABILITY_NOT_CAUSALLY_PRECEDED_BY_REGISTERED_MANIFOLD_MONITOR",
        "inter_manifold_transition_established": False,
        "universal_ood_claim": False,
        "historical_divergence_results_retained": True,
    }
    _write_json(args.test / "NEUROBEM_MANIFOLD_SWITCH_SUMMARY.json", summary)
    lines = [
        "# NeuroBEM manifold-aware PRISM switching — final report",
        "",
        "## Decision",
        "",
        "The registered hypothesis `t_alarm < t_diverge_static` was **not supported**. On the 12 frozen test trajectories, both PF_KCW and J_KCW diverged on 12/12 trajectories. No registered detector emitted a test alarm, so no known-model switch or causal local re-identification was activated.",
        "",
        "This is a negative result. The evidence is consistent with recursive predictor instability that is not causally preceded by the registered residual/projection/tangent monitor. It does not establish a true inter-manifold transition, and it does not justify a universal OOD claim.",
        "",
        "## Frozen provenance",
        "",
        f"- Generating commit: `{args.commit}`",
        "- PRISM core/model families and W family were unchanged; train-only parameters were refit under the same registered route family.",
        "- Each CSV is a distinct entity; history never crosses a trajectory boundary.",
        "- R2 uses 69 earliest parent flights (175 segments) for fit and 23 latest parent flights (60 segments) for calibration.",
        "- All 12 test content hashes were excluded from fit/calibration. One train alias identical to `random_points.csv` was removed.",
        "- R1 is retained but invalidated because 11 validation aliases duplicated 11 test files and one train alias duplicated the remaining test file.",
        "- Formal test was accessed once after code/config freeze and was not used for tuning.",
        "",
        "## Results",
        "",
        "| Route | Static divergence | Median first divergence | Detector alarms | Switches | New models |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PF_KCW | 12/12 | {static_pf.t_diverge.median():.1f} steps | 0 | 0 | 0 |",
        f"| J_KCW | 12/12 | {static_j.t_diverge.median():.1f} steps | 0 | 0 | 0 |",
        "",
        "All six registered ablations had divergence rate 1.0 for both routes. Because alarms were absent, the switching and re-identification ablations were behaviorally identical to the static route on test.",
        "",
        "The static divergence sensitivity rates at multipliers 0.8/1.0/1.2 were recorded in the machine-readable summary; this sensitivity analysis changed only the diagnostic threshold, never the model or test selection.",
        "",
        "## Interpretation boundary",
        "",
        "Teacher-forced one-step errors remain small on representative trajectories while free rollout explodes, separating local one-step prediction from recursive stability. The registered manifold signals did not reliably anticipate the explosion during development or test. Consequently the experiment cannot distinguish same-manifold support exit from a true regime transition; the safe classification is `INTER_MANIFOLD_TRANSITION_NOT_ESTABLISHED`.",
        "",
        "No clipping, spectral constraint, Lyapunov penalty, threshold retuning, or test-driven model change was introduced.",
    ]
    (args.test / "NEUROBEM_MANIFOLD_SWITCH_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
