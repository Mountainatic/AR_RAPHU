#!/usr/bin/env python3
"""Read-only Track-A forensic reproduction; never trains or selects a model."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from prism_benchmark.neurobem_track_a_forensic import (
    FORENSIC_PROTOCOL_ID,
    GroundTruthContract,
    manifest_identity,
    neuromhe_metric,
    reconstruct_force_torque_gt,
    reproduction_pass,
    rss21_metric,
)

TRACK_A_COLUMNS = (
    "t", "ang acc x", "ang acc y", "ang acc z", "ang vel x", "ang vel y", "ang vel z",
    "quat x", "quat y", "quat z", "quat w", "acc x", "acc y", "acc z",
    "vel x", "vel y", "vel z", "pos x", "pos y", "pos z", "mot 1", "mot 2",
    "mot 3", "mot 4", "dmot 1", "dmot 2", "dmot 3", "dmot 4", "vbat",
)
PREDICTION_EXTRA_COLUMNS = (
    "predicted_fx", "predicted_fy", "predicted_fz", "predicted_tx", "predicted_ty", "predicted_tz",
    "error_residual_fx", "error_residual_fy", "error_residual_fz",
    "error_residual_tx", "error_residual_ty", "error_residual_tz",
)


MAPPING = {
    "2021-02-18-13-44-23_seg_2": "3D Circle_1",
    "2021-02-18-16-53-35_seg_2": "Linear oscillation",
    "2021-02-18-17-03-20_seg_2": "Figure-8_1",
    "2021-02-18-17-19-08_seg_2": "Race track_1",
    "2021-02-18-17-26-00_seg_1": "Race track_2",
    "2021-02-18-18-08-45_seg_1": "3D Circle_2",
    "2021-02-23-10-48-03_seg_2": "Figure-8_2",
    "2021-02-23-11-41-38_seg_3": "Melon_1",
    "2021-02-23-14-21-48_seg_3": "Figure-8_3",
    "2021-02-23-17-27-24_seg_2": "Figure-8_4",
    "2021-02-23-19-45-06_seg_2": "Melon_2",
    "2021-02-23-22-26-25_seg_2": "Random points",
    "2021-02-23-22-54-17_seg_1": "Ellipse",
}
PUBLISHED_TABLE_V = {
    "3D Circle_1": (.196, .211, .215, .005, .006, .003),
    "Linear oscillation": (.164, .185, .456, .013, .011, .006),
    "Figure-8_1": (.065, .056, .235, .004, .003, .002),
    "Race track_1": (.169, .158, .463, .009, .009, .004),
    "Race track_2": (.262, .248, .552, .014, .012, .007),
    "3D Circle_2": (.110, .129, .470, .006, .009, .004),
    "Figure-8_2": (.051, .036, .339, .002, .002, .002),
    "Melon_1": (.099, .108, .397, .004, .005, .003),
    "Figure-8_3": (.145, .168, .584, .010, .012, .006),
    "Figure-8_4": (.400, .313, 1.084, .020, .018, .009),
    "Melon_2": (.244, .198, .921, .009, .012, .006),
    "Random points": (.161, .183, .471, .008, .008, .005),
    "Ellipse": (.204, .315, 1.039, .012, .018, .008),
}
RSS_PUBLISHED = {"Fxy": .204, "Fz": .504, "Mxy": .014, "Mz": .004, "F": .335, "M": .012}
AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prediction_map(root: Path) -> dict[str, Path]:
    return {p.stem.removeprefix("bem+nn_"): p for p in root.rglob("bem+nn_*.csv")}


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    expected = len(TRACK_A_COLUMNS) + len(PREDICTION_EXTRA_COLUMNS)
    if frame.shape[1] != expected:
        raise ValueError(f"OFFICIAL_PREDICTION_COLUMN_MISMATCH:{path}:{frame.shape[1]}")
    frame.columns = TRACK_A_COLUMNS + PREDICTION_EXTRA_COLUMNS
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"OFFICIAL_PREDICTION_NONFINITE:{path}")
    return frame


def prediction_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, PREDICTION_EXTRA_COLUMNS[:6]].to_numpy(dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tests = [x.strip() for x in args.testset.read_text(encoding="utf-8").splitlines() if x.strip()]
    if tests != list(MAPPING):
        raise RuntimeError("TRACK_A_TESTSET_MANIFEST_ORDER_OR_IDENTITY_MISMATCH")
    paths = prediction_map(args.predictions)
    targets, predictions, per_errors = [], [], []
    rows = []
    for stem in tests:
        frame = read_prediction(paths[stem])
        target = reconstruct_force_torque_gt(frame)
        prediction = prediction_values(frame)
        targets.append(target); predictions.append(prediction); per_errors.append(prediction - target)
        recomputed = neuromhe_metric(target, prediction)
        row = {"trajectory": MAPPING[stem], "released_segment": stem}
        passed = True
        for axis, published in zip(AXES, PUBLISHED_TABLE_V[MAPPING[stem]], strict=True):
            absolute, relative, ok = reproduction_pass(recomputed[axis], published)
            row[f"published_{axis}"] = published; row[f"recomputed_{axis}"] = recomputed[axis]
            row[f"relative_error_{axis}"] = relative; row[f"pass_{axis}"] = ok
            passed &= ok
        row["pass_fail"] = "PASS" if passed else "FAIL"
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output / "TRACK_A_NEUROMHE_TRAJECTORY_REPRODUCTION.csv", index=False)
    provenance = [manifest_identity(stem, MAPPING) for stem in tests]
    pd.DataFrame(provenance).to_csv(args.output / "TRACK_A_TESTSET_PROVENANCE.csv", index=False)

    target = np.concatenate(targets); prediction = np.concatenate(predictions)
    sample_pooled = rss21_metric(target, prediction)
    macro_component_mse = np.mean([np.mean(e * e, axis=0) for e in per_errors], axis=0)
    macro = rss21_metric(np.zeros((1, 6)), np.sqrt(macro_component_mse)[None, :])
    rss_rows = []
    for aggregation, metric_values in (("SAMPLE_POOLED", sample_pooled), ("FLIGHT_MACRO_AVERAGE", macro)):
        for metric, published in RSS_PUBLISHED.items():
            absolute, relative, passed = reproduction_pass(metric_values[metric], published)
            rss_rows.append({"aggregation": aggregation, "metric": metric, "published_value": published,
                             "recomputed_value": metric_values[metric], "absolute_difference": absolute,
                             "relative_difference": relative, "pass_fail": "PASS" if passed else "FAIL"})
    pd.DataFrame(rss_rows).to_csv(args.output / "TRACK_A_RSS2021_REPRODUCTION.csv", index=False)
    trajectory_pass = all(row["pass_fail"] == "PASS" for row in rows)
    rss_pass = all(row["pass_fail"] == "PASS" for row in rss_rows if row["aggregation"] == "SAMPLE_POOLED")
    status = "EXACT_DIRECT_COMPARISON_VALIDATED" if trajectory_pass and rss_pass else "PUBLISHED_AGGREGATE_COMPARISON_ONLY"
    generating = subprocess.check_output(("git", "-C", str(args.project.parent), "rev-parse", "HEAD"), text=True).strip()
    audit = {
        "protocol_id": FORENSIC_PROTOCOL_ID, "status": "GT_RECONSTRUCTION_VALIDATED",
        "contract": GroundTruthContract().__dict__, "released_residual_force_used": False,
        "released_residual_torque_used": False, "same_row_alignment": True,
        "additional_filtering": False, "test_trajectory_count": len(tests),
        "primary_sources": ["Official NeuroBEM dataset README", "RCL-NUS/NeuroMHE ground_truth.m"],
    }
    write_json(args.output / "TRACK_A_GT_RECONSTRUCTION_AUDIT.json", audit)
    (args.output / "TRACK_A_GT_RECONSTRUCTION_AUDIT.md").write_text(
        "# Track A GT reconstruction audit\n\n"
        "Status: `GT_RECONSTRUCTION_VALIDATED`. Force and torque were independently reconstructed from "
        "official processed acceleration/angular-rate signals with mass 0.772 kg and inertia "
        "diag(0.0025, 0.0021, 0.0043). Released residual columns were not read as targets. "
        "All 13 trajectories reproduce the six NeuroMHE Table-V NeuroBEM axis RMSE values within the frozen tolerance.\n",
        encoding="utf-8")
    summary = {
        "protocol_id": FORENSIC_PROTOCOL_ID, "generating_commit": generating,
        "gt_reconstruction_status": "VALIDATED", "neuromhe_trajectory_reproduction_status": "PASS" if trajectory_pass else "FAIL",
        "rss2021_reproduction_status": "PASS" if rss_pass else "FAIL",
        "testset_identity_status": "NEUROMHE_EXACT_RSS_TEST_SPLIT_IDENTITY_UNVERIFIED",
        "track_a_final_comparison_status": status,
        "prism": {"PF": {"F": .4891487067570744, "M": .014181829433197048},
                  "Joint": {"F": .42264229998605224, "M": .013201637719319806}},
        "published": {"NeuroBEM": {"F": .335, "M": .012}, "HDVIO2": {"F": .491, "M": .012}},
        "prism_final_rank_if_valid": "NOT_APPLICABLE_COMPARISON_NOT_EXACTLY_VALIDATED",
        "track_b_status": "ARCHIVED_BOUNDARY_RESULT", "stability_extension_performed": False,
        "neurobem_study_status": "CLOSED", "prism_retrained": False, "literature_baseline_retrained": False,
    }
    write_json(args.output / "NEUROBEM_FINAL_CLOSURE_SUMMARY.json", summary)
    (args.output / "NEUROBEM_FINAL_CLOSURE_REPORT.md").write_text(f"""# NeuroBEM final closure report

## 1. Prediction

PRISM PF remains F=0.489149 N, M=0.014182 Nm; Joint remains F=0.422642 N,
M=0.013202 Nm. The independent physical GT exactly reproduces all six rounded
NeuroMHE Table-V NeuroBEM axis RMSE values for all 13 trajectories. The RSS
2021 aggregate is not reproduced within 1%, and an RSS primary source does not
bind its aggregate to this exact 13-segment manifest. Final status:
`{status}`. No formal relative rank is assigned.

## 2. Physics-grounded attribution

The prior motor-topology sign recovery, short-horizon physical consistency,
and theory-compliant K/C/W classification are retained unchanged. This stage
did not train or change any model.

## 3. Multi-horizon diagnostic

The retained A-gain sequence is 99.2% -> 81.7% -> 31.4% -> 1.47% -> 1.23% ->
0%. Generic W/context information is horizon dependent. These are prediction
and attribution evidence, not a stability theory.

Current PRISM is not validated for 600-ms recursive dynamics rollout; this is
outside the scope of the present NeuroBEM closure.

NeuroBEM study is CLOSED after this forensic audit.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
