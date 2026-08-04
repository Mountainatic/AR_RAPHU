from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from prism_benchmark.c3_models import run_job
from prism_benchmark.v2_views import development_dynamic_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--availability-scenario", required=True)
    parser.add_argument("--proxy-policy", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    matches = [
        view for view in development_dynamic_views(args.shared)
        if view.head.head_id == args.head
        and view.availability_scenario == args.availability_scenario
        and view.proxy_policy == args.proxy_policy
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one view, got {len(matches)}")
    reference_root = args.reference_output / "BASELINE_DEVELOPMENT" / "C3"
    candidate_root = args.candidate_output / "C3"
    ar_source = reference_root / "PREDICTIONS" / "AR"
    ar_destination = candidate_root / "PREDICTIONS" / "AR"
    ar_destination.parent.mkdir(parents=True, exist_ok=True)
    if not ar_destination.exists():
        os.symlink(ar_source, ar_destination, target_is_directory=True)
    view = matches[0]
    candidate = run_job(args.shared, args.project, candidate_root, view, args.model)
    reference_path = reference_root / "PREDICTIONS" / args.model / view.relative_root / "RESULT.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    fields = ["status", "model", "parameter_count", "selection"]
    field_equal = {field: reference.get(field) == candidate.get(field) for field in fields}
    old = pd.read_parquet(reference_root / reference["prediction_path"])
    new = pd.read_parquet(candidate_root / candidate["prediction_path"])
    old = old.sort_values("view_sample_id").reset_index(drop=True)
    new = new.sort_values("view_sample_id").reset_index(drop=True)
    if not old["view_sample_id"].equals(new["view_sample_id"]):
        raise RuntimeError("immutable prediction sample IDs changed")
    difference = np.abs(
        old["y_pred"].to_numpy(dtype=np.float64) - new["y_pred"].to_numpy(dtype=np.float64)
    )
    report = {
        "status": "PASS" if all(field_equal.values()) and float(difference.max(initial=0.0)) <= 1e-8 else "FAIL",
        "field_equal": field_equal,
        "prediction_rows": len(new),
        "prediction_max_abs_difference": float(difference.max(initial=0.0)),
        "prediction_mean_abs_difference": float(difference.mean(dtype=np.float64)),
    }
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
