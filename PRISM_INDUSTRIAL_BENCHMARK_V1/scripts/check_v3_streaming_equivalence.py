from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from prism_benchmark.v2_c import run_c_view
from prism_benchmark.v2_views import development_input_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--proxy-policy", required=True)
    args = parser.parse_args()
    matches = [
        view for view in development_input_views(args.shared)
        if view.head.head_id == args.head and view.proxy_policy == args.proxy_policy
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one view, got {len(matches)}")
    source = args.reference_output / "DEVELOPMENT" / "CHANNEL_AUDIT"
    destination = args.candidate_output / "DEVELOPMENT" / "CHANNEL_AUDIT"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.symlink(source, destination, target_is_directory=True)
    candidate = run_c_view(args.shared, args.project, args.candidate_output, matches[0])
    reference_path = (
        args.reference_output / "DEVELOPMENT" / "JOINT_BASIS" /
        args.head / args.proxy_policy / "RESULT.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    fields = [
        "status", "selected_family", "selected_alpha", "selected_pairs",
        "active_channels", "global_joint_columns", "candidate_fold_losses",
    ]
    field_equal = {field: reference.get(field) == candidate.get(field) for field in fields}
    old = pd.read_parquet(args.reference_output / reference["prediction_path"])
    new = pd.read_parquet(args.candidate_output / candidate["prediction_path"])
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
