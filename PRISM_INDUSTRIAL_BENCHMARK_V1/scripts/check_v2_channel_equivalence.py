from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import BaseAccessor, input_columns
from prism_benchmark.v2_k import run_channel
from prism_benchmark.v2_views import development_input_views
import prism_benchmark.v2_k as v2_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--proxy-policy", required=True)
    parser.add_argument("--channel", required=True)
    args = parser.parse_args()
    matches = [
        view
        for view in development_input_views(args.shared)
        if view.head.head_id == args.head and view.proxy_policy == args.proxy_policy
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one view, got {len(matches)}")
    view = matches[0]
    if args.channel not in input_columns(args.shared, view.head.task_id, view.proxy_policy):
        raise SystemExit("channel is not present in the immutable input view")
    accessor = BaseAccessor(args.shared, view.head.dataset, "validation", [args.channel])
    accessor.warm_prefixes([args.channel])
    v2_k._PRELOADED_ACCESSORS = {view.head.dataset: accessor}
    candidate = run_channel(args.shared, args.project, args.candidate_output, view, args.channel)
    reference_path = (
        args.reference_output
        / "DEVELOPMENT"
        / "CHANNEL_AUDIT"
        / view.head.head_id
        / view.proxy_policy
        / args.channel
        / "RESULT.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    fields = [
        "status",
        "selected_profile",
        "retained_profiles",
        "selected_family",
        "selected_m_tau",
        "selected_m_x",
        "selected_lambdas",
        "active",
    ]
    field_equal = {field: reference.get(field) == candidate.get(field) for field in fields}
    old = pd.read_parquet(args.reference_output / reference["prediction_path"])
    new = pd.read_parquet(args.candidate_output / candidate["prediction_path"])
    old = old.sort_values("view_sample_id").reset_index(drop=True)
    new = new.sort_values("view_sample_id").reset_index(drop=True)
    if not old["view_sample_id"].equals(new["view_sample_id"]):
        raise RuntimeError("immutable prediction sample IDs changed")
    difference = np.abs(old["y_pred"].to_numpy(dtype=np.float64) - new["y_pred"].to_numpy(dtype=np.float64))
    report = {
        "status": "PASS" if all(field_equal.values()) and float(difference.max(initial=0.0)) <= 1e-8 else "FAIL",
        "field_equal": field_equal,
        "prediction_rows": len(new),
        "prediction_max_abs_difference": float(difference.max(initial=0.0)),
        "prediction_mean_abs_difference": float(difference.mean(dtype=np.float64)),
        "reference_prediction_sha256": reference["prediction_sha256"],
        "candidate_prediction_sha256": candidate["prediction_sha256"],
    }
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
