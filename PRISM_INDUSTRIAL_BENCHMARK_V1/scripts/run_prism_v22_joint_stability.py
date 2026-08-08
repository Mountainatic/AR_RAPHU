from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_DEFAULT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DEFAULT / "src"))
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from prism_benchmark.v22_config import OUTPUT_DIRECTORY  # noqa: E402
from prism_benchmark.v22_metro_runner import (  # noqa: E402
    V22Paths,
    prepare_v22_prerequisites,
    run_m5_v22,
    run_m6_v22,
)


DEFAULT_CORRECTION_ROOT = Path(
    "/root/autodl-tmp/PRISM_V212_JOINT_OOF_PROTOCOL_CORRECTION/"
    "PRISM_INDUSTRIAL_BENCHMARK_V1/"
    "results_prism_v2_1_2_metro_p60_joint_oof_correction"
)
DEFAULT_PF_FREEZE = Path(
    "/root/autodl-tmp/PRISM_V212_PF_INDEPENDENT_FREEZE_RESULTS/"
    "results_prism_v2_1_2_metro_p60_pf_independent_freeze_semantics/"
    "FREEZE/METRO_P60_V212_DEVELOPMENT_FREEZE.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PRISM v2.2 Metro-P60 development-only Joint stability stages."
    )
    parser.add_argument("stage", choices=("prepare", "m5", "m6", "auto"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PRISM_SHARED_ROOT", "/root/autodl-tmp/PRISM_SHARED_DATA_C1"
            )
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_CORRECTION_ROOT)
    parser.add_argument("--legacy-results", type=Path, default=DEFAULT_CORRECTION_ROOT)
    parser.add_argument("--inherited-pf-freeze", type=Path, default=DEFAULT_PF_FREEZE)
    args = parser.parse_args()
    project = args.project_root.resolve()
    paths = V22Paths(
        project=project,
        shared=args.shared_root.resolve(),
        output=(args.output_root or project / OUTPUT_DIRECTORY).resolve(),
        source_results=args.source_results.resolve(),
        legacy_results=args.legacy_results.resolve(),
        inherited_pf_freeze=args.inherited_pf_freeze.resolve(),
    )
    if args.stage == "prepare":
        result = prepare_v22_prerequisites(paths)
    elif args.stage == "m5":
        result = run_m5_v22(paths)
    elif args.stage == "m6":
        result = run_m6_v22(paths)
    else:
        reuse = prepare_v22_prerequisites(paths)
        m5 = run_m5_v22(paths)
        result = {"reuse": reuse, "m5": m5, "m6_run": False}
        if m5["status"] == "JOINT_V22_PREDICTIVE_STABILITY_SUPPORTED":
            result["m6"] = run_m6_v22(paths)
            result["m6_run"] = True
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
