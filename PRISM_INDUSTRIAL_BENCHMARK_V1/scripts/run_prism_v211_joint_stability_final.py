from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_DEFAULT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DEFAULT / "src"))
for variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
):
    os.environ[variable] = "1"

from prism_benchmark.stage0 import write_json  # noqa: E402
from prism_benchmark.v211_joint_stability_config import OUTPUT_DIRECTORY  # noqa: E402
from prism_benchmark.v211_joint_stability_runner import (  # noqa: E402
    FinalClosurePaths,
    audit_target_history_causality,
    migrate_development_evidence,
    run_m5_forbidden,
    run_m6_final,
    run_m7_preflight,
    write_lockbox_code_freeze,
)
from prism_benchmark.v211_metro_final import run_m7  # noqa: E402
from prism_benchmark.v211_metro_reporting import run_m8  # noqa: E402


DEFAULT_SOURCE_RESULTS = Path(
    "/root/autodl-tmp/PRISM_V22_JOINT_PREDICTIVE_STABILITY_RESULTS/"
    "results_prism_v2_2_metro_p60_joint_stability"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/PRISM_V211_METRO_P60_FINAL_RESULTS/"
    "results_prism_v2_1_1_metro_p60_joint_stability_final"
)


def _run_m7_guarded(paths: FinalClosurePaths) -> dict:
    try:
        return run_m7(paths.metro)
    except Exception as error:
        audit_path = paths.metro.test_access_audit_path
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit.update(
                {
                    "status": "LOCKBOX_ACCESSED_M7_RUNTIME_FAILURE",
                    "runtime_error_type": type(error).__name__,
                    "runtime_error": str(error),
                    "test_accessed": True,
                    "ood_accessed": True,
                }
            )
            write_json(audit_path, audit)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM v2.1.1 Metro-P60 final closure")
    parser.add_argument(
        "stage",
        choices=("prepare", "m5", "m6", "preflight", "causality", "lock", "m7", "m8", "auto"),
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument(
        "--shared-root", type=Path,
        default=Path(os.environ.get("PRISM_SHARED_ROOT", "/root/autodl-tmp/PRISM_SHARED_DATA_C1")),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    args = parser.parse_args()
    paths = FinalClosurePaths(
        project=args.project_root.resolve(), shared=args.shared_root.resolve(),
        output=args.output_root.resolve(), source_results=args.source_results.resolve(),
    )
    if args.stage == "prepare":
        result = migrate_development_evidence(paths)
    elif args.stage == "m5":
        result = run_m5_forbidden(paths)
    elif args.stage == "m6":
        result = run_m6_final(paths)
    elif args.stage == "preflight":
        result = run_m7_preflight(paths)
    elif args.stage == "causality":
        result = audit_target_history_causality(paths)
    elif args.stage == "lock":
        result = write_lockbox_code_freeze(paths)
    elif args.stage == "m7":
        result = _run_m7_guarded(paths)
    elif args.stage == "m8":
        result = run_m8(paths.metro)
    else:
        result = {"migration": migrate_development_evidence(paths)}
        result["m6"] = run_m6_final(paths)
        result["preflight"] = run_m7_preflight(paths)
        result["causality"] = audit_target_history_causality(paths)
        result["lockbox"] = write_lockbox_code_freeze(paths)
        result["m7"] = _run_m7_guarded(paths)
        result["m8"] = run_m8(paths.metro)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
