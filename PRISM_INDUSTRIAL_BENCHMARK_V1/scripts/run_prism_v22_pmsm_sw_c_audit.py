from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.pmsm_sw_c_audit import run_primary_corrected_c


def main() -> int:
    parser = argparse.ArgumentParser(description="Run train-only corrected-C audit for PMSM_SW")
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--d-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_primary_corrected_c(
        args.shared.resolve(),
        PROJECT_ROOT.resolve(),
        args.d_output.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({
        "status": result["status"],
        "stage": result["stage"],
        "structure": result["structure"],
        "validation_metrics": result["validation_metrics"],
        "persistence_validation_metrics": result["persistence_validation_metrics"],
        "test_accessed": result["test_accessed"],
        "validation_used_for_selection": result["validation_used_for_selection"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
