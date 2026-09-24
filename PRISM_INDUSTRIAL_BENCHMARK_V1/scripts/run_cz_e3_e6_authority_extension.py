from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run authority-aligned private-CZ E3--E6 development audits."
    )
    parser.add_argument(
        "stage", choices=("all", "e3", "e4", "e5", "e6", "status")
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    project = args.project.resolve()
    anchor = args.anchor.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(project / "src"))
    from prism_benchmark.cz_e3_e6_authority_extension import (
        run_cz_e3,
        run_cz_e5,
        run_cz_e6,
        verify_authority_sources,
        write_master_status,
    )
    from prism_benchmark.e1e6_phase_b import run_e4

    if output.exists() and args.stage == "all" and any(output.iterdir()):
        raise RuntimeError("STOP_REFUSE_EXISTING_NONEMPTY_OUTPUT")
    output.mkdir(parents=True, exist_ok=True)
    authority = verify_authority_sources(project)
    (output / "PROVENANCE").mkdir(parents=True, exist_ok=True)
    (output / "PROVENANCE/AUTHORITY_SOURCE_AUDIT.json").write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    e3 = None
    if args.stage in {"all", "e3"}:
        e3 = run_cz_e3(output, anchor, workers=min(args.workers, 2))
    if args.stage in {"all", "e4"}:
        run_e4(output, workers=args.workers)
    if args.stage in {"all", "e5"}:
        if e3 is None:
            import pandas as pd

            e3 = pd.read_csv(output / "E3_MULTISCALE/equal_budget_multiscale.csv")
        run_cz_e5(output, e3)
    if args.stage in {"all", "e6"}:
        run_cz_e6(output, anchor, workers=args.workers)
    status = write_master_status(output, authority)
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
