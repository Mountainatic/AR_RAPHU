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

from prism_benchmark.v211_metro_config import (  # noqa: E402
    OUTPUT_DIRECTORY,
    MetroV211Paths,
)
from prism_benchmark.v211_metro_runner import STAGES, run_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one frozen PRISM v2.1.1 Metro-P60 audit stage."
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=Path(os.environ.get("PRISM_SHARED_ROOT", "/root/autodl-tmp/PRISM_SHARED_DATA_C1")),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    shared = args.shared_root.resolve()
    output = (args.output_root or project / OUTPUT_DIRECTORY).resolve()
    os.environ.setdefault("PRISM_V211_MEMORY_GIB_PER_WORKER", "20")
    result = run_stage(
        args.stage,
        MetroV211Paths(project=project, shared=shared, output=output),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
