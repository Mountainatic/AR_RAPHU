from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.v211_public_all_audit import write_k_audit
from prism_benchmark.v211_public_all_closure import (
    build_common_support,
    run_public_all_test,
    write_development_freeze,
)
from prism_benchmark.v211_public_all_config import PublicAllPaths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze and materialize the PRISM v2.1.1 public-all rerun."
    )
    parser.add_argument(
        "stage", choices=("audit-k", "common-support", "freeze", "test")
    )
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    paths = PublicAllPaths(
        project=args.project.resolve(),
        shared=args.shared.resolve(),
        run_root=args.run_root.resolve(),
    )
    runners = {
        "audit-k": lambda: write_k_audit(paths),
        "common-support": lambda: build_common_support(paths),
        "freeze": lambda: write_development_freeze(paths),
        "test": lambda: run_public_all_test(paths),
    }
    print(json.dumps(runners[args.stage](), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
