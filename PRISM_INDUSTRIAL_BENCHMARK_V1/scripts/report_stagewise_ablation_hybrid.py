from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.stagewise_ablation_reporting import build_stagewise_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the validated aggregate table and incremental-gain figure."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_stagewise_report(args.manifest.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
