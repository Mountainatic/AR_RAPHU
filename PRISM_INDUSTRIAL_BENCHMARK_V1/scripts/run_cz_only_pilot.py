from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from prism_benchmark.cz_only_pilot import (
    create_cz_pilot_selection_freeze,
    fit_and_seal_cz_pilot_checkpoints,
    run_cz_pilot_test_inference,
    write_cz_pilot_scope,
)
from prism_benchmark.representative_formal import (
    assert_scope_request,
    run_all_cz_development,
    storage_preflight,
)
from prism_benchmark.v211_representative_stage1_config import (
    load_representative_stage1_descriptor,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated CZ L256 pilot with freeze, checkpoints and inference-only test."
    )
    parser.add_argument(
        "stage", choices=("scope", "preflight", "development", "freeze", "checkpoints", "test")
    )
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-cz", type=Path, required=True)
    args = parser.parse_args()
    args.project = args.project.resolve()
    args.run_root = args.run_root.resolve()
    args.raw_cz = args.raw_cz.resolve()
    if not args.raw_cz.is_file():
        raise RuntimeError("--raw-cz must identify the frozen CZ workbook")
    args.run_root.mkdir(parents=True, exist_ok=True)
    descriptor = load_representative_stage1_descriptor(args.project)
    assert_scope_request(datasets=("cz_czochralski",))
    if args.stage == "scope":
        result = write_cz_pilot_scope(args.project, args.run_root)
    elif args.stage == "preflight":
        result = storage_preflight(
            args.run_root.parent, float(descriptor["minimum_start_free_gib"])
        )
    elif args.stage == "development":
        result = run_all_cz_development(
            project=args.project, run_root=args.run_root, raw_path=args.raw_cz
        )
    elif args.stage == "freeze":
        result = create_cz_pilot_selection_freeze(args.project, args.run_root)
    elif args.stage == "checkpoints":
        result = fit_and_seal_cz_pilot_checkpoints(args.project, args.run_root)
    else:
        result = run_cz_pilot_test_inference(
            project=args.project, run_root=args.run_root, raw_cz=args.raw_cz
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
