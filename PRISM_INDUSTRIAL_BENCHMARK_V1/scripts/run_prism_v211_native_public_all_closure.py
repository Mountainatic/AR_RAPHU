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
from prism_benchmark.v211_public_all_partial_resume import (
    prepare_partial_resume,
    run_partial_resume,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze and materialize the PRISM v2.1.1 public-all rerun."
    )
    parser.add_argument(
        "stage",
        choices=(
            "audit-k",
            "common-support",
            "freeze",
            "test",
            "prepare-resume",
            "resume-test",
        ),
    )
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--materialization-commit")
    parser.add_argument("--memory-repair-commit")
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    paths = PublicAllPaths(
        project=args.project.resolve(),
        shared=args.shared.resolve(),
        run_root=args.run_root.resolve(),
    )
    if args.stage in {"prepare-resume", "resume-test"}:
        if args.resume_from is None or args.materialization_commit is None:
            parser.error(
                f"{args.stage} requires --resume-from and "
                "--materialization-commit"
            )
        parent = PublicAllPaths(
            project=args.project.resolve(),
            shared=args.shared.resolve(),
            run_root=args.resume_from.resolve(),
        )
        if args.stage == "prepare-resume":
            if (
                args.artifact_root is None
                or args.memory_repair_commit is None
            ):
                parser.error(
                    "prepare-resume requires --artifact-root and "
                    "--memory-repair-commit"
                )
            result = prepare_partial_resume(
                paths,
                parent,
                args.artifact_root.resolve(),
                materialization_commit=args.materialization_commit,
                memory_repair_commit=args.memory_repair_commit,
            )
        else:
            result = run_partial_resume(
                paths,
                parent,
                materialization_commit=args.materialization_commit,
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    runners = {
        "audit-k": lambda: write_k_audit(paths),
        "common-support": lambda: build_common_support(paths),
        "freeze": lambda: write_development_freeze(paths),
        "test": lambda: run_public_all_test(paths),
    }
    print(json.dumps(runners[args.stage](), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
