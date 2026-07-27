#!/usr/bin/env python3
"""Verify the pinned 2024 PB1 baseline paper/code profile without model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.datasets.literature_profile import literature_profile_gaps


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _remote_refs(repository: str) -> set[str]:
    output = subprocess.check_output(
        ["git", "ls-remote", repository],
        text=True,
        timeout=60,
    )
    return {line.split()[0] for line in output.splitlines() if line.split()}


def _audit_checkout(
    checkout: Path,
    *,
    commit: str,
    expected_paths: list[str],
) -> tuple[set[str], dict[str, str]]:
    observed_commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    hashes = {
        path: hashlib.sha256((checkout / path).read_bytes()).hexdigest()
        for path in expected_paths
    }
    return {observed_commit}, hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT
        / "configs/public_benchmarks/PB1_LITERATURE_PROFILES.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/public_benchmarks/pb1/protocol_audit"
        / "literature_profile_audit.json",
    )
    parser.add_argument(
        "--checkout",
        type=Path,
        help="Optional existing checkout at the pinned commit.",
    )
    args = parser.parse_args()
    profile = _read_json(args.profile)
    code = profile["companion_code"]
    repository = str(code["repository"])
    commit = str(code["commit"])
    if args.checkout is not None:
        refs, hashes = _audit_checkout(
            args.checkout,
            commit=commit,
            expected_paths=list(code["file_sha256"]),
        )
    else:
        refs = _remote_refs(repository)
        with tempfile.TemporaryDirectory(prefix="pb1-literature-") as temporary:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-checkout",
                    repository,
                    temporary,
                ],
                check=True,
                timeout=120,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    temporary,
                    "checkout",
                    "--quiet",
                    "--detach",
                    commit,
                ],
                check=True,
                timeout=60,
            )
            _, hashes = _audit_checkout(
                Path(temporary),
                commit=commit,
                expected_paths=list(code["file_sha256"]),
            )
    gaps = literature_profile_gaps(
        profile,
        observed_refs=refs,
        observed_file_sha256=hashes,
    )
    payload = {
        "schema_version": 6,
        "scope": "EXTERNAL_BASELINE_PROFILE_AUDIT_NO_MODEL_FIT",
        "paper": profile["paper"],
        "repository": repository,
        "commit": commit,
        "commit_present_in_remote_refs": commit in refs,
        "observed_file_sha256": hashes,
        "expected_file_sha256": code["file_sha256"],
        "gaps": gaps,
        "gates": {
            "LITERATURE_PROFILE_PINNED": not gaps,
            "EXTERNAL_BASELINE_COMMIT_UNPINNED": (
                "EXTERNAL_BASELINE_COMMIT_UNPINNED" in gaps
            ),
        },
        "status": "COMPLETED" if not gaps else "FAILED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(payload["status"])
    return 0 if not gaps else 2


if __name__ == "__main__":
    raise SystemExit(main())
