from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--shared", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    for name in ("src", "scripts", "configs", "tests", "README.md", "requirements.txt"):
        src = args.project / name
        if not src.exists():
            continue
        dst = args.output / name
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    shutil.copytree(args.results, args.output / "results_cpu")
    # Only metadata from the shared package is returned; no raw arrays or source files.
    shared_dst = args.output / "shared_metadata"
    shared_dst.mkdir()
    for name in ("PROTOCOL.json", "DATASET_HASHES.json", "TASK_REGISTRY.json", "SPLIT_REGISTRY.json", "VALIDATION_REPORT.md"):
        src = args.shared / name
        if src.exists():
            shutil.copy2(src, shared_dst / name)
    entries = []
    for p in sorted(args.output.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(args.output).as_posix()
        if p.name in {"PACKAGE_MANIFEST.json", "SHA256SUMS.txt"} or rel.startswith(".git/"):
            continue
        entries.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p), "type": "code" if rel.startswith(("src/", "scripts/", "tests/")) else "result_or_metadata"})
    (args.output / "PACKAGE_MANIFEST.json").write_text(json.dumps({"package": "PRISM_INDUSTRIAL_CPU_RESULTS_V1", "raw_data_included": False, "files": entries}, indent=2), encoding="utf-8")
    lines = [f"{e['sha256']}  {e['path']}" for e in entries]
    (args.output / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": len(entries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

