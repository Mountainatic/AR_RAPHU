from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.package_dir / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        p = args.package_dir / entry["path"]
        if not p.exists() or p.stat().st_size != entry["bytes"] or sha256(p) != entry["sha256"]:
            failures.append(entry["path"])
        # Derived CSV tables and predictions are allowed. Raw source workbooks,
        # archives and source directories are not allowed in the return package.
        rel = entry["path"].lower()
        if p.suffix.lower() in {".xlsx", ".xls", ".parquet"} or "/raw_sources/" in rel or rel.startswith("raw_sources/"):
            failures.append(f"forbidden_raw:{entry['path']}")
    print(json.dumps({"status": "PASS" if not failures else "FAIL", "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
