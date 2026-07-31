from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    root = Path(args.root).resolve(); output = Path(args.output).resolve()
    rows = [{"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.resolve() != output]
    output.write_text(json.dumps({"files": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"MANIFEST_FILE_COUNT={len(rows)}")
