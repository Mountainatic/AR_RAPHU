#!/usr/bin/env python3
"""Build a deterministic SHA256 package manifest."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() == output:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    payload = {
        "schema": "CZ_PACKAGE_MANIFEST_V1",
        "root": root.name,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
