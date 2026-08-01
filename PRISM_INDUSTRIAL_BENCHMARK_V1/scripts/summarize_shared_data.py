from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize immutable C1 row counts without reading target values")
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads((args.shared / "SAMPLE_ID_REGISTRY.json").read_text(encoding="utf-8"))
    rows = []
    for entry in registry["files"]:
        if not entry["path"].startswith("sample_ids/"):
            continue
        parts = Path(entry["path"]).parts
        rows.append(
            {
                "target_head": parts[1],
                "information_set": parts[2],
                "availability_scenario": parts[3],
                "proxy_policy": parts[4],
                "split": Path(parts[5]).stem,
                "rows": entry["rows"],
                "file_sha256": entry["sha256"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(str(value) for value in row.values())))
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
