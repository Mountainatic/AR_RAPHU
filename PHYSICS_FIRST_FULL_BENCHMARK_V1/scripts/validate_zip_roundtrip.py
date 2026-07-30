#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cpu_bundle_check_") as temp:
        with zipfile.ZipFile(args.zip) as bundle:
            forbidden = [
                name
                for name in bundle.namelist()
                if name.lower().endswith((".xlsx", ".xls"))
                or "/.git/" in name
                or "__pycache__" in name
            ]
            bundle.extractall(temp)
        result = {
            "status": "PASS" if not forbidden else "FAIL",
            "files": len(zipfile.ZipFile(args.zip).namelist()),
            "forbidden": forbidden,
        }
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
