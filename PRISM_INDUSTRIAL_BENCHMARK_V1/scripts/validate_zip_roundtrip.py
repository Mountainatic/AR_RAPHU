from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="prism_zip_") as d:
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(d)
        root = next(Path(d).iterdir())
        print(f"ROUNDTRIP_ROOT={root}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

