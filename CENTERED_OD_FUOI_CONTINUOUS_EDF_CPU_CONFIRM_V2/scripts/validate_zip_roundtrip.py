from __future__ import annotations

import argparse
import zipfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--zip", required=True); args = parser.parse_args()
    with zipfile.ZipFile(args.zip) as archive:
        assert archive.testzip() is None
        assert not any(name.lower().endswith((".xlsx", ".xls")) or "/.git/" in name for name in archive.namelist())
    print("ZIP_ROUNDTRIP=PASS")
