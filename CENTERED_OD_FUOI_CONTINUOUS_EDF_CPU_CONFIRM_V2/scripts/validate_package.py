from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.packaging import validate_package

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--package-dir", required=True); parser.add_argument("--forbid")
    args = parser.parse_args(); validate_package(Path(args.package_dir)); print("PACKAGE_VALIDATION=PASS")
