#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
NAME="CZ_5090_ORSS_COMPLETE_EXPERIMENT_RESULTS_bundle"
OUT="$ROOT/return/$NAME"
ZIP="$ROOT/return/$NAME.zip"

if [[ -e "$OUT" || -e "$ZIP" ]]; then
  echo "Refusing to overwrite an existing final package." >&2
  exit 73
fi

mkdir -p "$OUT"
cp -a README.md "$OUT/" 2>/dev/null || true
cp -a CHANGELOG.md "$OUT/" 2>/dev/null || true
cp -a src tools tests configs "$OUT/"
cp -a results/cz_real_data/complete_5090 "$OUT/results"
cp -a environment "$OUT/" 2>/dev/null || mkdir -p "$OUT/environment"
cp -a CZ_COMPLETE_5090_REPORT.md CZ_COMPLETE_5090_STATUS.json "$OUT/"

git rev-parse HEAD > "$OUT/FINAL_COMMIT.txt"
cp -a results/orss_reference/DENSE_REFERENCE_COMMIT.txt \
  "$OUT/SOURCE_COMMIT.txt"
git bundle create "$OUT/repository.bundle" --all

find "$OUT" -type d \
  \( -name "__pycache__" -o -name ".pytest_cache" \) \
  -prune -exec rm -rf {} +
find "$OUT" -type f \( -name "*.pyc" -o -name "*.tmp" \) -delete

python tools/build_manifest.py \
  --root "$OUT" \
  --output "$OUT/PACKAGE_MANIFEST.json"

(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS.txt
)

python - "$OUT" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
required = [
    "src", "tools", "tests", "configs", "results", "environment",
    "CZ_COMPLETE_5090_REPORT.md", "CZ_COMPLETE_5090_STATUS.json",
    "FINAL_COMMIT.txt", "SOURCE_COMMIT.txt", "repository.bundle",
    "PACKAGE_MANIFEST.json", "SHA256SUMS.txt",
]
missing = [item for item in required if not (root / item).exists()]
if missing:
    raise SystemExit(f"Missing required package entries: {missing}")
json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
print("PACKAGE_STRUCTURE_OK")
PY

(
  cd "$ROOT/return"
  zip -qr "$NAME.zip" "$NAME"
  unzip -t "$NAME.zip"
  sha256sum "$NAME.zip" > "$NAME.zip.sha256"
)

echo "FINAL_PACKAGE=$ZIP"
echo "FINAL_SHA256=$ZIP.sha256"
