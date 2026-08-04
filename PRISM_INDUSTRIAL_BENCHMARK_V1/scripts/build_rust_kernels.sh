#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m maturin build \
  --manifest-path "$PROJECT/rust/prism_kernels/Cargo.toml" \
  --release \
  --interpreter "$PYTHON" \
  --out "$PROJECT/dist/rust"
"$PYTHON" -m pip install --no-deps --force-reinstall "$PROJECT"/dist/rust/prism_kernels-*.whl
"$PYTHON" -c 'import _prism_rust; print("PRISM_RUST_KERNELS=AVAILABLE")'
