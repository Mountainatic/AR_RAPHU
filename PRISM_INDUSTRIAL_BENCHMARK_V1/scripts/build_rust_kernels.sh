#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export RUSTUP_DIST_SERVER="${RUSTUP_DIST_SERVER:-https://mirrors.tuna.tsinghua.edu.cn/rustup}"
export RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup}"

"$PYTHON" -m maturin build \
  --manifest-path "$PROJECT/rust/prism_kernels/Cargo.toml" \
  --release \
  --interpreter "$PYTHON" \
  --out "$PROJECT/dist/rust"
"$PYTHON" -m pip install --no-deps --force-reinstall "$PROJECT"/dist/rust/prism_kernels-*.whl
"$PYTHON" -c 'import _prism_rust; print("PRISM_RUST_KERNELS=AVAILABLE")'
