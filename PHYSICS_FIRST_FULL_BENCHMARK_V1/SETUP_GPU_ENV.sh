#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "$UV_BIN" ]]; then
  echo "uv is required; set UV_BIN=/absolute/path/to/uv" >&2
  exit 2
fi
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"
"$UV_BIN" sync --frozen --group dev --project "$REPO_ROOT"
PYTHON_BIN="$ENV_DIR/bin/python"
"$PYTHON_BIN" - <<'PY'
import sys
import torch
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Frozen lock requires Python 3.10, found {sys.version}")
print('torch=', torch.__version__)
print('cuda_runtime=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in the selected environment')
print('gpu=', torch.cuda.get_device_name(0))
print('capability=', torch.cuda.get_device_capability(0))
print('bf16=', torch.cuda.is_bf16_supported())
PY
printf 'GPU_ENV_READY=%s\n' "$ENV_DIR"
printf 'UV_BIN=%s\n' "$UV_BIN"
