#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-$ROOT/.venv-gpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 required, found {sys.version}")
PY

if [[ ! -d "$ENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements_gpu.txt"
python - <<'PY'
import torch
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
