#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
UV_BIN="${UV_BIN:-/root/AR_RAPHU_AUTODL/.autodl-tools/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/AR_RAPHU_AUTODL/.venv}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

mkdir -p "$HERE/logs" "$HERE/results/checkpoints"
exec "$UV_BIN" run --project "$REPO_ROOT" python "$HERE/scripts/run_cpu_confirm.py" "$@"
