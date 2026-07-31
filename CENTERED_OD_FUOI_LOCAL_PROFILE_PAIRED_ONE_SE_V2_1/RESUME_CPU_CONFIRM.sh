#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
UV_BIN="${UV_BIN:-/root/AR_RAPHU_AUTODL/.autodl-tools/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/AR_RAPHU_AUTODL/.venv}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
exec "$UV_BIN" run --project "$REPO_ROOT" python "$ROOT/scripts/run_v2_1.py" --resume
