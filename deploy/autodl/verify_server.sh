#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
UV_BIN="${PROJECT_ROOT}/.autodl-tools/uv"
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
fi
if test ! -x "${UV_BIN}"; then
  echo "uv executable not found: ${UV_BIN}" >&2
  echo "Run bash deploy/autodl/bootstrap.sh first." >&2
  exit 2
fi

export AR_RAPHU_RUNTIME_MANAGER=uv
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-100}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/ar_raphu_${UID}_pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/ar_raphu_${UID}_log}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ar_raphu_mpl_${UID}}"
export OMP_NUM_THREADS="${AR_RAPHU_CPU_THREADS_PER_WORKER:-3}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
export AR_RAPHU_TORCH_THREADS="${OMP_NUM_THREADS}"
export AR_RAPHU_TORCH_INTEROP_THREADS="${AR_RAPHU_TORCH_INTEROP_THREADS:-1}"

bash deploy/autodl/mps.sh start
trap 'bash deploy/autodl/mps.sh stop >/dev/null 2>&1 || true' EXIT

"${UV_BIN}" run python deploy/autodl/preflight.py --require-mps
"${UV_BIN}" run python -m pytest tests -q --ignore=tests/test_phase0_manifests.py
(
  cd STAGE1_DUAL_SOLVER_V20_bundle
  "${UV_BIN}" run --project "${PROJECT_ROOT}" python -m pytest \
    tests/test_stage1.py \
    tests/test_stage1_acceleration.py \
    tests/test_stage1_dual_solver_v20.py -q
)

mkdir -p results/runtime
"${UV_BIN}" run python -c \
  "import json,time; from pathlib import Path; Path('results/runtime/autodl_verified.json').write_text(json.dumps({'status':'COMPLETED','unix_time':time.time(),'mps_required':True,'public_only':True},indent=2)+'\n',encoding='utf-8')"
echo "AutoDL verification completed."
