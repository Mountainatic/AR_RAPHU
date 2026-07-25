#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
UV_BIN="${PROJECT_ROOT}/.autodl-tools/uv"
CPU_THREADS="${AR_RAPHU_CPU_THREADS_PER_WORKER:-1}"

export AR_RAPHU_RUNTIME_MANAGER=uv
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-100}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/ar_raphu_${UID}_pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/ar_raphu_${UID}_log}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export AR_RAPHU_TORCH_THREADS="${CPU_THREADS}"
export AR_RAPHU_TORCH_INTEROP_THREADS=1
export PYTHONHASHSEED=0

bash deploy/autodl/mps.sh start
cleanup() {
  bash deploy/autodl/mps.sh stop >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

"${UV_BIN}" run python deploy/autodl/preflight.py --require-mps

for seed in 0 1 2 3 4 5 6 7 8 9; do
  "${UV_BIN}" run python tools/run_phase1_m8_bootstrap.py seed \
    --seed "${seed}" --device cuda
done
"${UV_BIN}" run python tools/run_phase1_m8_bootstrap.py aggregate

echo "E2 M8 formal bootstrap rank audit completed."
