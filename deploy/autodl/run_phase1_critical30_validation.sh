#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
UV_BIN="${PROJECT_ROOT}/.autodl-tools/uv"
WORKERS="${AR_RAPHU_GPU_WORKERS:-16}"
CPU_THREADS="${AR_RAPHU_CPU_THREADS_PER_WORKER:-1}"
if (( WORKERS < 1 || CPU_THREADS < 1 || WORKERS * CPU_THREADS > 24 )); then
  echo "Invalid worker/thread budget: ${WORKERS}*${CPU_THREADS}" >&2
  exit 64
fi

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

run_pool() {
  local manifest="$1"
  "${UV_BIN}" run python STAGE1_DUAL_SOLVER_V20_bundle/tools/run_gpu_job_pool.py \
    --manifest "${manifest}" \
    --devices 0 \
    --workers-per-device "${WORKERS}" \
    --workdir . \
    --records-dir results/phase1/job_records \
    --resume
}

for scenario in AR-S0 AR-S1 AR-S2 AR-S3; do
  "${UV_BIN}" run python tools/make_phase1_scheme_a_manifest.py \
    --scenario "${scenario}" --track XAR --stage warmup --device cuda \
    --replicate-profile critical
  "${UV_BIN}" run python tools/make_phase1_scheme_a_manifest.py \
    --scenario "${scenario}" --track XAR --stage fork --device cuda \
    --replicate-profile critical
  run_pool "results/phase1/manifests/${scenario}_G2_XAR_warmup_critical.json"
  run_pool "results/phase1/manifests/${scenario}_G2_XAR_fork_critical.json"
  "${UV_BIN}" run python tools/run_phase1_scheme_a.py select \
    --scenario "${scenario}" --track XAR --device cuda \
    --replicate-profile critical
done

echo "E1 through E4 critical-30 validation-only confirmation completed."
