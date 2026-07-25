#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORKERS="${AR_RAPHU_GPU_WORKERS:-8}"
CPU_THREADS="${AR_RAPHU_CPU_THREADS_PER_WORKER:-3}"
if (( WORKERS < 1 || CPU_THREADS < 1 )); then
  echo "Worker and thread counts must be positive." >&2
  exit 64
fi
if (( WORKERS * CPU_THREADS > 25 )); then
  echo "Refusing CPU oversubscription: ${WORKERS}*${CPU_THREADS} > 25." >&2
  exit 64
fi

export AR_RAPHU_RUNTIME_MANAGER=uv
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-100}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/ar_raphu_${UID}_pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/ar_raphu_${UID}_log}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ar_raphu_mpl_${UID}}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export AR_RAPHU_TORCH_THREADS="${CPU_THREADS}"
export AR_RAPHU_TORCH_INTEROP_THREADS="${AR_RAPHU_TORCH_INTEROP_THREADS:-1}"
export PYTHONHASHSEED=0

if test ! -f results/runtime/autodl_verified.json; then
  echo "Run bash deploy/autodl/verify_server.sh before scientific training." >&2
  exit 3
fi

bash deploy/autodl/mps.sh start
cleanup() {
  bash deploy/autodl/mps.sh stop >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

uv run python deploy/autodl/preflight.py --require-mps

for track in X XAR; do
  uv run python tools/make_phase1_scheme_a_manifest.py \
    --scenario AR-S1 --track "${track}" --stage warmup --device cuda
  uv run python tools/make_phase1_scheme_a_manifest.py \
    --scenario AR-S1 --track "${track}" --stage fork --device cuda
done

run_pool() {
  local manifest="$1"
  uv run python STAGE1_DUAL_SOLVER_V20_bundle/tools/run_gpu_job_pool.py \
    --manifest "${manifest}" \
    --devices 0 \
    --workers-per-device "${WORKERS}" \
    --workdir . \
    --records-dir results/phase1/job_records \
    --resume
}

run_pool results/phase1/manifests/AR-S1_G2_X_warmup.json
run_pool results/phase1/manifests/AR-S1_G2_XAR_warmup.json
run_pool results/phase1/manifests/AR-S1_G2_X_fork.json
run_pool results/phase1/manifests/AR-S1_G2_XAR_fork.json

# Freeze both validation-only choices before either track opens test.
uv run python tools/run_phase1_scheme_a.py select \
  --scenario AR-S1 --track X --device cuda
uv run python tools/run_phase1_scheme_a.py select \
  --scenario AR-S1 --track XAR --device cuda

uv run python tools/run_phase1_scheme_a.py aggregate \
  --scenario AR-S1 --track X --device cuda
uv run python tools/run_phase1_scheme_a.py aggregate \
  --scenario AR-S1 --track XAR --device cuda

echo "E2 Scheme-A Track-X and Track-XAR completed."
