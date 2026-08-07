#!/usr/bin/env bash
set -euo pipefail

WORKTREE="${PRISM_METRO_WORKTREE:-/root/autodl-tmp/PRISM_V211_METRO_P60_W_AUDIT}"
PROJECT="${WORKTREE}/PRISM_INDUSTRIAL_BENCHMARK_V1"
SHARED="${PRISM_SHARED_ROOT:-/root/autodl-tmp/PRISM_SHARED_DATA_C1}"
UV="${UV_BIN:-/root/AR_RAPHU_AUTODL/.autodl-tools/uv}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=2
export PRISM_V211_WORKERS=2
export PRISM_V211_METRO_WORKERS="${PRISM_V211_METRO_WORKERS:-27}"
export PRISM_V211_K_MEMORY_GIB_PER_WORKER="${PRISM_V211_K_MEMORY_GIB_PER_WORKER:-1.75}"
export PRISM_V211_MEMORY_GIB_PER_WORKER=20
export PRISM_SHARED_ROOT="${SHARED}"

cd "${PROJECT}"
exec "${UV}" run --offline --frozen --project /root/AR_RAPHU_AUTODL \
  python scripts/run_prism_v211_metro_chain.py \
  --project-root "${PROJECT}" \
  --shared-root "${SHARED}" "$@"
