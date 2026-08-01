#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHARED=${PRISM_SHARED_DATA:-/root/autodl-tmp/PRISM_SHARED_DATA_C1}
RESULTS=${PRISM_CPU_RESULTS:-/root/autodl-tmp/PRISM_CPU_RESULTS_V1_STRICT}
PYTHON=${PRISM_PYTHON:-/root/AR_RAPHU_AUTODL/.venv/bin/python}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT_DIR}/src"

exec "${PYTHON}" "${PROJECT_DIR}/scripts/run_cpu_chain.py" \
  --project "${PROJECT_DIR}" \
  --shared "${SHARED}" \
  --results "${RESULTS}"
