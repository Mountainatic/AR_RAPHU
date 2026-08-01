#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHARED=${PRISM_SHARED_DATA:-/root/autodl-tmp/PRISM_SHARED_DATA_C1}
RESULTS=${PRISM_CPU_RESULTS:-/root/autodl-tmp/PRISM_CPU_RESULTS_V1_STRICT}
RETURN_ROOT=${PRISM_CPU_RETURN:-/root/autodl-tmp/PRISM_CPU_RETURN_V1}
PYTHON=${PRISM_PYTHON:-/root/AR_RAPHU_AUTODL/.venv/bin/python}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT_DIR}/src"

exec "${PYTHON}" "${PROJECT_DIR}/scripts/run_cpu_chain.py" \
  --project "${PROJECT_DIR}" \
  --shared "${SHARED}" \
  --results "${RESULTS}" \
  --return-root "${RETURN_ROOT}" \
  --c2-jobs "${PRISM_C2_JOBS:-8}" \
  --c3-jobs "${PRISM_C3_JOBS:-6}" \
  --c4-jobs "${PRISM_C4_JOBS:-8}" \
  --c5-jobs "${PRISM_C5_JOBS:-2}" \
  --publish
