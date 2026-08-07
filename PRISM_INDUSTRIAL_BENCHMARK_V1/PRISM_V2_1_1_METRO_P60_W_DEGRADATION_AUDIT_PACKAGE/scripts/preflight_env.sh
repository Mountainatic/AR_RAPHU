#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PRISM_WORKERS="${PRISM_WORKERS:-2}"
export PRISM_PREDICTION_CHUNK_ROWS="${PRISM_PREDICTION_CHUNK_ROWS:-50000}"
printf 'OMP_NUM_THREADS=%s\n' "$OMP_NUM_THREADS"
printf 'MKL_NUM_THREADS=%s\n' "$MKL_NUM_THREADS"
printf 'OPENBLAS_NUM_THREADS=%s\n' "$OPENBLAS_NUM_THREADS"
printf 'NUMEXPR_NUM_THREADS=%s\n' "$NUMEXPR_NUM_THREADS"
printf 'PRISM_WORKERS=%s\n' "$PRISM_WORKERS"
printf 'PRISM_PREDICTION_CHUNK_ROWS=%s\n' "$PRISM_PREDICTION_CHUNK_ROWS"
python - <<'PYENV'
import platform
print('python:', platform.python_version())
try:
 import numpy as np
 print('numpy:', np.__version__)
 np.show_config()
except Exception as e: print('numpy audit failed:', repr(e))
try:
 import psutil
 vm=psutil.virtual_memory()
 print('ram_total_gib:', round(vm.total/2**30,2))
 print('ram_available_gib:', round(vm.available/2**30,2))
except Exception as e: print('psutil audit unavailable:', repr(e))
PYENV
