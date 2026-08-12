#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
DATA_ROOT="${PRISM_NEUROBEM_DATA_ROOT:-$REPO_ROOT/data/private/neurobem}"
OUTPUT_ROOT="${PRISM_NEUROBEM_OUTPUT_ROOT:-$ROOT/results_prism_v2_1_1_neurobem_mimo_audit}"
PYTHON="${PRISM_NEUROBEM_PYTHON:-python}"
CONFIG="$ROOT/PRISM_V2_1_1_NEUROBEM_MIMO_AUDIT_PACKAGE/PRISM_V2_1_1_NEUROBEM_MIMO_CONFIG_FROZEN.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "$PYTHON" -m prism_benchmark.neurobem_runner "$@" \
  --repo-root "$REPO_ROOT" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT"
