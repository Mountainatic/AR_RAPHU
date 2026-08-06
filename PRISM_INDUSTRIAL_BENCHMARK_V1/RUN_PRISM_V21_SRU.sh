#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${PRISM_V21_UV_BIN:-/root/AR_RAPHU_AUTODL/.autodl-tools/uv}"
UV_ENV_PROJECT="${PRISM_V21_UV_ENV_PROJECT:-/root/AR_RAPHU_AUTODL}"
SHARED_ROOT="${PRISM_V21_SHARED_ROOT:?set PRISM_V21_SHARED_ROOT to the immutable C1 shared-data root}"
BASELINE_ROOT="${PRISM_V21_BASELINE_ROOT:?set PRISM_V21_BASELINE_ROOT to frozen sample-level predictions}"
OUTPUT_ROOT="${PRISM_V21_OUTPUT_ROOT:-${PROJECT_ROOT}/results_prism_v2_1_sru}"
THROUGH_STAGE="${PRISM_V21_THROUGH_STAGE:-e8}"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv executable is unavailable: ${UV_BIN}" >&2
  exit 2
fi

export AR_RAPHU_RUNTIME_MANAGER=uv
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${UV_BIN}" run --no-sync --project "${UV_ENV_PROJECT}" \
  python "${PROJECT_ROOT}/scripts/run_prism_v21_chain.py" \
  --project "${PROJECT_ROOT}" \
  --shared "${SHARED_ROOT}" \
  --baseline-root "${BASELINE_ROOT}" \
  --output "${OUTPUT_ROOT}" \
  --through "${THROUGH_STAGE}"
