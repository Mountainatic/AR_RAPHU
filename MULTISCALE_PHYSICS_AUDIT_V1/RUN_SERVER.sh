#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=""
SAMPLE_PERIOD=""
N_JOBS="12"
BOOTSTRAP="500"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --sample-period-sec) SAMPLE_PERIOD="$2"; shift 2 ;;
    --n-jobs) N_JOBS="$2"; shift 2 ;;
    --bootstrap) BOOTSTRAP="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$DATA" && -n "$SAMPLE_PERIOD" ]] || {
  echo "--data and --sample-period-sec are required" >&2
  exit 2
}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
mkdir -p "$ROOT/logs" "$ROOT/results/checkpoints"
PYTHON="${PYTHON:-python}"

"$PYTHON" "$ROOT/scripts/preflight.py" \
  --data "$DATA" --sample-period-sec "$SAMPLE_PERIOD" \
  --output "$ROOT/results/preflight"
"$PYTHON" "$ROOT/scripts/run_stage1_scale_scan.py" \
  --data "$DATA" --sample-period-sec "$SAMPLE_PERIOD" --n-jobs "$N_JOBS"
"$PYTHON" "$ROOT/scripts/run_stage2_confirm.py" \
  --data "$DATA" --sample-period-sec "$SAMPLE_PERIOD"
"$PYTHON" "$ROOT/scripts/run_stage3_nonlinear.py" \
  --data "$DATA" --sample-period-sec "$SAMPLE_PERIOD"
"$PYTHON" "$ROOT/scripts/build_return_bundle.py"
