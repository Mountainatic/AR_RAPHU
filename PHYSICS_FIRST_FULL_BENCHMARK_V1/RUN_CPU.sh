#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
SHARED="$ROOT/shared"
CONFIG="$ROOT/configs/cpu_models.yaml"
N_JOBS="20"
BOOTSTRAP="500"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shared-data) SHARED="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --n-jobs) N_JOBS="$2"; shift 2 ;;
    --bootstrap) BOOTSTRAP="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$BOOTSTRAP" == "500" ]] || {
  echo "The frozen protocol requires --bootstrap 500" >&2
  exit 2
}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export VIRTUAL_ENV=/root/AR_RAPHU_AUTODL/.venv
UV=/root/AR_RAPHU_AUTODL/.autodl-tools/uv
mkdir -p "$ROOT/logs" "$ROOT/results_cpu/checkpoints"

"$UV" run --active python "$ROOT/scripts/run_cpu_stage1.py" \
  --shared-data "$SHARED" \
  --config "$CONFIG" \
  --n-jobs "$N_JOBS"
"$UV" run --active python "$ROOT/scripts/run_cpu_stage2_confirm.py"
"$UV" run --active python "$ROOT/scripts/aggregate_cpu_results.py"
"$UV" run --active python "$ROOT/scripts/build_cpu_return_bundle.py"
