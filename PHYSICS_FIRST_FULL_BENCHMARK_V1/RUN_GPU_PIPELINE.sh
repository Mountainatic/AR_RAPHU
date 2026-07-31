#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS:-$ROOT/results_gpu}"
SHARED="${SHARED:-$ROOT/shared}"
CPU_K_OOF="${CPU_K_OOF:-/root/autodl-tmp/PHYSICS_FIRST_CPU_K_OOF}"
PYTHON_BIN="${PYTHON_BIN:-/root/AR_RAPHU_AUTODL/.venv/bin/python}"
CORE_PID_FILE="${CORE_PID_FILE:-$RESULTS/logs/core.pid}"
FULL_CONFIG="$ROOT/configs/gpu_models_full.yaml"
FINAL_CONFIG="$RESULTS/checkpoints/gpu_finalists.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$RESULTS/logs" "$RESULTS/checkpoints"

if [[ -f "$CORE_PID_FILE" ]]; then
  core_pid="$(cat "$CORE_PID_FILE")"
  while kill -0 "$core_pid" 2>/dev/null; do
    printf 'WAITING_FOR_CORE pid=%s utc=%s\n' "$core_pid" "$(date -u +%FT%TZ)"
    sleep 30
  done
fi

core_status="$("$PYTHON_BIN" -c \
  "import json; print(json.load(open('$RESULTS/checkpoints/latest.json'))['status'])")"
if [[ "$core_status" != "PASS" ]]; then
  printf 'PIPELINE_STOP=CORE_STATUS_%s\n' "$core_status" >&2
  exit 3
fi

"$PYTHON_BIN" "$ROOT/scripts/run_gpu_parallel_stage.py" \
  --stage frontier \
  --shared "$SHARED" \
  --config "$FULL_CONFIG" \
  --results "$RESULTS" \
  --cpu-results "$CPU_K_OOF" \
  --device cuda:0 \
  --seeds 0,1,2,3,4 \
  --parallel-workers "${GPU_PARALLEL_WORKERS:-12}" \
  --loader-workers 0 \
  --python-bin "$PYTHON_BIN" \
  --log-prefix frontier \
  2>&1 | tee "$RESULTS/logs/frontier.log"

"$PYTHON_BIN" "$ROOT/scripts/select_gpu_finalists.py" \
  --results "$RESULTS" \
  --config "$FULL_CONFIG" \
  --output "$FINAL_CONFIG" \
  --minimum-seeds 5 \
  2>&1 | tee "$RESULTS/logs/finalist_selection.log"

"$PYTHON_BIN" "$ROOT/scripts/run_gpu_parallel_stage.py" \
  --stage finalists \
  --shared "$SHARED" \
  --config "$FINAL_CONFIG" \
  --results "$RESULTS" \
  --cpu-results "$CPU_K_OOF" \
  --device cuda:0 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --parallel-workers "${GPU_PARALLEL_WORKERS:-12}" \
  --loader-workers 0 \
  --python-bin "$PYTHON_BIN" \
  --log-prefix finalists \
  2>&1 | tee "$RESULTS/logs/finalists.log"

"$PYTHON_BIN" "$ROOT/scripts/aggregate_gpu_results.py" --results "$RESULTS"
printf 'GPU_PIPELINE_STATUS=PASS\n'
