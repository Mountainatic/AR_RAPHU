#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS:-$ROOT/results_gpu}"
SHARED="${SHARED:-$ROOT/shared}"
CPU_K_OOF="${CPU_K_OOF:-/root/autodl-tmp/PHYSICS_FIRST_CPU_K_OOF}"
PYTHON_BIN="${PYTHON_BIN:-/root/AR_RAPHU_AUTODL/.venv/bin/python}"
PIPELINE_PID_FILE="${PIPELINE_PID_FILE:-$RESULTS/logs/pipeline.pid}"
FINAL_CONFIG="$RESULTS/checkpoints/gpu_finalists.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ -f "$PIPELINE_PID_FILE" ]]; then
  pipeline_pid="$(cat "$PIPELINE_PID_FILE")"
  while kill -0 "$pipeline_pid" 2>/dev/null; do
    printf 'WAITING_FOR_MAIN_PIPELINE pid=%s utc=%s\n' \
      "$pipeline_pid" "$(date -u +%FT%TZ)"
    sleep 60
  done
fi

final_status="$("$PYTHON_BIN" -c \
  "import json; print(json.load(open('$RESULTS/checkpoints/latest.json'))['status'])")"
if [[ "$final_status" != "PASS" || ! -f "$FINAL_CONFIG" ]]; then
  printf 'POSTPROCESS_STOP=FINAL_STATUS_%s\n' "$final_status" >&2
  exit 3
fi

non_residual_models="$("$PYTHON_BIN" -c \
  "import json; c=json.load(open('$FINAL_CONFIG')); print(','.join(x['id'] for x in c['models'] if x['mode'] != 'residual'))")"

for fraction in 0.25 0.50; do
  label="${fraction/./_}"
  ablation_results="$RESULTS/ABLATIONS/train_fraction_$label"
  "$PYTHON_BIN" "$ROOT/scripts/run_gpu_parallel_stage.py" \
    --stage finalists \
    --shared "$SHARED" \
    --config "$FINAL_CONFIG" \
    --results "$ablation_results" \
    --device cuda:0 \
    --models "$non_residual_models" \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --parallel-workers "${GPU_PARALLEL_WORKERS:-8}" \
    --loader-workers 0 \
    --python-bin "$PYTHON_BIN" \
    --log-prefix "train_fraction_$label" \
    --train-fraction "$fraction" \
    2>&1 | tee "$RESULTS/logs/train_fraction_$label.log"
done

"$PYTHON_BIN" "$ROOT/scripts/run_gpu_inference_ablations.py" \
  --shared "$SHARED" \
  --config "$FINAL_CONFIG" \
  --results "$RESULTS" \
  --cpu-results "$CPU_K_OOF" \
  --device cuda:0 \
  2>&1 | tee "$RESULTS/logs/inference_ablations.log"

"$PYTHON_BIN" "$ROOT/scripts/bootstrap_gpu_finalists.py" \
  --results "$RESULTS" \
  --config "$FINAL_CONFIG" \
  2>&1 | tee "$RESULTS/logs/finalist_bootstrap.log"

"$PYTHON_BIN" "$ROOT/scripts/aggregate_gpu_results.py" --results "$RESULTS"
"$PYTHON_BIN" "$ROOT/scripts/build_gpu_return_bundle.py" \
  --source-root "$ROOT" \
  --results "$RESULTS" \
  --output-dir "$ROOT/return/PHYSICS_FIRST_GPU_RESULTS" \
  --keep-best-checkpoints-only
"$PYTHON_BIN" "$ROOT/scripts/validate_package.py" \
  --package-dir "$ROOT/return/PHYSICS_FIRST_GPU_RESULTS"
"$PYTHON_BIN" "$ROOT/scripts/validate_zip_roundtrip.py" \
  --zip "$ROOT/return/PHYSICS_FIRST_GPU_RESULTS_bundle.zip"
printf 'GPU_POSTPROCESS_STATUS=PASS\n'
