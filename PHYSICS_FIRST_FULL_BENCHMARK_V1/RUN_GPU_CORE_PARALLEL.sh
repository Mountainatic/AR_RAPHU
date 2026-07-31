#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS:-$ROOT/results_gpu}"
SHARED="${SHARED:-$ROOT/shared}"
CPU_K_OOF="${CPU_K_OOF:-/root/autodl-tmp/PHYSICS_FIRST_CPU_K_OOF}"
PYTHON_BIN="${PYTHON_BIN:-/root/AR_RAPHU_AUTODL/.venv/bin/python}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$RESULTS/logs" "$RESULTS/checkpoints"
groups=(
  "mlp_u,lstm_u,gru_u"
  "tcn_u,dlinear_u,nlinear_u"
  "lstm_sa_u,lstm_uxy,gru_uxy"
  "tcn_uxy,k_residual_gru,k_residual_tcn"
)
pids=()
for index in 0 1 2 3; do
  env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 \
    "$PYTHON_BIN" "$ROOT/scripts/run_gpu_stage1_core.py" \
    --shared "$SHARED" \
    --results "$RESULTS" \
    --cpu-results "$CPU_K_OOF" \
    --device cuda:0 \
    --models "${groups[$index]}" \
    --seeds 0,1,2,3,4 \
    --strict-folds \
    --workers 0 \
    --checkpoint-name "core_shard_$index.json" \
    --skip-aggregate \
    > "$RESULTS/logs/core_shard_$index.log" 2>&1 &
  pids+=("$!")
  echo "${pids[-1]}" > "$RESULTS/logs/core_shard_$index.pid"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
"$PYTHON_BIN" "$ROOT/scripts/aggregate_gpu_results.py" --results "$RESULTS"
"$PYTHON_BIN" - "$RESULTS" <<'PY'
import collections
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
counter = collections.Counter()
for path in (root / "tasks" / "core").rglob("result.json"):
    counter[json.loads(path.read_text()).get("status", "UNKNOWN")] += 1
payload = {
    "status": "PASS" if counter == {"PASS": 120} else "PARTIAL",
    "stage": "core",
    "completed": counter.get("PASS", 0),
    "failed": counter.get("FAIL", 0),
    "expected": 120,
    "parallel_shards": 4,
}
(root / "checkpoints" / "latest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print("CORE_PARALLEL_RESULT=" + json.dumps(payload))
if payload["status"] != "PASS":
    raise SystemExit(2)
PY
exit "$status"
