#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS:-$ROOT/results_gpu}"
PYTHON_BIN="${PYTHON_BIN:-/root/AR_RAPHU_AUTODL/.venv/bin/python}"

echo "UTC=$(date -u +%FT%TZ)"
echo "GIT_COMMIT=$(git -C "$ROOT/.." rev-parse HEAD)"
for name in core pipeline postprocess; do
  pid_file="$RESULTS/logs/$name.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "${name^^}_PID=$pid RUNNING"
      ps -p "$pid" -o etime=,%cpu=,%mem=,stat= | \
        awk -v name="${name^^}" '{print name"_PROCESS elapsed="$1" cpu="$2"% mem="$3"% stat="$4}'
    else
      echo "${name^^}_PID=$pid EXITED"
    fi
  else
    echo "${name^^}_PID=NOT_STARTED"
  fi
done

"$PYTHON_BIN" - "$RESULTS" <<'PY'
import collections
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for stage in ("core", "frontier", "finalists"):
    counter = collections.Counter()
    for path in (root / "tasks" / stage).rglob("result.json"):
        try:
            counter[json.loads(path.read_text()).get("status", "UNKNOWN")] += 1
        except Exception:
            counter["UNREADABLE"] += 1
    print(f"{stage.upper()}_COUNTS={dict(counter)}")
checkpoint = root / "checkpoints" / "latest.json"
if checkpoint.is_file():
    value = json.loads(checkpoint.read_text())
    def count(field):
        item = value.get(field, [])
        return len(item) if isinstance(item, list) else int(item or 0)
    print(
        "LATEST_STAGE={stage} STATUS={status} COMPLETED={completed} FAILED={failed}".format(
            stage=value.get("stage"),
            status=value.get("status"),
            completed=count("completed"),
            failed=count("failed"),
        )
    )
PY

echo "GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)"
active_shards="$(pgrep -af 'run_gpu_stage[123]_|run_gpu_parallel_stage.py' | grep -v GPU_STATUS.sh || true)"
if [[ -n "$active_shards" ]]; then
  echo "ACTIVE_GPU_WORKERS=$(printf '%s\n' "$active_shards" | wc -l)"
  printf '%s\n' "$active_shards" | cut -c1-500
else
  echo "ACTIVE_GPU_WORKERS=0"
fi
for log in core_resume frontier finalists postprocess; do
  path="$RESULTS/logs/$log.log"
  if [[ -f "$path" ]]; then
    echo "LAST_${log^^}=$(grep -E 'TASK_PASS=|TASK_FAIL=|GPU_.*STATUS=|WAITING_' "$path" | tail -n 1 | cut -c1-500)"
  fi
done
