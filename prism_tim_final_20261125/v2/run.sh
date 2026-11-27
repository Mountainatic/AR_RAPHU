#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
umask 077
export PRISM_TIM_ARCHIVE="${PRISM_TIM_ARCHIVE:-$(cd .. && pwd)/PRISM_TIM_FINAL_20261125}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AR_RAPHU_RUNTIME_MANAGER=uv
python="${PRISM_TIM_PYTHON:-../PRISM_V06_SRU_CZ_SCALE_SEARCH_20261020/.venv/bin/python}"
out="${1:?output directory required}"
mkdir -p "$out"
"$python" -B experiments.py --out "$out" register
jobs=("sru 0" "rod1 16" "rod2 16" "rod1 8" "rod2 8" "rod1 32" "rod2 32" "rod1 48" "rod2 48" "rod1 64" "rod2 64")
for ((i=0; i<${#jobs[@]}; i+=8)); do
    pids=()
    for job in "${jobs[@]:i:8}"; do
        read -r entity horizon <<< "$job"
        "$python" -B -u experiments.py --out "$out" fit "$entity" "$horizon" >"$out/fit_${entity}_${horizon}.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if ((failed)); then
        echo "A fit in batch $i failed. See $out/fit_*.log." >&2
        exit 1
    fi
done
"$python" -B -u experiments.py --out "$out" evaluate >"$out/evaluate.log" 2>&1
echo "E1–E3 complete: $out/RESULTS.json"

