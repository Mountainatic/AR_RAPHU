#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: $0 audit|development|test|report}"
PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:?set SOURCE_ROOT to the private NeuroBEM root}"
TRACK_B_RELEASE_ROOT="${TRACK_B_RELEASE_ROOT:?set TRACK_B_RELEASE_ROOT to the official release root}"
OUTPUT="${OUTPUT:-$PROJECT/results_prism_v2_1_1_neurobem_literature_aligned_dual}"
PYTHON="${PYTHON:-python}"

export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

args=(
  "$STAGE"
  --project "$PROJECT"
  --source-root "$SOURCE_ROOT"
  --track-b-release-root "$TRACK_B_RELEASE_ROOT"
  --output "$OUTPUT"
)
if [[ "$STAGE" == audit ]]; then
  : "${PREDICTIONS_ROOT:?set PREDICTIONS_ROOT for the audit stage}"
  args+=(--predictions-root "$PREDICTIONS_ROOT")
fi

exec "$PYTHON" -m prism_benchmark.neurobem_literature_runner "${args[@]}"
