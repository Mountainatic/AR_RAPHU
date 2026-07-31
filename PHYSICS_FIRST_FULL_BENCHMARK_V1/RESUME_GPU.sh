#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Completed task result.json files are detected and skipped automatically.
exec bash "$ROOT/RUN_GPU.sh" "$@"
