#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p artifacts
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="artifacts/ar_raphu_public_results_${STAMP}.tar.gz"
tar -czf "${OUTPUT}" results logs configs/protocol_v2.yaml \
  deploy/autodl/runtime_profile.json
sha256sum "${OUTPUT}" >"${OUTPUT}.sha256"
echo "${OUTPUT}"
