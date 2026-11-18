#!/usr/bin/env bash
# Single retry: original Hi-Patch/MIMIC; unchanged model, batch size 1.
set -euo pipefail
if (( $# != 0 )); then
  echo 'Usage: GPU=1 bash scripts/run_hipatch_mimic_original.sh' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export GPU="${GPU:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
exec bash scripts/run_query_fusion.sh hipatch mimic original
