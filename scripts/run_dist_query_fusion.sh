#!/usr/bin/env bash
# DIST: three backbones x original/base/mlp/v1/strgcn, seed 3407.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/query_fusion_device.sh"
cd "$(dirname "$0")/.."
if (( $# > 1 )); then
  echo 'Usage: bash scripts/run_dist_query_fusion.sh [all|tpatchgnn|hipatch|mtand] [--device cuda:1]' >&2
  exit 2
fi
case "${1:-all}" in
  all) models=(TPatchQueryFusion HiPatchQueryFusion MTANQueryFusion) ;;
  tpatchgnn) models=(TPatchQueryFusion) ;;
  hipatch) models=(HiPatchQueryFusion) ;;
  mtand|mtan) models=(MTANQueryFusion) ;;
  *) echo 'Choose all, tpatchgnn, hipatch or mtand' >&2; exit 2 ;;
esac
archive="${DIST_DATA_PATH:-./data/processed/DIST.npz}"
if [[ ! -f "$archive" ]]; then echo "Missing DIST archive: $archive" >&2; exit 2; fi
case "${DRY_RUN:-0}" in
  0|1) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2 ;;
esac
output="${RUN_DIR:-results/query_fusion/dist_$(date +%Y%m%d_%H%M%S)_$$}"
command=("${PYTHON:-python}" -u scripts/run_query_fusion.py
  --data-root "${TPATCH_DATA_ROOT:-./data/tpatch-data}" --dist-data "$archive"
  --output "$output" --device "$DEVICE" --backbones "${models[@]}"
  --datasets dist --variants original base mlp v1 strgcn
  --seeds 3407 --epochs 30 --stream-logs)
if [[ "${DRY_RUN:-0}" == 1 ]]; then command+=(--plan-only); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then command+=(--batch-size "$BATCH_SIZE"); fi
printf 'DIST: original/base/mlp/v1/strgcn, sequential, seed 3407; device %s\n' "$DEVICE"
printf 'Output: %s\n' "$output"
exec "${command[@]}"
