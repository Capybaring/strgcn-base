#!/usr/bin/env bash
# One backbone, one dataset, one method, one fixed random seed.
# Default: GPU 0, t-PatchGNN, Activity, base, full data, 30 epochs.
# Examples:
#   bash scripts/run_query_fusion.sh
#   bash scripts/run_query_fusion.sh tpatchgnn activity v1
#   bash scripts/run_query_fusion.sh hipatch ushcn v2
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/query_fusion_device.sh"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    'Usage: bash scripts/run_query_fusion.sh [tpatchgnn|hipatch|mtand] [dataset] [method] [--device cuda:1]' \
    'Defaults: tpatchgnn activity base' \
    'Datasets: activity mimic physionet ushcn' \
    'Methods: original base mlp cross v1 v1_all strgcn v2 v2_static v2_no_transfer v1_no_time v1_no_variable' \
    'Optional environment: GPU, DEVICE, BATCH_SIZE, PYTHON, TPATCH_DATA_ROOT, RUN_DIR, DRY_RUN=1' \
    'Every invocation runs once, with seed 3407 and at most 30 epochs.'
  exit 0
fi
if (( $# > 3 )); then
  echo 'Expected at most three arguments: backbone dataset method' >&2
  exit 2
fi

cd "$(dirname "$0")/.."
backbone="${1:-tpatchgnn}"
dataset="${2:-activity}"
method="${3:-base}"
case "$backbone" in
  tpatchgnn) model=TPatchQueryFusion ;;
  hipatch) model=HiPatchQueryFusion ;;
  mtan|mtand) model=MTANQueryFusion ;;
  *) echo 'Backbone must be tpatchgnn, hipatch or mtand' >&2; exit 2 ;;
esac
case "$dataset" in
  activity|mimic|physionet|ushcn) ;;
  *) echo 'Dataset must be activity, mimic, physionet or ushcn' >&2; exit 2 ;;
esac
case "$method" in
  original|base|mlp|cross|v1|v1_all|strgcn|v2|v2_static|v2_no_transfer|v1_no_time|v1_no_variable) ;;
  *) echo 'Unknown method; run this script with --help' >&2; exit 2 ;;
esac
case "${DRY_RUN:-0}" in
  0|1) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2 ;;
esac

data_root="${TPATCH_DATA_ROOT:-./data/tpatch-data}"
if [[ ! -d "$data_root/$dataset/processed" ]]; then
  echo "Missing dataset directory: $data_root/$dataset/processed" >&2
  echo 'Set TPATCH_DATA_ROOT to your tpatch-data directory.' >&2
  exit 2
fi
output="${RUN_DIR:-results/query_fusion/${backbone}_${dataset}_${method}_$(date +%Y%m%d_%H%M%S)_$$}"
command=("${PYTHON:-python}" -u scripts/run_query_fusion.py
  --data-root "$data_root"
  --output "$output"
  --device "${DEVICE:-cuda:0}"
  --backbones "$model"
  --datasets "$dataset"
  --variants "$method"
  --seeds 3407
  --epochs 30
  --stream-logs)
if [[ "${DRY_RUN:-0}" == 1 ]]; then command+=(--plan-only); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then command+=(--batch-size "$BATCH_SIZE"); fi

printf 'Running: %s / %s / %s (seed 3407)\n' "$backbone" "$dataset" "$method"
printf 'Output: %s\n' "$output"
printf 'Physical GPU(s): %s; process device: %s; batch size: %s\n' "${CUDA_VISIBLE_DEVICES-unrestricted}" "$DEVICE" "${BATCH_SIZE:-config default}"
exec "${command[@]}"
