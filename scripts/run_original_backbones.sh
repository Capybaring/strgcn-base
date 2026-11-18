#!/usr/bin/env bash
# Native networks, all-variable training, endogenous-only validation/test.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/query_fusion_device.sh"
cd "$(dirname "$0")/.."
if (( $# > 1 )); then
  echo 'Usage: bash scripts/run_original_backbones.sh [all|tpatchgnn|hipatch|mtand] [--device cuda:1]' >&2
  exit 2
fi
case "${1:-all}" in
  all) models=(TPatchQueryFusion HiPatchQueryFusion MTANQueryFusion) ;;
  tpatchgnn) models=(TPatchQueryFusion) ;;
  hipatch) models=(HiPatchQueryFusion) ;;
  mtand|mtan) models=(MTANQueryFusion) ;;
  *) echo 'Choose all, tpatchgnn, hipatch or mtand' >&2; exit 2 ;;
esac
data_root="${TPATCH_DATA_ROOT:-./data/tpatch-data}"
for dataset in activity mimic physionet ushcn; do
  if [[ ! -d "$data_root/$dataset/processed" ]]; then
    echo "Missing dataset directory: $data_root/$dataset/processed" >&2
    exit 2
  fi
done
case "${DRY_RUN:-0}" in
  0|1) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2 ;;
esac
output="${RUN_DIR:-results/query_fusion/original_endo_eval_$(date +%Y%m%d_%H%M%S)_$$}"
command=("${PYTHON:-python}" -u scripts/run_query_fusion.py
  --data-root "$data_root" --output "$output" --device "${DEVICE:-cuda:0}"
  --backbones "${models[@]}" --datasets activity mimic physionet ushcn
  --variants original --seeds 3407 --epochs 30
  --input-mode FullExo --backbone-history yx --stream-logs)
if [[ "${DRY_RUN:-0}" == 1 ]]; then command+=(--plan-only); fi
printf 'Original backbones: all-variable history/training, endogenous-only evaluation.\n'
printf 'Output: %s\n' "$output"
exec "${command[@]}"
