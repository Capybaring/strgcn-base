#!/usr/bin/env bash
# Four datasets, one seed; preserve old runs and compare equal X visibility.
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# > 1 )); then
  echo 'Usage: bash scripts/run_strgcn_query_fusion.sh [hipatch|tpatchgnn|mtand]' >&2
  exit 2
fi
backbone="${1:-hipatch}"
case "$backbone" in
  hipatch) model=HiPatchQueryFusion ;;
  tpatchgnn) model=TPatchQueryFusion ;;
  mtand|mtan) model=MTANQueryFusion ;;
  *) echo 'Backbone must be hipatch, tpatchgnn or mtand' >&2; exit 2 ;;
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
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
output="${RUN_DIR:-results/query_fusion/${backbone}_strgcn_all_x_$(date +%Y%m%d_%H%M%S)_$$}"
command=("${PYTHON:-python}" -u scripts/run_query_fusion.py
  --data-root "$data_root" --output "$output" --device "${DEVICE:-cuda:0}"
  --backbones "$model" --datasets activity mimic physionet ushcn
  --variants base mlp v1_all strgcn --seeds 3407 --epochs 30
  --input-mode FullExo --backbone-history y --stream-logs)
if [[ "${DRY_RUN:-0}" == 1 ]]; then command+=(--plan-only); fi
printf '%s: 16 sequential runs, 4 datasets x base/mlp/v1_all/strgcn; seed 3407.\n' "$backbone"
printf 'Output: %s\n' "$output"
exec "${command[@]}"
