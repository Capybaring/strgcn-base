#!/usr/bin/env bash
# Sequential fixed-seed runs; neither model sizes nor batch sizes are changed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/query_fusion_device.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    'Usage: bash scripts/run_irgraph_query_fusion.sh [all|tpatchgnn|hipatch|mtand] [all|activity|mimic|physionet|ushcn|dist] [--device cuda:1]' \
    'Defaults: all all; 15 sequential irgraph runs, seed 3407, at most 30 epochs.' \
    'COMPARE=1 runs v1/v1_all/strgcn/graph/irgraph_no_node/irgraph_no_density/irgraph.' \
    'Optional: BATCH_SIZE, TPATCH_DATA_ROOT, DIST_DATA_PATH, RUN_DIR, DRY_RUN=1.'
  exit 0
fi
if (( $# > 2 )); then echo 'Expected backbone and dataset; see --help' >&2; exit 2; fi
cd "$(dirname "$0")/.."
case "${1:-all}" in
  all) backbones=(TPatchQueryFusion HiPatchQueryFusion MTANQueryFusion) ;;
  tpatchgnn) backbones=(TPatchQueryFusion) ;;
  hipatch) backbones=(HiPatchQueryFusion) ;;
  mtan|mtand) backbones=(MTANQueryFusion) ;;
  *) echo 'Unknown backbone; see --help' >&2; exit 2 ;;
esac
case "${2:-all}" in
  all) datasets=(activity mimic physionet ushcn dist) ;;
  activity|mimic|physionet|ushcn) datasets=("$2") ;;
  dist|DIST) datasets=(dist) ;;
  *) echo 'Unknown dataset; see --help' >&2; exit 2 ;;
esac
variants=(irgraph)
case "${COMPARE:-0}" in
  1) variants=(v1 v1_all strgcn graph irgraph_no_node irgraph_no_density irgraph) ;;
  0) ;;
  *) echo 'COMPARE must be 0 or 1' >&2; exit 2 ;;
esac
case "${DRY_RUN:-0}" in 0|1) ;; *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2 ;; esac
data_root="${TPATCH_DATA_ROOT:-./data/tpatch-data}"
dist_data="${DIST_DATA_PATH:-./data/processed/DIST.npz}"
for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == dist ]]; then
    if [[ ! -f "$dist_data" ]]; then echo "Missing DIST archive: $dist_data" >&2; exit 2; fi
  elif [[ ! -d "$data_root/$dataset/processed" ]]; then
    echo "Missing dataset directory: $data_root/$dataset/processed" >&2; exit 2
  fi
done
output="${RUN_DIR:-results/query_fusion/irgraph_$(date +%Y%m%d_%H%M%S)_$$}"
command=("${PYTHON:-python}" -u scripts/run_query_fusion.py
  --data-root "$data_root" --dist-data "$dist_data" --output "$output"
  --device "$DEVICE" --backbones "${backbones[@]}" --datasets "${datasets[@]}"
  --variants "${variants[@]}" --seeds 3407 --epochs 30
  --input-mode FullExo --backbone-history y --stream-logs)
if [[ "${DRY_RUN:-0}" == 1 ]]; then command+=(--plan-only); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then command+=(--batch-size "$BATCH_SIZE"); fi
printf '%s sequential runs; seed 3407; device %s; batch size %s\n' \
  "$((${#backbones[@]} * ${#datasets[@]} * ${#variants[@]}))" "$DEVICE" "${BATCH_SIZE:-config default}"
printf 'Output: %s\n' "$output"
exec "${command[@]}"
