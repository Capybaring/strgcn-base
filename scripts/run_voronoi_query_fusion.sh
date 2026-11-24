#!/usr/bin/env bash
# One model per batch by default; keep formal model and batch configurations.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/query_fusion_device.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    'Usage: bash scripts/run_voronoi_query_fusion.sh [tpatchgnn|hipatch|mtand|all] [all|activity|mimic|physionet|ushcn|dist] [--device cuda:1]' \
    'Defaults: tpatchgnn all; five sequential Voronoi runs, seed 3407, at most 30 epochs.' \
    'COMPARE=1 runs voronoi_no_density and voronoi (identical parameter layouts).' \
    'Optional: BATCH_SIZE, TPATCH_DATA_ROOT, DIST_DATA_PATH, RUN_DIR, PYTHON, DRY_RUN=1.' \
    'VORONOI_VARIANT=voronoi_h selects four-way node fusion; compare uses its matched no-density ablation.'
  exit 0
fi
if (( $# > 2 )); then echo 'Expected backbone and dataset; see --help' >&2; exit 2; fi
cd "$(dirname "$0")/.."
case "${1:-tpatchgnn}" in
  tpatchgnn) backbones=(TPatchQueryFusion) ;;
  hipatch) backbones=(HiPatchQueryFusion) ;;
  mtan|mtand) backbones=(MTANQueryFusion) ;;
  all) backbones=(TPatchQueryFusion HiPatchQueryFusion MTANQueryFusion) ;;
  *) echo 'Unknown backbone; see --help' >&2; exit 2 ;;
esac
case "${2:-all}" in
  all) datasets=(activity mimic physionet ushcn dist) ;;
  activity|mimic|physionet|ushcn) datasets=("$2") ;;
  dist|DIST) datasets=(dist) ;;
  *) echo 'Unknown dataset; see --help' >&2; exit 2 ;;
esac
variant="${VORONOI_VARIANT:-voronoi}"
case "$variant" in
  voronoi|voronoi_h) ;;
  *) echo 'VORONOI_VARIANT must be voronoi or voronoi_h' >&2; exit 2 ;;
esac
variants=("$variant")
case "${COMPARE:-0}" in
  1) variants=("${variant}_no_density" "$variant") ;;
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
output="${RUN_DIR:-results/query_fusion/${variant}_${1:-tpatchgnn}_$(date +%Y%m%d_%H%M%S)_$$}"
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
