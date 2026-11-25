#!/usr/bin/env bash
# Three new backbones, fixed seed and unchanged configured batch sizes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/query_fusion_device.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    'Usage: bash scripts/run_irregular_voronoi_h.sh [grafiti|cru|latentode|all] [all|activity|mimic|physionet|ushcn|dist] [--device cuda:1]' \
    'Defaults: grafiti all; seed 3407, configured batch size, up to 30 epochs.' \
    'COMPARE=1: base, voronoi_h_no_density, voronoi_h.' \
    'Optional: BATCH_SIZE, PYTHON, RUN_DIR, TPATCH_DATA_ROOT, DIST_DATA_PATH, DRY_RUN=1.'
  exit 0
fi
if (( $# > 2 )); then echo 'Expected backbone and dataset; see --help' >&2; exit 2; fi
cd "$(dirname "$0")/.."
case "${1:-grafiti}" in
  grafiti) backbones=(GraFITiQueryFusion) ;;
  cru) backbones=(CRUQueryFusion) ;;
  latentode) backbones=(LatentODEQueryFusion) ;;
  all) backbones=(GraFITiQueryFusion CRUQueryFusion LatentODEQueryFusion) ;;
  *) echo 'Unknown backbone; see --help' >&2; exit 2 ;;
esac
case "${2:-all}" in
  all) datasets=(activity mimic physionet ushcn dist) ;;
  activity|mimic|physionet|ushcn|dist) datasets=("$2") ;;
  *) echo 'Unknown dataset; see --help' >&2; exit 2 ;;
esac
case "${COMPARE:-0}" in
  0) variants=(voronoi_h) ;;
  1) variants=(base voronoi_h_no_density voronoi_h) ;;
  *) echo 'COMPARE must be 0 or 1' >&2; exit 2 ;;
esac
case "${DRY_RUN:-0}" in 0|1) ;; *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2 ;; esac
data_root="${TPATCH_DATA_ROOT:-./data/tpatch-data}"
dist_data="${DIST_DATA_PATH:-./data/processed/DIST.npz}"
for ds in "${datasets[@]}"; do
  if [[ "$ds" == dist ]]; then
    [[ -f "$dist_data" ]] || { echo "Missing DIST archive: $dist_data" >&2; exit 2; }
  else
    [[ -d "$data_root/$ds/processed" ]] || { echo "Missing dataset: $data_root/$ds/processed" >&2; exit 2; }
  fi
done
output="${RUN_DIR:-results/query_fusion/${1:-grafiti}_voronoi_h_$(date +%Y%m%d_%H%M%S)_$$}"
command=("${PYTHON:-python}" -u scripts/run_query_fusion.py --output "$output"
  --data-root "$data_root" --dist-data "$dist_data" --device "$DEVICE"
  --backbones "${backbones[@]}" --datasets "${datasets[@]}" --variants "${variants[@]}"
  --seeds 3407 --epochs 30 --input-mode FullExo --backbone-history y --stream-logs)
if [[ "${DRY_RUN:-0}" == 1 ]]; then command+=(--plan-only); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then command+=(--batch-size "$BATCH_SIZE"); fi
printf '%s sequential runs; seed 3407; device %s; batch size %s\n' \
  "$((${#backbones[@]} * ${#datasets[@]} * ${#variants[@]}))" "$DEVICE" "${BATCH_SIZE:-config default}"
printf 'Output: %s\n' "$output"
exec "${command[@]}"
