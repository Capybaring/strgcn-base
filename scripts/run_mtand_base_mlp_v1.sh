#!/usr/bin/env bash
# Sequential mTAN/mTAND experiments: 4 datasets x 3 methods x 1 seed = 12 runs.
# Run from the repository: bash scripts/run_mtand_base_mlp_v1.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/query_fusion_device.sh"

if (( $# != 0 )); then
  echo 'Usage: bash scripts/run_mtand_base_mlp_v1.sh [--device cuda:1]' >&2
  exit 2
fi
cd "$(dirname "$0")/.."

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

output="${RUN_DIR:-results/query_fusion/mtand_base_mlp_v1_$(date +%Y%m%d_%H%M%S)_$$}"
command=("${PYTHON:-python}" -u scripts/run_query_fusion.py
  --data-root "$data_root"
  --output "$output"
  --device "${DEVICE:-cuda:0}"
  --backbones MTANQueryFusion
  --datasets activity mimic physionet ushcn
  --variants base mlp v1
  --seeds 3407
  --epochs 30
  --input-mode FullExo
  --backbone-history y
  --stream-logs)
if [[ "${DRY_RUN:-0}" == 1 ]]; then command+=(--plan-only); fi

printf 'mTAN/mTAND: activity -> mimic -> physionet -> ushcn; base -> mlp -> v1 for each dataset.\n'
printf '12 sequential runs, fixed seed 3407. Output: %s\n' "$output"
exec "${command[@]}"
