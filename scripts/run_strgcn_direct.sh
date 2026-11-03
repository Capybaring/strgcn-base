#!/usr/bin/env bash
# Joint STRGCN time/variable relations; defaults to the 12 non-DIST tasks.
set -euo pipefail
cd "$(dirname "$0")/.."
python_cmd="${PYTHON:-python}"
run_tag="${RUN_TAG:-strgcn_direct_known_future}"
include_dist="${INCLUDE_DIST:-0}"
if [[ "$include_dist" != 0 && "$include_dist" != 1 ]]; then
  printf 'INCLUDE_DIST must be 0 or 1\n' >&2
  exit 2
fi
datasets=(BE NP PJM FR DE Energy Sdwpfm1 Sdwpfm2 Sdwpfh1 Sdwpfh2 Colbun Rapel)
if [[ "$include_dist" == 1 ]]; then
  datasets+=(DIST)
fi
for dataset in "${datasets[@]}"; do
  printf '\nRunning STRGCNDirect: %s\n' "$dataset"
  command=("$python_cmd" -u main.py
    --config_path "config/STRGCNDirect/processed/${dataset}.yaml"
    --model_save_dir_path "model_states/${run_tag}/${dataset}"
    --result_save_dir_path "results/${run_tag}/${dataset}")
  if [[ -n "${DEVICE:-}" ]]; then
    command+=(--device "$DEVICE")
  fi
  "${command[@]}"
done
