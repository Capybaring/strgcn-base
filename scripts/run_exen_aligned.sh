#!/usr/bin/env bash
# Aligned ExEn operator; 12 non-DIST datasets by default.
set -euo pipefail
cd "$(dirname "$0")/.."
python_cmd="${PYTHON:-python}"
no_exogenous="${NO_EXOGENOUS:-0}"
include_dist="${INCLUDE_DIST:-0}"
if [[ "$no_exogenous" != 0 && "$no_exogenous" != 1 ]]; then
  printf 'NO_EXOGENOUS must be 0 or 1\n' >&2
  exit 2
fi
if [[ "$include_dist" != 0 && "$include_dist" != 1 ]]; then
  printf 'INCLUDE_DIST must be 0 or 1\n' >&2
  exit 2
fi
if [[ "$no_exogenous" == 1 ]]; then
  run_tag="${RUN_TAG:-exen_aligned_no_exogenous}"
else
  run_tag="${RUN_TAG:-exen_aligned_known_future}"
fi
datasets=(BE NP PJM FR DE Energy Sdwpfm1 Sdwpfm2 Sdwpfh1 Sdwpfh2 Colbun Rapel)
if [[ "$include_dist" == 1 ]]; then
  datasets+=(DIST)
fi
for dataset in "${datasets[@]}"; do
  printf '\nRunning ExEnAligned: %s (NO_EXOGENOUS=%s)\n' "$dataset" "$no_exogenous"
  command=("$python_cmd" -u main.py
    --config_path "config/ExEnAligned/processed/${dataset}.yaml"
    --model_save_dir_path "model_states/${run_tag}/${dataset}"
    --result_save_dir_path "results/${run_tag}/${dataset}")
  if [[ -n "${DEVICE:-}" ]]; then
    command+=(--device "$DEVICE")
  fi
  if [[ "$no_exogenous" == 1 ]]; then
    command+=(--no-exogenous)
  fi
  "${command[@]}"
done
