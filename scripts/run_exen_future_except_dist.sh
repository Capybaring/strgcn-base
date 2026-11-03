#!/usr/bin/env bash
# Run all 12 non-DIST datasets with known future X, in new output directories.
set -euo pipefail
cd "$(dirname "$0")/.."
python_cmd="${PYTHON:-python}"
run_tag="${RUN_TAG:-exen_known_future}"
for dataset in BE NP PJM FR DE Energy Sdwpfm1 Sdwpfm2 Sdwpfh1 Sdwpfh2 Colbun Rapel; do
  printf '\nRunning %s with known future exogenous inputs\n' "$dataset"
  command=("$python_cmd" -u main.py
    --config_path "config/ExEn/processed/${dataset}.yaml"
    --model_save_dir_path "model_states/${run_tag}/${dataset}"
    --result_save_dir_path "results/${run_tag}/${dataset}")
  if [[ -n "${DEVICE:-}" ]]; then
    command+=(--device "$DEVICE")
  fi
  "${command[@]}"
done
