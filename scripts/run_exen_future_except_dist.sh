#!/usr/bin/env bash
# Run all 12 non-DIST datasets with known future X, in new output directories.
set -euo pipefail
cd "$(dirname "$0")/.."
python_cmd="${PYTHON:-python}"
no_exogenous="${NO_EXOGENOUS:-0}"
if [[ "$no_exogenous" != 0 && "$no_exogenous" != 1 ]]; then
  printf 'NO_EXOGENOUS must be 0 or 1\n' >&2
  exit 2
fi
if [[ "$no_exogenous" == 1 ]]; then
  run_tag="${RUN_TAG:-exen_no_exogenous}"
else
  run_tag="${RUN_TAG:-exen_known_future}"
fi
for dataset in BE NP PJM FR DE Energy Sdwpfm1 Sdwpfm2 Sdwpfh1 Sdwpfh2 Colbun Rapel; do
  printf '\nRunning %s (NO_EXOGENOUS=%s)\n' "$dataset" "$no_exogenous"
  command=("$python_cmd" -u main.py
    --config_path "config/ExEn/processed/${dataset}.yaml"
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
