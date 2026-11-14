#!/usr/bin/env bash
# Graph Transformer; 12 non-DIST datasets by default.
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
  run_tag="${RUN_TAG:-exen_graph_transformer_no_exogenous}"
else
  run_tag="${RUN_TAG:-exen_graph_transformer_known_future}"
fi
if [[ -z "${RUN_TAG:-}" && -n "${TIME_LAYERS:-}${VARIABLE_LAYERS:-}${CROSS_LAYERS:-}${WITHIN_VARIABLES:-}" ]]; then
  run_tag="${run_tag}_t${TIME_LAYERS:-cfg}_v${VARIABLE_LAYERS:-cfg}_c${CROSS_LAYERS:-cfg}_${WITHIN_VARIABLES:-cfg}"
fi
datasets=(BE NP PJM FR DE Energy Sdwpfm1 Sdwpfm2 Sdwpfh1 Sdwpfh2 Colbun Rapel)
if [[ "$include_dist" == 1 ]]; then
  datasets+=(DIST)
fi
for dataset in "${datasets[@]}"; do
  printf '\nRunning ExEnGraphTransformer: %s (NO_EXOGENOUS=%s)\n' "$dataset" "$no_exogenous"
  dataset_tag="$run_tag"
  if [[ "$dataset" == DIST ]]; then
    dataset_tag="${dataset_tag}_rolling_${ROLLING_WINDOW_SIZE:-cfg}"
  fi
  command=("$python_cmd" -u main.py
    --config_path "config/ExEnGraphTransformer/processed/${dataset}.yaml"
    --model_save_dir_path "model_states/${dataset_tag}/${dataset}"
    --result_save_dir_path "results/${dataset_tag}/${dataset}")
  if [[ -n "${ATTENTION_DTYPE:-}" ]]; then
    command+=(--attention-dtype "$ATTENTION_DTYPE")
  fi
  if [[ -n "${DEVICE:-}" ]]; then
    command+=(--device "$DEVICE")
  fi
  if [[ "$no_exogenous" == 1 ]]; then
    command+=(--no-exogenous)
  fi
  [[ -z "${TIME_LAYERS:-}" ]] || command+=(--time-layers "$TIME_LAYERS")
  [[ -z "${VARIABLE_LAYERS:-}" ]] || command+=(--variable-layers "$VARIABLE_LAYERS")
  [[ -z "${CROSS_LAYERS:-}" ]] || command+=(--cross-layers "$CROSS_LAYERS")
  [[ -z "${WITHIN_VARIABLES:-}" ]] || command+=(--within-variables "$WITHIN_VARIABLES")
  if [[ "$dataset" == DIST && -n "${ROLLING_WINDOW_SIZE:-}" ]]; then
    command+=(--rolling-window-size "$ROLLING_WINDOW_SIZE")
  fi
  "${command[@]}"
done
