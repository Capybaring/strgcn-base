#!/usr/bin/env bash
# Record-based supplementary oracle-X tasks; default: all four datasets.
set -euo pipefail
cd "$(dirname "$0")/.."
model="${MODEL:-ExEnGraphTransformer}"
case "$model" in ExEn|STRGCNDirect|ExEnAligned|ExEnGraphTransformer) ;; *) echo "Unsupported MODEL: $model" >&2; exit 2;; esac
no_exogenous="${NO_EXOGENOUS:-0}"
case "$no_exogenous" in 0|1) ;; *) echo 'NO_EXOGENOUS must be 0 or 1' >&2; exit 2;; esac
mode=oracle_x
[[ "$no_exogenous" == 0 ]] || mode=y_only
tag="${RUN_TAG:-${model}_tpatch_v2_${mode}}"
if [[ -z "${RUN_TAG:-}" && -n "${TIME_LAYERS:-}${VARIABLE_LAYERS:-}${CROSS_LAYERS:-}${WITHIN_VARIABLES:-}" ]]; then
  tag="${tag}_t${TIME_LAYERS:-cfg}_v${VARIABLE_LAYERS:-cfg}_c${CROSS_LAYERS:-cfg}_${WITHIN_VARIABLES:-cfg}"
fi
if [[ -z "${RUN_TAG:-}" && "$model" == ExEnGraphTransformer ]]; then
  tag="${tag}_rolling_${ROLLING_WINDOW_SIZE:-cfg}"
fi
if (( $# == 0 )); then set -- mimic physionet activity ushcn; fi
for dataset in "$@"; do
  case "$dataset" in mimic|physionet|activity|ushcn) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2;; esac
  command=("${PYTHON:-python}" -u main.py
    --config_path "config/${model}/tpatch/${dataset}.yaml"
    --model_save_dir_path "model_states/${tag}/${dataset}"
    --result_save_dir_path "results/${tag}/${dataset}")
  [[ -z "${DEVICE:-}" ]] || command+=(--device "$DEVICE")
  [[ -z "${TPATCH_DATA_ROOT:-}" ]] || command+=(--tpatch-data-root "$TPATCH_DATA_ROOT")
  [[ "$no_exogenous" == 0 ]] || command+=(--no-exogenous)
  if [[ -n "${ATTENTION_DTYPE:-}" ]]; then
    [[ "$model" == ExEnGraphTransformer ]] || { echo 'ATTENTION_DTYPE requires ExEnGraphTransformer' >&2; exit 2; }
    command+=(--attention-dtype "$ATTENTION_DTYPE")
  fi
  printf '\nRunning %s / %s (%s)\n' "$model" "$dataset" "$mode"
  [[ -z "${TIME_LAYERS:-}" ]] || command+=(--time-layers "$TIME_LAYERS")
  [[ -z "${VARIABLE_LAYERS:-}" ]] || command+=(--variable-layers "$VARIABLE_LAYERS")
  [[ -z "${CROSS_LAYERS:-}" ]] || command+=(--cross-layers "$CROSS_LAYERS")
  [[ -z "${WITHIN_VARIABLES:-}" ]] || command+=(--within-variables "$WITHIN_VARIABLES")
  if [[ -n "${ROLLING_WINDOW_SIZE:-}" ]]; then
    [[ "$model" == ExEnGraphTransformer ]] || { echo 'ROLLING_WINDOW_SIZE requires ExEnGraphTransformer' >&2; exit 2; }
    command+=(--rolling-window-size "$ROLLING_WINDOW_SIZE")
  fi
  "${command[@]}"
done
