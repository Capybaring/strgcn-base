#!/usr/bin/env bash
# DIST: identical windows, train-only MinMax and variable-macro evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${MODEL:-all}" in
  all) models=(ExEnGraphTransformer TPatchGNNFusion HiPatchFusion);;
  exen|ExEnGraphTransformer) models=(ExEnGraphTransformer);;
  tpatchgnn|TPatchGNNFusion) models=(TPatchGNNFusion);;
  hipatch|HiPatchFusion) models=(HiPatchFusion);;
  *) echo 'MODEL must be all, exen, tpatchgnn or hipatch' >&2; exit 2;;
esac
mode="${MODE:-FullExo}"
case "$mode" in FullExo|HistExo|NoExo) ;; *) echo 'MODE must be FullExo, HistExo or NoExo' >&2; exit 2;; esac
for model in "${models[@]}"; do
  tag="${RUN_TAG:-dist_variable_macro_${mode}}"
  if [[ -z "${RUN_TAG:-}" && -n "${TIME_LAYERS:-}${VARIABLE_LAYERS:-}${CROSS_LAYERS:-}${WITHIN_VARIABLES:-}" ]]; then
    tag="${tag}_t${TIME_LAYERS:-cfg}_v${VARIABLE_LAYERS:-cfg}_c${CROSS_LAYERS:-cfg}_${WITHIN_VARIABLES:-cfg}"
  fi
  if [[ -z "${RUN_TAG:-}" && "$model" == ExEnGraphTransformer ]]; then
    tag="${tag}_rolling_${ROLLING_WINDOW_SIZE:-cfg}"
  fi
  command=("${PYTHON:-python}" -u main.py
    --config_path "config/${model}/processed/DIST.yaml"
    --model_save_dir_path "model_states/${tag}/${model}/DIST"
    --result_save_dir_path "results/${tag}/${model}/DIST")
  [[ -z "${DEVICE:-}" ]] || command+=(--device "$DEVICE")
  if [[ "$model" == ExEnGraphTransformer && -n "${ATTENTION_DTYPE:-}" ]]; then
    command+=(--attention-dtype "$ATTENTION_DTYPE")
  fi
  if [[ "$mode" == HistExo ]]; then command+=(--history-only-exogenous); fi
  if [[ "$mode" == NoExo ]]; then command+=(--no-exogenous); fi
  printf '\nRunning %s / DIST (%s, variable-macro MinMax metrics)\n' "$model" "$mode"
  if [[ "$model" == ExEnGraphTransformer ]]; then
    [[ -z "${TIME_LAYERS:-}" ]] || command+=(--time-layers "$TIME_LAYERS")
    [[ -z "${VARIABLE_LAYERS:-}" ]] || command+=(--variable-layers "$VARIABLE_LAYERS")
    [[ -z "${CROSS_LAYERS:-}" ]] || command+=(--cross-layers "$CROSS_LAYERS")
    [[ -z "${WITHIN_VARIABLES:-}" ]] || command+=(--within-variables "$WITHIN_VARIABLES")
  fi
  if [[ "$model" == ExEnGraphTransformer && -n "${ROLLING_WINDOW_SIZE:-}" ]]; then
    command+=(--rolling-window-size "$ROLLING_WINDOW_SIZE")
  fi
  "${command[@]}"
done
