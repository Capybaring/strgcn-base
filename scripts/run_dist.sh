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
  "${command[@]}"
done
