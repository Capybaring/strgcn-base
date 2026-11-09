#!/usr/bin/env bash
# Both original patch backbones, with the same DAG pointwise MLP Fusion.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${BACKBONE:-all}" in
  all) models=(TPatchGNNFusion HiPatchFusion);;
  tpatchgnn) models=(TPatchGNNFusion);;
  hipatch) models=(HiPatchFusion);;
  *) echo 'BACKBONE must be all, tpatchgnn or hipatch' >&2; exit 2;;
esac
mode="${MODE:-FullExo}"
case "$mode" in FullExo|HistExo|NoExo) ;; *) echo 'MODE must be FullExo, HistExo or NoExo' >&2; exit 2;; esac
if (( $# == 0 )); then set -- mimic physionet activity ushcn; fi
for model in "${models[@]}"; do
  for dataset in "$@"; do
    case "$dataset" in mimic|physionet|activity|ushcn) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2;; esac
    tag="${RUN_TAG:-${model}_dag_mlp_${mode}}"
    command=("${PYTHON:-python}" -u main.py
      --config_path "config/${model}/tpatch/${dataset}.yaml"
      --model_save_dir_path "model_states/${tag}/${model}/${dataset}"
      --result_save_dir_path "results/${tag}/${model}/${dataset}")
    [[ -z "${DEVICE:-}" ]] || command+=(--device "$DEVICE")
    [[ -z "${TPATCH_DATA_ROOT:-}" ]] || command+=(--tpatch-data-root "$TPATCH_DATA_ROOT")
    if [[ "$mode" == NoExo ]]; then command+=(--no-exogenous); fi
    if [[ "$mode" == HistExo ]]; then command+=(--history-only-exogenous); fi
    printf '\nRunning %s / %s (%s)\n' "$model" "$dataset" "$mode"
    "${command[@]}"
  done
done
