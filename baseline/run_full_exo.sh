#!/usr/bin/env bash
# Run both new baselines on all processed dataset configurations, including DIST.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN_ROOT="${RUN_ROOT:-results/baseline/timexer_crosslinear_FullExo_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PYTHON="${PYTHON:-python}"
shopt -s nullglob
configs=(config/ExEn/processed/*.yaml)
if ((${#configs[@]} == 0)); then echo 'No processed dataset configurations found' >&2; exit 1; fi
mkdir -p "$RUN_ROOT"
echo "Results: $RUN_ROOT"
for method in timexer crosslinear; do
  mkdir -p "$RUN_ROOT/$method"
  for config in "${configs[@]}"; do
    dataset=$(basename "$config" .yaml)
    echo "========== $method / $dataset / FullExo =========="
    "$PYTHON" -u -m "baseline.${method}_irregular.run" \
      --config "$config" --mode FullExo --device "$DEVICE" \
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
      --output "$RUN_ROOT/$method/$dataset" \
      2>&1 | tee "$RUN_ROOT/$method/${dataset}.log"
  done
done
echo "All completed: $RUN_ROOT"
