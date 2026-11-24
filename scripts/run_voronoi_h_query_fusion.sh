#!/usr/bin/env bash
# Same data, batching and seed as voronoi; density enters four-way node fusion.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    'Usage: bash scripts/run_voronoi_h_query_fusion.sh [tpatchgnn|hipatch|mtand|all] [all|activity|mimic|physionet|ushcn|dist] [--device cuda:1]' \
    'Defaults: tpatchgnn all; five sequential voronoi_h runs, seed 3407, configured batch size.' \
    'COMPARE=1 runs voronoi_h_no_density and voronoi_h to isolate density from fusion capacity.' \
    'Optional: BATCH_SIZE, TPATCH_DATA_ROOT, DIST_DATA_PATH, RUN_DIR, PYTHON, DRY_RUN=1.'
  exit 0
fi
export VORONOI_VARIANT=voronoi_h
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_voronoi_query_fusion.sh" "$@"
