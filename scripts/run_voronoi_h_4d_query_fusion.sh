#!/usr/bin/env bash
# Equal-width density encoding, direct four-channel attention and matched ablation.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    'Usage: bash scripts/run_voronoi_h_4d_query_fusion.sh [tpatchgnn|hipatch|mtand|grafiti|cru|latentode|all_six] [all|activity|mimic|physionet|ushcn|dist] [--device cuda:1]' \
    'Defaults: tpatchgnn all; five sequential voronoi_h_4d runs, seed 3407, configured batch size.' \
    'COMPARE=1 runs voronoi_h_4d_no_density and voronoi_h_4d with identical network dimensions.' \
    'Optional: BATCH_SIZE, TPATCH_DATA_ROOT, DIST_DATA_PATH, RUN_DIR, PYTHON, DRY_RUN=1.'
  exit 0
fi
export VORONOI_VARIANT=voronoi_h_4d
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_voronoi_query_fusion.sh" "$@"
