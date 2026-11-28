#!/usr/bin/env bash
# Receiver-centered Voronoi-Graph Former; COMPARE=1 keeps identical star graphs.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    'Usage: bash scripts/run_voronoi_star_query_fusion.sh [tpatchgnn|hipatch|mtand|grafiti|cru|latentode|all|all_six] [all|activity|mimic|physionet|ushcn|dist] [--device cuda:1]' \
    'COMPARE=1 runs voronoi_star_no_density then voronoi_star; seed 3407, at most 30 epochs.' \
    'Reference density is fitted on training X only and saved in checkpoints.' \
    'Optional: BATCH_SIZE, TPATCH_DATA_ROOT, DIST_DATA_PATH, RUN_DIR, PYTHON, DRY_RUN=1.'
  exit 0
fi
export VORONOI_VARIANT=voronoi_star
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_voronoi_query_fusion.sh" "$@"
