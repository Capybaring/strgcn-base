#!/usr/bin/env bash
# Same STRGCNDirect backbone, with historical and future exogenous inputs removed.
set -euo pipefail
export NO_EXOGENOUS=1
export RUN_TAG="${RUN_TAG:-strgcn_direct_no_exogenous}"
exec bash "$(dirname "$0")/run_strgcn_direct.sh"
