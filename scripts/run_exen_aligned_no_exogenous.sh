#!/usr/bin/env bash
set -euo pipefail
export NO_EXOGENOUS=1
export RUN_TAG="${RUN_TAG:-exen_aligned_no_exogenous}"
exec bash "$(dirname "$0")/run_exen_aligned.sh"
