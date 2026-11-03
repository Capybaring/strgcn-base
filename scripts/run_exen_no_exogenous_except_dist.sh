#!/usr/bin/env bash
# Matched Y-only ablation: no historical or future X; excludes DIST.
set -euo pipefail
export NO_EXOGENOUS=1
export RUN_TAG="${RUN_TAG:-exen_no_exogenous}"
exec bash "$(dirname "$0")/run_exen_future_except_dist.sh"
