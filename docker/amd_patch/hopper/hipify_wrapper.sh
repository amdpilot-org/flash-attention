#!/usr/bin/env bash
set -euo pipefail
cd "${FLASH_ATTENTION_REPO:-/workspace/flash-attention}/hopper"
timeout "${HIPIFY_TIMEOUT_SECONDS:-200}" hipify-perl -inplace=false -no-output-stats *.cu *.cuh *.h > /tmp/fa3_hipify_dryrun.log 2>&1
grep -E "warning:|error:|HIPIFY_WARNING" /tmp/fa3_hipify_dryrun.log | sort | uniq -c | sort -rn > /tmp/fa3_intrinsic_catalog.txt || true
