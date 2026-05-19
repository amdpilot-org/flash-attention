#!/usr/bin/env bash
set -uo pipefail
: "${HIPIFY_TIMEOUT_SECONDS:=200}"
if ! command -v hipify-perl >/dev/null 2>&1; then
  echo "ERROR: hipify-perl not found" >&2
  exit 127
fi
status=0
for src in "$@"; do
  if [ ! -f "$src" ]; then
    echo "ERROR: missing source $src" >&2
    status=2
    continue
  fi
  timeout "${HIPIFY_TIMEOUT_SECONDS}s" hipify-perl -inplace=false -no-output-stats "$src" || status=$?
done
exit "$status"
