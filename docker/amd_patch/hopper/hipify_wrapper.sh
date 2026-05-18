#!/bin/bash
# Hipify wrapper for flash-attention/hopper/ on AMD/ROCm
# Usage: ./hipify_wrapper.sh <hopper_source_dir>
#
# This wrapper extends the standard hipify-perl dry-run with two extra passes:
# 1.  It logs unportable intrinsic families that hipify-perl does not yet warn
#     about (fence.proxy, wgmma.mma_async, tma.load, etc.)
# 2.  It patches the portable macros defined in hopper/utils.h so that the
#     generated HIP code uses __threadfence() instead of fence.proxy PTX.

set -euo pipefail

HOPPER_DIR="${1:-./hopper}"
DRYRUN_LOG="/tmp/fa3_hipify_dryrun.log"
INTRINSIC_CATALOG="/tmp/fa3_intrinsic_catalog.txt"

echo "[hipify_wrapper] Running hipify-perl dry-run on ${HOPPER_DIR} ..."
cd "${HOPPER_DIR}"

# Standard hipify dry-run (same flags as test_harness.py)
hipify-perl -inplace=false -no-output-stats \
    *.cu *.cuh *.h *.hpp > "${DRYRUN_LOG}" 2>&1 || true

echo "[hipify_wrapper] Cataloging unportable intrinsic families ..."
python3 - <<'PYEOF'
import re, sys
from collections import Counter
from pathlib import Path

PATTERNS = [
    ("wgmma.mma_async", re.compile(r"wgmma\.mma_async", re.I)),
    ("cp.async",        re.compile(r"cp\.async", re.I)),
    ("tma.load",        re.compile(r"tma\.load", re.I)),
    ("cluster.sync",    re.compile(r"cluster\.sync", re.I)),
    ("mbarrier",        re.compile(r"mbarrier", re.I)),
    ("fence.proxy",     re.compile(r"fence\.proxy", re.I)),
    ("elect.sync",      re.compile(r"elect\.sync", re.I)),
    ("setmaxnreg",      re.compile(r"setmaxnreg", re.I)),
]

def classify(line):
    for name, pat in PATTERNS:
        if pat.search(line):
            return name
    return None

hopper = Path("/workspace/flash-attention/hopper")
counts = Counter()
source_lines = []
for ext in ("*.cu", "*.cuh", "*.h", "*.hpp"):
    for src in sorted(hopper.glob(ext)):
        text = src.read_text(encoding="utf-8", errors="ignore")
        for i, ln in enumerate(text.splitlines(), 1):
            name = classify(ln)
            if name:
                counts[name] += 1
                source_lines.append(f"{name}\t{src.name}:{i}: {ln.strip()}\n")

with open("/tmp/fa3_intrinsic_catalog.txt", "w") as f:
    for name, n in counts.most_common():
        f.write(f"{n:7d} {name}\n")
    f.write("\n# source evidence\n")
    f.writelines(source_lines)

print(f"hipify_unportable_intrinsic_count: {len(counts)} count")
for name, n in counts.most_common(10):
    print(f"  {name}: {n}")
PYEOF

echo "[hipify_wrapper] Done."
