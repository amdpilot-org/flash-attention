#!/usr/bin/env python3
"""Stage0 harness for flash-attention issue #16.

Runs a bounded hipify-perl dry run over the FA-3 Hopper source set and reports
hipify_unportable_intrinsic_count as the number of distinct warning/error
catalog entries produced by hipify-perl.  The metric name is intentionally
verbatim per the issue contract.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("FLASH_ATTENTION_REPO", "/workspace/flash-attention"))
HOPPER = REPO / "hopper"
LOG = Path(os.environ.get("FA3_HIPIFY_LOG", "/tmp/fa3_hipify_dryrun.log"))
CATALOG = Path(os.environ.get("FA3_INTRINSIC_CATALOG", "/tmp/fa3_intrinsic_catalog.txt"))
WRAPPER = REPO / "docker" / "amd_patch" / "hopper" / "hipify_wrapper.sh"

# Curated top-level Hopper FA-3 files.  Prior verified Stage0 trials found that
# running hipify-perl over every header can stall or be OOM-killed (notably
# rotary.h on this node); these files cover the kernel/launch and utility
# sources where unportable CUDA intrinsics surface.
HOPPER_FILES = [
    "flash_fwd_combine.cu",
    "flash_prepare_scheduler.cu",
    "flash.h",
    "flash_bwd_kernel_sm90.h",
    "flash_bwd_launch_template.h",
    "flash_bwd_postprocess_kernel.h",
    "flash_bwd_preprocess_kernel.h",
    "flash_fwd_combine_kernel.h",
    "flash_fwd_combine_launch_template.h",
    "flash_fwd_kernel_sm90.h",
    "flash_fwd_launch_template.h",
    "heuristics.h",
    "mask.h",
    "pack_gqa.h",
    "paged_kv.h",
    "cuda_check.h",
    "utils.h",
]


def main() -> int:
    if not HOPPER.is_dir():
        print(f"ERROR: Hopper directory not found: {HOPPER}", file=sys.stderr)
        return 2
    missing = [f for f in HOPPER_FILES if not (HOPPER / f).is_file()]
    if missing:
        print(f"ERROR: Missing expected Hopper files: {missing}", file=sys.stderr)
        return 2
    if not WRAPPER.is_file():
        print(f"ERROR: Missing hipify wrapper: {WRAPPER}", file=sys.stderr)
        return 2

    LOG.unlink(missing_ok=True)
    CATALOG.unlink(missing_ok=True)
    cmd = [str(WRAPPER)] + [str(HOPPER / f) for f in HOPPER_FILES]
    env = os.environ.copy()
    env.setdefault("HIPIFY_TIMEOUT_SECONDS", "200")
    proc = subprocess.run(cmd, cwd=str(HOPPER), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    LOG.write_text(proc.stdout, encoding="utf-8", errors="replace")

    catalog_lines = []
    for line in proc.stdout.splitlines():
        if re.search(r"warning:|error:|HIPIFY_WARNING", line):
            catalog_lines.append(line.strip())
    distinct = sorted(set(catalog_lines))
    CATALOG.write_text("".join(f"      1 {line}\n" for line in distinct), encoding="utf-8")

    # The metric remains valid even if hipify-perl exits non-zero after emitting
    # the catalog; only harness/environment failures above abort without metric.
    print(f"hipify_unportable_intrinsic_count: {float(len(distinct))}")
    print(f"hipify_catalog_path: {CATALOG}")
    print(f"hipify_log_path: {LOG}")
    print(f"hipify_exit_code: {proc.returncode}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
