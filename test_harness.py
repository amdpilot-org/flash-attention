#!/usr/bin/env python3
"""Cycle-80h Phase 2B harness shim — auto-injected by docker_manager.

When the container is launched in tool_call mode, the LLM-emitted
test_harness.py is moved to test_harness_inner.py and this shim takes
its place at /workspace/test_harness.py. The supervisor's canonical
command (``python3 /workspace/test_harness.py``) is unchanged; the
shim acquires GPUs via amdpilot.runtime.harness_runner.acquire_for_self
BEFORE runpy'ing the inner harness, so the inner's first ``import
torch`` initializes with only the chosen GPUs visible.

GPU count comes from $AMDPILOT_HARNESS_GPU_COUNT (injected by
docker_manager at container start). Defaults to 1 if unset.
"""
import os
import runpy
import sys

count = int(os.environ.get("AMDPILOT_HARNESS_GPU_COUNT", "1"))
timeout_s = float(os.environ.get("AMDPILOT_HARNESS_GPU_TIMEOUT_S", "300"))

try:
    from amdpilot.runtime.harness_runner import acquire_for_self
except ImportError as exc:
    print(f"[tool_call shim] amdpilot.runtime.harness_runner not importable: {exc}",
          file=sys.stderr)
    print(f"[tool_call shim] PYTHONPATH={os.environ.get('PYTHONPATH','<unset>')}",
          file=sys.stderr)
    sys.exit(70)  # EX_SOFTWARE

chosen = acquire_for_self(count=count, timeout_s=timeout_s)
print(f"[tool_call shim] acquired GPU(s) {chosen}; "
      f"running inner harness at /workspace/test_harness_inner.py",
      file=sys.stderr, flush=True)

runpy.run_path("/workspace/test_harness_inner.py", run_name="__main__")
