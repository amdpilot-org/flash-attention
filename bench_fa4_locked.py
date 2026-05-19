#!/usr/bin/env python3
"""Locked FA-4 Stage0 benchmark entrypoint.

Delegates to test_harness.py so the bundle provides a stable bench_* command
while preserving the exact required metric semantics.
"""
import runpy
import sys

if __name__ == "__main__":
    ns = runpy.run_path("/workspace/flash-attention/test_harness.py", run_name="stage0_harness")
    main = ns.get("main")
    if main is None:
        raise SystemExit("test_harness.py did not define main()")
    sys.exit(main())
