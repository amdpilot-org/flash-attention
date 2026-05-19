#!/usr/bin/env python3
"""Stage0 harness for issue-flash-attention-22.

Runs a small faithful MI355X bf16 no-causal head_dim=128 check. The harness
prints the required metric name verbatim and exits non-zero if the FA-4 CUTE API
is absent/unimplemented or exceeds the parity tolerance.
"""
import json
import os
import sys
import time
import traceback

import torch

METRIC = "fa4_forward_correctness_max_abs_diff"
TOL = 8e-3

def sdpa_reference(q, k, v):
    # q/k/v: [B, S, H, D], bf16. SDPA wants [B, H, S, D].
    qh, kh, vh = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=False)
    return out.transpose(1, 2).contiguous()

def main():
    os.environ.setdefault("TMPDIR", "/scratch/tmp")
    torch.manual_seed(1234)
    if not torch.cuda.is_available():
        print(json.dumps({"status": "failed", "reason": "torch.cuda.is_available false"}))
        return 2
    dev_name = torch.cuda.get_device_name(0)
    # Single required acceptance shape: head_dim=128, bf16, causal=False.
    b, s, h, d = 1, 256, 1, 128
    q = torch.randn((b, s, h, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((b, s, h, d), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((b, s, h, d), device="cuda", dtype=torch.bfloat16)
    ref = sdpa_reference(q, k, v)
    torch.cuda.synchronize()

    try:
        from flash_attn.cute import flash_attn_func
    except Exception as exc:
        print(f"{METRIC}: 1e9 max_abs_diff  # import_error {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 3

    try:
        t0 = time.perf_counter()
        out = flash_attn_func(q, k, v, causal=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        print(f"{METRIC}: 1e9 max_abs_diff  # runtime_error {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 4

    if isinstance(out, tuple):
        out = out[0]
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"{METRIC}: {diff:.8g} max_abs_diff")
    print(f"fa4_forward_time_ms: {elapsed_ms:.6f} ms")
    print(json.dumps({"device": dev_name, "shape": [b, s, h, d], "dtype": "bf16", "causal": False, METRIC: diff, "t_ms": elapsed_ms}))
    return 0 if diff < TOL else 5

if __name__ == "__main__":
    sys.exit(main())
