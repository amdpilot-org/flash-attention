#!/usr/bin/env python3
"""Small Stage0 timing probe for the locked FA-4 forward shape.

This is not the acceptance metric; test_harness.py emits the required
correctness metric. The timing probe is included to give executor agents a
stable smoke benchmark for the same shape.
"""
import json
import os
import statistics
import time

import torch


def sdpa_reference(q, k, v):
    qh, kh, vh = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=False)
    return out.transpose(1, 2).contiguous()


def time_call(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), min(samples)


def main():
    os.environ.setdefault("TMPDIR", "/scratch/tmp")
    torch.manual_seed(1234)
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is false")
    from flash_attn.cute import flash_attn_func

    b, s, h, d = 1, 256, 1, 128
    q = torch.randn((b, s, h, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((b, s, h, d), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((b, s, h, d), device="cuda", dtype=torch.bfloat16)

    fa4_median, fa4_best = time_call(lambda: flash_attn_func(q, k, v, causal=False))
    sdpa_median, sdpa_best = time_call(lambda: sdpa_reference(q, k, v))
    print(f"fa4_forward_median_ms: {fa4_median:.6f} ms")
    print(f"sdpa_forward_median_ms: {sdpa_median:.6f} ms")
    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "shape": [b, s, h, d],
        "dtype": "bf16",
        "causal": False,
        "fa4_forward_median_ms": fa4_median,
        "fa4_forward_best_ms": fa4_best,
        "sdpa_forward_median_ms": sdpa_median,
        "sdpa_forward_best_ms": sdpa_best,
    }))


if __name__ == "__main__":
    main()
