#!/usr/bin/env python3
"""Locked FA-4 CUTE forward correctness harness for issue #22.

Runs the issue-specific MI355X/gfx950 shape and compares
`flash_attn.cute.flash_attn_func` against the no-causal bf16 attention semantic
reference.  The required metric line is intentionally stable for the
orchestrator:

    fa4_forward_correctness_max_abs_diff: <value> max_abs_diff_vs_fa3_bf16
"""
import argparse
import importlib
import json
import math
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

REPO_ROOT = os.environ.get("FLASH_ATTENTION_REPO", "/workspace/flash-attention")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def find_fa4_func():
    candidates = [
        ("flash_attn.cute", "flash_attn_func"),
        ("flash_attn.cute.interface", "flash_attn_func"),
        ("flash_attn.cute.flash_attn_interface", "flash_attn_func"),
        ("flash_attn.cute.flash_attn_func", "flash_attn_func"),
    ]
    errors = []
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr)
            return fn, f"{mod_name}.{attr}"
        except Exception as exc:  # keep trying alternate public locations
            errors.append(f"{mod_name}.{attr}: {type(exc).__name__}: {exc}")
    raise ImportError("FA-4 CUTE flash_attn_func not importable; " + " | ".join(errors))


def time_call(fn, iters):
    # A small bounded timing loop is included for context only.  Correctness is
    # the required Stage0 metric.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / max(iters, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=256)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timing-iters", type=int, default=10)
    p.add_argument("--fail-on-target-miss", action="store_true")
    args = p.parse_args()

    result = {
        "metric_name": "fa4_forward_correctness_max_abs_diff",
        "metric_unit": "max_abs_diff_vs_fa3_bf16",
        "metric_direction": "lower",
        "target_threshold": 8e-3,
        "shape": {"batch": args.batch, "seqlen": args.seqlen, "heads": args.heads, "head_dim": args.head_dim},
        "causal": False,
        "dtype": "bf16",
        "api": None,
        "status": "unknown",
        "error": None,
    }

    metric = math.inf
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is not available")
        if args.head_dim != 128:
            raise ValueError("issue #22 Stage0 harness is locked to head_dim=128")

        torch.manual_seed(args.seed)
        device = "cuda"
        q = torch.randn(args.batch, args.seqlen, args.heads, args.head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def sdpa_call():
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, is_causal=False,
            ).transpose(1, 2).contiguous()

        ref = sdpa_call()
        torch.cuda.synchronize()

        fn, api = find_fa4_func()
        result["api"] = api

        def fa4_call():
            out = fn(q, k, v, causal=False)
            return out[0] if isinstance(out, tuple) else out

        out = fa4_call()
        torch.cuda.synchronize()
        metric = (out.float() - ref.float()).abs().max().item()
        result["status"] = "ran"
        result["fa4_latency_ms"] = time_call(fa4_call, args.timing_iters)
        result["sdpa_reference_latency_ms"] = time_call(sdpa_call, args.timing_iters)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback_tail"] = traceback.format_exc(limit=4).splitlines()[-12:]

    printable = "inf" if math.isinf(metric) else f"{metric:.10g}"
    result["fa4_forward_correctness_max_abs_diff"] = printable if math.isinf(metric) else metric
    result["metric_value_is_finite"] = math.isfinite(metric)
    result["target_passed"] = math.isfinite(metric) and metric < 8e-3

    print(f"fa4_forward_correctness_max_abs_diff: {printable} max_abs_diff_vs_fa3_bf16")
    if "fa4_latency_ms" in result:
        print(f"fa4_forward_latency_ms: {result['fa4_latency_ms']:.6f} ms")
        print(f"sdpa_reference_latency_ms: {result['sdpa_reference_latency_ms']:.6f} ms")
    print("STAGE0_RESULT_JSON=" + json.dumps(result, sort_keys=True, default=str, allow_nan=False))

    if args.fail_on_target_miss and not result["target_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
