#!/usr/bin/env python3
"""Stage0 harness for issue-flash-attention-22-r58.

Validates the issue-specific FA-4 CUTE API path on a single MI355X GPU and
emits the required metric name with a real numerical comparison. This harness
intentionally bypasses sglang server startup and uses direct PyTorch tensors.
"""
import importlib
import json
import os
import sys
import time
import traceback

import torch

METRIC_NAME = "fa4_forward_correctness_max_abs_diff"
METRIC_UNIT = "max_abs_diff_vs_reference_bf16"
THRESHOLD = 8e-3


def _sdpa_reference(q, k, v):
    qh, kh, vh = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, dropout_p=0.0, is_causal=False
    )
    return out.transpose(1, 2).contiguous()


def _call_flash_attn_func(q, k, v):
    mod = importlib.import_module("flash_attn.cute")
    fn = getattr(mod, "flash_attn_func", None)
    if fn is None:
        raise RuntimeError("flash_attn.cute.flash_attn_func is missing")
    return fn(q, k, v, softmax_scale=None, causal=False)


def main():
    os.makedirs("/scratch/tmp", exist_ok=True)
    print(json.dumps({
        "event": "env",
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    }))
    if not torch.cuda.is_available():
        print("STAGE0_FAILURE: gpu_not_visible", file=sys.stderr)
        return 2

    torch.manual_seed(22)
    B, S, H, D = 1, 1024, 1, 128
    q = torch.randn((B, S, H, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    torch.cuda.synchronize()
    ref = _sdpa_reference(q, k, v)
    torch.cuda.synchronize()

    try:
        with torch.no_grad():
            out = _call_flash_attn_func(q, k, v)
        if isinstance(out, tuple):
            out = out[0]
        torch.cuda.synchronize()
    except Exception as exc:
        print("STAGE0_FAILURE: fa4_api_unimplemented_or_runtime_error", file=sys.stderr)
        print(repr(exc), file=sys.stderr)
        traceback.print_exc()
        return 3

    max_abs = (out.float() - ref.float()).abs().max().item()

    with torch.no_grad():
        for _ in range(5):
            _ = _call_flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    with torch.no_grad():
        for _ in range(iters):
            _ = _call_flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) * 1000.0 / iters

    print(f"{METRIC_NAME}: {max_abs:.8g}")
    print(f"fa4_forward_latency_ms: {t_ms:.6f}")
    print(json.dumps({
        "metric_name": METRIC_NAME,
        "metric_unit": METRIC_UNIT,
        "metric_value": max_abs,
        "threshold": THRESHOLD,
        "latency_ms": t_ms,
    }))
    return 0 if max_abs < THRESHOLD else 4


if __name__ == "__main__":
    raise SystemExit(main())
