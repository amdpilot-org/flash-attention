#!/usr/bin/env python3
"""Locked direct-PyTorch FA-4 baseline harness for issue #22.

This intentionally bypasses sglang server launch.  It checks the required CUTE
FA-4 API on MI355X for bf16/head_dim=128/non-causal and compares to a stable
PyTorch SDPA reference.  When the baseline API is absent/unimplemented, it emits
`inf` for the required metric; that is the expected Stage0 baseline condition.
"""
import argparse
import importlib
import json
import math
import os
import sys
import traceback

import torch
import torch.nn.functional as F

# Running this script by absolute path sets sys.path[0] to /workspace; ensure the
# checked-out repo is importable without requiring pip install/build.
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
        except Exception as exc:
            errors.append(f"{mod_name}.{attr}: {type(exc).__name__}: {exc}")
    raise ImportError("FA-4 CUTE flash_attn_func not importable; " + " | ".join(errors))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=256)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fail-on-target-miss", action="store_true")
    args = p.parse_args()

    result = {
        "metric_name": "fa4_forward_correctness_max_abs_diff",
        "metric_unit": "max_abs_diff_vs_fa3_bf16",
        "metric_direction": "lower",
        "shape": {"batch": args.batch, "seqlen": args.seqlen, "heads": args.heads, "head_dim": args.head_dim},
        "causal": False,
        "dtype": "bf16",
        "status": "unknown",
        "api": None,
        "error": None,
    }

    if not torch.cuda.is_available():
        result.update(status="environment_error", error="torch.cuda is not available")
        metric = math.inf
    else:
        torch.manual_seed(args.seed)
        device = "cuda"
        q = torch.randn(args.batch, args.seqlen, args.heads, args.head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        # Reference approximates the Tier-5 FA-3 forward semantics for this no-causal single shape.
        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).contiguous()
        torch.cuda.synchronize()
        try:
            fn, api = find_fa4_func()
            result["api"] = api
            out = fn(q, k, v, causal=False)
            if isinstance(out, tuple):
                out = out[0]
            torch.cuda.synchronize()
            metric = (out.float() - ref.float()).abs().max().item()
            result["status"] = "ran"
        except Exception as exc:
            metric = math.inf
            result["status"] = "fa4_unimplemented_or_failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback_tail"] = traceback.format_exc(limit=4).splitlines()[-12:]

    printable = "inf" if math.isinf(metric) else f"{metric:.8g}"
    result["fa4_forward_correctness_max_abs_diff"] = printable if math.isinf(metric) else metric
    result["metric_value_is_finite"] = math.isfinite(metric)
    print(f"fa4_forward_correctness_max_abs_diff: {printable} max_abs_diff_vs_fa3_bf16")
    print("STAGE0_RESULT_JSON=" + json.dumps(result, sort_keys=True, default=str, allow_nan=False))
    if args.fail_on_target_miss and not (math.isfinite(metric) and metric < 8e-3):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
