#!/usr/bin/env python3
"""Locked direct Stage0 harness for issue-flash-attention-22-r34.

The issue is about the Python/CUTE FA-4 forward API on MI355X/gfx950.  This
harness intentionally bypasses sglang server startup and validates the requested
`flash_attn.cute.flash_attn_func` API on a fixed bf16, non-causal,
head_dim=128 forward shape.  It always emits the required metric name.
"""
import json
import math
import traceback
import sys

METRIC = "fa4_forward_correctness_max_abs_diff"
UNIT = "max abs diff vs FA-3 port reference (bf16)"


def emit(value, status, detail):
    if isinstance(value, float) and math.isinf(value):
        printable = "inf"
    else:
        printable = f"{float(value):.10g}"
    print(f"{METRIC}: {printable}", flush=True)
    print(f"metric_unit: {UNIT}", flush=True)
    print(
        "stage0_status_json: "
        + json.dumps(
            {
                "metric_name": METRIC,
                "metric_unit": UNIT,
                "metric_direction": "lower",
                "measured_value": printable,
                "status": status,
                "detail": detail,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    try:
        import torch

        print(f"torch_version: {torch.__version__}", flush=True)
        print(f"cuda_is_available: {torch.cuda.is_available()}", flush=True)
        if not torch.cuda.is_available():
            emit(float("inf"), "no_gpu", "torch.cuda.is_available() is false")
            return 2
        print(f"device_name: {torch.cuda.get_device_name(0)}", flush=True)

        try:
            import aiter  # noqa: F401
            print("aiter_import: ok", flush=True)
        except Exception as e:
            emit(float("inf"), "aiter_import_failed", repr(e))
            return 1

        # Conservative single-shape problem: bf16, non-causal, head_dim=128.
        torch.manual_seed(22)
        B, S, H, D = 1, 256, 8, 128
        q = torch.randn((B, S, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((B, S, H, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((B, S, H, D), device="cuda", dtype=torch.bfloat16)
        scale = D ** -0.5
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        ).transpose(1, 2).contiguous()

        try:
            from flash_attn.cute import flash_attn_func
        except Exception as e:
            emit(float("inf"), "api_import_failed", repr(e))
            return 1

        try:
            out = flash_attn_func(q, k, v, causal=False)
        except TypeError:
            try:
                out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False, softmax_scale=scale)
            except Exception as e:
                emit(float("inf"), "api_call_failed", repr(e))
                return 1
        except Exception as e:
            emit(float("inf"), "api_call_failed", repr(e))
            return 1

        if isinstance(out, (tuple, list)):
            out = out[0]
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs().max().item()
        status = "pass" if diff < 8e-3 else "numeric_mismatch"
        emit(diff, status, "finite FA-4/CUTE output compared with PyTorch SDPA algorithmic reference")
        return 0 if diff < 8e-3 else 1
    except Exception:
        emit(float("inf"), "harness_exception", traceback.format_exc(limit=4))
        return 1


if __name__ == "__main__":
    sys.exit(main())
