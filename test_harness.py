#!/usr/bin/env python3
"""Stage0 harness for issue-flash-attention-17-r3.

This task is specifically about native FlashAttention-3 (hopper/flash_attn_3)
being ported to gfx950/MI355X via AMD MFMA/global_load_lds. The unmodified ROCm
checkout can fall back to AITER/Triton, but that is not the requested FA-3 MFMA
kernel. Therefore this harness first requires the native `flash_attn_3._C`
extension/operator to exist. If it is missing, Stage0 records the faithful
baseline as UNIMPLEMENTED and does not report a fabricated numeric diff.

After an executor implements/builds native FA-3 for ROCm, this same script runs a
head_dim=128 bf16 forward-only no-causal single-batch correctness check vs
PyTorch eager SDPA and emits the required metric name and unit.
"""
import importlib
import json
import os
import subprocess
import sys
import traceback

METRIC = "fa3_forward_correctness_max_abs_diff"
UNIT = "max_abs_diff"


def env_report():
    out = {
        "flash_attention_amd_fa3": os.environ.get("FLASH_ATTENTION_AMD_FA3"),
        "flash_attention_triton_amd_enable": os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE"),
    }
    try:
        import torch
        out.update({
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "cuda_is_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        })
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            out["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            out["gcnArchName"] = getattr(props, "gcnArchName", None)
    except Exception as exc:
        out["torch_error"] = repr(exc)
    try:
        out["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        out["git_status_short"] = subprocess.check_output(["git", "status", "--short"], text=True).strip().splitlines()
    except Exception as exc:
        out["git_error"] = repr(exc)
    return out


def require_native_fa3(report):
    """Require the native FA-3 extension; do not accept ROCm Triton fallback."""
    try:
        mod = importlib.import_module("flash_attn_3._C")
        report["native_fa3_extension"] = getattr(mod, "__file__", "loaded")
    except Exception as exc:
        raise RuntimeError(
            "native flash_attn_3._C extension is unavailable on ROCm/gfx950; "
            "AITER/Triton fallback is not the FA-3 MFMA port requested by this issue"
        ) from exc


def native_fa3_correctness():
    import torch
    require_native_fa3({})
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda/HIP is not available; cannot run MI355X correctness check")
    # Import after native extension is proven present so hopper interface binds torch.ops.flash_attn_3.
    sys.path.insert(0, os.path.join(os.getcwd(), "hopper"))
    from flash_attn_interface import flash_attn_func

    torch.manual_seed(17)
    dev = "cuda"
    B, S, H, D = 1, 256, 1, 128
    q = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    ).transpose(1, 2)
    got = flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), deterministic=False)
    return float((got.float() - ref.float()).abs().max().item())


def main():
    report = env_report()
    try:
        require_native_fa3(report)
        value = native_fa3_correctness()
        report.update({"status": "native_fa3_metric_observed", "metric_value": value, "metric_unit": UNIT})
        print(f"{METRIC}: {value:.8g} {UNIT}")
        print("HARNESS_JSON=" + json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        report.update({
            "status": "baseline_native_fa3_unimplemented",
            "exception": repr(exc),
            "traceback_tail": traceback.format_exc(limit=8),
            "metric_value": None,
            "metric_unit": UNIT,
        })
        print(f"{METRIC}: UNIMPLEMENTED {UNIT}")
        print("HARNESS_JSON=" + json.dumps(report, sort_keys=True))
        return 42


if __name__ == "__main__":
    raise SystemExit(main())
