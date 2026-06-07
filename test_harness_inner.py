#!/usr/bin/env python3
"""Stage0 FA-2 paged-KV decode baseline harness for flash-attention#74.

The harness intentionally benchmarks the existing FA-2 paged-KV implementation.
It does not implement FA-3. It verifies the AMD/ROCm Python environment, model
cache presence, import hygiene, and a 32k-context paged decode metric. The final
line emits an AMDPILOT_METRIC JSON block for pipeline extraction.
"""
import json
import math
import os
import pathlib
import re
import sys
import time

# Prior trials showed /sgl-workspace/mori/python and accidental cwd paths can
# shadow aiter/flash_attn namespaces. Keep imports deterministic.
BAD_PATH_FRAGMENTS = ("/sgl-workspace/mori/python",)
sys.path[:] = [p for p in sys.path if p and not any(bad in p for bad in BAD_PATH_FRAGMENTS)]
FA_REPO = pathlib.Path(os.environ.get("FLASH_ATTENTION_REPO", "/workspace/flash-attention"))
if FA_REPO.exists():
    sys.path = [str(FA_REPO)] + [p for p in sys.path if pathlib.Path(p or ".").resolve() != FA_REPO.resolve()]
AITER_SRC = pathlib.Path("/workspace/aiter_src")
if AITER_SRC.exists():
    sys.path.insert(0, str(AITER_SRC))

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", os.environ.get("ROCR_VISIBLE_DEVICES", "0"))

MODEL_CACHE = pathlib.Path(os.environ.get(
    "MODEL_CACHE",
    "/root/.cache/huggingface/hub/models--amd--Qwen3-235B-A22B-Instruct-2507-MXFP4",
))


def _find_callable():
    try:
        import torch  # noqa: F401
        from flash_attn import flash_attn_with_kvcache
        return flash_attn_with_kvcache
    except Exception as exc1:
        try:
            from flash_attn.flash_attn_interface import flash_attn_with_kvcache
            return flash_attn_with_kvcache
        except Exception as exc2:
            raise RuntimeError(f"could not import flash_attn_with_kvcache: {exc1!r}; {exc2!r}")


def _call_paged(fn, q, k_cache, v_cache, cache_seqlens, block_table, page_size):
    attempts = [
        dict(k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens, block_table=block_table),
        dict(k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens, block_table=block_table, causal=True),
        dict(k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens, block_table=block_table, page_table=block_table),
    ]
    last = None
    for kwargs in attempts:
        try:
            return fn(q, **kwargs)
        except TypeError as exc:
            last = exc
    raise last


def main():
    import torch

    info = {
        "python": sys.executable,
        "torch": getattr(torch, "__version__", "unknown"),
        "hip": getattr(torch.version, "hip", None),
        "model_cache_exists": MODEL_CACHE.exists(),
        "model_cache": str(MODEL_CACHE),
    }
    if not MODEL_CACHE.exists():
        raise FileNotFoundError(f"required model cache missing: {MODEL_CACHE}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda/ROCm is not available in container")

    fn = _find_callable()

    # Qwen3-235B-MXFP4 GQA grouping: 16 query heads per KV head. Keep head_dim
    # small enough for Stage0 runtime while preserving 32k paged decode shape.
    batch = int(os.environ.get("BATCH", "8"))
    context = int(os.environ.get("CONTEXT", "32768"))
    page_size = int(os.environ.get("PAGE_SIZE", "16"))
    nheads_k = int(os.environ.get("NHEADS_K", "1"))
    gqa = int(os.environ.get("GQA_GROUP", "16"))
    nheads_q = nheads_k * gqa
    head_dim = int(os.environ.get("HEAD_DIM", "128"))
    warmup = int(os.environ.get("WARMUP", "3"))
    iters = int(os.environ.get("ITERS", "10"))
    dtype = torch.bfloat16
    device = "cuda"

    torch.manual_seed(17)
    nblocks_per_seq = math.ceil(context / page_size)
    total_blocks = batch * nblocks_per_seq
    q = torch.randn(batch, 1, nheads_q, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(total_blocks, page_size, nheads_k, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.full((batch,), context, device=device, dtype=torch.int32)
    block_table = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(batch, nblocks_per_seq)

    # Preflight import/ABI sanity; previous trials saw module_aiter_core segfaults.
    torch.cuda.synchronize()
    out = _call_paged(fn, q, k_cache, v_cache, cache_seqlens, block_table, page_size)
    torch.cuda.synchronize()
    if out is None or not torch.isfinite(out).all().item():
        raise RuntimeError("FA-2 paged decode produced non-finite or empty output")

    for _ in range(warmup):
        _call_paged(fn, q, k_cache, v_cache, cache_seqlens, block_table, page_size)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _call_paged(fn, q, k_cache, v_cache, cache_seqlens, block_table, page_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_token = elapsed * 1000.0 / (iters * batch)

    metric = {
        "schema_version": "amdpilot.metric.v1",
        "metric_name": "fa2_decode_ms_per_token_32k",
        "metric_value": ms_per_token,
        "metric_direction": "lower",
        "context_length": context,
        "batch": batch,
        "page_size": page_size,
        "gqa_group": gqa,
        "num_query_heads": nheads_q,
        "num_kv_heads": nheads_k,
        "head_dim": head_dim,
        "warmup": warmup,
        "iters": iters,
        "info": info,
    }
    print("AMDPILOT_METRIC " + json.dumps(metric, sort_keys=True))


if __name__ == "__main__":
    main()
