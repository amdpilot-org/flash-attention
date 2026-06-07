#!/usr/bin/env python3
"""Minimal torch.profiler wrapper around the FA-2 paged decode baseline."""
import os
import pathlib
import sys
import json

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

import torch
import torch.profiler as profiler

# Replicate harness parameters
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

# Import after path setup
from flash_attn import flash_attn_with_kvcache

torch.manual_seed(17)
nblocks_per_seq = (context + page_size - 1) // page_size
total_blocks = batch * nblocks_per_seq
q = torch.randn(batch, 1, nheads_q, head_dim, device=device, dtype=dtype)
k_cache = torch.randn(total_blocks, page_size, nheads_k, head_dim, device=device, dtype=dtype)
v_cache = torch.randn_like(k_cache)
cache_seqlens = torch.full((batch,), context, device=device, dtype=torch.int32)
block_table = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(batch, nblocks_per_seq)

# Warmup outside profile
for _ in range(warmup):
    flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens, block_table=block_table)
torch.cuda.synchronize()

# Profile
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
) as prof:
    for _ in range(iters):
        flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens, block_table=block_table)
    torch.cuda.synchronize()

trace_path = "/workspace/traces/fa2_baseline_32k_trace.json"
prof.export_chrome_trace(trace_path)
print(f"Trace saved to {trace_path}")

# Print kernel table sorted by CUDA time
print("\n=== CUDA Kernel Breakdown ===")
events = prof.key_averages()
events = sorted(events, key=lambda e: e.cuda_time_total, reverse=True)
total_cuda = sum(e.cuda_time_total for e in events)
for e in events[:20]:
    pct = e.cuda_time_total / total_cuda * 100 if total_cuda else 0
    print(f"{e.key:60s} {e.cuda_time_total/1e3:8.2f} ms  {pct:5.1f}%")
