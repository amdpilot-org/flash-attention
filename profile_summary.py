#!/usr/bin/env python3
import os, pathlib, sys
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
from flash_attn import flash_attn_with_kvcache

batch, context, page_size = 8, 32768, 16
nheads_k, gqa = 1, 16
nheads_q = nheads_k * gqa
head_dim = 128
dtype = torch.bfloat16
device = "cuda"

nblocks_per_seq = (context + page_size - 1) // page_size
total_blocks = batch * nblocks_per_seq
q = torch.randn(batch, 1, nheads_q, head_dim, device=device, dtype=dtype)
k_cache = torch.randn(total_blocks, page_size, nheads_k, head_dim, device=device, dtype=dtype)
v_cache = torch.randn_like(k_cache)
cache_seqlens = torch.full((batch,), context, device=device, dtype=torch.int32)
block_table = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(batch, nblocks_per_seq)

# single warmup
flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens, block_table=block_table)
torch.cuda.synchronize()

with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens, block_table=block_table)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
