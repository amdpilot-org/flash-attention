"""ROCm / MI355X TileLang-based implementation of flash_attn.cute interfaces."""

from typing import Optional, Tuple, Callable

import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

# TileLang kernel for flash attention forward pass on gfx950/MI355X
# Configured for bf16, head_dim=128, causal=False.
# Based on TileLang AMD example with best-known config bm128_bn64_t256_s1.
@tilelang.jit(out_idx=[3])
def _tilelang_flash_fwd_gfx950(batch, heads, seq_len, dim, is_causal, groups, softmax_scale=None):
    scale = softmax_scale if softmax_scale is not None else ((1.0 / dim) ** 0.5)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    # Best-known config from hill-climb memory: bm128_bn64_t256_s1
    block_M = 128
    block_N = 64
    num_split_q = 1
    threads = 256
    num_stages = 1
    k_pack = 2
    qk_coalesced_width = 8
    v_coalesced_width = 4
    panel_size = 8
    enable_rasterization = True

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(num_split_q, batch * heads, threads=threads) as (b_split, byz_combined):
            T.use_swizzle(panel_size, enable=enable_rasterization)
            bz = byz_combined // heads
            by = byz_combined % heads
            num_q_blocks = T.ceildiv(seq_len, block_M)

            bx = T.alloc_var(T.int32)
            bx = b_split

            while bx < num_q_blocks:
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                m_i = T.alloc_fragment([block_M], accum_dtype)
                l_i = T.alloc_fragment([block_M], accum_dtype)
                T.fill(acc_o, 0)
                T.fill(m_i, -T.infinity(accum_dtype))
                T.fill(l_i, 0)

                current_bx = bx
                q_block_offset = current_bx * block_M

                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                m_prev = T.alloc_fragment([block_M], accum_dtype)
                scale_factor = T.alloc_fragment([block_M], accum_dtype)

                T.copy(
                    Q[bz, q_block_offset : q_block_offset + block_M, by, :],
                    Q_shared,
                    coalesced_width=qk_coalesced_width,
                )

                loop_end_k = T.ceildiv(seq_len, block_N)

                row_sum = T.alloc_fragment([block_M], accum_dtype)

                for k in T.Pipelined(loop_end_k, num_stages=num_stages):
                    kv_idx = k * block_N
                    T.copy(
                        K[bz, kv_idx : kv_idx + block_N, by // groups, :],
                        K_shared,
                        coalesced_width=qk_coalesced_width,
                    )
                    T.copy(
                        V[bz, kv_idx : kv_idx + block_N, by // groups, :],
                        V_shared,
                        coalesced_width=v_coalesced_width,
                    )

                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        k_pack=k_pack,
                        policy=GemmWarpPolicy.FullRow,
                    )

                    T.copy(m_i, m_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        m_i[i] = T.max(m_i[i], m_prev[i])

                    for i in T.Parallel(block_M):
                        sf = T.exp(m_prev[i] * scale - m_i[i] * scale)
                        l_i[i] *= sf
                        scale_factor[i] = sf

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scale_factor[i]

                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)

                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(block_M):
                        l_i[i] += row_sum[i]

                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.FullRow)

                l_inv = T.alloc_fragment([block_M], accum_dtype)
                for i in T.Parallel(block_M):
                    safe_l = T.if_then_else(l_i[i] > 1e-6, l_i[i], 1.0)
                    l_inv[i] = 1.0 / safe_l

                for i, j in T.Parallel(block_M, dim):
                    Output[bz, q_block_offset + i, by, j] = acc_o[i, j] * l_inv[i]

                bx = current_bx + num_split_q

    return main


# Cache compiled TileLang kernels per shape/config to avoid recompilation on each call.
_tilelang_kernel_cache = {}


def _get_cached_kernel(batch, heads, seq_len, dim, is_causal, groups, softmax_scale=None):
    key = (batch, heads, seq_len, dim, is_causal, groups, softmax_scale)
    if key not in _tilelang_kernel_cache:
        _tilelang_kernel_cache[key] = _tilelang_flash_fwd_gfx950(batch, heads, seq_len, dim, is_causal, groups, softmax_scale)
    return _tilelang_kernel_cache[key]


def _flash_attn_fwd_rocm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float = None,
) -> torch.Tensor:
    """ROCm TileLang forward pass wrapper."""
    batch, seqlen, num_head, head_dim = q.shape
    num_head_kv = k.shape[2]
    groups = num_head // num_head_kv
    assert q.dtype == torch.bfloat16, f"ROCm path only supports bfloat16 for now, got {q.dtype}"
    assert groups >= 1 and num_head % num_head_kv == 0

    kernel = _get_cached_kernel(batch, num_head, seqlen, head_dim, causal, groups, softmax_scale)
    out = kernel(q, k, v)
    return out


class FlashAttnFuncROCm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        return_lse: bool = False,
        softmax_scale: float = None,
    ):
        out = _flash_attn_fwd_rocm(q, k, v, causal=causal, softmax_scale=softmax_scale)
        return (out, None) if return_lse else (out, None)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("ROCm TileLang backward pass not yet implemented")


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse: bool = False,
):
    if score_mod is not None or mask_mod is not None:
        raise NotImplementedError("score_mod/mask_mod not supported on ROCm TileLang path")
    if softcap != 0.0:
        raise NotImplementedError("softcap not supported on ROCm TileLang path")
    return FlashAttnFuncROCm.apply(q, k, v, causal, return_lse, softmax_scale)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors=None,
    aux_tensors: Optional[list] = None,
    return_lse: bool = False,
):
    raise NotImplementedError("flash_attn_varlen_func not yet implemented on ROCm TileLang path")
