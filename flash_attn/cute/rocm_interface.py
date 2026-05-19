"""ROCm / MI355X implementation of flash_attn.cute interfaces.

Stage1 brings up aiter-backed forward and varlen-forward paths for gfx950
(MI355X) while the TileLang/CUTE-DSL kernel path is finished in a later
stage.
"""

import math
import os
from typing import Optional, Tuple, Callable

import torch

from aiter.ops.mha import fmha_v3_fwd as _fa_backend_fmha_v3_fwd
from aiter.ops.mha import fmha_v3_varlen_fwd as _fa_backend_fmha_v3_varlen_fwd

# Conditional TileLang import – only used when FLASH_ATTENTION_ROCM_BACKEND=tilelang
_tilelang = None
_tilelang_flashattn_cache = {}

try:
    import tilelang
    import tilelang.language as T
    from tilelang.tileop.base import GemmWarpPolicy
    _tilelang = tilelang
except Exception:
    pass


def _build_tilelang_flashattn(batch, heads, seq_len, dim, is_causal):
    """Compile a TileLang FA-2 style forward kernel for bf16 on ROCm.

    Hardcoded tuning config targeting the Stage0 harness shape
    (batch=1, heads=1, seq_len=1024, dim=128, non-causal).
    Compilation is cached by (batch, heads, seq_len, dim, is_causal).
    """
    if _tilelang is None:
        raise RuntimeError("TileLang is not available")
    # Guard shapes that the hardcoded template can safely handle.
    if is_causal:
        raise RuntimeError("TileLang path does not yet support causal masking")
    if seq_len % 128 != 0:
        raise RuntimeError(f"TileLang path requires seq_len divisible by 128, got {seq_len}")
    if dim != 128:
        raise RuntimeError(f"TileLang path only supports head_dim=128 for now, got {dim}")

    block_M = 128
    block_N = 128
    threads = 256
    num_stages = 1
    enable_rasterization = True
    k_pack = 2
    panel_size = 8
    qk_coalesced_width = 8
    v_coalesced_width = 4
    num_split_q = 8
    scale = (1.0 / dim) ** 0.5
    dtype = T.bfloat16
    accum_dtype = T.float32
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, heads, dim]

    @_tilelang.jit(out_idx=[3])
    def tilelang_flashattn_bf16(batch, heads, seq_len, dim):
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
                            K[bz, kv_idx : kv_idx + block_N, by, :],
                            K_shared,
                            coalesced_width=qk_coalesced_width,
                        )
                        T.copy(
                            V[bz, kv_idx : kv_idx + block_N, by, :],
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

    return tilelang_flashattn_bf16(batch, heads, seq_len, dim)


def _flash_attn_fwd_rocm_tilelang(q, k, v, causal=False, softmax_scale=None):
    """TileLang forward wrapper for the Stage0 harness shape on gfx950."""
    batch, seqlen, num_head, head_dim = q.shape
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)
    cache_key = (batch, num_head, seqlen, head_dim, causal)
    if cache_key not in _tilelang_flashattn_cache:
        _tilelang_flashattn_cache[cache_key] = _build_tilelang_flashattn(
            batch, num_head, seqlen, head_dim, causal
        )
    kernel = _tilelang_flashattn_cache[cache_key]
    out = kernel(q, k, v)
    return out[0] if isinstance(out, (list, tuple)) else out


def _flash_attn_fwd_rocm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ROCm forward pass wrapper – delegates to aiter fmha_v3_fwd or TileLang on gfx950."""
    batch, seqlen, num_head, head_dim = q.shape
    num_head_kv = k.shape[2]
    groups = num_head // num_head_kv
    assert q.dtype == torch.bfloat16, f"ROCm path only supports bfloat16 for now, got {q.dtype}"
    assert groups >= 1 and num_head % num_head_kv == 0

    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)

    # Optional TileLang fast-path (experimental, env-gated).
    if os.environ.get("FLASH_ATTENTION_ROCM_BACKEND", "aiter").lower() == "tilelang":
        try:
            return _flash_attn_fwd_rocm_tilelang(q, k, v, causal=causal, softmax_scale=softmax_scale)
        except Exception:
            pass

    # Pre-allocate output to skip the allocation inside aiter and avoid an
    # extra host-device sync when the caller already has gradient disabled.
    if out is None:
        out = torch.empty(batch, seqlen, num_head, head_dim, dtype=q.dtype, device=q.device)
    aiter_out = _fa_backend_fmha_v3_fwd(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        return_softmax_lse=False,
        return_dropout_randval=False,
        how_v3_bf16_cvt=0,
        out=out,
    )
    # fmha_v3_fwd returns a list/tuple; unwrap the attention output tensor.
    return aiter_out[0] if isinstance(aiter_out, (list, tuple)) else aiter_out


class FlashAttnFuncROCm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        return_lse: bool = False,
        softmax_scale: Optional[float] = None,
    ):
        out = _flash_attn_fwd_rocm(q, k, v, causal=causal, softmax_scale=softmax_scale)
        return (out, None) if return_lse else (out, None)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("ROCm backward pass not yet implemented")


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
    # Fast-path: bypass autograd Function wrapper when backward is unnecessary.
    # This happens either when grad is globally disabled or when none of the
    # input tensors require gradients (the common inference case).
    if not return_lse and (not torch.is_grad_enabled() or (not q.requires_grad and not k.requires_grad and not v.requires_grad)):
        out = _flash_attn_fwd_rocm(q, k, v, causal=causal, softmax_scale=softmax_scale)
        return out
    return FlashAttnFuncROCm.apply(q, k, v, causal, return_lse, softmax_scale)


def _flash_attn_varlen_fwd_rocm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """ROCm varlen forward pass wrapper – delegates to aiter fmha_v3_varlen_fwd on gfx950."""
    head_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)
    # aiter varlen fwd requires cu_seqlens as int32 on device
    if cu_seqlens_q.dtype != torch.int32:
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    if cu_seqlens_k.dtype != torch.int32:
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
    # aiter also expects a min_seqlen_q argument; default to 0 when unknown.
    min_seqlen_q = 0
    out = _fa_backend_fmha_v3_varlen_fwd(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        min_seqlen_q,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        logits_soft_cap=0.0,
        zero_tensors=False,
        is_causal=causal,
        window_size_left=window_size_left if window_size_left is not None else -1,
        window_size_right=window_size_right if window_size_right is not None else -1,
        return_softmax_lse=False,
        return_dropout_randval=False,
        how_v3_bf16_cvt=0,
    )
    return out[0] if isinstance(out, (list, tuple)) else out


class FlashAttnVarlenFuncROCm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        return_lse: bool = False,
        softmax_scale: Optional[float] = None,
    ):
        out = _flash_attn_varlen_fwd_rocm(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            softmax_scale=softmax_scale,
        )
        return (out, None) if return_lse else (out, None)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("ROCm backward pass not yet implemented")


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
    if score_mod is not None or mask_mod is not None:
        raise NotImplementedError("score_mod/mask_mod not supported on ROCm TileLang path")
    if softcap != 0.0:
        raise NotImplementedError("softcap not supported on ROCm TileLang path")
    if qv is not None:
        raise NotImplementedError("qv not supported on ROCm TileLang path")
    if page_table is not None:
        raise NotImplementedError("page_table not supported on ROCm varlen path")
    if seqused_q is not None or seqused_k is not None:
        raise NotImplementedError("seqused_q/seqused_k not supported on ROCm varlen path")
    if gather_kv_indices is not None:
        raise NotImplementedError("gather_kv_indices not supported on ROCm varlen path")
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("cu_seqlens_q and cu_seqlens_k are required for varlen")
    if max_seqlen_q is None or max_seqlen_k is None:
        raise ValueError("max_seqlen_q and max_seqlen_k are required for varlen")
    window_size_left = window_size[0] if window_size is not None and window_size[0] is not None else -1
    window_size_right = window_size[1] if window_size is not None and window_size[1] is not None else -1
    if not return_lse and (not torch.is_grad_enabled() or (not q.requires_grad and not k.requires_grad and not v.requires_grad)):
        return _flash_attn_varlen_fwd_rocm(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            softmax_scale=softmax_scale,
        )
    return FlashAttnVarlenFuncROCm.apply(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        causal, window_size_left, window_size_right,
        return_lse, softmax_scale,
    )
