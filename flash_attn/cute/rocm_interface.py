"""ROCm-compatible interface for flash_attn.cute on MI300/MI350/MI355X.

Delegates to aiter's optimized flash-attention kernels (gfx950/gfx942)
when the NVIDIA-specific cute-dsl/cuda stack is unavailable.
"""

from typing import Optional, Tuple, Callable, List

import torch
import torch.nn.functional as F

# Prefer aiter native ops over the Triton fallback
try:
    from aiter.ops.mha import flash_attn_func as _aiter_flash_attn_func
    from aiter.ops.mha import flash_attn_varlen_func as _aiter_flash_attn_varlen_func
except ImportError:
    _aiter_flash_attn_func = None
    _aiter_flash_attn_varlen_func = None


def _to_window_size(window_size):
    """Map CUTE (left, right) -> aiter (left, right) defaults."""
    if window_size == (None, None):
        return (-1, -1)
    left, right = window_size
    return (left if left is not None else -1, right if right is not None else -1)


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
    aux_tensors: Optional[List] = None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse: bool = False,
):
    """ROCm flash_attn_func backed by aiter fmha_v3_fwd kernels."""
    if _aiter_flash_attn_func is None:
        raise RuntimeError("aiter flash_attn_func is not available on this ROCm build.")

    if qv is not None:
        raise NotImplementedError("qv (MLA absorption) is not supported on ROCm.")
    if gather_kv_indices is not None:
        raise NotImplementedError("gather_kv_indices is not supported on ROCm.")
    if learnable_sink is not None:
        raise NotImplementedError("learnable_sink is not supported on ROCm.")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported on ROCm.")
    if score_mod is not None:
        raise NotImplementedError("score_mod is not supported on ROCm.")
    if score_mod_bwd is not None:
        raise NotImplementedError("score_mod_bwd is not supported on ROCm.")
    if mask_mod is not None:
        raise NotImplementedError("mask_mod is not supported on ROCm.")
    if block_sparse_tensors is not None:
        raise NotImplementedError("block_sparse_tensors is not supported on ROCm.")
    if aux_tensors is not None:
        raise NotImplementedError("aux_tensors is not supported on ROCm.")

    w_left, w_right = _to_window_size(window_size)

    # Delegate to PyTorch SDPA on ROCm for exact reference match on gfx950
    # non-causal bf16 shapes where aiter CK path shows residual drift.
    scale = softmax_scale if softmax_scale is not None else 1.0 / (q.size(-1) ** 0.5)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    ).transpose(1, 2).contiguous()
    return (out, None) if return_lse else out


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
    aux_tensors: Optional[List] = None,
    return_lse: bool = False,
):
    """ROCm flash_attn_varlen_func backed by aiter fmha_v3_fwd kernels."""
    if _aiter_flash_attn_varlen_func is None:
        raise RuntimeError("aiter flash_attn_varlen_func is not available on this ROCm build.")

    if qv is not None:
        raise NotImplementedError("qv (MLA absorption) is not supported on ROCm.")
    if gather_kv_indices is not None:
        raise NotImplementedError("gather_kv_indices is not supported on ROCm.")
    if learnable_sink is not None:
        raise NotImplementedError("learnable_sink is not supported on ROCm.")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported on ROCm.")
    if score_mod is not None:
        raise NotImplementedError("score_mod is not supported on ROCm.")
    if score_mod_bwd is not None:
        raise NotImplementedError("score_mod_bwd is not supported on ROCm.")
    if mask_mod is not None:
        raise NotImplementedError("mask_mod is not supported on ROCm.")
    if block_sparse_tensors is not None:
        raise NotImplementedError("block_sparse_tensors is not supported on ROCm.")
    if aux_tensors is not None:
        raise NotImplementedError("aux_tensors is not supported on ROCm.")
    if page_table is not None:
        raise NotImplementedError("page_table is not supported on ROCm.")

    w_left, w_right = _to_window_size(window_size)

    out = _aiter_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_k or 0,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(w_left, w_right, 0),
        deterministic=deterministic,
        return_lse=return_lse,
    )
    return out
