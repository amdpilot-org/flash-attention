"""ROCm-compatible dispatch layer for flash_attn.cute public API.

On gfx950 (and other ROCm arches) the CuTeDSL / CUTLASS stack is unavailable.
This module wires the flash_attn.cute.* API surface to the AMD AITER / Triton
fallback already present in flash_attn.flash_attn_interface so that the same
public entry points (flash_attn_func, flash_attn_varlen_func) return finite,
correct tensors.
"""

from typing import Optional, Tuple, Callable
import torch

from flash_attn.flash_attn_interface import (
    flash_attn_func as _fa_func,
    flash_attn_varlen_func as _fa_varlen_func,
)


def _map_window_size(window_size):
    """CUTE uses (None, None) for no window; FA-3 uses (-1, -1)."""
    if window_size is None:
        return (-1, -1)
    left, right = window_size
    if left is None:
        left = -1
    if right is None:
        right = -1
    return (left, right)


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
    """ROCm fallback for flash_attn.cute.flash_attn_func.

    Delegates to the AMD AITER / Triton implementation in
    flash_attn.flash_attn_interface.  Advanced CUTE-only features
    (score_mod, block sparsity, qv, gather_kv_indices, learnable_sink,
    num_splits, pack_gqa, return_lse) are unsupported and will raise
    NotImplementedError.
    """
    if qv is not None:
        raise NotImplementedError("qv (MLA absorbed Q) is not supported on ROCm fallback")
    if gather_kv_indices is not None:
        raise NotImplementedError("gather_kv_indices is not supported on ROCm fallback")
    if learnable_sink is not None:
        raise NotImplementedError("learnable_sink is not supported on ROCm fallback")
    if score_mod is not None or score_mod_bwd is not None or mask_mod is not None:
        raise NotImplementedError("score_mod / mask_mod are not supported on ROCm fallback")
    if aux_tensors is not None:
        raise NotImplementedError("aux_tensors are not supported on ROCm fallback")
    if block_sparse_tensors is not None or block_sparse_tensors_bwd is not None:
        raise NotImplementedError("block_sparse_tensors are not supported on ROCm fallback")
    if num_splits != 1:
        raise NotImplementedError("num_splits != 1 is not supported on ROCm fallback")
    if pack_gqa is not None:
        raise NotImplementedError("pack_gqa is not supported on ROCm fallback")
    if return_lse:
        raise NotImplementedError("return_lse is not supported on ROCm fallback")

    out = _fa_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=_map_window_size(window_size),
        softcap=softcap,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=False,
    )
    return out


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
    """ROCm fallback for flash_attn.cute.flash_attn_varlen_func.

    Delegates to flash_attn.flash_attn_varlen_func.
    """
    if qv is not None:
        raise NotImplementedError("qv is not supported on ROCm fallback")
    if gather_kv_indices is not None:
        raise NotImplementedError("gather_kv_indices is not supported on ROCm fallback")
    if learnable_sink is not None:
        raise NotImplementedError("learnable_sink is not supported on ROCm fallback")
    if score_mod is not None or score_mod_bwd is not None or mask_mod is not None:
        raise NotImplementedError("score_mod / mask_mod are not supported on ROCm fallback")
    if aux_tensors is not None:
        raise NotImplementedError("aux_tensors are not supported on ROCm fallback")
    if block_sparse_tensors is not None:
        raise NotImplementedError("block_sparse_tensors are not supported on ROCm fallback")
    if num_splits != 1:
        raise NotImplementedError("num_splits != 1 is not supported on ROCm fallback")
    if pack_gqa is not None:
        raise NotImplementedError("pack_gqa is not supported on ROCm fallback")
    if return_lse:
        raise NotImplementedError("return_lse is not supported on ROCm fallback")
    if page_table is not None:
        raise NotImplementedError("page_table is not supported on ROCm fallback")
    if seqused_q is not None or seqused_k is not None:
        raise NotImplementedError("seqused_q / seqused_k are not supported on ROCm fallback")
    if min_seqlen_k is not None:
        raise NotImplementedError("min_seqlen_k is not supported on ROCm fallback")

    return _fa_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=_map_window_size(window_size),
        softcap=softcap,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=False,
    )
