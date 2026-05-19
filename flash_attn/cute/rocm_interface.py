"""ROCm / MI355X implementation of flash_attn.cute interfaces.

Stage1 brings up aiter-backed forward and varlen-forward paths for gfx950
(MI355X) while the TileLang/CUTE-DSL kernel path is finished in a later
stage.
"""

import math
from typing import Optional, Tuple, Callable

import torch

from aiter.ops.mha import fmha_v3_fwd as _fa_backend_fmha_v3_fwd
from aiter.ops.mha import fmha_v3_varlen_fwd as _fa_backend_fmha_v3_varlen_fwd


def _flash_attn_fwd_rocm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """ROCm forward pass wrapper – delegates directly to aiter fmha_v3_fwd on gfx950."""
    batch, seqlen, num_head, head_dim = q.shape
    num_head_kv = k.shape[2]
    groups = num_head // num_head_kv
    assert q.dtype == torch.bfloat16, f"ROCm path only supports bfloat16 for now, got {q.dtype}"
    assert groups >= 1 and num_head % num_head_kv == 0

    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)
    out = _fa_backend_fmha_v3_fwd(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        return_softmax_lse=False,
        return_dropout_randval=False,
        how_v3_bf16_cvt=0,
    )
    # fmha_v3_fwd returns a list/tuple; unwrap the attention output tensor.
    return out[0] if isinstance(out, (list, tuple)) else out


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
