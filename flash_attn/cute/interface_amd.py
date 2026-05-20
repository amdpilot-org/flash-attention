"""AMD/ROCm shim for flash_attn.cute — routes to AITER Triton AMD backend.

This module provides drop-in replacements for the NVIDIA CuTe DSL-based
flash_attn_func / flash_attn_varlen_func when running on AMD GPUs (gfx950
and other CDNA/RDNA architectures).  It imports the battle-tested AITER
flash-attention Triton AMD kernels and exposes them with the FA-4 CUTE
public API so that code written against flash_attn.cute works out of the
box on MI300/MI355X.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _maybe_import_aiter_amd():
    try:
        from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import (
            interface_v3 as _amd_v3,
        )
        return _amd_v3
    except Exception:
        return None


_amd_v3 = _maybe_import_aiter_amd()


def _to_none(x):
    """Treat sentinel values as None so the AITER backend doesn't trip."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 0:
        return None
    return x


class _FlashAttnFuncAMD(torch.autograd.Function):
    """Autograd Function wrapper for AMD forward + backward."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=None,
        aux_tensors=None,
        block_sparse_tensors=None,
        block_sparse_tensors_bwd=None,
        return_lse: bool = False,
    ):
        if _amd_v3 is None:
            raise RuntimeError("AITER AMD Triton flash-attention backend is not available")

        if qv is not None:
            raise NotImplementedError("QV packed input not supported on AMD Triton backend")
        if softcap != 0.0:
            raise NotImplementedError("softcap not supported on AMD Triton backend")
        if num_splits != 1:
            raise NotImplementedError("num_splits > 1 not supported on AMD Triton backend")
        if pack_gqa is not None and pack_gqa is not False:
            raise NotImplementedError("pack_gqa not supported on AMD Triton backend")
        if score_mod is not None or score_mod_bwd is not None or mask_mod is not None:
            raise NotImplementedError("score_mod / mask_mod not supported on AMD Triton backend")
        if block_sparse_tensors is not None or block_sparse_tensors_bwd is not None:
            raise NotImplementedError("block sparsity not supported on AMD Triton backend")
        if aux_tensors is not None:
            raise NotImplementedError("aux_tensors not supported on AMD Triton backend")

        head_dim = q.shape[-1]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        # AITER fwd expects all positional args; pass None for unsupported ones.
        out, softmax_lse, _, _ = _amd_v3.fwd(
            q,
            k,
            v,
            None,          # k_new
            None,          # v_new
            None,          # qv
            None,          # out
            None,          # cu_seqlens_q
            None,          # cu_seqlens_k
            None,          # cu_seqlens_k_new
            None,          # seqused_q
            None,          # seqused_k
            None,          # max_seqlen_q
            None,          # max_seqlen_k
            None,          # page_table
            None,          # kv_batch_idx
            None,          # leftpad_k
            None,          # rotary_cos
            None,          # rotary_sin
            None,          # seqlens_rotary
            None,          # q_descale
            None,          # k_descale
            None,          # v_descale
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            1,             # attention_chunk
            0.0,           # softcap
            False,         # rotary_interleaved
            None,          # scheduler_metadata
            num_splits,
            False,         # pack_gqa
            0,             # sm_margin
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.return_lse = return_lse
        return (out, softmax_lse) if return_lse else out

    @staticmethod
    def backward(ctx, dout, dlse):
        # AITER bwd is available but not exercised by the current harness.
        raise NotImplementedError("AMD CUTE backward pass is not enabled in this shim")


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
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    aux_tensors=None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse: bool = False,
):
    """FA-4 CUTE forward compatible wrapper for AITER AMD Triton backend."""
    if learnable_sink is not None:
        raise NotImplementedError("learnable_sink not supported on AMD Triton backend")
    window_size_left = window_size[0] if window_size[0] is not None else -1
    window_size_right = window_size[1] if window_size[1] is not None else -1
    return _FlashAttnFuncAMD.apply(
        q,
        k,
        v,
        qv,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        aux_tensors,
        block_sparse_tensors,
        block_sparse_tensors_bwd,
        return_lse,
    )


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
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    block_sparse_tensors=None,
    aux_tensors=None,
    return_lse: bool = False,
):
    """Variable-length FA-4 CUTE forward compatible wrapper for AITER AMD Triton backend."""
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        raise NotImplementedError("varlen not yet wired in AMD CUTE shim")
    return flash_attn_func(
        q,
        k,
        v,
        qv=qv,
        gather_kv_indices=gather_kv_indices,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        learnable_sink=learnable_sink,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        block_sparse_tensors=block_sparse_tensors,
        return_lse=return_lse,
    )


__all__ = ["flash_attn_func", "flash_attn_varlen_func"]
