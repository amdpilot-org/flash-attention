"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import torch

if torch.version.hip is not None:
    # ROCm / gfx950 fallback using torch SDPA
    from typing import Optional, Tuple, Callable

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
        """ROCm fallback for flash_attn_func using torch.nn.functional.scaled_dot_product_attention.

        Expects q, k, v in (batch, seqlen, nheads, headdim) layout.
        Internally transposes to (batch, nheads, seqlen, headdim) for SDPA.
        """
        if qv is not None:
            raise NotImplementedError("ROCm SDPA fallback does not support qv argument")
        if gather_kv_indices is not None:
            raise NotImplementedError("ROCm SDPA fallback does not support gather_kv_indices")
        if window_size != (None, None):
            raise NotImplementedError("ROCm SDPA fallback does not support window_size")
        if learnable_sink is not None:
            raise NotImplementedError("ROCm SDPA fallback does not support learnable_sink")
        if softcap != 0.0:
            raise NotImplementedError("ROCm SDPA fallback does not support softcap")
        if score_mod is not None:
            raise NotImplementedError("ROCm SDPA fallback does not support score_mod")
        if mask_mod is not None:
            raise NotImplementedError("ROCm SDPA fallback does not support mask_mod")
        if block_sparse_tensors is not None:
            raise NotImplementedError("ROCm SDPA fallback does not support block_sparse_tensors")
        if return_lse:
            raise NotImplementedError("ROCm SDPA fallback does not support return_lse")

        # flash_attn uses (batch, seqlen, nheads, headdim) layout
        # SDPA expects (batch, nheads, seqlen, headdim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        scale = softmax_scale if softmax_scale is not None else (q.size(-1) ** -0.5)

        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
        )
        return out.transpose(1, 2)

    def flash_attn_varlen_func(
        q,
        k,
        v,
        qv=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        min_seqlen_k=None,
        seqused_q=None,
        seqused_k=None,
        gather_kv_indices=None,
        page_table=None,
        softmax_scale=None,
        causal=False,
        window_size=(None, None),
        learnable_sink=None,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=None,
        block_sparse_tensors=None,
        aux_tensors=None,
        return_lse=False,
    ):
        raise NotImplementedError("ROCm SDPA fallback does not support flash_attn_varlen_func")
else:
    from .interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
