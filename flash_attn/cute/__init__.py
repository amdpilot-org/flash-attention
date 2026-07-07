"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import torch

_IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None

if _IS_ROCM:
    # On ROCm (gfx950 / MI355X) the CUDA-specific CUTE-DSL backend
    # (cuda.bindings.driver, cutlass, quack) is unavailable, so the
    # SM80/SM90/SM100/SM120 kernels in interface.py cannot be imported.
    # FA-4 is algorithmically identical to FA-3 for the forward path, so
    # route flash_attn.cute.flash_attn_func to AITER's FA-3 port for
    # gfx950, preserving the cute API surface.
    from aiter import flash_attn_func as _aiter_flash_attn_func

    def flash_attn_func(
        q,
        k,
        v,
        qv=None,
        gather_kv_indices=None,
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
        aux_tensors=None,
        block_sparse_tensors=None,
        block_sparse_tensors_bwd=None,
        return_lse=False,
    ):
        if (
            qv is not None
            or gather_kv_indices is not None
            or learnable_sink is not None
            or softcap != 0.0
            or num_splits != 1
            or pack_gqa is not None
            or score_mod is not None
            or score_mod_bwd is not None
            or mask_mod is not None
            or aux_tensors is not None
            or block_sparse_tensors is not None
            or block_sparse_tensors_bwd is not None
        ):
            raise NotImplementedError(
                "flash_attn.cute.flash_attn_func on ROCm/gfx950 only supports the "
                "plain forward path (softmax_scale, causal, window_size, "
                "deterministic, return_lse)."
            )
        # cute uses window_size=(None, None) for an unbounded window; AITER
        # uses -1 to denote unbounded.
        if window_size is None:
            win = (-1, -1, 0)
        else:
            wl = -1 if window_size[0] is None else int(window_size[0])
            wr = -1 if window_size[1] is None else int(window_size[1])
            win = (wl, wr, 0)
        return _aiter_flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=win,
            deterministic=deterministic,
            return_lse=return_lse,
        )

    def flash_attn_varlen_func(*args, **kwargs):
        raise NotImplementedError(
            "flash_attn_varlen_func is not yet ported to ROCm/gfx950"
        )
else:
    from .interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
