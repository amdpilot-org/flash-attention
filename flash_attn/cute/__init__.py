"""Flash Attention CUTE (CUDA Template Engine) implementation."""

import torch

is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

if is_rocm:
    # On ROCm, the CUTE CUDA DSL kernels are unavailable. Dispatch to the existing
    # ROCm-capable flash_attn backend (FA-2 Triton/CK) so that flash_attn.cute APIs work.
    from flash_attn.flash_attn_interface import (
        flash_attn_func as _rocm_fa_func,
        flash_attn_varlen_func as _rocm_fa_varlen_func,
    )

    def flash_attn_func(q, k, v, *args, **kwargs):
        """ROCm dispatch wrapper for flash_attn.cute.flash_attn_func."""
        return _rocm_fa_func(q, k, v, *args, **kwargs)

    def flash_attn_varlen_func(q, k, v, *args, **kwargs):
        """ROCm dispatch wrapper for flash_attn.cute.flash_attn_varlen_func."""
        return _rocm_fa_varlen_func(q, k, v, *args, **kwargs)

    __all__ = ["flash_attn_func", "flash_attn_varlen_func"]
else:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("fa4")
    except PackageNotFoundError:
        __version__ = "0.0.0"

    from .interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )

    __all__ = [
        "flash_attn_func",
        "flash_attn_varlen_func",
    ]
