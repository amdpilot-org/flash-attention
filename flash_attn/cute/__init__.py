"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import torch

if getattr(torch.version, 'hip', None) is not None:
    # ROCm / gfx950 path: the CuTeDSL / CUTLASS stack is CUDA-only, so we
    # dispatch to the AITER / Triton fallback already present in
    # flash_attn.flash_attn_interface.
    from .rocm_dispatch import (
        flash_attn_func,
        flash_attn_varlen_func,
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
