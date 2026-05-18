"""Flash Attention CUTE (CUDA Template Engine) implementation."""

import torch
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

if getattr(torch.version, "hip", None) is not None:
    from .rocm_interface import (
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
