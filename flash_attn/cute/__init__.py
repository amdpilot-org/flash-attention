"""Flash Attention CUTE (CUDA Template Engine) implementation."""

import os
import torch

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

# On AMD ROCm the CuTe DSL stack is unavailable; delegate to the ROCm dispatch layer.
_IS_ROCM = getattr(torch.version, "hip", None) is not None
_ROCM_ENABLE = _IS_ROCM and os.environ.get("FLASH_ATTENTION_CUTE_ROCM_ENABLE", "1") != "0"

if _ROCM_ENABLE:
    from .rocm_dispatch import flash_attn_func, flash_attn_varlen_func
    ROCM_DELEGATE = True
else:
    ROCM_DELEGATE = False

if not ROCM_DELEGATE:
    from .interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
