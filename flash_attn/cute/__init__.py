"""Flash Attention CUTE implementation."""

from importlib.metadata import PackageNotFoundError, version
import torch

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

if is_rocm:
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
