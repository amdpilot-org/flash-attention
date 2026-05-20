"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version
import torch

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

if getattr(torch.version, "hip", None) is not None or not torch.cuda.is_available():
    # ROCm / HIP path: CuTe CUDA kernels are unavailable.
    # Use a pure-PyTorch shim backed by torch.nn.functional.scaled_dot_product_attention.
    from .rocm_shim import (
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
