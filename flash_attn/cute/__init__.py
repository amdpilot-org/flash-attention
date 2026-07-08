"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import torch as _torch

if getattr(_torch.version, "hip", None) is not None:
    # ROCm (e.g. MI355X / gfx950): the upstream CUTLASS/CUTE-DSL path depends on
    # NVIDIA-only modules (cuda.bindings.driver, cutlass, quack). Use the
    # TileLang-backed port instead so flash_attn.cute is importable & runnable.
    from .interface_rocm import (
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
