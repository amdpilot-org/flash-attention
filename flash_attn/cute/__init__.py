"""AMD MI355X shim for the NVIDIA-only flash_attn.cute FA-4 API.

Implements flash_attn_func with the upstream [B, S, H, D] layout. On ROCm
this dispatches to a TileLang-based flash-attention kernel; on CUDA falls
back to the upstream CUTE implementation.
"""
import torch

if torch.version.hip is not None:
    from .rocm_interface import flash_attn_func, flash_attn_varlen_func
else:
    from .interface import flash_attn_func, flash_attn_varlen_func

__all__ = ["flash_attn_func", "flash_attn_varlen_func"]
