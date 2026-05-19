"""AMD MI355X TileLang implementation for the NVIDIA-only flash_attn.cute FA-4 API.

Dispatches to the TileLang-based FlashAttention forward kernel on ROCm / gfx950
when inputs are bf16 and head_dim=128, falling back to torch SDPA otherwise.
Returned tensor matches the layout/dtype convention expected by test harnesses.
"""
from .rocm_interface import flash_attn_func, flash_attn_varlen_func

