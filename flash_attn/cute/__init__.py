"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

# On ROCm the CUTE implementation depends on the NVIDIA cuda.bindings driver
# package which is not available.  Provide a thin shim that routes to the
# already-working FA-3 / Triton backend on AMD GPUs.
def _is_rocm_missing_cuda(exc):
    msg = str(exc)
    return "No module named 'cuda'" in msg or "cuda.bindings" in msg or "No module named 'cuda.bindings'" in msg

try:
    from .interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError as _exc:
    if _is_rocm_missing_cuda(_exc):
        import torch

        # ROCm fallback: use FA-3 (flash_attn.flash_attn_func) which on this
        # image already falls back to the Triton AMD kernel when
        # flash_attn_2_cuda is absent.
        from flash_attn.flash_attn_interface import (
            flash_attn_func as _fa3_flash_attn_func,
            flash_attn_varlen_func as _fa3_flash_attn_varlen_func,
        )

        def flash_attn_func(q, k, v, causal=False, **kwargs):
            """ROCm shim: route CUTE-style flash_attn_func call to FA-3/Triton."""
            out = _fa3_flash_attn_func(q, k, v, causal=causal, **kwargs)
            # The harness expects either a bare tensor or a tuple; FA-3 returns
            # a tensor, so return it directly.
            return out

        def flash_attn_varlen_func(q, k, v, causal=False, **kwargs):
            """ROCm shim: route CUTE-style flash_attn_varlen_func call to FA-3/Triton."""
            return _fa3_flash_attn_varlen_func(q, k, v, causal=causal, **kwargs)
    else:
        raise

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
