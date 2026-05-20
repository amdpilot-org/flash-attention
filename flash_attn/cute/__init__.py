"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import torch

# On AMD/ROCm the NVIDIA CuTe DSL imports (cuda.bindings.driver, cutlass, …)
# are not available.  Route to the AITER Triton AMD backend instead.
_is_amd = False
if torch.cuda.is_available():
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        _is_amd = getattr(props, "gcnArchName", "").startswith("gfx")
    except Exception:
        pass

if _is_amd:
    from .interface_amd import (
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
