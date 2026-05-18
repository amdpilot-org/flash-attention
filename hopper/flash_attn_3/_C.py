# Python shim for flash_attn_3._C on ROCm / HIP.
# Upstream hopper code expects a C++ extension registering torch.ops.flash_attn_3.
# On gfx950 in this image, route those calls to the AITER Triton backend.
import os
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "1")
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import interface_v3
fwd = interface_v3.fwd
bwd = interface_v3.bwd
fwd_combine = interface_v3.fwd_combine
