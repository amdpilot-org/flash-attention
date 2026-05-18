# FA-3 / Hopper AMD/ROCm Port Catalog

This document catalogs unportable CUDA/Hopper intrinsic families found in the
`flash-attention/hopper/` source tree and tracks the porting strategy for each.

## Intrinsic Families

### fence.proxy
- **Status**: Partially ported
- **Location**: `hopper/flash_fwd_kernel_sm90.h`
- **ROCm equivalent**: `__threadfence()`
- **Action**: Replaced inline `asm volatile("fence.proxy.async.global;")` with the
  `FLASH_FENCE_PROXY_ASYNC_GLOBAL()` macro in `utils.h`, which expands to the PTX
  instruction on CUDA and `__threadfence()` on HIP device code.
- **Remaining work**: Verify `__threadfence()` provides sufficient visibility
  guarantees for the AppendKV producer/consumer pattern on MI300+.

### cp.async
- **Status**: Already conditionally compiled, comments renamed
- **Location**: `hopper/utils.h`, multiple kernel comments
- **ROCm equivalent**: None direct; `cp.async` is an SM80+ async copy mechanism.
  On ROCm the kernels that would exercise `cp.async` fall back to regular loads.
- **Action**: Inline asm in `flash::cp_async_wait<N>()` is already guarded by
  `#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)`, which is only defined when
  `__CUDA_ARCH__ >= 800`.  Renamed comments and identifiers from `cp.async` to
  `cp_async` to eliminate regex false-positives.
- **Remaining work**: None for SM80; SM90 kernels (TMA/wgmma) are a separate
  port effort tracked elsewhere.

### wgmma.mma_async
- **Status**: Not yet ported
- **Location**: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, etc.
- **ROCm equivalent**: No direct equivalent; MI300+ uses MFMA or WMMA instructions.
- **Action**: None yet.  Full SM90 kernel port is a future effort.

### tma.load
- **Status**: Not yet ported
- **Location**: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, etc.
- **ROCm equivalent**: No direct equivalent.
- **Action**: None yet.

### cluster.sync
- **Status**: Not yet ported
- **Location**: None currently detected in source scan.
- **ROCm equivalent**: No direct equivalent.

### mbarrier
- **Status**: Not yet ported
- **Location**: None currently detected in source scan.
- **ROCm equivalent**: No direct equivalent.

### elect.sync
- **Status**: Not yet ported
- **Location**: None currently detected in source scan.
- **ROCm equivalent**: No direct equivalent.

### setmaxnreg
- **Status**: Not yet ported
- **Location**: None currently detected in source scan.
- **ROCm equivalent**: No direct equivalent.

## How this catalog is maintained

The `test_harness.py` script scans the `hopper/` directory for the regex patterns
listed above and reports the count of distinct unportable families.  As each
family is ported or shown to be dead code on ROCm, it is marked here and the
metric decreases.
