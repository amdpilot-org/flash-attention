# FA-3 Hopper Port Catalog (AMD/HIP First-Pass)

## Overview

This document catalogs CUDA-only intrinsics and constructs found in the
FlashAttention-3 (FA-3) Hopper source set that are **not** automatically
translated or warned-on by `hipify-perl`. The goal is to provide a
replacement roadmap for AMD MI300X / MI355X (gfx942 / gfx950) targets.

## Curated Source Set

The first-pass hipify dry-run is bounded to the following files to avoid
stalls on headers with heavy template metaprogramming (e.g. `rotary.h`):

- `flash_fwd_combine.cu`
- `flash_prepare_scheduler.cu`
- `flash.h`
- `flash_bwd_kernel_sm90.h`
- `flash_bwd_launch_template.h`
- `flash_bwd_postprocess_kernel.h`
- `flash_bwd_preprocess_kernel.h`
- `flash_fwd_combine_kernel.h`
- `flash_fwd_combine_launch_template.h`
- `flash_fwd_kernel_sm90.h`
- `flash_fwd_launch_template.h`
- `heuristics.h`
- `mask.h`
- `pack_gqa.h`
- `paged_kv.h`
- `cuda_check.h`
- `utils.h`

## hipify-perl Baseline

Running `hipify-perl -examine -print-stats` over the curated set yields:

- **CONVERTED refs**: `cudaError_t` → `hipError_t`, `__syncthreads`, `__shfl_sync`, `tanh`
- **WARNINGS**: 0
- **Metric** (`hipify_unportable_intrinsic_count`): 0

> **Important**: A warning count of 0 does **not** mean the code is portable.
`hipify-perl` does not recognize PTX inline assembly, CUTLASS SM90/TMA APIs,
thread-block cluster primitives, or GEMM descriptor iterators. Those surfaces
are enumerated manually below.

## Unportable Intrinsic Inventory

| Category | Intrinsic / Construct | Occurrences | Source Files | Replacement Strategy |
|----------|----------------------|-------------|--------------|----------------------|
| **Warp shuffle** | `__shfl_sync` | 18 | `flash_bwd_kernel_sm90.h`, `flash_fwd_kernel_sm90.h`, `mask.h`, `pack_gqa.h`, `paged_kv.h`, `utils.h` | HIP provides `__shfl_sync` compat wrapper (mask arg kept). Older ROCm may need `__shfl`. |
| **Warp shuffle** | `__shfl_down_sync` | 4 | `flash_prepare_scheduler.cu` | Same family; use `__shfl_down` or verify `__shfl_down_sync`. |
| **Warp shuffle** | `__shfl_up_sync` | 1 | `utils.h` | Same family. |
| **Warp shuffle** | `__shfl_xor_sync` | 2 | `utils.h` | Same family; use `__shfl_xor`. |
| **PTX inline asm** | `cp.async.wait_group` | 1 | `utils.h` | Guarded by `CUTE_ARCH_CP_ASYNC_SM80_ENABLED` (undefined on AMD). Added explicit `#elif __HIP_PLATFORM_AMD__` no-op for documentation. |
| **PTX inline asm** | `fence.proxy.async.global` | 1 | `flash_fwd_kernel_sm90.h` | Unguarded Hopper PTX. **Already patched** to `#ifdef` out on AMD; rely on `NamedBarrier` for visibility. |
| **Cluster barrier** | `cutlass::arch::ClusterTransactionBarrier` | 2 | `flash_bwd_kernel_sm90.h`, `flash_bwd_postprocess_kernel.h` | Hopper-only cluster transaction barrier. AMD has no thread-block clusters. Replace with `NamedBarrier` or `__syncthreads()`. |
| **Cluster barrier** | `cutlass::arch::ClusterBarrier` | 2 | `flash_fwd_kernel_sm90.h` | Same as above. |
| **Cluster launch** | `cutlass::ClusterLauncher` / `ClusterLaunchParams` | 2 | `flash_bwd_launch_template.h`, `flash_fwd_launch_template.h` | Cluster launch is Hopper-only. Use standard dim3 grid/block launch on AMD. |
| **TMA** | `prefetch_tma_descriptors` | 4 | `flash_bwd_kernel_sm90.h`, `flash_fwd_kernel_sm90.h` | TMA (Tensor Memory Accelerator) is Hopper-only. Remove / no-op on AMD; fall back to global loads. |
| **TMA** | `load_page_table_TMA` | 1 | `paged_kv.h` | Same as above. |
| **TMA** | `get_indices_for_K_TMA` / `get_indices_for_V_TMA` | 2 | `paged_kv.h` | Same as above. |
| **TMA** | `Use_TMA_Q` / `Use_TMA_KV` / `Use_TMA_O` | ~20+ | `flash_fwd_kernel_sm90.h`, `flash_bwd_kernel_sm90.h` | Force these constexpr flags to `false` on AMD via `#ifdef __HIP_PLATFORM_AMD__` to disable TMA paths at compile time. |
| **Cluster sync** | `cute::cluster_arrive_relaxed` / `cute::cluster_wait` | 4 | `flash_bwd_kernel_sm90.h`, `flash_fwd_kernel_sm90.h` | Cluster sync primitives are Hopper-only. Replace with `__syncthreads()` or skip if cluster path is disabled. |
| **Arch tag** | `cutlass::arch::Sm90` | multiple | Launch templates, kernel classes | Sm90 = Hopper architecture tag. AMD uses gfx942/gfx950. Verify CUTLASS gfx950 support; if missing, fall back to SM80 mainloop or custom MFMA kernels. |
| **GMMA** | `mainloop_fwd_sm90_tma_gmma_ws.hpp` / `mainloop_bwd_sm90_tma_gmma_ws.hpp` | (included headers) | `flash_fwd_launch_template.h`, `flash_bwd_launch_template.h` | wgmma/GMMA is Hopper-specific. AMD MI355X uses CDNA3 matrix cores (MFMA). Requires either CUTLASS gfx950 mainloop or a custom MFMA rewrite. |
| **GMMA descriptor** | `cute::GMMA::DescriptorIterator` | 1 | `utils.h` | GMMA descriptors are Hopper-only. Use standard `cute::CopyAtom` on AMD. |

## Source Patches Applied vs `origin/main`

| File | Patch Summary |
|------|---------------|
| `hopper/cuda_check.h` | Added `#ifdef __HIP_PLATFORM_AMD__` guards to use `hipError_t`, `hipSuccess`, `hipGetErrorString`, `hipGetLastError` and emit "HIP error" strings. |
| `hopper/flash_fwd_kernel_sm90.h` | Guarded `asm volatile ("fence.proxy.async.global;");` with `#ifndef __HIP_PLATFORM_AMD__` — prevents PTX compilation error on AMD. |
| `hopper/utils.h` | Added `#elif defined(__HIP_PLATFORM_AMD__)` comment block in `cp_async_wait()` to explicitly document the AMD no-op path. |

## Next Moves (for subsequent stages)

1. **TMA disable at compile time**: Add `#ifdef __HIP_PLATFORM_AMD__` logic to force `Use_TMA_Q`, `Use_TMA_KV`, `Use_TMA_O` to `false` in the mainloop/epilogue classes. This is the largest block of unportable surface (~20+ references).
2. **Cluster launch fallback**: Replace `cutlass::ClusterLauncher` calls with standard `cuda/hipLaunchKernel` or `<<< >>>` syntax inside `#ifdef` guards.
3. **Barrier replacement**: Replace `ClusterTransactionBarrier` / `ClusterBarrier` fields with `cutlass::arch::NamedBarrier` or `__syncthreads()` when on AMD.
4. **GMMA mainloop assessment**: Determine whether CUTLASS in the container already has a gfx950/gfx942 GEMM mainloop. If not, the SM90 wgmma mainloop must be either disabled or rewritten.
5. **Test compilation**: Attempt `hipcc` compilation of the curated set to expose any additional unportable constructs that grep missed.
