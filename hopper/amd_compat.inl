// Portable fence for async proxy operations across CUDA (SM90 TMA) and HIP
#if defined(__CUDA_ARCH__)
    #define FLASH_FENCE_PROXY_ASYNC_GLOBAL() asm volatile ("fence.proxy.async.global;")
#elif defined(__HIP_DEVICE_COMPILE__)
    #define FLASH_FENCE_PROXY_ASYNC_GLOBAL() __threadfence()
#else
    #define FLASH_FENCE_PROXY_ASYNC_GLOBAL()
#endif

// Portable cp.async.wait_group intrinsic
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    #define FLASH_CP_ASYNC_WAIT_GROUP(N) asm volatile("cp" ".async.wait_group %0;\n" :: "n"(N))
#else
    #define FLASH_CP_ASYNC_WAIT_GROUP(N)
#endif

// Portable named barrier arrival
#if defined(__CUDA_ARCH__)
    #define FLASH_NAMED_BARRIER_ARRIVE(barrier_id, num_threads) \
        asm volatile("bar.arrive %0, %1;" :: "r"(barrier_id), "r"(num_threads))
    #define FLASH_NAMED_BARRIER_SYNC(barrier_id, num_threads) \
        asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(num_threads))
#elif defined(__HIP_DEVICE_COMPILE__)
    // On AMD, named barriers do not exist; use regular __syncthreads() scoped by thread block
    #define FLASH_NAMED_BARRIER_ARRIVE(barrier_id, num_threads) __syncthreads()
    #define FLASH_NAMED_BARRIER_SYNC(barrier_id, num_threads) __syncthreads()
#else
    #define FLASH_NAMED_BARRIER_ARRIVE(barrier_id, num_threads)
    #define FLASH_NAMED_BARRIER_SYNC(barrier_id, num_threads)
#endif

// Portable cp.async.reduce.bulk intrinsic (SM90 TMA bulk reduce)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    #define FLASH_BULK_REDUCE_ADD_ASM(global_ptr, smem_ptr, bytes) \
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n" \
                     :: "l"(global_ptr), "r"(smem_ptr), "r"(bytes) : "memory")
#else
    #define FLASH_BULK_REDUCE_ADD_ASM(global_ptr, smem_ptr, bytes)
#endif
