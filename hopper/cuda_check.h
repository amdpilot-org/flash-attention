/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdlib.h>

#include <cutlass/cutlass.h>

#ifdef __HIP_PLATFORM_AMD__
#define CHECK_CUDA(call)                        \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                       \
        if (status_ != hipSuccess) {                                                                     \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(hipGetLastError())
#else
#define CHECK_CUDA(call)                        \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())
#endif

#define CHECK_CUTLASS(call)                                                                               \
    do {                                                                                                  \
        cutlass::Status status_ = (call);                                                                 \
        if (status_ != cutlass::Status::kSuccess) {                                                        \
            fprintf(stderr, "CUTLASS error (%s:%d): %s\n", __FILE__, __LINE__, cutlass::cutlassGetStatusString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)
