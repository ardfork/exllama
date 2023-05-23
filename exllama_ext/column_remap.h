#ifndef _column_remap_h
#define _column_remap_h

#if USE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define cudaError_t hipError_t
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include <cstdint>

cudaError_t column_remap_cuda
(
    const half* x,
    half* x_new,
    const int x_height,
    const int x_width,
    const uint32_t* x_map
);

#endif