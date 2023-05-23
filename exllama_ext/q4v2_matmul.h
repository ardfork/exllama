#ifndef _q4v2_matmul_h
#define _q4v2_matmul_h

#if USE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define cudaError_t hipError_t
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include <cstdint>
#include <cstdio>

cudaError_t q4v2_matmul_cuda
(
    const half* x,
    const uint32_t* w,
    half* out,  // y
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int dim,
    const int width,
    const int groupsize,
    const uint16_t* seq_g_idx,
    const uint32_t* x_map
);

#endif