#ifndef _q4v2_recons_h
#define _q4v2_recons_h

#if USE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define cudaError_t hipError_t
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include <cstdint>

cudaError_t q4v2_recons_cuda
(
    const uint32_t* w,
    half* out,  // y
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int width,
    const int groupsize,
    const uint16_t* seq_g_idx
);

#endif

