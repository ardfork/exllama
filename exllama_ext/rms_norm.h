#ifndef _rms_norm_h
#define _rms_norm_h

#if USE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define cudaError_t hipError_t
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include <cstdint>

cudaError_t rms_norm_cuda
(
    half* x,
    const half* w,
    half* out,
    float* scratch,
    const float epsilon,
    const int rows,
    const int dim
);

#endif