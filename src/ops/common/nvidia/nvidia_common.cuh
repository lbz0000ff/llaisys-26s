#pragma once

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
constexpr size_t CUDA_BLOCK_SIZE = 256;

inline size_t gridSize(size_t count) {
    return (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
}

inline void checkCudaLaunch(const char *operation) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
__device__ float toFloat(T value);

template <>
__device__ inline float toFloat(float value) {
    return value;
}

template <>
__device__ inline float toFloat(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float toFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T fromFloat(float value);

template <>
__device__ inline float fromFloat(float value) {
    return value;
}

template <>
__device__ inline __half fromFloat(float value) {
    return __float2half_rn(value);
}

template <>
__device__ inline __nv_bfloat16 fromFloat(float value) {
    return __float2bfloat16_rn(value);
}
} // namespace llaisys::ops::nvidia
