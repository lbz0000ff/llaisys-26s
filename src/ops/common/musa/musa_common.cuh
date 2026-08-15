#pragma once

#include "../../../utils.hpp"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::ops::musa {
constexpr size_t MUSA_BLOCK_SIZE = 256;

inline size_t gridSize(size_t count) {
    return (count + MUSA_BLOCK_SIZE - 1) / MUSA_BLOCK_SIZE;
}

inline void checkMusaLaunch(const char *operation) {
    const musaError_t status = musaGetLastError();
    if (status != musaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + musaGetErrorString(status));
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
__device__ inline float toFloat(__mt_bfloat16 value) {
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
__device__ inline __mt_bfloat16 fromFloat(float value) {
    return __float2bfloat16_rn(value);
}
} // namespace llaisys::ops::musa
