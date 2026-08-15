#include "argmax_musa.cuh"

#include "../../common/musa/musa_common.cuh"

#include <cmath>

namespace llaisys::ops::musa {
namespace {
template <typename T>
__global__ void argmaxKernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    __shared__ float shared_values[MUSA_BLOCK_SIZE];
    __shared__ size_t shared_indices[MUSA_BLOCK_SIZE];

    float best_value = -3.402823466e+38F;
    size_t best_index = 0;
    for (size_t index = threadIdx.x; index < numel; index += blockDim.x) {
        const float value = toFloat(vals[index]);
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    shared_values[threadIdx.x] = best_value;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other_value = shared_values[threadIdx.x + stride];
            const size_t other_index = shared_indices[threadIdx.x + stride];
            if (other_value > shared_values[threadIdx.x]
                || (other_value == shared_values[threadIdx.x]
                    && other_index < shared_indices[threadIdx.x])) {
                shared_values[threadIdx.x] = other_value;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_idx[0] = static_cast<int64_t>(shared_indices[0]);
        max_val[0] = vals[shared_indices[0]];
    }
}

template <typename T>
void launchArgmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    size_t numel) {
    argmaxKernel<<<1, MUSA_BLOCK_SIZE>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);
    checkMusaLaunch("argmax kernel");
}
} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__mt_bfloat16>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::musa
