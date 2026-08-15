#include "rms_norm_nvidia.cuh"

#include "../../common/nvidia/nvidia_common.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void rmsNormKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t cols,
    float eps) {
    __shared__ float reduction[CUDA_BLOCK_SIZE];
    const size_t row = blockIdx.x;
    const size_t base = row * cols;

    float sum = 0.0F;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = toFloat(in[base + col]);
        sum += value * value;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(reduction[0] / static_cast<float>(cols) + eps);
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        out[base + col] = fromFloat<T>(
            toFloat(in[base + col]) * inv_rms * toFloat(weight[col]));
    }
}

template <typename T>
void launchRmsNorm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t rows,
    size_t cols,
    float eps) {
    rmsNormKernel<<<rows, CUDA_BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        cols,
        eps);
    checkCudaLaunch("rms_norm kernel");
}
} // namespace

void rmsNorm(std::byte *out, const std::byte *in, const std::byte *weight,
             llaisysDataType_t dtype, size_t rows, size_t cols, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<__half>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<__nv_bfloat16>(out, in, weight, rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
