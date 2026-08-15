#include "swiglu_musa.cuh"

#include "../../common/musa/musa_common.cuh"

#include <cmath>

namespace llaisys::ops::musa {
namespace {
template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        const float gate_value = toFloat(gate[index]);
        const T silu = fromFloat<T>(gate_value / (1.0F + expf(-gate_value)));
        out[index] = fromFloat<T>(toFloat(up[index]) * toFloat(silu));
    }
}

template <typename T>
void launchSwiGLU(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    size_t numel) {
    swigluKernel<<<gridSize(numel), MUSA_BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
    checkMusaLaunch("swiglu kernel");
}
} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchSwiGLU<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return launchSwiGLU<__half>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return launchSwiGLU<__mt_bfloat16>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::musa
