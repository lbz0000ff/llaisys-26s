#include "rope_nvidia.cuh"

#include "../../common/nvidia/nvidia_common.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void ropeKernel(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t n_heads,
    size_t head_dim,
    size_t pair_count,
    float theta) {
    const size_t pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_index >= pair_count) {
        return;
    }

    const size_t half_dim = head_dim / 2;
    const size_t pair = pair_index % half_dim;
    const size_t token_head = pair_index / half_dim;
    const size_t token = token_head / n_heads;
    const size_t base = token_head * head_dim;
    const float exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(head_dim);
    const float angle = static_cast<float>(pos_ids[token]) / powf(theta, exponent);
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);

    const float a = toFloat(in[base + pair]);
    const float b = toFloat(in[base + half_dim + pair]);
    out[base + pair] = fromFloat<T>(a * cosine - b * sine);
    out[base + half_dim + pair] = fromFloat<T>(b * cosine + a * sine);
}

template <typename T>
void launchRoPE(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta) {
    const size_t pair_count = seq_len * n_heads * (head_dim / 2);
    ropeKernel<<<gridSize(pair_count), CUDA_BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids),
        n_heads,
        head_dim,
        pair_count,
        theta);
    checkCudaLaunch("rope kernel");
}
} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads,
          size_t head_dim, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchRoPE<float>(out, in, pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return launchRoPE<__half>(out, in, pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return launchRoPE<__nv_bfloat16>(out, in, pos_ids, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
