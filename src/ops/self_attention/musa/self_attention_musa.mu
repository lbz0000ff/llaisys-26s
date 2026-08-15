#include "self_attention_musa.cuh"

#include "../../common/musa/musa_common.cuh"

#include <cmath>

namespace llaisys::ops::musa {
namespace {
template <typename T>
__global__ void selfAttentionKernel(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    size_t query_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {
    extern __shared__ float scores[];
    __shared__ float reduction[MUSA_BLOCK_SIZE];

    const size_t head = blockIdx.x;
    const size_t query_pos = blockIdx.y;
    const size_t heads_per_kv = n_heads / n_kv_heads;
    const size_t kv_head = head / heads_per_kv;
    const size_t prefix_len = kv_len - query_len;
    const size_t valid_keys = prefix_len + query_pos + 1;
    const size_t q_base = (query_pos * n_heads + head) * qk_dim;

    for (size_t key_pos = 0; key_pos < valid_keys; ++key_pos) {
        const size_t k_base = (key_pos * n_kv_heads + kv_head) * qk_dim;
        float partial = 0.0F;
        for (size_t dim = threadIdx.x; dim < qk_dim; dim += blockDim.x) {
            partial += toFloat(q[q_base + dim]) * toFloat(k[k_base + dim]);
        }
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            scores[key_pos] = reduction[0] * scale;
        }
        __syncthreads();
    }

    float local_max = -3.402823466e+38F;
    for (size_t key_pos = threadIdx.x; key_pos < valid_keys; key_pos += blockDim.x) {
        local_max = fmaxf(local_max, scores[key_pos]);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduction[0];

    float local_sum = 0.0F;
    for (size_t key_pos = threadIdx.x; key_pos < valid_keys; key_pos += blockDim.x) {
        const float probability = expf(scores[key_pos] - max_score);
        scores[key_pos] = probability;
        local_sum += probability;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denominator = reduction[0];
    for (size_t key_pos = threadIdx.x; key_pos < valid_keys; key_pos += blockDim.x) {
        scores[key_pos] /= denominator;
    }
    __syncthreads();

    const size_t out_base = (query_pos * n_heads + head) * value_dim;
    for (size_t dim = threadIdx.x; dim < value_dim; dim += blockDim.x) {
        float value = 0.0F;
        for (size_t key_pos = 0; key_pos < valid_keys; ++key_pos) {
            const size_t v_base = (key_pos * n_kv_heads + kv_head) * value_dim;
            value += scores[key_pos] * toFloat(v[v_base + dim]);
        }
        out[out_base + dim] = fromFloat<T>(value);
    }
}

template <typename T>
void launchSelfAttention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    size_t query_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {
    const dim3 grid(static_cast<unsigned int>(n_heads), static_cast<unsigned int>(query_len));
    selfAttentionKernel<<<grid, MUSA_BLOCK_SIZE, kv_len * sizeof(float)>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        query_len,
        kv_len,
        n_heads,
        n_kv_heads,
        qk_dim,
        value_dim,
        scale);
    checkMusaLaunch("self_attention kernel");
}
} // namespace

void selfAttention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t dtype,
    size_t query_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchSelfAttention<float>(out, q, k, v, query_len, kv_len, n_heads, n_kv_heads,
                                          qk_dim, value_dim, scale);
    case LLAISYS_DTYPE_F16:
        return launchSelfAttention<__half>(out, q, k, v, query_len, kv_len, n_heads, n_kv_heads,
                                           qk_dim, value_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return launchSelfAttention<__mt_bfloat16>(out, q, k, v, query_len, kv_len, n_heads, n_kv_heads,
                                                  qk_dim, value_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::musa
