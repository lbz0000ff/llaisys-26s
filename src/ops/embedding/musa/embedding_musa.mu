#include "embedding_musa.cuh"

#include "../../common/musa/musa_common.cuh"

namespace llaisys::ops::musa {
namespace {
template <typename T>
__global__ void embeddingKernel(
    T *out,
    const int64_t *index,
    const T *weight,
    size_t numel,
    size_t embedding_dim) {
    const size_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index < numel) {
        const size_t row = output_index / embedding_dim;
        const size_t column = output_index % embedding_dim;
        out[output_index] = weight[static_cast<size_t>(index[row]) * embedding_dim + column];
    }
}

template <typename T>
void launchEmbedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    size_t index_count,
    size_t embedding_dim) {
    const size_t numel = index_count * embedding_dim;
    embeddingKernel<<<gridSize(numel), MUSA_BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const int64_t *>(index),
        reinterpret_cast<const T *>(weight),
        numel,
        embedding_dim);
    checkMusaLaunch("embedding kernel");
}
} // namespace

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t index_count, size_t embedding_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchEmbedding<float>(out, index, weight, index_count, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return launchEmbedding<__half>(out, index, weight, index_count, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return launchEmbedding<__mt_bfloat16>(out, index, weight, index_count, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::musa
