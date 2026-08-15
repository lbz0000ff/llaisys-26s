#include "rearrange_musa.cuh"

#include "../../common/musa/musa_common.cuh"

namespace llaisys::ops::musa {
namespace {
constexpr size_t MAX_TENSOR_DIMS = 8;

struct TensorIndexMeta {
    size_t ndim;
    size_t shape[MAX_TENSOR_DIMS];
    ptrdiff_t strides[MAX_TENSOR_DIMS];
};

template <typename T>
__global__ void rearrangeKernel(
    T *out,
    const T *in,
    size_t numel,
    TensorIndexMeta meta) {
    const size_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= numel) {
        return;
    }

    size_t remaining = output_index;
    ptrdiff_t source_index = 0;
    for (size_t i = meta.ndim; i > 0; --i) {
        const size_t dim = i - 1;
        const size_t index = remaining % meta.shape[dim];
        remaining /= meta.shape[dim];
        source_index += static_cast<ptrdiff_t>(index) * meta.strides[dim];
    }
    out[output_index] = in[source_index];
}

template <typename T>
void launchRearrange(
    std::byte *out,
    const std::byte *in,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &strides) {
    CHECK_ARGUMENT(shape.size() <= MAX_TENSOR_DIMS, "MUSA rearrange supports at most 8 dimensions");
    TensorIndexMeta meta{};
    meta.ndim = shape.size();
    size_t numel = 1;
    for (size_t dim = 0; dim < shape.size(); ++dim) {
        meta.shape[dim] = shape[dim];
        meta.strides[dim] = strides[dim];
        numel *= shape[dim];
    }

    rearrangeKernel<<<gridSize(numel), MUSA_BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        numel,
        meta);
    checkMusaLaunch("rearrange kernel");
}
} // namespace

void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t dtype,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &strides) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchRearrange<float>(out, in, shape, strides);
    case LLAISYS_DTYPE_F16:
        return launchRearrange<__half>(out, in, shape, strides);
    case LLAISYS_DTYPE_BF16:
        return launchRearrange<__mt_bfloat16>(out, in, shape, strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::musa
