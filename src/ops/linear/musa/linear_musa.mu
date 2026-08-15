#include "linear_musa.cuh"

#include "../../common/musa/musa_common.cuh"

#include <mublas.h>

#include <stdexcept>
#include <string>

namespace llaisys::ops::musa {
namespace {
const char *mublasStatusName(mublasStatus status) {
    switch (status) {
    case MUBLAS_STATUS_SUCCESS:
        return "MUBLAS_STATUS_SUCCESS";
    case MUBLAS_STATUS_INVALID_HANDLE:
        return "MUBLAS_STATUS_INVALID_HANDLE";
    case MUBLAS_STATUS_NOT_IMPLEMENTED:
        return "MUBLAS_STATUS_NOT_IMPLEMENTED";
    case MUBLAS_STATUS_INVALID_POINTER:
        return "MUBLAS_STATUS_INVALID_POINTER";
    case MUBLAS_STATUS_INVALID_SIZE:
        return "MUBLAS_STATUS_INVALID_SIZE";
    case MUBLAS_STATUS_MEMORY_ERROR:
        return "MUBLAS_STATUS_MEMORY_ERROR";
    case MUBLAS_STATUS_INVALID_VALUE:
        return "MUBLAS_STATUS_INVALID_VALUE";
    case MUBLAS_STATUS_INTERNAL_ERROR:
        return "MUBLAS_STATUS_INTERNAL_ERROR";
    default:
        return "MUBLAS_STATUS_UNKNOWN";
    }
}

void checkMublas(mublasStatus status, const char *operation) {
    if (status != MUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + mublasStatusName(status));
    }
}

class MublasHandle {
private:
    mublasHandle_t _handle;

public:
    MublasHandle() {
        checkMublas(mublasCreate(&_handle), "mublasCreate");
        checkMublas(mublasSetMathMode(_handle, MUBLAS_MATH_MODE_DEFAULT), "mublasSetMathMode");
    }

    ~MublasHandle() {
        mublasDestroy(_handle);
    }

    mublasHandle_t get() const {
        return _handle;
    }
};

mublasHandle_t mublasHandle() {
    thread_local MublasHandle handle;
    return handle.get();
}

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t numel, size_t out_features) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        out[index] = fromFloat<T>(toFloat(out[index]) + toFloat(bias[index % out_features]));
    }
}

template <typename T>
__global__ void matmulKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = rows * out_features;
    if (index >= numel) {
        return;
    }
    const size_t row = index / out_features;
    const size_t column = index % out_features;
    float value = 0.0F;
    for (size_t inner = 0; inner < in_features; ++inner) {
        value += toFloat(in[row * in_features + inner])
                 * toFloat(weight[column * in_features + inner]);
    }
    out[index] = fromFloat<T>(value);
}

template <typename T>
void launchMatmul(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    const size_t numel = rows * out_features;
    matmulKernel<<<gridSize(numel), MUSA_BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        rows,
        in_features,
        out_features);
    checkMusaLaunch("linear matmul kernel");
}

constexpr size_t MATMUL_TILE = 16;

template <typename T>
__global__ void tiledMatmulKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    __shared__ T input_tile[MATMUL_TILE][MATMUL_TILE];
    __shared__ T weight_tile[MATMUL_TILE][MATMUL_TILE];

    const size_t row = blockIdx.y * MATMUL_TILE + threadIdx.y;
    const size_t column = blockIdx.x * MATMUL_TILE + threadIdx.x;
    float value = 0.0F;

    for (size_t offset = 0; offset < in_features; offset += MATMUL_TILE) {
        const size_t input_column = offset + threadIdx.x;
        const size_t weight_column = offset + threadIdx.x;
        const size_t weight_row = blockIdx.x * MATMUL_TILE + threadIdx.y;
        input_tile[threadIdx.y][threadIdx.x] =
            row < rows && input_column < in_features
                ? in[row * in_features + input_column]
                : fromFloat<T>(0.0F);
        weight_tile[threadIdx.y][threadIdx.x] =
            weight_row < out_features && weight_column < in_features
                ? weight[weight_row * in_features + weight_column]
                : fromFloat<T>(0.0F);
        __syncthreads();

        for (size_t inner = 0; inner < MATMUL_TILE; ++inner) {
            value += toFloat(input_tile[threadIdx.y][inner])
                     * toFloat(weight_tile[threadIdx.x][inner]);
        }
        __syncthreads();
    }

    if (row < rows && column < out_features) {
        out[row * out_features + column] = fromFloat<T>(value);
    }
}

template <typename T>
void launchTiledMatmul(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    const dim3 block(MATMUL_TILE, MATMUL_TILE);
    const dim3 grid(
        static_cast<unsigned int>((out_features + MATMUL_TILE - 1) / MATMUL_TILE),
        static_cast<unsigned int>((rows + MATMUL_TILE - 1) / MATMUL_TILE));
    tiledMatmulKernel<<<grid, block>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight), rows, in_features, out_features);
    checkMusaLaunch("linear tiled matmul kernel");
}

template <typename T>
void launchBias(std::byte *out, const std::byte *bias, size_t numel, size_t out_features) {
    addBiasKernel<<<gridSize(numel), MUSA_BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(bias),
        numel,
        out_features);
    checkMusaLaunch("linear bias kernel");
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype, size_t rows,
            size_t in_features, size_t out_features) {
    // muBLAS does not provide kernels for some very small GEMM shapes on
    // current MUSA releases. Use a simple kernel for those boundary cases.
    if (rows * out_features <= 4096) {
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launchMatmul<float>(out, in, weight, rows, in_features, out_features);
            break;
        case LLAISYS_DTYPE_F16:
            launchMatmul<__half>(out, in, weight, rows, in_features, out_features);
            break;
        case LLAISYS_DTYPE_BF16:
            launchMatmul<__mt_bfloat16>(out, in, weight, rows, in_features, out_features);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    } else {
        const float alpha = 1.0F;
        const float beta = 0.0F;
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            checkMublas(
                mublasSgemm(
                    mublasHandle(), MUBLAS_OP_T, MUBLAS_OP_N,
                    static_cast<int>(out_features), static_cast<int>(rows),
                    static_cast<int>(in_features), &alpha,
                    reinterpret_cast<const float *>(weight), static_cast<int>(in_features),
                    reinterpret_cast<const float *>(in), static_cast<int>(in_features), &beta,
                    reinterpret_cast<float *>(out), static_cast<int>(out_features)),
                "mublasSgemm");
            break;
        case LLAISYS_DTYPE_F16: {
            const __half half_alpha = __float2half(1.0F);
            const __half half_beta = __float2half(0.0F);
            checkMublas(
                mublasHgemm(
                    mublasHandle(), MUBLAS_OP_T, MUBLAS_OP_N,
                    static_cast<int>(out_features), static_cast<int>(rows),
                    static_cast<int>(in_features), &half_alpha,
                    reinterpret_cast<const __half *>(weight), static_cast<int>(in_features),
                    reinterpret_cast<const __half *>(in), static_cast<int>(in_features), &half_beta,
                    reinterpret_cast<__half *>(out), static_cast<int>(out_features)),
                "mublasHgemm");
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            launchTiledMatmul<__mt_bfloat16>(
                out, in, weight, rows, in_features, out_features);
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    }

    if (bias == nullptr) {
        return;
    }
    const size_t numel = rows * out_features;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchBias<float>(out, bias, numel, out_features);
    case LLAISYS_DTYPE_F16:
        return launchBias<__half>(out, bias, numel, out_features);
    case LLAISYS_DTYPE_BF16:
        return launchBias<__mt_bfloat16>(out, bias, numel, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::musa
