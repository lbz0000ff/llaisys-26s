#include "linear_nvidia.cuh"

#include "../../common/nvidia/nvidia_common.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {
const char *cublasStatusName(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

void checkCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cublasStatusName(status));
    }
}

class CublasHandle {
private:
    cublasHandle_t _handle;

public:
    CublasHandle() {
        checkCublas(cublasCreate(&_handle), "cublasCreate");
        checkCublas(cublasSetMathMode(_handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
    }

    ~CublasHandle() {
        cublasDestroy(_handle);
    }

    cublasHandle_t get() const {
        return _handle;
    }
};

cublasHandle_t cublasHandle() {
    thread_local CublasHandle handle;
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
void launchBias(std::byte *out, const std::byte *bias, size_t numel, size_t out_features) {
    addBiasKernel<<<gridSize(numel), CUDA_BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(bias),
        numel,
        out_features);
    checkCudaLaunch("linear bias kernel");
}

cudaDataType_t cudaDataType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype, size_t rows,
            size_t in_features, size_t out_features) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const cudaDataType_t data_type = cudaDataType(dtype);

    checkCublas(
        cublasGemmEx(
            cublasHandle(),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(out_features),
            static_cast<int>(rows),
            static_cast<int>(in_features),
            &alpha,
            weight,
            data_type,
            static_cast<int>(in_features),
            in,
            data_type,
            static_cast<int>(in_features),
            &beta,
            out,
            data_type,
            static_cast<int>(out_features),
            CUBLAS_COMPUTE_32F_PEDANTIC,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");

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
        return launchBias<__nv_bfloat16>(out, bias, numel, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
