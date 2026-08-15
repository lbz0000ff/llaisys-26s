#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../common/cpu/cpu_kernels.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/rms_norm_musa.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(in->ndim() == 2 && weight->shape() == std::vector<size_t>{in->shape()[1]},
                   "rms_norm expects a 2D input and matching 1D weight");
    CHECK_ARGUMENT(eps >= 0.0F, "rms_norm epsilon must be non-negative");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RmsNorm: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), in->shape()[0],
                             in->shape()[1], eps);
    }
    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), in->shape()[0],
                             in->shape()[1], eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rmsNorm(out->data(), in->data(), weight->data(), out->dtype(), in->shape()[0],
                               in->shape()[1], eps);
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::rmsNorm(out->data(), in->data(), weight->data(), out->dtype(), in->shape()[0],
                               in->shape()[1], eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
