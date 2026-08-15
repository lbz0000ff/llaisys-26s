#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../common/cpu/cpu_kernels.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/linear_musa.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
                   "linear input, weight, and output must be 2D");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1]
                       && out->shape()[0] == in->shape()[0]
                       && out->shape()[1] == weight->shape()[0],
                   "linear tensor shapes are incompatible");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: input, weight, and output must be contiguous.");
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        CHECK_ARGUMENT(bias->shape() == std::vector<size_t>{weight->shape()[0]}, "linear bias shape mismatch");
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(),
                           out->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
    }
    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(),
                           out->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(),
                              out->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::linear(out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(),
                              out->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
