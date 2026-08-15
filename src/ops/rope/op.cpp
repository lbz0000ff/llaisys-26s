#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../common/cpu/cpu_kernels.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/rope_musa.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(in->ndim() == 3 && in->shape()[2] % 2 == 0, "rope input must be 3D with even head_dim");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64
                       && pos_ids->shape() == std::vector<size_t>{in->shape()[0]},
                   "rope position ids must be int64 with shape [seq_len]");
    CHECK_ARGUMENT(theta > 0.0F, "rope theta must be positive");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0],
                         in->shape()[1], in->shape()[2], theta);
    }
    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0],
                         in->shape()[1], in->shape()[2], theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0],
                            in->shape()[1], in->shape()[2], theta);
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0],
                            in->shape()[1], in->shape()[2], theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
