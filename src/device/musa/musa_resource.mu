#include "musa_resource.cuh"

namespace llaisys::device::musa {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_MUSA, device_id) {}

} // namespace llaisys::device::musa
