#pragma once

#include "../device_resource.hpp"

namespace llaisys::device::musa {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource() = default;
};
} // namespace llaisys::device::musa
