#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::musa {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel);
}
