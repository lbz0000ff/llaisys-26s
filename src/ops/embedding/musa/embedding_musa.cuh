#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::musa {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t index_count, size_t embedding_dim);
}
