#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void selfAttention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t dtype,
    size_t query_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale);
}
