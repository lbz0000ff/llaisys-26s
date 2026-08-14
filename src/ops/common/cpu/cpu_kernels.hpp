#pragma once

#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t numel);
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype,
               size_t index_count, size_t embedding_dim);
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t rows, size_t in_features, size_t out_features);
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t dtype,
              size_t rows, size_t cols, float eps);
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t dtype,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta);
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t query_len, size_t kv_len, size_t n_heads,
                    size_t n_kv_heads, size_t qk_dim, size_t value_dim, float scale);
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t numel);
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t dtype,
               const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides);
} // namespace llaisys::ops::cpu
