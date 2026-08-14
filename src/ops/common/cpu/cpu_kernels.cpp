#include "cpu_kernels.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {
template <typename T>
float as_float(T value) {
    return llaisys::utils::cast<float>(value);
}

template <typename T>
T from_float(float value) {
    return llaisys::utils::cast<T>(value);
}

template <typename T>
void argmax_impl(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    size_t best_idx = 0;
    float best_val = as_float(vals[0]);
    for (size_t i = 1; i < numel; ++i) {
        const float value = as_float(vals[i]);
        if (value > best_val) {
            best_val = value;
            best_idx = i;
        }
    }
    max_idx[0] = static_cast<int64_t>(best_idx);
    max_val[0] = vals[best_idx];
}

template <typename T>
void embedding_impl(T *out, const int64_t *index, const T *weight, size_t index_count, size_t embedding_dim) {
    for (size_t row = 0; row < index_count; ++row) {
        std::memcpy(out + row * embedding_dim, weight + index[row] * embedding_dim, embedding_dim * sizeof(T));
    }
}

template <typename T>
void convert_to_float(float *out, const T *in, size_t count);

template <typename Function>
void parallel_for(size_t count, Function function) {
    if (count == 0) {
        return;
    }
    const size_t available_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t thread_count = count < 4096 ? 1 : std::min(count, available_threads);
    const size_t chunk_size = (count + thread_count - 1) / thread_count;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (size_t thread = 0; thread < thread_count; ++thread) {
        const size_t begin = thread * chunk_size;
        const size_t end = std::min(count, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        workers.emplace_back([begin, end, &function]() { function(begin, end); });
    }
    for (auto &worker : workers) {
        worker.join();
    }
}

template <typename T>
void convert_to_float(float *out, const T *in, size_t count) {
    parallel_for(count, [out, in](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            out[i] = as_float(in[i]);
        }
    });
}

template <typename T>
void linear_impl(T *out, const T *in, const T *weight, const T *bias,
                 size_t rows, size_t in_features, size_t out_features) {
    std::vector<float> in_buffer;
    std::vector<float> weight_buffer;
    std::vector<float> bias_buffer;
    const float *in_values;
    const float *weight_values;

    if constexpr (std::is_same_v<T, float>) {
        in_values = in;
        weight_values = weight;
    } else {
        in_buffer.resize(rows * in_features);
        weight_buffer.resize(out_features * in_features);
        convert_to_float(in_buffer.data(), in, in_buffer.size());
        convert_to_float(weight_buffer.data(), weight, weight_buffer.size());
        in_values = in_buffer.data();
        weight_values = weight_buffer.data();
    }
    if (bias != nullptr) {
        bias_buffer.resize(out_features);
        convert_to_float(bias_buffer.data(), bias, out_features);
    }

    parallel_for(rows * out_features, [&](size_t begin, size_t end) {
        for (size_t output_index = begin; output_index < end; ++output_index) {
            const size_t row = output_index / out_features;
            const size_t column = output_index % out_features;
            const float *input_row = in_values + row * in_features;
            const float *weight_row = weight_values + column * in_features;
            float value = bias == nullptr ? 0.0F : bias_buffer[column];
            for (size_t inner = 0; inner < in_features; ++inner) {
                value += input_row[inner] * weight_row[inner];
            }
            out[output_index] = from_float<T>(value);
        }
    });
}

template <typename T>
void rms_norm_impl(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t row = 0; row < rows; ++row) {
        float sum_squares = 0.0F;
        for (size_t col = 0; col < cols; ++col) {
            const float value = as_float(in[row * cols + col]);
            sum_squares += value * value;
        }
        const float inv_rms = 1.0F / std::sqrt(sum_squares / static_cast<float>(cols) + eps);
        for (size_t col = 0; col < cols; ++col) {
            out[row * cols + col]
                = from_float<T>(as_float(in[row * cols + col]) * inv_rms * as_float(weight[col]));
        }
    }
}

template <typename T>
void rope_impl(T *out, const T *in, const int64_t *pos_ids,
               size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    const size_t half_dim = head_dim / 2;
    for (size_t token = 0; token < seq_len; ++token) {
        for (size_t head = 0; head < n_heads; ++head) {
            const size_t base = (token * n_heads + head) * head_dim;
            for (size_t i = 0; i < half_dim; ++i) {
                const float exponent = 2.0F * static_cast<float>(i) / static_cast<float>(head_dim);
                const float angle = static_cast<float>(pos_ids[token]) / std::pow(theta, exponent);
                const float sin_value = std::sin(angle);
                const float cos_value = std::cos(angle);
                const float a = as_float(in[base + i]);
                const float b = as_float(in[base + half_dim + i]);
                out[base + i] = from_float<T>(a * cos_value - b * sin_value);
                out[base + half_dim + i] = from_float<T>(b * cos_value + a * sin_value);
            }
        }
    }
}

template <typename T>
void self_attention_impl(T *out, const T *q, const T *k, const T *v,
                         size_t query_len, size_t kv_len, size_t n_heads, size_t n_kv_heads,
                         size_t qk_dim, size_t value_dim, float scale) {
    const size_t heads_per_kv = n_heads / n_kv_heads;
    const size_t prefix_len = kv_len - query_len;
    std::vector<float> scores(kv_len);

    for (size_t query_pos = 0; query_pos < query_len; ++query_pos) {
        const size_t valid_keys = std::min(kv_len, prefix_len + query_pos + 1);
        for (size_t head = 0; head < n_heads; ++head) {
            const size_t kv_head = head / heads_per_kv;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t key_pos = 0; key_pos < valid_keys; ++key_pos) {
                float score = 0.0F;
                const size_t q_base = (query_pos * n_heads + head) * qk_dim;
                const size_t k_base = (key_pos * n_kv_heads + kv_head) * qk_dim;
                for (size_t dim = 0; dim < qk_dim; ++dim) {
                    score += as_float(q[q_base + dim]) * as_float(k[k_base + dim]);
                }
                scores[key_pos] = score * scale;
                max_score = std::max(max_score, scores[key_pos]);
            }

            float denominator = 0.0F;
            for (size_t key_pos = 0; key_pos < valid_keys; ++key_pos) {
                scores[key_pos] = std::exp(scores[key_pos] - max_score);
                denominator += scores[key_pos];
            }
            for (size_t key_pos = 0; key_pos < valid_keys; ++key_pos) {
                scores[key_pos] /= denominator;
            }

            const size_t out_base = (query_pos * n_heads + head) * value_dim;
            for (size_t dim = 0; dim < value_dim; ++dim) {
                float value = 0.0F;
                for (size_t key_pos = 0; key_pos < valid_keys; ++key_pos) {
                    const size_t v_base = (key_pos * n_kv_heads + kv_head) * value_dim;
                    value += scores[key_pos] * as_float(v[v_base + dim]);
                }
                out[out_base + dim] = from_float<T>(value);
            }
        }
    }
}

template <typename T>
void swiglu_impl(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        const float gate_value = as_float(gate[i]);
        const T silu = from_float<T>(gate_value / (1.0F + std::exp(-gate_value)));
        out[i] = from_float<T>(as_float(up[i]) * as_float(silu));
    }
}

template <typename T>
void rearrange_impl(T *out, const T *in, const std::vector<size_t> &shape,
                    const std::vector<ptrdiff_t> &strides) {
    size_t numel = 1;
    for (const size_t dim : shape) {
        numel *= dim;
    }
    for (size_t linear_index = 0; linear_index < numel; ++linear_index) {
        size_t remaining = linear_index;
        ptrdiff_t source_index = 0;
        for (size_t i = shape.size(); i > 0; --i) {
            const size_t dim = i - 1;
            const size_t index = remaining % shape[dim];
            remaining /= shape[dim];
            source_index += static_cast<ptrdiff_t>(index) * strides[dim];
        }
        out[linear_index] = in[source_index];
    }
}
} // namespace

namespace llaisys::ops::cpu {
#define DISPATCH_FLOAT_TYPES(DTYPE, CALL)     \
    switch (DTYPE) {                          \
    case LLAISYS_DTYPE_F32:                   \
        return CALL(float);                   \
    case LLAISYS_DTYPE_F16:                   \
        return CALL(llaisys::fp16_t);         \
    case LLAISYS_DTYPE_BF16:                  \
        return CALL(llaisys::bf16_t);         \
    default:                                  \
        EXCEPTION_UNSUPPORTED_DATATYPE(DTYPE); \
    }

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t numel) {
#define CALL(T) argmax_impl(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<T *>(max_val), reinterpret_cast<const T *>(vals), numel)
    DISPATCH_FLOAT_TYPES(dtype, CALL)
#undef CALL
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype,
               size_t index_count, size_t embedding_dim) {
#define CALL(T) embedding_impl(reinterpret_cast<T *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const T *>(weight), index_count, embedding_dim)
    DISPATCH_FLOAT_TYPES(dtype, CALL)
#undef CALL
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t rows, size_t in_features, size_t out_features) {
#define CALL(T) linear_impl(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(bias), rows, in_features, out_features)
    DISPATCH_FLOAT_TYPES(dtype, CALL)
#undef CALL
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t dtype,
              size_t rows, size_t cols, float eps) {
#define CALL(T) rms_norm_impl(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), reinterpret_cast<const T *>(weight), rows, cols, eps)
    DISPATCH_FLOAT_TYPES(dtype, CALL)
#undef CALL
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t dtype,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
#define CALL(T) rope_impl(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta)
    DISPATCH_FLOAT_TYPES(dtype, CALL)
#undef CALL
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t query_len, size_t kv_len, size_t n_heads,
                    size_t n_kv_heads, size_t qk_dim, size_t value_dim, float scale) {
#define CALL(T) self_attention_impl(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(q), reinterpret_cast<const T *>(k), reinterpret_cast<const T *>(v), query_len, kv_len, n_heads, n_kv_heads, qk_dim, value_dim, scale)
    DISPATCH_FLOAT_TYPES(dtype, CALL)
#undef CALL
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t numel) {
#define CALL(T) swiglu_impl(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate), reinterpret_cast<const T *>(up), numel)
    DISPATCH_FLOAT_TYPES(dtype, CALL)
#undef CALL
}

void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t dtype,
               const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides) {
#define CALL(T) rearrange_impl(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), shape, strides)
    DISPATCH_FLOAT_TYPES(dtype, CALL)
#undef CALL
}

#undef DISPATCH_FLOAT_TYPES
} // namespace llaisys::ops::cpu
