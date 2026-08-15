#pragma once

#include "../../../include/llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaisys::models {

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);

    const LlaisysQwen2Meta &meta() const;
    const Qwen2Weights &weights() const;
    size_t expectedWeightCount() const;
    size_t loadedWeightCount() const;
    llaisysQwen2WeightLoadStatus_t loadWeight(
        const std::string &name,
        const void *data,
        llaisysDataType_t dtype,
        const size_t *shape,
        size_t ndim);

private:
    tensor_t createWeight(const std::string &name, const std::vector<size_t> &shape);

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    Qwen2Weights _weights;
    std::unordered_map<std::string, tensor_t> _weights_by_name;
    std::unordered_set<std::string> _loaded_weights;
};

} // namespace llaisys::models
