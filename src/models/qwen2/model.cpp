#include "model.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    CHECK_ARGUMENT(_meta.nlayer > 0, "Qwen2 must have at least one layer");
    CHECK_ARGUMENT(_meta.hs > 0 && _meta.nh > 0 && _meta.nkvh > 0, "invalid Qwen2 attention dimensions");
    CHECK_ARGUMENT(_meta.dh > 0 && _meta.di > 0 && _meta.voc > 0, "invalid Qwen2 model dimensions");
    CHECK_ARGUMENT(_meta.hs == _meta.nh * _meta.dh, "hidden size must equal attention heads times head size");
    CHECK_ARGUMENT(_meta.nh % _meta.nkvh == 0, "attention heads must be divisible by KV heads");

    _weights.attn_norm_w.resize(_meta.nlayer);
    _weights.attn_q_w.resize(_meta.nlayer);
    _weights.attn_q_b.resize(_meta.nlayer);
    _weights.attn_k_w.resize(_meta.nlayer);
    _weights.attn_k_b.resize(_meta.nlayer);
    _weights.attn_v_w.resize(_meta.nlayer);
    _weights.attn_v_b.resize(_meta.nlayer);
    _weights.attn_o_w.resize(_meta.nlayer);
    _weights.mlp_norm_w.resize(_meta.nlayer);
    _weights.mlp_gate_w.resize(_meta.nlayer);
    _weights.mlp_up_w.resize(_meta.nlayer);
    _weights.mlp_down_w.resize(_meta.nlayer);

    _weights.in_embed = createWeight("model.embed_tokens.weight", {_meta.voc, _meta.hs});
    _weights.out_norm_w = createWeight("model.norm.weight", {_meta.hs});
    _weights.out_embed = createWeight("lm_head.weight", {_meta.voc, _meta.hs});

    const size_t q_size = _meta.nh * _meta.dh;
    const size_t kv_size = _meta.nkvh * _meta.dh;
    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        _weights.attn_norm_w[layer] = createWeight(prefix + "input_layernorm.weight", {_meta.hs});
        _weights.attn_q_w[layer] = createWeight(prefix + "self_attn.q_proj.weight", {q_size, _meta.hs});
        _weights.attn_q_b[layer] = createWeight(prefix + "self_attn.q_proj.bias", {q_size});
        _weights.attn_k_w[layer] = createWeight(prefix + "self_attn.k_proj.weight", {kv_size, _meta.hs});
        _weights.attn_k_b[layer] = createWeight(prefix + "self_attn.k_proj.bias", {kv_size});
        _weights.attn_v_w[layer] = createWeight(prefix + "self_attn.v_proj.weight", {kv_size, _meta.hs});
        _weights.attn_v_b[layer] = createWeight(prefix + "self_attn.v_proj.bias", {kv_size});
        _weights.attn_o_w[layer] = createWeight(prefix + "self_attn.o_proj.weight", {_meta.hs, q_size});
        _weights.mlp_norm_w[layer] = createWeight(prefix + "post_attention_layernorm.weight", {_meta.hs});
        _weights.mlp_gate_w[layer] = createWeight(prefix + "mlp.gate_proj.weight", {_meta.di, _meta.hs});
        _weights.mlp_up_w[layer] = createWeight(prefix + "mlp.up_proj.weight", {_meta.di, _meta.hs});
        _weights.mlp_down_w[layer] = createWeight(prefix + "mlp.down_proj.weight", {_meta.hs, _meta.di});
    }
}

const LlaisysQwen2Meta &Qwen2Model::meta() const {
    return _meta;
}

const Qwen2Weights &Qwen2Model::weights() const {
    return _weights;
}

size_t Qwen2Model::expectedWeightCount() const {
    return _weights_by_name.size();
}

size_t Qwen2Model::loadedWeightCount() const {
    return _loaded_weights.size();
}

llaisysQwen2WeightLoadStatus_t Qwen2Model::loadWeight(
    const std::string &name,
    const void *data,
    llaisysDataType_t dtype,
    const size_t *shape,
    size_t ndim) {
    if (data == nullptr || (shape == nullptr && ndim != 0)) {
        return LLAISYS_QWEN2_WEIGHT_LOAD_INVALID_ARGUMENT;
    }
    const auto found = _weights_by_name.find(name);
    if (found == _weights_by_name.end()) {
        return LLAISYS_QWEN2_WEIGHT_LOAD_UNKNOWN_NAME;
    }
    if (_loaded_weights.count(name) != 0) {
        return LLAISYS_QWEN2_WEIGHT_LOAD_DUPLICATE;
    }

    const tensor_t &target = found->second;
    if (dtype != target->dtype()) {
        return LLAISYS_QWEN2_WEIGHT_LOAD_DTYPE_MISMATCH;
    }
    if (ndim != target->ndim()) {
        return LLAISYS_QWEN2_WEIGHT_LOAD_SHAPE_MISMATCH;
    }
    for (size_t dim = 0; dim < ndim; ++dim) {
        if (shape[dim] != target->shape()[dim]) {
            return LLAISYS_QWEN2_WEIGHT_LOAD_SHAPE_MISMATCH;
        }
    }

    target->load(data);
    _loaded_weights.insert(name);
    return LLAISYS_QWEN2_WEIGHT_LOAD_SUCCESS;
}

tensor_t Qwen2Model::createWeight(const std::string &name, const std::vector<size_t> &shape) {
    tensor_t tensor = Tensor::create(shape, _meta.dtype, _device, _device_id);
    const auto inserted = _weights_by_name.emplace(name, tensor);
    if (!inserted.second) {
        throw std::logic_error("duplicate Qwen2 weight name: " + name);
    }
    return tensor;
}

} // namespace llaisys::models
