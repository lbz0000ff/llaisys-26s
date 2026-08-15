#include "model.hpp"

#include "../../utils.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>
#include <numeric>
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

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(_device == LLAISYS_DEVICE_CPU, "Qwen2 prefill currently supports CPU only");
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2 token input must not be null");
    CHECK_ARGUMENT(ntoken > 0 && ntoken <= _meta.maxseq, "invalid Qwen2 input sequence length");
    CHECK_ARGUMENT(loadedWeightCount() == expectedWeightCount(), "Qwen2 weights are not fully loaded");

    tensor_t token_tensor = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    token_tensor->load(token_ids);
    tensor_t position_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    std::vector<int64_t> positions(ntoken);
    std::iota(positions.begin(), positions.end(), int64_t{0});
    position_ids->load(positions.data());

    tensor_t hidden = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
    ops::embedding(hidden, token_tensor, _weights.in_embed);

    const size_t q_size = _meta.nh * _meta.dh;
    const size_t kv_size = _meta.nkvh * _meta.dh;
    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(_meta.dh));

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        tensor_t attn_norm = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::rms_norm(attn_norm, hidden, _weights.attn_norm_w[layer], _meta.epsilon);

        tensor_t q_flat = Tensor::create({ntoken, q_size}, _meta.dtype, _device, _device_id);
        tensor_t k_flat = Tensor::create({ntoken, kv_size}, _meta.dtype, _device, _device_id);
        tensor_t v_flat = Tensor::create({ntoken, kv_size}, _meta.dtype, _device, _device_id);
        ops::linear(q_flat, attn_norm, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        ops::linear(k_flat, attn_norm, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        ops::linear(v_flat, attn_norm, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);

        tensor_t q = q_flat->view({ntoken, _meta.nh, _meta.dh});
        tensor_t k = k_flat->view({ntoken, _meta.nkvh, _meta.dh});
        tensor_t v = v_flat->view({ntoken, _meta.nkvh, _meta.dh});
        tensor_t rotated_q = Tensor::create(q->shape(), _meta.dtype, _device, _device_id);
        tensor_t rotated_k = Tensor::create(k->shape(), _meta.dtype, _device, _device_id);
        ops::rope(rotated_q, q, position_ids, _meta.theta);
        ops::rope(rotated_k, k, position_ids, _meta.theta);

        tensor_t attention = Tensor::create({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
        ops::self_attention(attention, rotated_q, rotated_k, v, attention_scale);
        tensor_t attention_flat = attention->view({ntoken, _meta.hs});
        tensor_t attention_out = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::linear(attention_out, attention_flat, _weights.attn_o_w[layer], nullptr);

        tensor_t post_attention = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::add(post_attention, hidden, attention_out);
        tensor_t mlp_norm = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::rms_norm(mlp_norm, post_attention, _weights.mlp_norm_w[layer], _meta.epsilon);

        tensor_t gate = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
        tensor_t up = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
        ops::linear(gate, mlp_norm, _weights.mlp_gate_w[layer], nullptr);
        ops::linear(up, mlp_norm, _weights.mlp_up_w[layer], nullptr);
        tensor_t activated = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
        ops::swiglu(activated, gate, up);
        tensor_t mlp_out = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::linear(mlp_out, activated, _weights.mlp_down_w[layer], nullptr);

        tensor_t next_hidden = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::add(next_hidden, post_attention, mlp_out);
        hidden = std::move(next_hidden);
    }

    tensor_t last_hidden = hidden->slice(0, ntoken - 1, ntoken);
    tensor_t normalized = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_id);
    ops::rms_norm(normalized, last_hidden, _weights.out_norm_w, _meta.epsilon);
    tensor_t logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device, _device_id);
    ops::linear(logits, normalized, _weights.out_embed, nullptr);
    tensor_t max_index = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    tensor_t max_value = Tensor::create({1}, _meta.dtype, _device, _device_id);
    ops::argmax(max_index, max_value, logits->view({_meta.voc}));
    return *reinterpret_cast<const int64_t *>(max_index->data());
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
