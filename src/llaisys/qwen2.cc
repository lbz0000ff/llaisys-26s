#include "llaisys/models/qwen2.h"

#include "../models/qwen2/model.hpp"
#include "llaisys_tensor.hpp"

#include <memory>
#include <vector>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> model;
    LlaisysQwen2Weights weights{};
    std::vector<std::unique_ptr<LlaisysTensor>> handles;
    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;
};

namespace {

llaisysTensor_t makeHandle(LlaisysQwen2Model &model, const llaisys::tensor_t &tensor) {
    model.handles.push_back(std::make_unique<LlaisysTensor>(LlaisysTensor{tensor}));
    return model.handles.back().get();
}

void exposeWeights(LlaisysQwen2Model &model) {
    const auto &source = model.model->weights();
    const size_t nlayer = model.model->meta().nlayer;

    model.weights.in_embed = makeHandle(model, source.in_embed);
    model.weights.out_embed = makeHandle(model, source.out_embed);
    model.weights.out_norm_w = makeHandle(model, source.out_norm_w);

#define EXPOSE_LAYER_WEIGHTS(field)                                      \
    model.field.resize(nlayer);                                          \
    for (size_t layer = 0; layer < nlayer; ++layer) {                    \
        model.field[layer] = makeHandle(model, source.field[layer]);     \
    }                                                                    \
    model.weights.field = model.field.data()

    EXPOSE_LAYER_WEIGHTS(attn_norm_w);
    EXPOSE_LAYER_WEIGHTS(attn_q_w);
    EXPOSE_LAYER_WEIGHTS(attn_q_b);
    EXPOSE_LAYER_WEIGHTS(attn_k_w);
    EXPOSE_LAYER_WEIGHTS(attn_k_b);
    EXPOSE_LAYER_WEIGHTS(attn_v_w);
    EXPOSE_LAYER_WEIGHTS(attn_v_b);
    EXPOSE_LAYER_WEIGHTS(attn_o_w);
    EXPOSE_LAYER_WEIGHTS(mlp_norm_w);
    EXPOSE_LAYER_WEIGHTS(mlp_gate_w);
    EXPOSE_LAYER_WEIGHTS(mlp_up_w);
    EXPOSE_LAYER_WEIGHTS(mlp_down_w);

#undef EXPOSE_LAYER_WEIGHTS
}

} // namespace

__C {

LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    if (meta == nullptr || device_ids == nullptr || ndevice != 1) {
        return nullptr;
    }
    auto result = std::make_unique<LlaisysQwen2Model>();
    result->model = std::make_unique<llaisys::models::Qwen2Model>(*meta, device, device_ids[0]);
    exposeWeights(*result);
    return result.release();
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    return model == nullptr ? nullptr : &model->weights;
}

llaisysQwen2WeightLoadStatus_t llaisysQwen2ModelLoadWeight(
    LlaisysQwen2Model *model,
    const char *name,
    const void *data,
    llaisysDataType_t dtype,
    const size_t *shape,
    size_t ndim) {
    if (model == nullptr || name == nullptr) {
        return LLAISYS_QWEN2_WEIGHT_LOAD_INVALID_ARGUMENT;
    }
    return model->model->loadWeight(name, data, dtype, shape, ndim);
}

size_t llaisysQwen2ModelExpectedWeightCount(const LlaisysQwen2Model *model) {
    return model == nullptr ? 0 : model->model->expectedWeightCount();
}

size_t llaisysQwen2ModelLoadedWeightCount(const LlaisysQwen2Model *model) {
    return model == nullptr ? 0 : model->model->loadedWeightCount();
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (model == nullptr) {
        return -1;
    }
    return model->model->infer(token_ids, ntoken);
}

} // extern "C"
