#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/gemma4/geometry.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::gemma4 {

struct TensorRef {
    const TensorInfo* info = nullptr;
    const uint8_t* data = nullptr;
};

struct LayerTensors {
    TensorRef attention_norm;
    TensorRef q;
    TensorRef k;
    TensorRef v;
    TensorRef attention_output;
    TensorRef q_norm;
    TensorRef k_norm;
    TensorRef post_attention_norm;
    TensorRef ffn_norm;
    TensorRef shared_gate;
    TensorRef shared_up;
    TensorRef shared_down;
    TensorRef post_ffw_norm;
    TensorRef router;
    TensorRef router_scale;
    TensorRef pre_ffw_norm_2;
    TensorRef post_ffw_norm_1;
    TensorRef post_ffw_norm_2;
    TensorRef expert_gate_up;
    TensorRef expert_down;
    TensorRef expert_down_scale;
    TensorRef layer_output_scale;
};

struct Gemma4Model {
    std::shared_ptr<GgufModel> file;
    Gemma4Geometry geometry;
    TensorRef token_embedding;
    TensorRef output_norm;
    TensorRef rope_freqs;
    std::vector<LayerTensors> layers;
};

/// Open and validate the target text GGUF, then map its tensor directory without copying or reading weight payloads.
inline bool load_model(const std::string& path, Gemma4Model& out, std::string& err) {
    Gemma4Model model;
    try {
        model.file = std::make_shared<GgufModel>(gguf_split_paths(path));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    err = check_gemma4_layout(*model.file);
    if (!err.empty()) return false;
    if (!geometry_from_gguf(model.file->meta(), model.geometry, err)) return false;

    auto tensor = [&](const std::string& name, TensorRef& dst, bool required = true) {
        size_t shard = 0;
        const TensorInfo* info = model.file->find(name, &shard);
        if (!info) {
            if (required) err = "missing " + name;
            return !required;
        }
        dst = {info, model.file->shard(shard).tensor_data(*info)};
        return true;
    };
    if (!tensor("token_embd.weight", model.token_embedding) ||
        !tensor("output_norm.weight", model.output_norm) ||
        !tensor("rope_freqs.weight", model.rope_freqs))
        return false;

    model.layers.resize(static_cast<size_t>(model.geometry.n_layers));
    for (int64_t i = 0; i < model.geometry.n_layers; ++i) {
        LayerTensors& l = model.layers[static_cast<size_t>(i)];
        const std::string p = "blk." + std::to_string(i) + ".";
        const bool swa = model.geometry.layers[static_cast<size_t>(i)].sliding_window;
        if (!tensor(p + "attn_norm.weight", l.attention_norm) ||
            !tensor(p + "attn_q.weight", l.q) ||
            !tensor(p + "attn_k.weight", l.k) ||
            !tensor(p + "attn_v.weight", l.v, swa) ||
            !tensor(p + "attn_output.weight", l.attention_output) ||
            !tensor(p + "attn_q_norm.weight", l.q_norm) ||
            !tensor(p + "attn_k_norm.weight", l.k_norm) ||
            !tensor(p + "post_attention_norm.weight", l.post_attention_norm) ||
            !tensor(p + "ffn_norm.weight", l.ffn_norm) ||
            !tensor(p + "ffn_gate.weight", l.shared_gate) ||
            !tensor(p + "ffn_up.weight", l.shared_up) ||
            !tensor(p + "ffn_down.weight", l.shared_down) ||
            !tensor(p + "post_ffw_norm.weight", l.post_ffw_norm) ||
            !tensor(p + "ffn_gate_inp.weight", l.router) ||
            !tensor(p + "ffn_gate_inp.scale", l.router_scale) ||
            !tensor(p + "pre_ffw_norm_2.weight", l.pre_ffw_norm_2) ||
            !tensor(p + "post_ffw_norm_1.weight", l.post_ffw_norm_1) ||
            !tensor(p + "post_ffw_norm_2.weight", l.post_ffw_norm_2) ||
            !tensor(p + "ffn_gate_up_exps.weight", l.expert_gate_up) ||
            !tensor(p + "ffn_down_exps.weight", l.expert_down) ||
            !tensor(p + "ffn_down_exps.scale", l.expert_down_scale) ||
            !tensor(p + "layer_output_scale.weight", l.layer_output_scale))
            return false;
    }
    out = std::move(model);
    err.clear();
    return true;
}

}  // namespace strata::gemma4
