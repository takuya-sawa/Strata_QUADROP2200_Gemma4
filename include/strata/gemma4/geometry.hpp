#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::gemma4 {

struct LayerGeometry {
    bool sliding_window = false;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t rope_dim = 0;
};

struct Gemma4Geometry {
    int64_t n_layers = 0;
    int64_t context_length = 0;
    int64_t n_embd = 0;
    int64_t n_vocab = 0;
    int64_t n_heads = 0;
    int64_t n_experts = 0;
    int64_t experts_per_token = 0;
    int64_t ffn_dim = 0;
    int64_t expert_ffn_dim = 0;
    int64_t sliding_window = 0;
    int64_t shared_kv_layers = 0;
    int64_t per_layer_input_dim = 0;
    int64_t key_dim = 0;
    int64_t value_dim = 0;
    int64_t key_dim_swa = 0;
    int64_t value_dim_swa = 0;
    int64_t rope_dim = 0;
    int64_t rope_dim_swa = 0;
    double rope_base = 0;
    double rope_base_swa = 0;
    double rms_epsilon = 0;
    double logit_softcap = 0;
    std::vector<LayerGeometry> layers;
};

inline bool geometry_from_gguf(const GgufFile& meta, Gemma4Geometry& out, std::string& err) {
    err = check_architecture_gemma4(meta);
    if (!err.empty()) return false;

    auto u32 = [&](const char* key, int64_t& dst) {
        const MetaValue* value = meta.get(key);
        if (!value || value->type != MetaType::U32) {
            err = std::string(key) + " is missing or is not U32";
            return false;
        }
        dst = static_cast<int64_t>(value->u);
        return true;
    };
    auto f32 = [&](const char* key, double& dst) {
        const MetaValue* value = meta.get(key);
        if (!value || value->type != MetaType::F32) {
            err = std::string(key) + " is missing or is not F32";
            return false;
        }
        dst = value->f;
        return true;
    };
    const std::string p = "gemma4.";
    if (!u32((p + "block_count").c_str(), out.n_layers) ||
        !u32((p + "context_length").c_str(), out.context_length) ||
        !u32((p + "embedding_length").c_str(), out.n_embd) ||
        !u32((p + "attention.head_count").c_str(), out.n_heads) ||
        !u32((p + "expert_count").c_str(), out.n_experts) ||
        !u32((p + "expert_used_count").c_str(), out.experts_per_token) ||
        !u32((p + "feed_forward_length").c_str(), out.ffn_dim) ||
        !u32((p + "expert_feed_forward_length").c_str(), out.expert_ffn_dim) ||
        !u32((p + "attention.sliding_window").c_str(), out.sliding_window) ||
        !u32((p + "attention.shared_kv_layers").c_str(), out.shared_kv_layers) ||
        !u32((p + "embedding_length_per_layer_input").c_str(), out.per_layer_input_dim) ||
        !u32((p + "attention.key_length").c_str(), out.key_dim) ||
        !u32((p + "attention.value_length").c_str(), out.value_dim) ||
        !u32((p + "attention.key_length_swa").c_str(), out.key_dim_swa) ||
        !u32((p + "attention.value_length_swa").c_str(), out.value_dim_swa) ||
        !u32((p + "rope.dimension_count").c_str(), out.rope_dim) ||
        !u32((p + "rope.dimension_count_swa").c_str(), out.rope_dim_swa) ||
        !f32((p + "rope.freq_base").c_str(), out.rope_base) ||
        !f32((p + "rope.freq_base_swa").c_str(), out.rope_base_swa) ||
        !f32((p + "attention.layer_norm_rms_epsilon").c_str(), out.rms_epsilon) ||
        !f32((p + "final_logit_softcapping").c_str(), out.logit_softcap))
        return false;

    const MetaValue* tokens = meta.get("tokenizer.ggml.tokens");
    const MetaValue* kv_heads = meta.get(p + "attention.head_count_kv");
    const MetaValue* sliding = meta.get(p + "attention.sliding_window_pattern");
    if (!tokens || tokens->type != MetaType::ARRAY || !kv_heads || kv_heads->type != MetaType::ARRAY ||
        kv_heads->elem != MetaType::I32 || !sliding || sliding->type != MetaType::ARRAY ||
        sliding->elem != MetaType::BOOL || kv_heads->items.size() != static_cast<size_t>(out.n_layers) ||
        sliding->items.size() != static_cast<size_t>(out.n_layers)) {
        err = "Gemma 4 tokenizer or per-layer attention metadata is incomplete";
        return false;
    }

    out.n_vocab = static_cast<int64_t>(tokens->count);
    out.layers.clear();
    out.layers.reserve(static_cast<size_t>(out.n_layers));
    for (int64_t i = 0; i < out.n_layers; ++i) {
        const bool swa = sliding->items[static_cast<size_t>(i)].u != 0;
        out.layers.push_back({swa,
                              static_cast<int64_t>(kv_heads->items[static_cast<size_t>(i)].u),
                              swa ? out.key_dim_swa : out.key_dim,
                              swa ? out.rope_dim_swa : out.rope_dim});
    }
    return true;
}

}  // namespace strata::gemma4
