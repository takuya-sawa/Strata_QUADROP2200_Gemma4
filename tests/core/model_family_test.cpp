// tests/core/model_family_test.cpp - the model families (NEW-MODELS-PLAN.md T0) on synthetic GGUFs: no model, no GPU.
//
//   * the architecture guard accepts qwen4exp and qwen35moe and refuses anything else; the engine's guard refuses
//     qwen35moe with "not ... yet" (no engine path for Qwen3.6 before T1);
//   * geometry_from_gguf reads Qwen3.6-35B-A3B's real metadata values (hidden 2048, 40 + 1 MTP layers, 32 GDN V
//     heads, conv 8192, no hyper-connections, no indexer) and Flash-Next's as exactly the compiled defaults;
//   * the layout tables, on tiny files of either family: a complete file passes, a wrong shape, a missing tensor
//     and (qwen35moe) a tensor the table does not name are refused, naming the tensor.
//
// `model_family_test --real SHARD`: the same view of a real model (headers only): family, geometry, the kernels'
// verdict and the layout check over every shard.
#include "gguf_fixture.hpp"

#include "strata/core/family_layout.hpp"
#include "strata/gemma4/geometry.hpp"
#include "strata/gemma4/model.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::current_path() /
               ("strata-model-family-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
using fixture::Tensor;
const uint32_t F32 = 0;

std::vector<Kv> gemma_metadata() {
    std::vector<Kv> kv = {
        fixture::str("general.architecture", "gemma4"),
        fixture::u32("general.file_type", 2),
        fixture::u32("general.quantization_version", 2),
        fixture::str("tokenizer.ggml.model", "gemma4"),
        fixture::strs("tokenizer.ggml.tokens", std::vector<std::string>(262144)),
        fixture::strs("tokenizer.ggml.merges", std::vector<std::string>(514906)),
        fixture::u32("tokenizer.ggml.bos_token_id", 2),
        fixture::u32("tokenizer.ggml.eos_token_id", 1),
        fixture::u32("tokenizer.ggml.unknown_token_id", 3),
        fixture::u32("tokenizer.ggml.padding_token_id", 0),
        fixture::u32("tokenizer.ggml.mask_token_id", 4),
    };
    Kv scores = fixture::u32s("tokenizer.ggml.scores", std::vector<uint64_t>(262144));
    scores.elem = 6;
    kv.push_back(std::move(scores));
    Kv types = fixture::u32s("tokenizer.ggml.token_type", std::vector<uint64_t>(262144));
    types.elem = 5;
    kv.push_back(std::move(types));
    Kv prefix = fixture::make_kv("tokenizer.ggml.add_space_prefix", 7);
    Kv bos = fixture::make_kv("tokenizer.ggml.add_bos_token", 7);
    bos.u = 1;
    kv.push_back(std::move(prefix));
    kv.push_back(std::move(bos));
    for (const auto& k : std::vector<std::pair<std::string, uint64_t>>{
             {"block_count", 30}, {"context_length", 262144}, {"embedding_length", 2816},
             {"feed_forward_length", 2112}, {"attention.head_count", 16}, {"expert_count", 128},
             {"expert_used_count", 8}, {"expert_feed_forward_length", 704}, {"attention.key_length", 512},
             {"attention.value_length", 512}, {"attention.key_length_swa", 256}, {"attention.value_length_swa", 256},
             {"attention.sliding_window", 1024}, {"attention.shared_kv_layers", 0},
             {"embedding_length_per_layer_input", 0}, {"rope.dimension_count", 512}, {"rope.dimension_count_swa", 256}})
        kv.push_back(fixture::u32("gemma4." + k.first, k.second));
    kv.push_back(fixture::f32("gemma4.rope.freq_base", 1000000.f));
    kv.push_back(fixture::f32("gemma4.rope.freq_base_swa", 10000.f));
    kv.push_back(fixture::f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6f));
    kv.push_back(fixture::f32("gemma4.final_logit_softcapping", 30.f));
    Kv heads = fixture::u32s("gemma4.attention.head_count_kv", {});
    Kv swa = fixture::u32s("gemma4.attention.sliding_window_pattern", {});
    heads.elem = 5;
    swa.elem = 7;
    for (int i = 0; i < 30; ++i) {
        heads.arr.push_back(i % 6 == 5 ? 2 : 8);
        swa.arr.push_back(i % 6 == 5 ? 0 : 1);
    }
    kv.push_back(std::move(heads));
    kv.push_back(std::move(swa));
    return kv;
}

std::vector<Tensor> gemma_tensors() {
    std::vector<Tensor> ts;
    auto add = [&](std::string name, std::vector<uint64_t> shape, uint32_t type) {
        ts.push_back({std::move(name), std::move(shape), type, 0});
    };
    add("token_embd.weight", {2816, 262144}, 14);
    add("output_norm.weight", {2816}, 0);
    add("rope_freqs.weight", {256}, 0);
    for (int i = 0; i < 30; ++i) {
        const bool full = i % 6 == 5;
        const uint64_t h = full ? 512 : 256, nkv = full ? 2 : 8;
        const std::string p = "blk." + std::to_string(i) + ".";
        add(p + "attn_q.weight", {2816, h * 16}, 2);
        add(p + "attn_k.weight", {2816, h * nkv}, 2);
        if (!full) add(p + "attn_v.weight", {2816, h * nkv}, 2);
        add(p + "attn_output.weight", {h * 16, 2816}, 2);
        add(p + "attn_q_norm.weight", {h}, 0);
        add(p + "attn_k_norm.weight", {h}, 0);
        for (const char* n : {"attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight",
                              "post_ffw_norm.weight", "post_ffw_norm_1.weight", "post_ffw_norm_2.weight",
                              "pre_ffw_norm_2.weight", "ffn_gate_inp.scale"}) add(p + n, {2816}, 0);
        add(p + "layer_output_scale.weight", {1}, 0);
        add(p + "ffn_gate_inp.weight", {2816, 128}, 0);
        add(p + "ffn_down_exps.scale", {128}, 0);
        add(p + "ffn_gate.weight", {2816, 2112}, 2);
        add(p + "ffn_up.weight", {2816, 2112}, 2);
        add(p + "ffn_down.weight", {2112, 2816}, 2);
        add(p + "ffn_gate_up_exps.weight", {2816, 1408, 128}, 2);
        add(p + "ffn_down_exps.weight", {704, 2816, 128}, 2);
    }
    return ts;
}

// Sparse payload extents let a full-sized directory be checked without writing or reading model weights.
void write_gemma_fixture(const fs::path& path, const std::vector<Kv>& kv, const std::vector<Tensor>& ts,
                         bool overlap = false, bool truncate = false) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    auto raw = [&](const void* p, size_t n) { f.write(static_cast<const char*>(p), (std::streamsize) n); };
    auto u32 = [&](uint32_t x) { raw(&x, sizeof(x)); };
    auto u64 = [&](uint64_t x) { raw(&x, sizeof(x)); };
    auto str = [&](const std::string& x) { u64(x.size()); raw(x.data(), x.size()); };
    u32(0x46554747u); u32(3); u64(ts.size()); u64(kv.size());
    for (const Kv& k : kv) {
        str(k.key); u32(k.type);
        if (k.type == 8) str(k.s);
        else if (k.type == 4 || k.type == 5) u32((uint32_t) k.u);
        else if (k.type == 6) raw(&k.f, sizeof(k.f));
        else if (k.type == 7) { uint8_t b = (uint8_t) k.u; raw(&b, 1); }
        else if (k.type == 9) {
            u32(k.elem);
            if (k.elem == 8) {
                u64(k.sarr.size());
                for (const auto& x : k.sarr) str(x);
            } else {
                u64(k.arr.size());
                for (uint64_t x : k.arr) {
                    if (k.elem == 7) { uint8_t b = (uint8_t) x; raw(&b, 1); }
                    else u32((uint32_t) x);
                }
            }
        } else throw std::runtime_error("unsupported synthetic metadata type");
    }
    uint64_t offset = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        const auto& t = ts[i];
        strata::TensorInfo ti;
        ti.shape = t.shape; ti.type = t.type;
        const uint64_t bytes = strata::tensor_payload_bytes(ti);
        if (!bytes) throw std::runtime_error("invalid synthetic tensor");
        str(t.name); u32((uint32_t) t.shape.size());
        for (uint64_t dim : t.shape) u64(dim);
        u32(t.type); u64(overlap && i == 1 ? 0 : offset);
        offset = (offset + bytes + 31) / 32 * 32;
    }
    const uint64_t data = ((uint64_t) f.tellp() + 31) / 32 * 32;
    const uint64_t size = data + offset - (truncate ? 64 : 0);
    f.seekp((std::streamoff) (size - 1));
    f.put('\0');
    if (!f) throw std::runtime_error("cannot write sparse Gemma fixture");
}

struct Dims {
    uint64_t block_count, embd, heads, heads_kv, key_len, experts, used, ff, ff_sh, conv, state, groups, v_heads,
        inner, interval, nextn, hc, hc_lr, idx_heads, idx_key;
};
// Qwen3.6-35B-A3B (unsloth/Qwen3.6-35B-A3B-MTP-GGUF, every quant) and Qwen3.8-Flash-Next, as their files say
const Dims kQwen36 = {41, 2048, 16, 2, 256, 256, 8, 512, 512, 4, 128, 16, 32, 4096, 4, 1, 0, 0, 0, 0};
const Dims kFlashNext = {48, 2560, 24, 2, 256, 512, 10, 640, 640, 4, 128, 16, 48, 6144, 4, 0, 4, 320, 4, 128};
// tiny models of each family for the layout checks (every tensor F32, a few KB)
const Dims kTiny36 = {5, 64, 2, 1, 16, 4, 2, 32, 32, 4, 8, 2, 4, 32, 4, 1, 0, 0, 0, 0};
const Dims kTinyFlash = {8, 64, 2, 1, 16, 4, 2, 32, 32, 4, 8, 2, 4, 32, 4, 0, 2, 8, 2, 8};

std::vector<Kv> metadata(const char* arch, const Dims& d, uint64_t vocab) {
    const std::string p = std::string(arch) + ".";
    std::vector<Kv> kv = {fixture::str("general.architecture", arch),
                          fixture::u32(p + "block_count", d.block_count),
                          fixture::u32(p + "embedding_length", d.embd),
                          fixture::u32(p + "attention.head_count", d.heads),
                          fixture::u32(p + "attention.head_count_kv", d.heads_kv),
                          fixture::u32(p + "attention.key_length", d.key_len),
                          fixture::u32(p + "attention.value_length", d.key_len),
                          fixture::u32(p + "expert_count", d.experts),
                          fixture::u32(p + "expert_used_count", d.used),
                          fixture::u32(p + "expert_feed_forward_length", d.ff),
                          fixture::u32(p + "expert_shared_feed_forward_length", d.ff_sh),
                          fixture::u32(p + "ssm.conv_kernel", d.conv),
                          fixture::u32(p + "ssm.state_size", d.state),
                          fixture::u32(p + "ssm.group_count", d.groups),
                          fixture::u32(p + "ssm.time_step_rank", d.v_heads),
                          fixture::u32(p + "ssm.inner_size", d.inner),
                          fixture::u32(p + "full_attention_interval", d.interval)};
    if (std::string(arch) == "qwen35moe") {
        kv.push_back(fixture::u32(p + "nextn_predict_layers", d.nextn));
    } else {
        kv.push_back(fixture::u32(p + "hyper_connection.count", d.hc));
        kv.push_back(fixture::u32(p + "hyper_connection.low_rank", d.hc_lr));
        kv.push_back(fixture::u32(p + "attention.indexer.head_count", d.idx_heads));
        kv.push_back(fixture::u32(p + "attention.indexer.key_length", d.idx_key));
    }
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < vocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

/// The tensors the layout table of `g` names, as F32 fixtures.
std::vector<Tensor> tensors_of(const strata::core::ModelGeometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        std::vector<uint64_t> s(shape.begin(), shape.end());
        ts.push_back({name, s, F32, seed++});
    };
    for (const auto& w : strata::core::global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const auto& w : strata::core::layer_tensors(g, l)) add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

bool geometry_of(const fs::path& p, strata::core::ModelGeometry& g, int64_t& k, std::string& err) {
    const strata::GgufFile f(p.string());
    return strata::core::geometry_from_gguf(f, g, k, err);
}

std::string layout_error(const fs::path& dir, const char* arch, const Dims& d, uint64_t vocab,
                         void (*mutate)(std::vector<Tensor>&)) {
    const fs::path meta = dir / "meta.gguf";
    fixture::write(meta, metadata(arch, d, vocab), {});
    strata::core::ModelGeometry g;
    int64_t k = 0;
    std::string err;
    if (std::string(arch) == "qwen35moe") {
        if (!geometry_of(meta, g, k, err)) return "geometry: " + err;
    } else {
        // the qwen4exp guard pins Flash-Next's own block count and widths, so the tiny model's geometry is set here
        g.n_layers = (int64_t) d.block_count;
        g.n_embd = (int64_t) d.embd;
        g.n_head = (int64_t) d.heads;
        g.n_head_kv = (int64_t) d.heads_kv;
        g.head_dim = (int64_t) d.key_len;
        g.n_expert = (int64_t) d.experts;
        g.n_ff = (int64_t) d.ff;
        g.n_ff_shexp = (int64_t) d.ff_sh;
        g.ssm_d_conv = (int64_t) d.conv;
        g.ssm_state_size = (int64_t) d.state;
        g.ssm_k_heads = (int64_t) d.groups;
        g.ssm_v_heads = (int64_t) d.v_heads;
        g.ssm_value_dim = (int64_t) d.inner;
        g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
        g.hc = (int64_t) d.hc;
        g.hc_lr = (int64_t) d.hc_lr;
        g.idx_q_heads = (int64_t) d.idx_heads;
        g.idx_key_dim = (int64_t) d.idx_key;
        g.n_vocab = (int64_t) vocab;
    }
    std::vector<Tensor> ts = tensors_of(g);
    if (mutate) mutate(ts);
    const fs::path model = dir / (std::string(arch) + "-tiny.gguf");
    fixture::write(model, metadata(arch, d, vocab), ts);
    const strata::GgufModel m({model.string()});
    size_t n = 0;
    if (!strata::core::check_gguf_layout(m, g, err, &n)) return err;
    return {};
}

int real_model(const std::string& any) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(any);
        const std::string arch = strata::check_architecture(m.meta());
        std::printf("  %zu shard(s); architecture guard: %s; engine guard: %s\n", m.size(),
                    arch.empty() ? "ok" : arch.c_str(),
                    strata::check_engine_architecture(m.meta()).empty() ? "runs"
                                                                         : strata::check_engine_architecture(m.meta()).c_str());
        if (!arch.empty()) return 1;
        if (strata::model_family(m.meta()) == strata::ModelFamily::Gemma4) {
            strata::gemma4::Gemma4Geometry g;
            std::string err;
            if (!strata::gemma4::geometry_from_gguf(m.meta(), g, err)) {
                std::printf("  Gemma 4 geometry: %s\n", err.c_str());
                return 1;
            }
            err = strata::check_gemma4_layout(m);
            if (!err.empty()) {
                std::printf("  Gemma 4 directory: %s\n", err.c_str());
                return 1;
            }
            strata::gemma4::Gemma4Model loaded;
            if (!strata::gemma4::load_model(any, loaded, err)) {
                std::printf("  Gemma 4 tensor map: %s\n", err.c_str());
                return 1;
            }
            std::printf("  Gemma 4 geometry: %lld layers, hidden %lld, vocab %lld, SWA %lld/%lld KV heads: ok\n",
                        (long long) g.n_layers, (long long) g.n_embd, (long long) g.n_vocab,
                        (long long) g.layers.front().kv_heads, (long long) g.layers[5].kv_heads);
            std::printf("  Gemma 4 tensor map: %zu layers, optional global-layer V, Q4_0 weights: ok\n",
                        loaded.layers.size());
            std::printf("  Gemma 4 directory: ok\n");
            return 0;
        }
        strata::core::ModelGeometry g;
        int64_t k = 0;
        std::string err;
        if (!strata::core::geometry_from_gguf(m.meta(), g, k, err)) {
            std::printf("  geometry: %s\n", err.c_str());
            return 1;
        }
        std::printf("  family %s: layers %lld (+%lld MTP), hidden %lld, attention %lld/%lld x %lld every %lld, GDN K %lld"
                    " V %lld x %lld conv %lld (%lld), MoE %lld experts top-%lld FF %lld shared %lld, hc %lld, indexer"
                    " %lld, vocab %lld\n",
                    strata::model_family_title(g.family), (long long) g.n_layers, (long long) g.n_mtp_layers,
                    (long long) g.n_embd, (long long) g.n_head, (long long) g.n_head_kv, (long long) g.head_dim,
                    (long long) g.qsa_interval, (long long) g.ssm_k_heads, (long long) g.ssm_v_heads,
                    (long long) g.ssm_state_size, (long long) g.ssm_conv_channels, (long long) g.ssm_d_conv,
                    (long long) g.n_expert, (long long) k, (long long) g.n_ff, (long long) g.n_ff_shexp,
                    (long long) g.hc, (long long) g.idx_q_heads, (long long) g.n_vocab);
        const std::string mm = strata::core::compiled_geometry_mismatch(g);
        std::printf("  compiled kernels: %s\n", mm.empty() ? "match" : mm.c_str());
        size_t n = 0;
        const bool ok = strata::core::check_gguf_layout(m, g, err, &n);
        std::printf("  layout: %s (%zu tensors checked)\n", ok ? "ok" : err.c_str(), n);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("model_family_test\n");
    TempDir d;
    {
        const auto base_kv = gemma_metadata();
        const auto base_ts = gemma_tensors();
        const fs::path path = d.path / "gemma.gguf";
        auto validate = [&](const std::vector<Kv>& kv, const std::vector<Tensor>& ts, bool overlap = false,
                            bool truncate = false) {
            write_gemma_fixture(path, kv, ts, overlap, truncate);
            const strata::GgufModel m({path.string()});
            return strata::check_gemma4_layout(m);
        };
        check(validate(base_kv, base_ts).empty(), "Gemma 4: complete sparse Q4_0 directory passes");
        {
            strata::gemma4::Gemma4Model model;
            std::string err;
            check(strata::gemma4::load_model(path.string(), model, err) && model.layers.size() == 30 &&
                      model.token_embedding.info && model.layers[0].v.info && !model.layers[5].v.info &&
                      model.layers[0].expert_gate_up.info && model.layers[0].expert_down.info,
                  "Gemma 4 loader maps required tensors and allows absent global-layer V");
        }
        {
            const strata::GgufFile f(path.string());
            check(strata::model_family(f) == strata::ModelFamily::Gemma4 &&
                      strata::check_architecture(f).empty(),
                  "Gemma 4 is recognised with valid metadata");
            check(!strata::model_family_runs(strata::ModelFamily::Gemma4) &&
                      strata::check_engine_architecture(f).find("no Strata engine") != std::string::npos,
                  "recognition does not claim engine support");
            strata::gemma4::Gemma4Geometry g;
            std::string err;
            check(strata::gemma4::geometry_from_gguf(f, g, err) && g.n_layers == 30 && g.n_embd == 2816 &&
                      g.n_vocab == 262144 && g.n_experts == 128 && g.experts_per_token == 8 &&
                      g.layers.size() == 30 && g.layers[0].sliding_window && g.layers[0].kv_heads == 8 &&
                      g.layers[0].head_dim == 256 && !g.layers[5].sliding_window && g.layers[5].kv_heads == 2 &&
                      g.layers[5].head_dim == 512,
                  "Gemma 4 geometry preserves per-layer SWA/global attention dimensions");
            strata::core::ModelGeometry qwen_geometry;
            int64_t top_k = 0;
            check(!strata::core::geometry_from_gguf(f, qwen_geometry, top_k, err) &&
                      err.find("no Strata engine") != std::string::npos,
                  "Gemma 4 is not interpreted as the Qwen kernel geometry");
        }
        auto kv = base_kv;
        for (auto& k : kv) if (k.key == "gemma4.attention.head_count_kv") k.arr[5] = 8;
        check(validate(kv, base_ts).find("head_count_kv") != std::string::npos,
              "Gemma 4: wrong full-attention KV heads rejected");
        kv = base_kv;
        for (auto& k : kv) if (k.key == "gemma4.attention.sliding_window_pattern") k.arr[0] = 0;
        check(validate(kv, base_ts).find("sliding_window_pattern") != std::string::npos,
              "Gemma 4: wrong sliding-window layer rejected");
        kv = base_kv;
        for (auto& k : kv) if (k.key == "gemma4.attention.head_count_kv") k.elem = 4;
        check(validate(kv, base_ts).find("head_count_kv") != std::string::npos,
              "Gemma 4: wrong array element type rejected");
        kv = base_kv;
        for (auto& k : kv) if (k.key == "tokenizer.ggml.tokens") k.sarr.pop_back();
        check(validate(kv, base_ts).find("tokenizer.ggml.tokens") != std::string::npos,
              "Gemma 4: wrong vocabulary count rejected");
        kv = base_kv;
        for (auto& k : kv) if (k.key == "gemma4.final_logit_softcapping") k.f = 0.f;
        check(validate(kv, base_ts).find("final_logit_softcapping") != std::string::npos,
              "Gemma 4: wrong softcap rejected");
        kv = base_kv;
        kv.erase(std::remove_if(kv.begin(), kv.end(), [](const Kv& k) {
                     return k.key == "gemma4.rope.freq_base_swa";
                 }), kv.end());
        check(validate(kv, base_ts).find("rope.freq_base_swa") != std::string::npos,
              "Gemma 4: missing RoPE metadata rejected");
        kv = base_kv;
        kv.push_back(fixture::u32("gemma4.block_count", 29));
        check(validate(kv, base_ts).find("duplicate metadata") != std::string::npos,
              "Gemma 4: duplicate metadata rejected");
        kv = base_kv;
        for (auto& k : kv) if (k.key == "tokenizer.ggml.eos_token_id") k.u = 2;
        check(validate(kv, base_ts).find("eos_token_id") != std::string::npos,
              "Gemma 4: wrong EOS token metadata rejected");
        auto ts = base_ts;
        for (auto& t : ts) if (t.name == "blk.5.attn_q.weight") t.shape[1] = 4096;
        check(validate(base_kv, ts).find("blk.5.attn_q.weight") != std::string::npos,
              "Gemma 4: wrong full-attention tensor shape rejected");
        ts = base_ts;
        for (auto& t : ts) if (t.name == "token_embd.weight") t.type = 2;
        check(validate(base_kv, ts).find("token_embd.weight") != std::string::npos,
              "Gemma 4: wrong tied-head quantization rejected");
        ts = base_ts;
        for (auto& t : ts) if (t.name == "blk.7.ffn_gate_up_exps.weight") t.shape[2] = 127;
        check(validate(base_kv, ts).find("blk.7.ffn_gate_up_exps.weight") != std::string::npos,
              "Gemma 4: wrong expert layout rejected");
        ts = base_ts;
        ts.erase(std::remove_if(ts.begin(), ts.end(), [](const Tensor& t) {
                     return t.name == "blk.0.attn_v.weight";
                 }), ts.end());
        check(validate(base_kv, ts).find("missing blk.0.attn_v.weight") != std::string::npos,
              "Gemma 4: missing SWA value projection rejected");
        ts = base_ts;
        ts.push_back({"output.weight", {2816, 262144}, 14, 0});
        check(validate(base_kv, ts).find("unexpected tensor output.weight") != std::string::npos,
              "Gemma 4: separate output tensor rejected");
        check(validate(base_kv, base_ts, true).find("overlapping") != std::string::npos,
              "Gemma 4: overlapping payloads rejected");
        check(validate(base_kv, base_ts, false, true).find("out-of-bounds") != std::string::npos,
              "Gemma 4: truncated payload extent rejected without reading it");
    }
    {
        // the guards
        const fs::path q36 = d.path / "q36.gguf", fn = d.path / "fn.gguf", other = d.path / "other.gguf";
        fixture::write(q36, metadata("qwen35moe", kQwen36, 8), {});
        fixture::write(fn, metadata("qwen4exp", kFlashNext, 8), {});
        fixture::write(other, {fixture::str("general.architecture", "llama")}, {});
        const strata::GgufFile a(q36.string()), b(fn.string()), c(other.string());
        check(strata::model_family(a) == strata::ModelFamily::Qwen35Moe &&
                  strata::model_family(b) == strata::ModelFamily::Qwen4Exp &&
                  strata::model_family(c) == strata::ModelFamily::Unknown,
              "model_family: qwen35moe, qwen4exp, unknown");
        check(strata::check_architecture(a).empty(), "the architecture guard accepts qwen35moe");
        check(strata::check_architecture(b).empty(), "the architecture guard accepts qwen4exp");
        check(strata::check_architecture(c).find("'llama'") != std::string::npos, "and refuses llama, naming it");
        const std::string e = strata::check_engine_architecture(a);
        check(e.find("Qwen3.6-35B-A3B") != std::string::npos && e.find("does not run it yet") != std::string::npos,
              "the engine guard refuses Qwen3.6: recognised, not supported yet");
        check(strata::check_engine_architecture(b).empty(), "the engine guard runs qwen4exp");
        std::vector<Kv> kv = metadata("qwen35moe", kQwen36, 8);
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "qwen35moe.ssm.time_step_rank") { kv.erase(it); break; }
        const fs::path miss = d.path / "miss.gguf";
        fixture::write(miss, kv, {});
        check(strata::check_architecture(strata::GgufFile(miss.string())).find("ssm.time_step_rank") != std::string::npos,
              "a qwen35moe file without ssm.time_step_rank is refused, naming the key");
    }
    {
        // the geometry of the real models, from their metadata values
        const fs::path q36 = d.path / "q36g.gguf", fn = d.path / "fng.gguf";
        fixture::write(q36, metadata("qwen35moe", kQwen36, 248320), {});
        fixture::write(fn, metadata("qwen4exp", kFlashNext, 248320), {});
        strata::core::ModelGeometry g;
        int64_t k = 0;
        std::string err;
        const bool ok = geometry_of(q36, g, k, err);
        check(ok, "Qwen3.6 geometry reads" + (ok ? std::string() : ": " + err));
        check(g.family == strata::ModelFamily::Qwen35Moe && g.n_layers == 40 && g.n_mtp_layers == 1 && g.n_embd == 2048,
              "Qwen3.6: 40 trunk layers + 1 MTP block, hidden 2048");
        check(g.n_qsa_layers() == 10 && g.n_gdn_layers() == 30 && strata::core::layer_kind(g, 3) ==
                  strata::core::LayerKind::Attention && strata::core::layer_kind(g, 40) == strata::core::LayerKind::Mtp,
              "Qwen3.6: 10 attention / 30 GDN layers, layer 40 is the MTP block");
        check(g.ssm_k_heads == 16 && g.ssm_v_heads == 32 && g.ssm_conv_channels == 8192 && g.ssm_value_dim == 4096,
              "Qwen3.6 GDN: 16 K groups, 32 V heads, conv 8192, inner 4096");
        check(g.n_head == 16 && g.n_head_kv == 2 && g.head_dim == 256 && !g.has_indexer() && !g.has_hyper_connections(),
              "Qwen3.6 attention: 16 Q / 2 KV x 256, no indexer, no hyper-connections");
        check(g.n_expert == 256 && k == 8 && g.n_ff == 512 && g.n_ff_shexp == 512 && g.n_vocab == 248320,
              "Qwen3.6 MoE: 256 experts top-8, FF 512, shared 512; vocab 248320");
        check(!strata::core::compiled_geometry_mismatch(g).empty(), "Qwen3.6 does not match the compiled kernels");
        strata::core::ModelGeometry f;
        const bool okf = geometry_of(fn, f, k, err);
        const strata::core::ModelGeometry c;
        check(okf && strata::core::compiled_geometry_mismatch(f).empty() && k == 10 && f.n_expert == c.n_expert &&
                  f.ssm_conv_channels == c.ssm_conv_channels && f.n_vocab == c.n_vocab && f.hc_dim() == c.hc_dim(),
              "Flash-Next's metadata gives exactly the compiled geometry");
        Dims coder = kFlashNext;
        coder.experts = 256;
        const fs::path cp = d.path / "coder.gguf";
        fixture::write(cp, metadata("qwen4exp", coder, 8), {});
        check(geometry_of(cp, f, k, err) && f.n_expert == 256 && strata::core::compiled_geometry_mismatch(f).empty(),
              "the Coder's 256 experts are a run-time size, not a kernel mismatch");
        Dims wrong = kQwen36;
        wrong.inner = 4000;
        const fs::path wp = d.path / "wrong.gguf";
        fixture::write(wp, metadata("qwen35moe", wrong, 8), {});
        check(!geometry_of(wp, f, k, err) && err.find("ssm.inner_size") != std::string::npos,
              "an inconsistent ssm.inner_size is refused");
    }
    {
        // the layout tables
        check(layout_error(d.path, "qwen35moe", kTiny36, 16, nullptr).empty(),
              "a complete qwen35moe file passes its layout (incl. the nextn MTP block)");
        const std::string ef = layout_error(d.path, "qwen4exp", kTinyFlash, 16, nullptr);
        check(ef.empty(), "a complete qwen4exp file passes its layout (hyper-connections, indexer)" +
                              (ef.empty() ? std::string() : ": " + ef));
        std::string e = layout_error(d.path, "qwen35moe", kTiny36, 16, [](std::vector<Tensor>& ts) {
            for (auto& t : ts)
                if (t.name == "blk.1.ssm_conv1d.weight") t.shape[1] += 1;
        });
        check(e.find("blk.1.ssm_conv1d.weight") != std::string::npos && e.find("wants") != std::string::npos,
              "a wrong shape is refused, naming the tensor and both shapes");
        e = layout_error(d.path, "qwen35moe", kTiny36, 16, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.4.nextn.eh_proj.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.4.nextn.eh_proj.weight", "a missing MTP tensor is refused");
        e = layout_error(d.path, "qwen35moe", kTiny36, 16, [](std::vector<Tensor>& ts) {
            ts.push_back({"blk.0.hc_attn_norm.weight", {64}, F32, 99});
        });
        check(e.find("blk.0.hc_attn_norm.weight") != std::string::npos && e.find("not in the") != std::string::npos,
              "a tensor the qwen35moe table does not name is refused");
        e = layout_error(d.path, "qwen4exp", kTinyFlash, 16, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.3.indexer.k_proj.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.3.indexer.k_proj.weight", "Flash-Next: a missing indexer tensor is refused (" + e + ")");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
