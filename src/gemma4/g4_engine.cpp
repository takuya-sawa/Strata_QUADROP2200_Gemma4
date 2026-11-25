// src/gemma4/g4_engine.cpp - the Gemma 4 (GGUF architecture gemma4) engine path: Gemma-4-26B-A4B (30 layers, 128 experts,
// top-8, 5 sliding-window layers then one global layer), a port of llama.cpp's src/models/gemma4.cpp.
//
// The compute graph is llama.cpp's, op for op, built on ggml (Strata's vendored copy):
//   * x = tok_embd[token] * sqrt(n_embd);
//   * per layer: attention (pre-norm, per-head q/k norms, V = rms(K) on layers without attn_v, NEOX rope with the layer's
//     own base and width, attention scale 1.0, sliding window on the SWA layers, rope_freqs on the global ones), post
//     norm, residual; then the FFN: a dense GELU branch and a routed branch (softmax router on rms(x) / sqrt(n_embd) *
//     scale, top-8, weights renormalised, GELU experts, per-expert output scale), each with its own post norm, summed,
//     post norm, residual, layer_output_scale;
//   * output_norm, the tied token embedding as head, logits = 30 * tanh(logits / 30).
// The weights stay in the mmapped GGUF (Q4_0 / Q6_K blocks read in place by ggml-cpu); the KV cache is Q4_0 (--kv-type) and a ring for the sliding-window layers.  With ggml's
// CUDA backend (STRATA_G4_CUDA) the tensors every token reads (attention, shared FFN, router, head) and that layer's KV
// cache are copied to the GPU, layer by layer, as far as the free VRAM goes (--gpu-mem / --gpu-reserve MiB, --no-gpu);
// the routed experts (12.6 GB) stay on the CPU unless a whole layer's still fit.  ggml_backend_sched runs each op where
// its weights are.
//
// Modes: --serve (the engine protocol of serve/server.py: READY / GEN / T / DONE / STOP), --tokens IDS [--max-new N].
#include "strata/gemma4/engine.hpp"

#include "strata/gemma4/model.hpp"
#include "strata/qwen35/sampling.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#if defined(STRATA_G4_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kMaskPad = 64;   // flash attention wants the mask rows padded to this
constexpr int64_t kFattnStride = 256;   // ggml's CUDA flash-attn blocks; K/V must span a multiple of it
constexpr int kBatch = 256;    // prompt tokens per graph
constexpr int kMaxEmbdBatch = 1280;   // the most tokens one picture may take (the vision tower gives up to 1120)
constexpr int kGraphNodes = 8192;   // the graph's node capacity; the engine builds ~2900 nodes per step
std::vector<int32_t> g_stop{1, 106};   // <eos> and <turn|>, the end of a model turn

// The physical cores of this box and whether each of them has more than one hardware thread; 0 if unknowable.
// ggml's CPU ops saturate the physical cores, so a worker per logical CPU only spins on the hyperthreads.
int detect_physical_cores(int& threads_per_core) {
    FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (!f) return 0;
    std::map<int, int> cores_of;   // physical id -> the cores it reports
    int phys = 0, best_siblings = 0, n_proc = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        int v;
        if (std::sscanf(line, "processor : %d", &v) == 1) ++n_proc;
        else if (std::sscanf(line, "physical id : %d", &v) == 1) phys = v;
        else if (std::sscanf(line, "siblings : %d", &v) == 1) best_siblings = std::max(best_siblings, v);
        else if (std::sscanf(line, "cpu cores : %d", &v) == 1) cores_of[phys] = std::max(cores_of[phys], v);
    }
    std::fclose(f);
    int physical = 0;
    for (const auto& kv : cores_of) physical += kv.second;
    if (physical < 1) return 0;
    const int per_core = cores_of.begin()->second > 0 ? best_siblings / cores_of.begin()->second : 1;
    threads_per_core = std::max(1, per_core);
    return n_proc > 0 ? physical : 0;
}

// A decode step is a chain of tiny ops: past the physical cores each extra worker adds barrier and cache traffic
// instead of work (measured here: 3 threads 72-91 ms/token, 4 threads 84-190 ms on the same build).  Prefill is
// the opposite - a big batch does scale to every core.
int decode_threads(int want) {
    int tpc = 0;
    const int physical = detect_physical_cores(tpc);
    int best = physical > 0 ? physical - (tpc >= 2 ? 0 : 1) : want / 2;   // without SMT, leave a core for the rest
    if (want <= 2) best = want;                                          // never collapse a small request to one thread
    if (const char* e = std::getenv("STRATA_G4_THREADS_DECODE")) { const int v = std::atoi(e); if (v > 0) best = v; }
    return std::max(1, std::min(want, best));
}

// libgomp reads OMP_WAIT_POLICY / GOMP_SPINCOUNT when the process starts, so a running process cannot change them.
// Re-exec once with the workers asked to sleep rather than spin, which is what the barrier between ggml's ~2900
// per-token ops needs (measured on this box: 4-thread decode 108-346 ms/token with the default policy, 82-98 with it).
void reexec_with_passive_wait(int argc, char** argv) {
#ifndef _WIN32
    for (const char* v : {"GOMP_SPINCOUNT", "OMP_WAIT_POLICY"})
        if (std::getenv(v)) return;                                   // the caller already chose a policy
    if (std::getenv("STRATA_G4_PASSIVE_WAIT")) return;                 // already re-executed once
    if (std::thread::hardware_concurrency() < 2) return;
    if (setenv("STRATA_G4_PASSIVE_WAIT", "1", 1) != 0) return;
    setenv("GOMP_SPINCOUNT", "0", 1);
    std::vector<char*> av;
    av.reserve((size_t) argc + 1);
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);   // argv[0] is not always a usable path
    if (n > 0) { self[n] = '\0'; av.push_back(self); } else av.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) av.push_back(argv[i]);
    av.push_back(nullptr);
    execvp(av[0], av.data());                                         // the new process keeps the same environment
    unsetenv("STRATA_G4_PASSIVE_WAIT");
    unsetenv("GOMP_SPINCOUNT");
    std::fprintf(stderr, "strata-gemma4: cannot re-exec with a passive wait policy\n");
#endif
}
bool is_stop(int32_t t) { return std::find(g_stop.begin(), g_stop.end(), t) != g_stop.end(); }

// A piece of a prompt: text tokens, or the embeddings of a picture (read as one batch).
struct Segment {
    bool embd = false;
    std::vector<int32_t> tokens;
    std::vector<float> data;   // embd: count x n_embd
    int count = 0;             // embd: positions
    uint64_t key = 0;          // embd: identifies the picture, so a cache that holds it can be reused
    size_t size() const { return embd ? (size_t) count : tokens.size(); }
};

// what the KV cache holds at a position: the token id, or a number made from the picture's key and the position in it
int64_t image_key(uint64_t key, int idx) { return (1LL << 62) | (int64_t) ((key & ((1ULL << 40) - 1)) << 16) | idx; }

struct LayerW {
    ggml_tensor *attn_norm = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *q_norm = nullptr,
                *k_norm = nullptr, *attn_post_norm = nullptr, *ffn_norm = nullptr, *ffn_gate = nullptr,
                *ffn_up = nullptr, *ffn_down = nullptr, *ffn_post_norm = nullptr, *router = nullptr,
                *router_scale = nullptr, *pre_norm_2 = nullptr, *post_norm_1 = nullptr, *post_norm_2 = nullptr,
                *gate_up_exps = nullptr, *down_exps = nullptr, *down_scale = nullptr, *out_scale = nullptr;
};

struct Engine {
    strata::gemma4::Gemma4Model m;
    ggml_context* wctx = nullptr;   // weights read in place from the mapped file (host)
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_context* gctx = nullptr;   // weights copied to the GPU
    ggml_backend_buffer_t gbuf = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_context* kvctx[2] = {nullptr, nullptr};   // the KV cache: [0] host, [1] GPU
    ggml_backend_buffer_t kvbuf[2] = {nullptr, nullptr};
    ggml_backend_sched_t sched = nullptr;
    ggml_context* fctx = nullptr;   // the graph metadata, kept for the whole run and rewound by ggml_reset
    std::vector<uint8_t> fbuf;
    size_t gpu_bytes = 0;
    int gpu_layers = 0;       // layers whose attention and shared FFN run on the GPU
    int gpu_expert_layers = 0;
    std::vector<LayerW> L;
    std::vector<ggml_tensor*> kc, vc;
    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *rope_freqs = nullptr;
    int n_ctx = 0;
    ggml_type kv_type = GGML_TYPE_Q4_0;
    int swa_cells = 0;   // cells of a sliding-window layer's cache: a ring over the last positions
    int n_threads = 4;    // the prefill / batch phase
    int n_threads_dec = 3;   // a decode step (see reexec_with_passive_wait for why the two phases differ)
    std::vector<int64_t> live;   // what the KV cache holds, by position (token ids; see image_key)
    std::vector<float> logits;

    // STRATA_G4_TIME: sums the time of each phase of forward() so the slow parts can be found. Off unless set.
    // [0] = a decode step (one token), [1] = a prefill batch (several tokens).
    struct Phase {
        long long n = 0, n_tok = 0;
        double build = 0, alloc = 0, fill = 0, upload = 0, compute = 0, tail = 0, sample = 0;
    };
    struct Timing {
        bool on = false;
        Phase p[2];
        void add_sample(int nt, double ms) { if (on) p[nt > 1].sample += ms; }
    };
    Timing tm;
    void print_timing(bool last) {
        if (!tm.on) return;
        for (int i = 0; i < 2; ++i) {
            const Phase& q = tm.p[i];
            if (q.n == 0) continue;
            std::fprintf(stderr, "timing %s (%lld fwd, %lld tok): build %.3f alloc %.3f fill %.3f upload %.3f compute %.3f tail %.3f sample %.3f ms/fwd\n",
                         i ? "prefill" : "decode", q.n, q.n_tok, q.build / q.n, q.alloc / q.n, q.fill / q.n,
                         q.upload / q.n, q.compute / q.n, q.tail / q.n, q.sample / q.n);
        }
        if (last) tm.p[0] = tm.p[1] = Phase{};
    }

    ~Engine() {
        print_timing(true);
        if (sched) ggml_backend_sched_free(sched);
        if (fctx) ggml_free(fctx);
        for (int i = 0; i < 2; ++i) {
            if (kvbuf[i]) ggml_backend_buffer_free(kvbuf[i]);
            if (kvctx[i]) ggml_free(kvctx[i]);
        }
        if (gbuf) ggml_backend_buffer_free(gbuf);
        if (gctx) ggml_free(gctx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (gpu) ggml_backend_free(gpu);
        if (cpu) ggml_backend_free(cpu);
    }

    struct Options {
        int ctx = 4096;
        int threads = 4;
        bool use_gpu = true;
        long long gpu_mem_mib = -1;       // the VRAM to fill with weights and KV cache; -1: what is free less the reserve
        long long gpu_reserve_mib = 400;  // kept free for the compute buffers
        std::string kv_type = "q4_0";   // the KV cache's type: f16, q8_0 or q4_0
        bool op_offload = false;          // let the GPU run host-weight ops of big batches (weights copied per batch)
    };

    static size_t bytes_of(const strata::gemma4::TensorRef& r) {
        if (!r.info) return 0;
        size_t n = ggml_row_size((ggml_type) r.info->type, (int64_t) r.info->shape[0]);
        for (size_t i = 1; i < r.info->shape.size(); ++i) n *= (size_t) r.info->shape[i];
        return n;
    }

    bool load(const std::string& path, const Options& opt, std::string& err) {
        if (!strata::gemma4::load_model(path, m, err)) return false;
        n_ctx = opt.ctx;
        n_threads = opt.threads;
        n_threads_dec = std::min(n_threads, decode_threads(n_threads));
        const auto& g = m.geometry;
        if (opt.kv_type == "f16") kv_type = GGML_TYPE_F16;
        else if (opt.kv_type == "q8_0") kv_type = GGML_TYPE_Q8_0;
        else if (opt.kv_type == "q4_0") kv_type = GGML_TYPE_Q4_0;
        else { err = "--kv-type is f16, q8_0 or q4_0"; return false; }
        if (n_ctx > g.context_length) { err = "--max-context is beyond the model's context length"; return false; }
        // a sliding-window layer keeps the last `window` positions plus room for one batch and a short rewind
        swa_cells = (int) std::min<int64_t>(n_ctx, g.sliding_window + std::max(kBatch, kMaxEmbdBatch) + 256);
        if (const char* e = std::getenv("STRATA_G4_SWA_CELLS")) swa_cells = std::min(n_ctx, std::atoi(e));   // test: 0 / the context = no ring
        if (g.n_layers < 1 || g.shared_kv_layers != 0 || g.per_layer_input_dim != 0) {
            err = "this engine runs Gemma 4 files without shared-KV layers and per-layer embeddings (the 26B-A4B)";
            return false;
        }
        cpu = ggml_backend_cpu_init();
        if (!cpu) { err = "cannot start the ggml CPU backend"; return false; }
        ggml_backend_cpu_set_n_threads(cpu, n_threads);

        // --- what goes to the GPU: the tensors every token reads (attention, shared FFN, router, norms) and that
        // layer's KV cache, layer by layer, the token embedding / head, then whole layers' experts, while the budget lasts
        std::vector<char> layer_gpu(m.layers.size(), 0), experts_gpu(m.layers.size(), 0);
        size_t expert_layers_total = 0;
        for (const auto& layer : m.layers)
            if (bytes_of(layer.expert_gate_up) && bytes_of(layer.expert_down)) ++expert_layers_total;
        size_t gpu_expert_bytes = 0;
        bool head_gpu = false;
        auto cells_of = [&](size_t il) { return g.layers[il].sliding_window ? swa_cells : n_ctx; };
        auto kv_bytes = [&](size_t il) {
            return (size_t) 2 * ggml_row_size(kv_type, g.layers[il].kv_heads * g.layers[il].head_dim) * (size_t) cells_of(il);
        };
        auto dense_bytes = [&](const strata::gemma4::LayerTensors& s) {
            size_t b = 0;
            for (const auto* r : {&s.attention_norm, &s.q, &s.k, &s.v, &s.attention_output, &s.q_norm, &s.k_norm,
                                  &s.post_attention_norm, &s.ffn_norm, &s.shared_gate, &s.shared_up, &s.shared_down,
                                  &s.post_ffw_norm, &s.router, &s.router_scale, &s.pre_ffw_norm_2, &s.post_ffw_norm_1,
                                  &s.post_ffw_norm_2, &s.expert_down_scale, &s.layer_output_scale})
                b += bytes_of(*r);
            return b;
        };
#if defined(STRATA_G4_CUDA)
        if (opt.use_gpu) {
            gpu = ggml_backend_cuda_init(0);
            if (!gpu) std::fprintf(stderr, "strata-gemma4: no usable CUDA device, running on the CPU\n");
        }
        if (gpu) {
            size_t fr = 0, tot = 0;
            ggml_backend_cuda_get_device_memory(0, &fr, &tot);
            long long budget = opt.gpu_mem_mib >= 0 ? opt.gpu_mem_mib * 1048576LL
                                                    : (long long) fr - opt.gpu_reserve_mib * 1048576LL;
            const size_t head = bytes_of(m.token_embedding) + bytes_of(m.output_norm);
            if ((long long) head <= budget) { head_gpu = true; budget -= (long long) head; }
            for (size_t il = 0; il < m.layers.size(); ++il) {
                const size_t need = dense_bytes(m.layers[il]) + kv_bytes(il);
                if ((long long) need > budget) break;
                layer_gpu[il] = 1;
                budget -= (long long) need;
                ++gpu_layers;
            }
            for (size_t il = 0; il < m.layers.size() && gpu_layers == (int) m.layers.size(); ++il) {
                const size_t need = bytes_of(m.layers[il].expert_gate_up) + bytes_of(m.layers[il].expert_down);
                if ((long long) need > budget) break;
                experts_gpu[il] = 1;
                budget -= (long long) need;
                ++gpu_expert_layers;
                gpu_expert_bytes += need;
            }
        }
#else
        (void) opt; (void) dense_bytes; (void) kv_bytes;
#endif

        ggml_init_params ip{ggml_tensor_overhead() * 1024 + 4096, nullptr, true};
        wctx = ggml_init(ip);
        gctx = ggml_init(ip);
        std::vector<std::pair<ggml_tensor*, const uint8_t*>> placed, to_gpu;
        auto mk = [&](const strata::gemma4::TensorRef& r, bool on_gpu) -> ggml_tensor* {
            if (!r.info) return nullptr;
            int64_t ne[4] = {1, 1, 1, 1};
            for (size_t i = 0; i < r.info->shape.size() && i < 4; ++i) ne[i] = (int64_t) r.info->shape[i];
            ggml_tensor* t = ggml_new_tensor(on_gpu ? gctx : wctx, (ggml_type) r.info->type, (int) r.info->shape.size(), ne);
            ggml_set_name(t, r.info->name.c_str());
            (on_gpu ? to_gpu : placed).emplace_back(t, r.data);
            if (on_gpu) gpu_bytes += ggml_nbytes(t);
            return t;
        };
        tok_embd = mk(m.token_embedding, head_gpu);
        out_norm = mk(m.output_norm, head_gpu);
        rope_freqs = mk(m.rope_freqs, gpu_layers > 0);
        L.resize(m.layers.size());
        for (size_t i = 0; i < m.layers.size(); ++i) {
            const auto& s = m.layers[i];
            LayerW& w = L[i];
            const bool d = layer_gpu[i], x = experts_gpu[i];
            w.attn_norm = mk(s.attention_norm, d); w.wq = mk(s.q, d); w.wk = mk(s.k, d); w.wv = mk(s.v, d);
            w.wo = mk(s.attention_output, d); w.q_norm = mk(s.q_norm, d); w.k_norm = mk(s.k_norm, d);
            w.attn_post_norm = mk(s.post_attention_norm, d); w.ffn_norm = mk(s.ffn_norm, d);
            w.ffn_gate = mk(s.shared_gate, d); w.ffn_up = mk(s.shared_up, d); w.ffn_down = mk(s.shared_down, d);
            w.ffn_post_norm = mk(s.post_ffw_norm, d); w.router = mk(s.router, d); w.router_scale = mk(s.router_scale, d);
            w.pre_norm_2 = mk(s.pre_ffw_norm_2, d); w.post_norm_1 = mk(s.post_ffw_norm_1, d);
            w.post_norm_2 = mk(s.post_ffw_norm_2, d); w.gate_up_exps = mk(s.expert_gate_up, x);
            w.down_exps = mk(s.expert_down, x); w.down_scale = mk(s.expert_down_scale, d);
            w.out_scale = mk(s.layer_output_scale, d);
        }
        // the host tensors point into one buffer over the mapped file: nothing is copied
        if (!placed.empty()) {
            const uint8_t* lo = nullptr;
            const uint8_t* hi = nullptr;
            for (auto& [t, p] : placed) {
                if (!lo || p < lo) lo = p;
                if (!hi || p + ggml_nbytes(t) > hi) hi = p + ggml_nbytes(t);
            }
            wbuf = ggml_backend_cpu_buffer_from_ptr(const_cast<uint8_t*>(lo), (size_t) (hi - lo));
            if (!wbuf) { err = "cannot wrap the mapped weights"; return false; }
            ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);   // ops follow their weights
            for (auto& [t, p] : placed) ggml_backend_tensor_alloc(wbuf, t, const_cast<uint8_t*>(p));
        }
        if (!to_gpu.empty()) {
            gbuf = ggml_backend_alloc_ctx_tensors(gctx, gpu);
            if (!gbuf) { err = "cannot allocate the GPU weights (lower --gpu-mem)"; return false; }
            ggml_backend_buffer_set_usage(gbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (auto& [t, p] : to_gpu) ggml_backend_tensor_set(t, p, 0, ggml_nbytes(t));
        }

        // the KV cache: per layer one row of n_kv_heads * head_dim per position, K and V, F16; on the GPU for the layers
        // whose attention runs there
        for (int i = 0; i < 2; ++i) kvctx[i] = ggml_init(ggml_init_params{ggml_tensor_overhead() * (size_t) (2 * g.n_layers + 8), nullptr, true});
        for (int64_t il = 0; il < g.n_layers; ++il) {
            const auto& lg = g.layers[(size_t) il];
            ggml_context* c = kvctx[layer_gpu[(size_t) il] ? 1 : 0];
            const int64_t cells = lg.sliding_window ? swa_cells : n_ctx;
            kc.push_back(ggml_new_tensor_2d(c, kv_type, lg.kv_heads * lg.head_dim, cells));
            vc.push_back(ggml_new_tensor_2d(c, kv_type, lg.kv_heads * lg.head_dim, cells));
            if (layer_gpu[(size_t) il]) gpu_bytes += kv_bytes((size_t) il);
        }
        for (int i = 0; i < 2; ++i) {
            if (!ggml_get_first_tensor(kvctx[i])) continue;   // no layer's cache lives there
            kvbuf[i] = ggml_backend_alloc_ctx_tensors(kvctx[i], i ? gpu : cpu);
            if (!kvbuf[i]) { err = "cannot allocate the KV cache"; return false; }
            ggml_backend_buffer_clear(kvbuf[i], 0);
        }

        tm.on = std::getenv("STRATA_G4_TIME") != nullptr;

        std::vector<ggml_backend_t> backends;
        if (gpu) backends.push_back(gpu);
        backends.push_back(cpu);
        sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), kGraphNodes, false,
                                       gpu != nullptr && opt.op_offload);
        if (!sched) { err = "cannot create the backend scheduler"; return false; }

        // the graph metadata lives in one buffer for the whole run: ggml_reset rewinds it instead of the allocator
        // seeing a fresh malloc and free of ~1 MiB on every step
        fbuf.resize(ggml_tensor_overhead() * (size_t) kGraphNodes * 2 + ggml_graph_overhead_custom(kGraphNodes, false));
        fctx = ggml_init(ggml_init_params{fbuf.size(), fbuf.data(), true});
        if (!fctx) { err = "cannot create the graph context"; return false; }
        if (gpu)
            std::fprintf(stderr, "strata-gemma4: GPU: %d of %zu layers (attention + shared FFN + KV)%s, experts of %d layers, "
                         "%.0f MiB of weights and cache\n", gpu_layers, m.layers.size(), head_gpu ? ", token embedding / head" : "",
                         gpu_expert_layers, (double) gpu_bytes / 1048576.0);
        std::printf("INFO expert_slots=%lld experts=%lld expert_cache_mib=%lld expert_layers_gpu=%d expert_layers_total=%zu\n",
                    (long long) gpu_expert_layers * g.n_experts,
                    (long long) expert_layers_total * g.n_experts,
                    (long long) (gpu_expert_bytes >> 20), gpu_expert_layers, expert_layers_total);
        std::fflush(stdout);
        return true;
    }

    // ---------------------------------------------------------------------------------------------- the graph
    // Runs `nt` tokens at positions [n_past, n_past + nt); fills `logits` with the last token's.
    // With `emb` (nt x n_embd floats) the batch is a picture's embeddings instead of tokens: they enter as they are, and the
    // sliding-window layers attend across the whole picture (llama.cpp's non-causal batch); the global layers stay causal.
    bool forward(const int32_t* toks, int nt, int n_past, const float* emb = nullptr) {
        const auto& g = m.geometry;
        const int64_t n_embd = g.n_embd, n_head = g.n_heads, n_kv = n_past + nt;
        const int64_t n_exp = g.n_experts, n_used = g.experts_per_token;
        const float eps = (float) g.rms_epsilon;
        auto t_ph = Clock::now();
        const int ph = nt > 1;   // 0 = a decode step, 1 = a prefill batch
        auto lap = [&](double Phase::* member) { const auto n = Clock::now(); tm.p[ph].*member += ms_since(t_ph); t_ph = n; };
        ggml_context* ctx = fctx;
        ggml_reset(ctx);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphNodes, false);

        ggml_tensor* t_tok = emb ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
        ggml_tensor* t_emb = emb ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, nt) : nullptr;
        ggml_tensor* t_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
        const int64_t nt_pad = ((nt + kMaskPad - 1) / kMaskPad) * kMaskPad;
        // CUDA only has flash-attn kernels for 512-wide heads in the GQA-optimised variants, and those require the
        // key/value span to be a multiple of FATTN_KQ_STRIDE (256) or the op is run on the CPU. Round the global
        // layers' span up to that block size; the extra cells are past the cache's valid rows, so they stay masked
        // out and the kernel trims them with the mask scan.
        const int64_t n_att_full = std::min(((n_kv + kFattnStride - 1) / kFattnStride) * kFattnStride, (int64_t) n_ctx);
        // the sliding-window span is the cache length as it is: rounding it up to the flash-attn block size selects a
        // different kernel variant (the GQA-optimised one), which changes the numbers, so the key/value span stays exact
        const int64_t n_kv_swa = std::min(n_kv, (int64_t) swa_cells);
        // What the allocator's graph plan actually has to keep valid is the *mask* leaf: the key/value tensors are views
        // into the cache and claim no buffer, so the mask's width - which grew one element per decoded token and threw
        // the plan away every step - is the only size that matters. Rounding just that up keeps the plan (the kernels
        // walk the mask through nb[1], so a wider row is read as the same values) and costs nothing.
        const int64_t swa_max = ((swa_cells + kFattnStride - 1) / kFattnStride) * kFattnStride;
        const int64_t n_mask_swa = std::min(((n_kv_swa + kFattnStride - 1) / kFattnStride) * kFattnStride, swa_max);
        ggml_tensor* t_mask_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_att_full, nt_pad);
        ggml_tensor* t_mask_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_mask_swa, nt_pad);
        ggml_tensor* t_out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        for (ggml_tensor* t : {t_tok, t_emb, t_pos, t_mask_full, t_mask_swa, t_out})
            if (t) ggml_set_input(t);

        auto rms = [&](ggml_tensor* x, ggml_tensor* w) {
            x = ggml_rms_norm(ctx, x, eps);
            return w ? ggml_mul(ctx, x, w) : x;
        };
        auto geglu = [&](ggml_tensor* gate, ggml_tensor* up) { return ggml_geglu_split(ctx, gate, up); };
        static const bool debug = std::getenv("STRATA_G4_DEBUG") != nullptr;
        std::vector<ggml_tensor*> probes;   // STRATA_G4_DEBUG: the sum of each is printed, to compare with llama.cpp's
        auto probe = [&](ggml_tensor* t, const char* name, int64_t il) {
            if (!debug) return;
            ggml_set_name(t, (std::string(name) + "-" + std::to_string(il)).c_str());
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
            probes.push_back(t);
        };

        ggml_tensor* inp = t_emb ? t_emb : ggml_scale(ctx, ggml_get_rows(ctx, tok_embd, t_tok), std::sqrt((float) n_embd));

        for (int64_t il = 0; il < g.n_layers; ++il) {
            const LayerW& w = L[(size_t) il];
            const auto& lg = g.layers[(size_t) il];
            const int64_t hd = lg.head_dim, nkv = lg.kv_heads;
            const float base = (float) (lg.sliding_window ? g.rope_base_swa : g.rope_base);
            ggml_tensor* factors = lg.sliding_window ? nullptr : rope_freqs;
            const int rope_mode = GGML_ROPE_TYPE_NEOX;
            const int n_orig = (int) g.context_length;

            ggml_tensor* cur = rms(inp, w.attn_norm);
            ggml_tensor* q = ggml_mul_mat(ctx, w.wq, cur);
            q = ggml_reshape_3d(ctx, q, hd, n_head, nt);
            q = rms(q, w.q_norm);
            q = ggml_rope_ext(ctx, q, t_pos, factors, (int) lg.rope_dim, rope_mode, n_orig, base, 1.0f, 0.0f, 1.0f,
                              32.0f, 1.0f);

            ggml_tensor* k = ggml_mul_mat(ctx, w.wk, cur);
            ggml_tensor* v = w.wv ? ggml_mul_mat(ctx, w.wv, cur) : k;   // no attn_v: V is the K projection
            k = ggml_reshape_3d(ctx, k, hd, nkv, nt);
            v = ggml_reshape_3d(ctx, v, hd, nkv, nt);
            k = rms(k, w.k_norm);
            v = ggml_rms_norm(ctx, v, eps);
            k = ggml_rope_ext(ctx, k, t_pos, factors, (int) lg.rope_dim, rope_mode, n_orig, base, 1.0f, 0.0f, 1.0f,
                              32.0f, 1.0f);

            probe(q, "Qcur_pos", il);
            probe(k, "Kcur_pos", il);
            // the new rows go to cell (position mod cells): two pieces where a ring wraps
            const int cells = lg.sliding_window ? swa_cells : n_ctx;
            auto store = [&](ggml_tensor* src, ggml_tensor* cache) {
                src = ggml_reshape_2d(ctx, src, hd * nkv, nt);
                const int first = (int) std::min<int64_t>(nt, cells - (int64_t) n_past % cells);
                for (int part = 0; part < 2; ++part) {
                    const int len = part ? nt - first : first;
                    if (len <= 0) continue;
                    ggml_tensor* from = ggml_view_2d(ctx, src, hd * nkv, len, src->nb[1], part ? (size_t) first * src->nb[1] : 0);
                    ggml_tensor* to = ggml_view_2d(ctx, cache, hd * nkv, len, cache->nb[1],
                                                   part ? 0 : (size_t) (n_past % cells) * cache->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, from, to));
                }
            };
            store(k, kc[(size_t) il]);
            store(v, vc[(size_t) il]);

            ggml_tensor* qa = ggml_permute(ctx, q, 0, 2, 1, 3);
            const int64_t n_att = lg.sliding_window ? n_kv_swa : n_att_full;
            ggml_tensor* ka = ggml_view_3d(ctx, kc[(size_t) il], hd, n_att, nkv, kc[(size_t) il]->nb[1],
                                           ggml_row_size(kv_type, hd), 0);
            ggml_tensor* va = ggml_view_3d(ctx, vc[(size_t) il], hd, n_att, nkv, vc[(size_t) il]->nb[1],
                                           ggml_row_size(kv_type, hd), 0);
            ggml_tensor* att = ggml_flash_attn_ext(ctx, qa, ka, va, lg.sliding_window ? t_mask_swa : t_mask_full,
                                                   1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(att, GGML_PREC_F32);
            att = ggml_reshape_2d(ctx, att, hd * n_head, nt);
            probe(att, "kqv_out", il);
            att = ggml_mul_mat(ctx, w.wo, att);
            att = rms(att, w.attn_post_norm);
            ggml_tensor* attn_out = ggml_add(ctx, att, inp);
            probe(attn_out, "attn_out", il);

            // the dense branch
            ggml_tensor* mlp = rms(attn_out, w.ffn_norm);
            mlp = ggml_mul_mat(ctx, w.ffn_down, geglu(ggml_mul_mat(ctx, w.ffn_gate, mlp), ggml_mul_mat(ctx, w.ffn_up, mlp)));
            mlp = rms(mlp, w.post_norm_1);
            probe(mlp, "ffn_mlp", il);

            // the routed branch: the router reads rms(attn_out) / sqrt(n_embd) * scale
            ggml_tensor* moe_in = rms(attn_out, w.pre_norm_2);
            ggml_tensor* rin = ggml_rms_norm(ctx, attn_out, eps);
            rin = ggml_scale(ctx, rin, 1.0f / std::sqrt((float) n_embd));
            rin = ggml_mul(ctx, rin, w.router_scale);
            ggml_tensor* probs = ggml_soft_max(ctx, ggml_mul_mat(ctx, w.router, rin));   // [n_exp, nt]
            ggml_tensor* sel = ggml_argsort_top_k(ctx, probs, (int) n_used);              // [n_used, nt]
            ggml_tensor* wts = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_exp, nt), sel);
            wts = ggml_reshape_2d(ctx, wts, n_used, nt);
            wts = ggml_div(ctx, wts, ggml_clamp(ctx, ggml_sum_rows(ctx, wts), 6.103515625e-5f, INFINITY));
            wts = ggml_reshape_3d(ctx, wts, 1, n_used, nt);
            ggml_build_forward_expand(gf, wts);

            ggml_tensor* x3 = ggml_reshape_3d(ctx, moe_in, n_embd, 1, nt);
            ggml_tensor* gu = ggml_mul_mat_id(ctx, w.gate_up_exps, x3, sel);   // [2 * n_ff, n_used, nt]
            const int64_t n_ff = gu->ne[0] / 2;
            ggml_tensor* gate = ggml_view_3d(ctx, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
            ggml_tensor* up = ggml_view_3d(ctx, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], n_ff * gu->nb[0]);
            ggml_tensor* ex = ggml_mul_mat_id(ctx, w.down_exps, geglu(gate, up), sel);   // [n_embd, n_used, nt]
            ggml_tensor* sc = ggml_reshape_3d(ctx, w.down_scale, 1, n_exp, 1);
            sc = ggml_get_rows(ctx, ggml_repeat_4d(ctx, sc, 1, n_exp, nt, 1), sel);
            ex = ggml_mul(ctx, ggml_mul(ctx, ex, sc), wts);
            ggml_build_forward_expand(gf, ex);
            ggml_tensor* moe = ggml_view_2d(ctx, ex, n_embd, nt, ex->nb[2], 0);
            for (int64_t e = 1; e < n_used; ++e)
                moe = ggml_add(ctx, moe, ggml_view_2d(ctx, ex, n_embd, nt, ex->nb[2], (size_t) e * ex->nb[1]));
            moe = rms(moe, w.post_norm_2);
            probe(moe, "ffn_moe", il);

            ggml_tensor* ffn = rms(ggml_add(ctx, mlp, moe), w.ffn_post_norm);
            ffn = ggml_add(ctx, ffn, attn_out);
            inp = ggml_mul(ctx, ffn, w.out_scale);
            probe(inp, "l_out", il);
        }

        ggml_tensor* cur = rms(inp, out_norm);
        cur = ggml_get_rows(ctx, cur, t_out);   // only the last token's row reaches the head
        cur = ggml_mul_mat(ctx, tok_embd, cur);
        if (g.logit_softcap > 0.0) {
            const float cap = (float) g.logit_softcap;
            cur = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, cur, 1.0f / cap)), cap);
        }
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        lap(&Phase::build);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            std::fprintf(stderr, "strata-gemma4: cannot allocate the compute graph\n");
            return false;
        }
        lap(&Phase::alloc);
        std::vector<int32_t> pos((size_t) nt);
        std::iota(pos.begin(), pos.end(), n_past);
        const ggml_fp16_t neg = ggml_fp32_to_fp16(-INFINITY), zero = ggml_fp32_to_fp16(0.0f);
        std::vector<ggml_fp16_t> mfull((size_t) (n_att_full * nt_pad), neg), mswa((size_t) (n_mask_swa * nt_pad), neg);
        const int64_t newest = (int64_t) n_past + nt - 1;
        for (int j = 0; j < nt; ++j) {
            const int64_t qj = (int64_t) n_past + j;
            for (int64_t i = 0; i <= qj; ++i) mfull[(size_t) (j * n_att_full + i)] = zero;
            // cell c holds the newest position congruent to c modulo the ring size; the padding columns past the
            // cache stay masked out (the vector was built with -inf) because no kernel reads past K->ne[1]
            for (int64_t c = 0; c < n_kv_swa; ++c) {
                const int64_t pc = newest - (((newest - c) % swa_cells) + swa_cells) % swa_cells;
                if (pc >= 0 && (emb || pc <= qj) && qj - pc < g.sliding_window) mswa[(size_t) (j * n_mask_swa + c)] = zero;
            }
        }
        const int32_t last = nt - 1;
        lap(&Phase::fill);
        // a decode step scales best with fewer workers than a prefill batch (measured on this 4-core box: 3 threads
        // beat 4 by 1.5-2x per token, while prefill wants every core), so switch the CPU backend per phase
        if (n_threads_dec != n_threads) ggml_backend_cpu_set_n_threads(cpu, nt > 1 ? n_threads : n_threads_dec);
        if (emb) ggml_backend_tensor_set(t_emb, emb, 0, (size_t) nt * (size_t) n_embd * sizeof(float));
        else ggml_backend_tensor_set(t_tok, toks, 0, (size_t) nt * sizeof(int32_t));
        ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
        ggml_backend_tensor_set(t_mask_full, mfull.data(), 0, mfull.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_mask_swa, mswa.data(), 0, mswa.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_out, &last, 0, sizeof(last));
        lap(&Phase::upload);

        const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        lap(&Phase::compute);
        if (ok && debug && nt > 1)
            for (ggml_tensor* t : probes) {
                std::vector<float> v((size_t) ggml_nelements(t));
                ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
                double sum = 0;
                for (float x : v) sum += x;
                std::fprintf(stderr, "dbg %s %.6f\n", ggml_get_name(t), sum);
            }
        if (ok) {
            logits.resize((size_t) g.n_vocab);
            ggml_backend_tensor_get(cur, logits.data(), 0, logits.size() * sizeof(float));
        }
        lap(&Phase::tail);
        if (tm.on) { ++tm.p[ph].n; tm.p[ph].n_tok += nt; }
        return ok;
    }

    // Reads a prompt (text and pictures) after the longest start the cache already holds; leaves the last token's logits
    // in `logits`.
    bool read_segments(const std::vector<Segment>& segs, int64_t& fresh, int64_t& reused,
                       const std::atomic<bool>* stop = nullptr) {
        std::vector<int64_t> keys;
        std::vector<size_t> begin;
        for (const Segment& sg : segs) {
            begin.push_back(keys.size());
            if (sg.embd) for (int i = 0; i < sg.count; ++i) keys.push_back(image_key(sg.key, i));
            else keys.insert(keys.end(), sg.tokens.begin(), sg.tokens.end());
        }
        size_t common = 0;
        while (common < live.size() && common < keys.size() && live[common] == keys[common]) ++common;
        if (common == keys.size()) --common;   // the last token is read again for its logits
        // a picture is read whole: back to its first position
        for (size_t i = 0; i < segs.size(); ++i)
            if (segs[i].embd && common > begin[i] && common < begin[i] + segs[i].size()) { common = begin[i]; break; }
        // a sliding-window ring only takes a short rewind: the cells of the positions before `common` must be intact
        if (swa_cells < n_ctx && live.size() - common > (size_t) (swa_cells - m.geometry.sliding_window)) common = 0;
        live.resize(common);
        reused = (int64_t) common;
        fresh = (int64_t) (keys.size() - common);
        size_t at = common;
        for (size_t i = 0; i < segs.size(); ++i) {
            const Segment& sg = segs[i];
            if (begin[i] + sg.size() <= at) continue;
            if (sg.embd) {
                if (sg.count > kMaxEmbdBatch) return false;
                if (!forward(nullptr, sg.count, (int) begin[i], sg.data.data())) return false;
                live.insert(live.end(), keys.begin() + (long) begin[i], keys.begin() + (long) (begin[i] + sg.size()));
                at = begin[i] + sg.size();
                continue;
            }
            for (size_t off = at - begin[i]; off < sg.tokens.size();) {
                const int nt = (int) std::min<size_t>(kBatch, sg.tokens.size() - off);
                if (!forward(sg.tokens.data() + off, nt, (int) (begin[i] + off))) return false;
                live.insert(live.end(), sg.tokens.begin() + (long) off, sg.tokens.begin() + (long) off + nt);
                off += (size_t) nt;
                at = begin[i] + off;
                if (stop && stop->load() && at < keys.size()) return false;
            }
        }
        return true;
    }
    bool read(const std::vector<int32_t>& ids, int64_t& fresh, int64_t& reused, const std::atomic<bool>* stop = nullptr) {
        Segment sg;
        sg.tokens = ids;
        return read_segments({sg}, fresh, reused, stop);
    }
    bool step(int32_t token) {
        const int32_t one = token;
        if (!forward(&one, 1, (int) live.size())) return false;
        live.push_back(token);
        return true;
    }
};

int32_t argmax(const std::vector<float>& v) {
    return (int32_t) (std::max_element(v.begin(), v.end()) - v.begin());
}

int32_t sample(const std::vector<float>& logits, const strata::qwen35::Sampling& s, const std::vector<int32_t>& hist,
               std::mt19937_64& rng) {
    std::vector<float> penalized;
    const std::vector<float>* candidate = &logits;
    if (s.needs_penalties() && !hist.empty()) {
        penalized = logits;
        strata::qwen35::apply_penalties(penalized, s, hist);
        candidate = &penalized;
    }
    const std::vector<float>& lg = *candidate;
    if (s.temperature <= 0.f) return argmax(lg);
    std::vector<int32_t> idx(lg.size());
    std::iota(idx.begin(), idx.end(), 0);
    const size_t k = s.top_k > 0 ? std::min<size_t>((size_t) s.top_k, lg.size()) : lg.size();
    std::partial_sort(idx.begin(), idx.begin() + (long) k, idx.end(), [&](int32_t a, int32_t b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize(k);
    std::vector<double> p(k);
    const double top = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < k; ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - top) / s.temperature));
    for (double& x : p) x /= sum;
    size_t n = k;
    if (s.min_p > 0.f)
        while (n > 1 && p[n - 1] < (double) s.min_p * p[0]) --n;
    if (s.top_p < 1.f) {
        double acc = 0;
        for (size_t i = 0; i < n; ++i)
            if ((acc += p[i]) >= (double) s.top_p) { n = i + 1; break; }
    }
    std::discrete_distribution<size_t> d(p.begin(), p.begin() + (long) n);
    return idx[d(rng)];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> out;
    while (*s) {
        char* e = nullptr;
        const long v = std::strtol(s, &e, 10);
        if (e == s) { ++s; continue; }
        out.push_back((int32_t) v);
        s = e;
    }
    return out;
}

// A prompt file the server writes ("G4P1"): u32 n_segments, then per segment u32 kind (0 text tokens, 1 embeddings),
// u32 count, [u32 n_embd, u64 key (the picture's)], and the payload (count x i32, or count x n_embd x f32).
bool load_prompt(const char* path, int64_t n_embd, std::vector<Segment>& out, std::string& err) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { err = std::string("cannot open the prompt file ") + path; return false; }
    auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
    char magic[4];
    uint32_t n = 0;
    bool ok = rd(magic, 4) && std::memcmp(magic, "G4P1", 4) == 0 && rd(&n, 4) && n > 0 && n < 100000;
    for (uint32_t i = 0; ok && i < n; ++i) {
        uint32_t kind = 0, count = 0;
        ok = rd(&kind, 4) && rd(&count, 4) && kind <= 1 && count > 0 && count < (1u << 24);
        if (!ok) break;
        Segment sg;
        if (kind == 0) {
            sg.tokens.resize(count);
            ok = rd(sg.tokens.data(), count * sizeof(int32_t));
        } else {
            uint32_t ne = 0;
            ok = rd(&ne, 4) && rd(&sg.key, 8) && (int64_t) ne == n_embd && count <= (uint32_t) kMaxEmbdBatch;
            if (ok) {
                sg.embd = true;
                sg.count = (int) count;
                sg.data.resize((size_t) count * ne);
                ok = rd(sg.data.data(), sg.data.size() * sizeof(float));
            }
        }
        if (ok) out.push_back(std::move(sg));
    }
    std::fclose(f);
    if (!ok) err = "the prompt file is damaged, or a picture is too big";
    return ok;
}

struct Serve {
    Engine& E;
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;

    explicit Serve(Engine& e) : E(e) {}

    void request(const std::string& line) {
        const bool gene = line.rfind("GENE ", 0) == 0;
        if (!gene && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENE <max_new> [key=value ...] <prompt file>\n");
            std::fflush(stdout);
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + (gene ? 5 : 4);
        const long long max_new = std::strtoll(p, &p, 10);
        strata::qwen35::Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st, (size_t) (p - st));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }
        std::vector<Segment> segs;
        if (gene) {
            std::string err;
            while (*p == ' ') ++p;
            if (!load_prompt(p, E.m.geometry.n_embd, segs, err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                return;
            }
        } else {
            Segment sg;
            sg.tokens = parse_ids(p);
            segs.push_back(std::move(sg));
        }
        std::vector<int32_t> ids;   // the text tokens (what the penalties see)
        size_t total = 0;
        for (const Segment& sg : segs) {
            total += sg.size();
            if (!sg.embd) ids.insert(ids.end(), sg.tokens.begin(), sg.tokens.end());
        }
        if (max_new < 1 || total == 0) {
            std::printf("ERR bad request: max_new or the prompt\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) total + max_new > E.n_ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", total, max_new, E.n_ctx);
            std::fflush(stdout);
            return;
        }
        stop.store(false);
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = strata::qwen35::penalty_history_start(ids, false);
        const auto t0 = Clock::now();
        int64_t fresh = 0, reused = 0;
        if (!E.read_segments(segs, fresh, reused, &stop)) {
            E.live.clear();
            std::printf("DONE 0 %zu 0.0 0.0 %s 0 0 %lld 0 0 0 0 0.0 %lld 0\n", total, stop.load() ? "cancel" : "error",
                        (long long) reused, (long long) fresh);
            std::fflush(stdout);
            return;
        }
        const double prompt_ms = ms_since(t0);
        std::fprintf(stderr, "strata-gemma4: prompt %zu tokens = %lld reused + %lld read in %.0f ms (%.2f tok/s)\n",
                     total, (long long) reused, (long long) fresh, prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0);
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        for (;;) {
            const auto s0 = Clock::now();
            const int32_t nx = sample(E.logits, smp, hist, rng);
            E.tm.add_sample(1, ms_since(s0));
            std::printf("T %d\n", nx);
            std::fflush(stdout);
            ++produced;
            hist.push_back(nx);
            if (is_stop(nx)) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (stop.load()) { finish = "cancel"; break; }
            if (!E.step(nx)) { E.live.clear(); finish = "error"; break; }
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %lld 0 0 0 0 0.0 %lld 0\n", produced, total, prompt_ms, decode_ms,
                    finish, (long long) reused, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-gemma4: %lld generated in %.0f ms (%.2f tok/s) (%s)\n", produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, finish);
        E.print_timing(true);
    }

    int run() {
        std::printf("READY %d stop\n", E.n_ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (lines.empty()) return 0;
                line = lines.front();
                lines.pop_front();
            }
            if (!line.empty()) request(line);
        }
    }
};

}  // namespace

int strata::gemma4::run(int argc, char** argv) {
    reexec_with_passive_wait(argc, argv);
    std::string model, tokens, dump;
    Engine::Options opt;
    opt.threads = (int) std::max(1u, std::thread::hardware_concurrency());
    {   // ggml's CPU kernels scale with the physical cores; the extra logical CPUs only add spin and cache traffic
        int tpc = 0;
        const int physical = detect_physical_cores(tpc);
        if (physical > 0) opt.threads = std::min<int>(std::max(1u, std::thread::hardware_concurrency()), physical * std::max(1, tpc));
    }
    int& ctx = opt.ctx;
    int& threads = opt.threads;
    int n_gen = 64;
    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-gemma4: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--max-new" || a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--threads" || a == "--cpu-threads" || a == "--pool-workers") threads = std::atoi(next().c_str());
        else if (a == "--no-gpu") opt.use_gpu = false;
        else if (a == "--kv-type") opt.kv_type = next();
        else if (a == "--gpu-mem") opt.gpu_mem_mib = std::atoll(next().c_str());
        else if (a == "--gpu-reserve") opt.gpu_reserve_mib = std::atoll(next().c_str());
        else if (a == "--offload-ops") opt.op_offload = true;
        else if (a == "--tokens") tokens = next();
        else if (a == "--dump-logits") dump = next();
        else if (a == "--stop-ids") g_stop = parse_ids(next().c_str());
        else if (a == "--serve") serve = true;
        else if (a == "--greedy") {}
        else if (a.rfind("--", 0) == 0) {
            // the callers (strata's own command line, serve/server.py) pass options this engine does not have
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-gemma4: ignoring %s%s%s\n", a.c_str(), val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr, "usage: strata-gemma4 --model GGUF [--max-context 4096] [--threads N] [--no-gpu] [--kv-type q4_0] [--gpu-mem MiB] "
                                 "[--gpu-reserve MiB] (--serve | --tokens IDS [--max-new 64] [--dump-logits F])\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-gemma4: --model is required\n"); return 2; }
    if (ctx < 64 || threads < 1) { std::fprintf(stderr, "strata-gemma4: bad --max-context / --threads\n"); return 2; }
    Engine E;
    std::string err;
    try {
        if (!E.load(model, opt, err)) { std::fprintf(stderr, "strata-gemma4: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-gemma4: %s\n", e.what());
        return 1;
    }
    if (serve) {
        Serve sv(E);
        return sv.run();
    }
    const std::vector<int32_t> ids = parse_ids(tokens.c_str());
    if (ids.empty()) { std::fprintf(stderr, "strata-gemma4: --tokens or --serve is required\n"); return 2; }
    if ((int64_t) ids.size() + n_gen > ctx) { std::fprintf(stderr, "strata-gemma4: prompt + --max-new exceeds --max-context\n"); return 2; }
    int64_t fresh = 0, reused = 0;
    const auto t0 = Clock::now();
    if (!E.read(ids, fresh, reused)) { std::fprintf(stderr, "strata-gemma4: the forward pass failed\n"); return 1; }
    std::fprintf(stderr, "prompt: %zu tokens in %.0f ms\n", ids.size(), ms_since(t0));
    if (!dump.empty()) {
        if (FILE* f = std::fopen(dump.c_str(), "wb")) {
            std::fwrite(E.logits.data(), sizeof(float), E.logits.size(), f);
            std::fclose(f);
        }
    }
    const auto d0 = Clock::now();
    int produced = 0;
    for (; produced < n_gen; ++produced) {
        const int32_t nx = argmax(E.logits);
        std::printf("%s%d", produced ? " " : "", nx);
        std::fflush(stdout);
        if (is_stop(nx) || produced + 1 == n_gen) { ++produced; break; }
        if (!E.step(nx)) { std::fprintf(stderr, "\nstrata-gemma4: the forward pass failed\n"); return 1; }
    }
    std::printf("\n");
    std::fprintf(stderr, "decode: %d tokens in %.0f ms (%.2f tok/s)\n", produced, ms_since(d0),
                 1000.0 * produced / std::max(1.0, ms_since(d0)));
    return 0;
}
