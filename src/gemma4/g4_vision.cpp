// src/gemma4/g4_vision.cpp - strata-gemma4-vision: one image through Gemma 4's vision tower (llama.cpp's libmtmd, run on the
// CPU so the VRAM stays with the language model), written as the pieces of the prompt it stands for:
//
//     strata-gemma4-vision TEXT.gguf MMPROJ.gguf IMAGE OUT.emb [threads]
//
// OUT.emb ("G4V1"): u32 n_segments, then per segment u32 kind (0: text tokens, 1: embeddings), u32 count, [u32 n_embd for
// kind 1], and the payload (count x i32 token ids, or count x n_embd x f32).  For Gemma 4 the segments are
// `<|image>`, the image's embeddings, `<image|>`; the text model is loaded for its vocabulary only.
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
void put(std::FILE* f, const void* p, size_t n) { std::fwrite(p, 1, n, f); }
void put_u32(std::FILE* f, uint32_t v) { put(f, &v, sizeof v); }

// the text model's embedding width (a vocabulary-only model does not carry it)
int embedding_length(const char* gguf_path) {
    gguf_init_params gp{true, nullptr};
    gguf_context* g = gguf_init_from_file(gguf_path, gp);
    if (!g) return 0;
    int n = 0;
    const int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch >= 0) {
        const std::string key = std::string(gguf_get_val_str(g, arch)) + ".embedding_length";
        const int64_t k = gguf_find_key(g, key.c_str());
        if (k >= 0) n = (int) gguf_get_val_u32(g, k);
    }
    gguf_free(g);
    return n;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: strata-gemma4-vision TEXT.gguf MMPROJ.gguf IMAGE OUT.emb [threads]\n");
        return 2;
    }
    const int threads = argc > 5 ? std::atoi(argv[5]) : 4;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "cannot read the vocabulary of %s\n", argv[1]); return 1; }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = false;
    cp.n_threads = threads;
    cp.warmup = false;
    mtmd_context* ctx = mtmd_init_from_file(argv[2], model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot load the vision projector %s\n", argv[2]); return 1; }
    if (!mtmd_support_vision(ctx)) { std::fprintf(stderr, "%s has no vision tower\n", argv[2]); return 1; }

    mtmd_helper_bitmap_wrapper bm = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default());
    if (!bm.bitmap) { std::fprintf(stderr, "cannot decode the image %s\n", argv[3]); return 1; }

    const std::string text = mtmd_default_marker();
    mtmd_input_text in{text.c_str(), text.size(), false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = {bm.bitmap};
    if (mtmd_tokenize(ctx, chunks, &in, bitmaps, 1) != 0) { std::fprintf(stderr, "cannot prepare the image\n"); return 1; }

    std::FILE* out = std::fopen(argv[4], "wb");
    if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
    put(out, "G4V1", 4);
    const size_t n = mtmd_input_chunks_size(chunks);
    put_u32(out, (uint32_t) n);
    for (size_t i = 0; i < n; ++i) {
        const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t count = 0;
            const llama_token* t = mtmd_input_chunk_get_tokens_text(c, &count);
            put_u32(out, 0);
            put_u32(out, (uint32_t) count);
            put(out, t, count * sizeof(llama_token));
        } else if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            if (mtmd_encode_chunk(ctx, c) != 0) { std::fprintf(stderr, "the vision tower failed\n"); return 1; }
            const size_t count = mtmd_input_chunk_get_n_tokens(c);
            const int n_embd = embedding_length(argv[1]);
            if (n_embd <= 0) { std::fprintf(stderr, "no embedding_length in %s\n", argv[1]); return 1; }
            put_u32(out, 1);
            put_u32(out, (uint32_t) count);
            put_u32(out, (uint32_t) n_embd);
            put(out, mtmd_get_output_embd(ctx), count * (size_t) n_embd * sizeof(float));
        } else {
            std::fprintf(stderr, "an unexpected chunk type\n");
            return 1;
        }
    }
    std::fclose(out);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bm.bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    return 0;
}
