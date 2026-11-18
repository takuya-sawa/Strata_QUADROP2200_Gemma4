// include/strata/gemma4/engine.hpp - the Gemma 4 (GGUF architecture gemma4) engine path as a library: `strata` hands a
// gemma4 model to it at the top of main() (the family switch), and strata-gemma4 wraps the same entry point.
// See src/gemma4/g4_engine.cpp.
#pragma once

namespace strata::gemma4 {

/// Command line: --model / --native GGUF, --max-context N, --threads N, --no-gpu, --gpu-mem MiB, --gpu-reserve MiB, --serve | --tokens IDS [--max-new N]
/// [--stop-ids IDS] [--dump-logits FILE].  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::gemma4
