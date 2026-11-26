# Upstream engine snapshot

This directory is the `engine/` source tree from Niko1221/Strata v0.2.22,
commit `b28ca95e3e6847f2d5e317994f253b66716deeaa`. Its CMake project reports
engine version 0.1.69. See `LICENSE` for the upstream license.

It is kept separate from this fork's root CMake project and P2200 Gemma 4
runtime. Upstream v0.2.22 has no `strata-gemma4` target; the Gemma 4 code in
its bundled `strata-llama` tree is not this fork's serving executable.

The upstream CUDA source supports Pascal `sm_61` only through its experimental
CUDA 12 build path. This snapshot has not yet been built or tested on the P2200.
