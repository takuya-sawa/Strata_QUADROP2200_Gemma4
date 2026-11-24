> [!NOTE]
> **This repository is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT), not the upstream project.**
> **このリポジトリは [Niko1221/Strata](https://github.com/Niko1221/Strata) のフォークです。** It is one old desktop machine's build of
> Strata: **Intel Core i5-3470 (4 cores, no AVX2) · 16 GB RAM · NVIDIA Quadro P2200 5 GB (sm_61, running at PCIe Gen1 x16)**, serving
> **Gemma 4 26B-A4B** as an OpenAI-compatible server on `http://localhost:8088`. Upstream Strata - its models, installers, hardware
> guides and support - is at **[Niko1221/Strata](https://github.com/Niko1221/Strata)**. Japanese: [README.ja.md](README.ja.md), full fork guide:
> **[docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md)**.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Fork build: Gemma 4 26B-A4B on a 5 GB Quadro P2200</h3>
<p align="center">One weak PC, one local model server. 古い PC 1 台で Gemma 4 26B-A4B をローカル起動するフォーク。</p>

<p align="center">
  <a href="#the-machine"><b>The machine</b></a> ·
  <a href="#measured-speed"><b>Measured speed</b></a> ·
  <a href="#run-it"><b>Run it</b></a> ·
  <a href="#vs-code"><b>VS Code</b></a> ·
  <a href="#fork-only-files">Fork files</a> ·
  <a href="#upstream-strata">Upstream Strata</a>
</p>

<p align="center">
  <a href="https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4"><img src="https://img.shields.io/badge/fork-Niko1221%2FStrata-0ab887" alt="forked from Niko1221/Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=upstream+release&color=0ab887" alt="upstream Strata release"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.ja.md">日本語</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**This page describes only this fork's machine.** The upstream guides are still in this tree too: [docs/README.md](docs/README.md) (guide index),
[docs/MODELS.md](docs/MODELS.md), [docs/CAPABILITIES.md](docs/CAPABILITIES.md), [docs/INSTALL.md](docs/INSTALL.md), [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md).

<a id="the-machine"></a>

## The machine (how weak it is)

| | |
| --- | --- |
| CPU | Intel Core i5-3470 @ 3.20 GHz · 4 cores / 4 threads (Ivy Bridge) · **AVX yes, AVX2 no** |
| RAM | 16 GB (15.06 GiB) + 4 GB swap, about 3 GB of it in use all the time |
| Disk | 440 GB SATA SSD (TOSHIBA Q300), 336 GB free in `/home` |
| GPU | **NVIDIA Quadro P2200** (GP106GL) · 5,120 MiB VRAM · compute capability 6.1 (sm_61) · 75 W · **link runs at PCIe Gen1 x16** |
| OS | Ubuntu 24.04.4 LTS · NVIDIA driver 580.178.04 · CUDA 12.2 · gcc-12 · CMake 3.28.3 · Python 3.12.3 |
| Engine | Strata 0.1.59.1 built here (`build-local-p2200/strata`) |
| Weights | `gemma-4-26B_q4_0-it.gguf` 14.4 GB + `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` 0.81 GB in `~/models/` (never committed) |

The model **does not fit in 5 GB of VRAM or in 16 GB of RAM**. It runs anyway because the MoE experts stay on the CPU (native / IQ-quantized)
while only the attention layers sit in VRAM (about 4.7 GB used). `tools/gemma4_server.py` starts `strata` with `--native`; `--no-gpu` makes it
CPU-only and slower. Live usage: GPU **4,696 / 5,120 MiB** at 34 °C, 11,712 MiB of RAM available, 11 requests served with 0 errors.

<a id="measured-speed"></a>

## Measured speed (this machine)

Measured by sending real requests to `http://localhost:8088` (2026-11-18, Strata 0.1.59.1).

| Situation | Measured |
| --- | --- |
| Output (decode) | **8.3-8.5 tokens/s** |
| First read of 1,898 input tokens (prefill) | **87.1 s (about 22 tokens/s)** |
| Short request, 41 input tokens | 3.3 s prompt, 8.5 tokens/s output |
| Follow-up over an already-read 32,280-token prefix (KV cache reuse) | **0.34-0.37 s** |
| Output in the middle of a long conversation | **3.1 tokens/s** |
| First read of 32,000 input tokens (nothing cached) | **about 4,640 s (77 minutes)** |

So this is not a fork that makes Strata fast. It is a fork that makes it **run at all, locally, on hardware the upstream guides treat as
unsupported** (see [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)). Upstream speed numbers for RTX 5070 / RTX 3060 cards are in
[docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) and [docs/CAPABILITIES.md](docs/CAPABILITIES.md) - they differ by an order of magnitude.

<a id="run-it"></a>

## Run it

```bash
# once: build llama.cpp with CUDA for sm_61 (the Gemma 4 engine uses ggml-cuda, and the vision helper is built from it)
git clone https://github.com/ggml-org/llama.cpp ~/llama_cpp
cmake -S ~/llama_cpp -B ~/llama_cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build ~/llama_cpp/build -j4

# build this repository for the P2200
cmake -S . -B build-local-p2200 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GEMMA4_GGML_DIR=$HOME/llama_cpp -DSTRATA_BUILD_TESTS=OFF
cmake --build build-local-p2200 -j4

./gemma4-server.sh        # MODEL= STRATA= PORT= MAX_CONTEXT= are optional overrides
```

| Endpoint | What it is |
| --- | --- |
| `http://localhost:8088/` | chat page |
| `http://localhost:8088/monitor` | live page with GPU, running request, speed and history (`/status` returns the same as JSON) |
| `http://localhost:8088/v1/chat/completions` | OpenAI-compatible API (`stream`, `tools`); `/v1/models` answers `gemma-4-26b-a4b` |
| `http://localhost:8088/v1/status` | JSON the upstream Strata web UI uses to recognise the engine (`"service": "strata"`, context, vision, engine version) |
| `http://localhost:8088/health` | liveness check (`{"status":"ok","service":"strata", ...}`) |

Requests without `max_tokens` (or `max_completion_tokens`) may generate up to `DEFAULT_MAX_TOKENS = 4096` tokens, clamped to the room left in the context.

Pictures work when `mmproj-gemma-4-26B*.gguf` sits next to the model file (otherwise image requests return 400). **There is no
authentication** and the server listens on `0.0.0.0`, so keep it on a trusted network; for a local-only server start
`python3 tools/gemma4_server.py` with `--host 127.0.0.1`. Requests are served one at a time (there is a single KV cache).

<a id="web-ui"></a>

## The upstream web UI

`apps/host/` (the upstream Strata web UI) runs against this fork's engine on `:8088`:

```bash
./strata-web.sh          # → http://127.0.0.1:8791/  (prepares Node 24 and dist when missing)
```

Dismiss the setup modal with "Go straight to the app", then pick **"Gemma 4 26B-A4B"** under API models in the chat model
picker (per chat). `inference · Strata engine` on the console is the healthy state; token stats are on
`http://localhost:8088/monitor`. The UI listens on loopback only — reach it from another machine with
`ssh -L 8791:localhost:8791 <this PC>`. Full steps and caveats: [docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md#web-ui).

<a id="vs-code"></a>

## VS Code

This is the entry this machine actually uses in `chatLanguageModels.json` (Setting: **Language Models: Chat Language Models** → *Edit in Settings JSON*).

```jsonc
{
  "name": "STRATA-Quadro",
  "vendor": "customendpoint",
  "apiKey": "Strata",
  "apiType": "chat-completions",
  "models": [
    {
      "id": "gemma-4-26b-a4b",
      "name": "(Quadro) Gemma 4 26B-A4B",
      "url": "http://localhost:8088/v1",
      "toolCalling": true,
      "vision": true,
      "maxInputTokens": 122880,
      "maxOutputTokens": 8192
    }
  ]
}
```

`maxInputTokens` is set to **122880** here, but a cold 32k prompt takes over an hour on this box, so **start at 12288** and raise it only for
conversations whose prefix is already cached. `maxOutputTokens: 8192` can mean 15 minutes for one answer. Other machines in this setup are added
to the same file as `customendpoint` entries with `"apiType": "messages"` and `http://<that machine's IP>:<port>/v1`. The API is
OpenAI-compatible, so Claude Code, Codex CLI, OpenCode and similar tools work too (upstream: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md),
[docs/AI_SETUP.md](docs/AI_SETUP.md)).

<a id="fork-only-files"></a>

## Fork-only files

- [`gemma4-server.sh`](gemma4-server.sh) - the launcher used on this machine (repository root).
- [`strata-web.sh`](strata-web.sh) - starts the upstream web UI (`apps/host/`) against the engine above; prepares Node 24, `dist` and the chat model connection.
- [`tools/gemma4_server.py`](tools/gemma4_server.py) - the Gemma 4 HTTP server: OpenAI API, monitor page, tokenizer, chat template. Not in upstream.
- [`docs/P2200_GEMMA4.md`](docs/P2200_GEMMA4.md) - the Japanese guide for this machine: environment, build flags, startup, troubleshooting.
- Gemma 4 code in `src/gemma4/` and `include/strata/gemma4/`, plus `tools/gemma4_*` helpers and `data/dsv41/`.
- `build-local-p2200/` (gitignored) and the 15 GB of weights in `~/models/` (`/models/` is gitignored) - never committed.

<a id="upstream-strata"></a>

## Upstream Strata (app, Terminal, other models)

Upstream code, docs and tests are still in this tree. The Strata desktop app, the Terminal and the other model families
(DeepSeek V4.1-Flash, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B) are released by upstream, not by this fork.

- Guides: [docs/README.md](docs/README.md) · [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) · [docs/DETAILS.md](docs/DETAILS.md) · [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) · [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)
- Upstream Terminal one-liner (installs the upstream release, not this fork's build):

```bash
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

<sub>Upstream installers: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg` ([releases/latest](https://github.com/Niko1221/Strata/releases/latest))</sub>

Sponsoring and contacting **the upstream author** happens at [Niko1221/Strata](https://github.com/Niko1221/Strata)
([GitHub Sponsors](https://github.com/sponsors/Niko1221), [Buy Me a Coffee](https://buymeacoffee.com/strataengine)); this fork does not take
support. MIT license, code you can read and change. Privacy: [docs/SAFETY.md](docs/SAFETY.md). Upstream:
[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) · This fork: [github.com/takuya-sawa/Strata_QUADROP2200_Gemma4](https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4).
