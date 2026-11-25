> [!NOTE]
> **这个仓库是 [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT) 的分支 (fork)，不是本项目本身。** 它是在一台旧的台式机上构建的 Strata:
> **Intel Core i5-3470 (4 核、无 AVX2) · 16 GB 内存 · NVIDIA Quadro P2200 5 GB** (sm_61、实际链路 PCIe Gen1 x16), 把 **Gemma 4 26B-A4B** 作为
> OpenAI 兼容服务器运行在 `http://localhost:8089`。本家的模型、安装程序、硬件指南和支持都在 **[Niko1221/Strata](https://github.com/Niko1221/Strata)**。
> 完整指南 (日语): **[docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md)**。本页面只描述这个分支的机器; 本家指南在这个代码树里也保留着:
> [docs/README.md](docs/README.md), [docs/MODELS.md](docs/MODELS.md), [docs/CAPABILITIES.md](docs/CAPABILITIES.md), [docs/INSTALL.md](docs/INSTALL.md), [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">分支的构成: 在 5 GB 的 Quadro P2200 上运行 Gemma 4 26B-A4B</h3>
<p align="center">一台很弱的电脑, 一个本地模型服务器。</p>

<p align="center">
  <a href="#this-machine"><b>这台机器</b></a> ·
  <a href="#measured-speed"><b>实测速度</b></a> ·
  <a href="#run-it"><b>启动</b></a> ·
  <a href="#vs-code"><b>VS Code</b></a> ·
  <a href="#fork-only-files"><b>分支独有的文件</b></a> ·
  <a href="#upstream-strata"><b>本家 Strata</b></a>
</p>

<p align="center">
  <a href="https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4"><img src="https://img.shields.io/badge/fork-Niko1221%2FStrata-0ab887" alt="forked from Niko1221/Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=upstream+release&color=0ab887" alt="upstream Strata release"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <b>简体中文</b> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

<a id="this-machine"></a>

## 这台机器 (有多弱)

| 部件 | 说明 |
| --- | --- |
| CPU | Intel Core i5-3470 @ 3.20 GHz · 4 核心 / 4 线程 (Ivy Bridge) · **有 AVX、无 AVX2** |
| 内存 | 16 GB (15.06 GiB) + 4 GB swap, 常驻约 3 GB 在使用 |
| 磁盘 | 440 GB SATA SSD (TOSHIBA Q300), 空闲 336 GB |
| GPU | **NVIDIA Quadro P2200** (GP106GL) · 5,120 MiB VRAM · compute capability 6.1 (sm_61) · 75 W · **链路实际运行于 PCIe Gen1 x16** |
| 操作系统 | Ubuntu 24.04.4 LTS · NVIDIA driver 580.178.04 · CUDA 12.2 · gcc-12 · CMake 3.28.3 · Python 3.12.3 |
| 引擎 | Strata 0.1.59.1 在本机构建 (`build-local-p2200/strata`) |
| 权重 | `gemma-4-26B_q4_0-it.gguf` 14.4 GB + `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` 0.81 GB 位于 `~/models/` (从不提交) |

模型 **既放不进 5 GB 显存, 也放不进 16 GB 内存**。它能运行, 是因为 MoE 的专家留在 CPU 上 (原生 / IQ 量化), 只有 attention 部分放在显存里 (约 4.7 GB)。`tools/gemma4_server.py` 用 `--native` 启动 `strata`; `--no-gpu` 会更慢。实际使用: GPU **4,696 / 5,120 MiB**、34 °C、可用内存 11,712 MiB。

<a id="measured-speed"></a>

## 实测速度 (这台机器)

向 `http://localhost:8089` 实际发送请求测出的数值 (2026-11-18、Strata 0.1.59.1)。

| 情况 | 实测 |
| --- | --- |
| 输出 (decode) | **8.3-8.5 tokens/s** |
| 首次读取 1,898 个输入 token (prefill) | **87.1 s (约 22 token/s)** |
| 41 个输入 token 的短请求 | 提示 3.3 秒, 输出 8.5 token/s |
| 再次读取已读过的 32,280 token 前缀 (KV 缓存复用) | **0.34-0.37 s** |
| 长对话中的输出 | **3.1 tokens/s** |
| 首次读取 32,000 个输入 token (无缓存) | **约 4,640 秒 (77 分)** |

所以这个分支不是让 Strata 变快, 而是让它在**本家指南视为不受支持的旧卡**上运行 (见 [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md))。本家 RTX 5070 / RTX 3060 的速度在 [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) 与 [docs/CAPABILITIES.md](docs/CAPABILITIES.md), 相差一个数量级。

<a id="run-it"></a>

## 启动

```bash
# 只做一次: 为 sm_61 构建带 CUDA 的 llama.cpp
git clone https://github.com/ggml-org/llama.cpp ~/llama_cpp
cmake -S ~/llama_cpp -B ~/llama_cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build ~/llama_cpp/build -j4

# 把这个仓库为 P2200 构建
cmake -S . -B build-local-p2200 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GEMMA4_GGML_DIR=$HOME/llama_cpp -DSTRATA_BUILD_TESTS=OFF
cmake --build build-local-p2200 -j4

./gemma4-server.sh        # MODEL= STRATA= PORT= MAX_CONTEXT= 可以覆盖
```

| 地址 | 内容 |
| --- | --- |
| `http://localhost:8089/` | 聊天页面 |
| `http://127.0.0.1:8088/#/monitor` | Web 界面的监视器（用 `./strata-web.sh` 启动）；引擎指标：`http://localhost:8089/metrics` |
| `http://localhost:8089/v1/chat/completions` | OpenAI 兼容 API (`stream`、`tools`); `/v1/models` 返回 `gemma-4-26b-a4b` |

**没有身份验证**, 服务器监听 `0.0.0.0`, 所以只在可信网络中使用; 只想本机可用时给 `python3 tools/gemma4_server.py` 加 `--host 127.0.0.1`。请求一次只处理一个 (只有一个 KV 缓存)。图片只在模型旁边有 `mmproj-gemma-4-26B*.gguf` 时可用。

<a id="vs-code"></a>

## VS Code

这台机器实际使用的 `chatLanguageModels.json` 条目如下 (设置: **Language Models: Chat Language Models** -> *Edit in Settings JSON*)。`maxInputTokens` 实际设成 **122880**, 但冷启动读 32k 要一个多小时, **建议从 12288 开始**; `maxOutputTokens: 8192` 时一个回答可能等 15 分钟以上。其他机器以 `customendpoint` 加在同一文件里 (`apiType: messages`, 地址是 `http://<那台机器的 IP>:<端口>/v1`)。API 兼容 OpenAI, 因此 Claude Code、Codex CLI、OpenCode 等也能用 (本家: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md)、[docs/AI_SETUP.md](docs/AI_SETUP.md))。

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
      "url": "http://localhost:8089/v1",
      "toolCalling": true,
      "vision": true,
      "maxInputTokens": 122880,
      "maxOutputTokens": 8192
    }
  ]
}
```

<a id="fork-only-files"></a>

## 这个分支独有的文件

[`gemma4-server.sh`](gemma4-server.sh) - 这台机器使用的启动脚本 (仓库根目录)。
[`tools/gemma4_server.py`](tools/gemma4_server.py) - Gemma 4 的 HTTP 服务器: OpenAI API、监视页面、分词器、聊天模板。本家没有。
[`docs/P2200_GEMMA4.md`](docs/P2200_GEMMA4.md) - 这台机器的日语指南: 环境、编译参数、启动、故障排查。
`src/gemma4/` 与 `include/strata/gemma4/` 里的 Gemma 4 代码, `tools/gemma4_*` 辅助脚本和 `data/dsv41/`。
`build-local-p2200/` (已 gitignore) 和 `~/models/` 里约 15 GB 的权重 (`/models/` 已 gitignore) - 从不提交。

<a id="upstream-strata"></a>

## 本家 Strata (应用、Terminal、其他模型)

本家的代码、文档和测试仍在这个代码树里。Strata 桌面应用、Terminal 和其他模型家族 (DeepSeek V4.1-Flash、Qwen3.8-Flash-Next、Qwen3.6-35B-A3B) 由本家发布, 不是这个分支。

- 指南: [docs/README.md](docs/README.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/CAPABILITIES.md](docs/CAPABILITIES.md) · [docs/INSTALL.md](docs/INSTALL.md) · [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) · [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)
- 本家 Terminal 的一行安装命令 (装的是本家发布版, 不是这个分支的构建):

```bash
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

<sub>本家安装文件: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg` ([releases/latest](https://github.com/Niko1221/Strata/releases/latest))</sub>

给**本家作者**的支持与联系方式在 [Niko1221/Strata](https://github.com/Niko1221/Strata) ([GitHub Sponsors](https://github.com/sponsors/Niko1221)、[Buy Me a Coffee](https://buymeacoffee.com/strataengine)); 这个分支不接受支持。隐私: [docs/SAFETY.md](docs/SAFETY.md)。

本家: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) · 这个分支: [github.com/takuya-sawa/Strata_QUADROP2200_Gemma4](https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4)
