> [!NOTE]
> **Este repositório é um fork de [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT), não o projeto original.** É a versão construída para
> um único desktop antigo: **Intel Core i5-3470 (4 núcleos, sem AVX2) · 16 GB de RAM · NVIDIA Quadro P2200 5 GB** (sm_61, com link PCIe Gen1 x16),
> que atende **Gemma 4 26B-A4B** como servidor compatível com OpenAI em `http://localhost:8088`. Modelos, instaladores, guias de hardware e o suporte
> do original estão em **[Niko1221/Strata](https://github.com/Niko1221/Strata)**. Guia completo (japonês): **[docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md)**.
> Esta página descreve só a máquina deste fork; os guias do original continuam na árvore: [docs/README.md](docs/README.md), [docs/MODELS.md](docs/MODELS.md), [docs/CAPABILITIES.md](docs/CAPABILITIES.md), [docs/INSTALL.md](docs/INSTALL.md), [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Versão deste fork: Gemma 4 26B-A4B em uma Quadro P2200 de 5 GB</h3>
<p align="center">Um PC fraco, um servidor local de modelo.</p>

<p align="center">
  <a href="#this-machine"><b>A máquina</b></a> ·
  <a href="#measured-speed"><b>Velocidade medida</b></a> ·
  <a href="#run-it"><b>Executar</b></a> ·
  <a href="#vs-code"><b>VS Code</b></a> ·
  <a href="#fork-only-files"><b>Só no fork</b></a> ·
  <a href="#upstream-strata"><b>Strata original</b></a>
</p>

<p align="center">
  <a href="https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4"><img src="https://img.shields.io/badge/fork-Niko1221%2FStrata-0ab887" alt="forked from Niko1221/Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=upstream+release&color=0ab887" alt="upstream Strata release"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <b>Português</b> · <a href="README.pl.md">Polski</a></sub></p>

<a id="this-machine"></a>

## A máquina (quão fraca ela é)

| Componente | Detalhe |
| --- | --- |
| CPU | Intel Core i5-3470 @ 3.20 GHz · 4 núcleos / 4 threads (Ivy Bridge) · **AVX sim, AVX2 não** |
| RAM | 16 GB (15.06 GiB) + 4 GB swap, cerca de 3 GB sempre em uso |
| Disco | 440 GB SATA SSD (TOSHIBA Q300), 336 GB livres |
| GPU | **NVIDIA Quadro P2200** (GP106GL) · 5,120 MiB VRAM · compute capability 6.1 (sm_61) · 75 W · **o link opera em PCIe Gen1 x16** |
| Sistema | Ubuntu 24.04.4 LTS · NVIDIA driver 580.178.04 · CUDA 12.2 · gcc-12 · CMake 3.28.3 · Python 3.12.3 |
| Motor | Strata 0.1.59.1 compilado aqui (`build-local-p2200/strata`) |
| Pesos | `gemma-4-26B_q4_0-it.gguf` 14.4 GB + `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` 0.81 GB em `~/models/` (nunca commitado) |

O modelo **não cabe em 5 GB de VRAM nem em 16 GB de RAM**. Ele roda porque os especialistas MoE ficam na CPU (nativos / quantizados IQ) e apenas as camadas de atenção ficam na VRAM (cerca de 4,7 GB). `tools/gemma4_server.py` inicia o `strata` com `--native`; `--no-gpu` é ainda mais lento. Em uso real: GPU **4.696 / 5.120 MiB** a 34 °C, 11.712 MiB de RAM disponível.

<a id="measured-speed"></a>

## Velocidade medida (nesta máquina)

Medido enviando pedidos reais a `http://localhost:8088` (2026-11-18, Strata 0.1.59.1).

| Situação | Medido |
| --- | --- |
| Saída (decode) | **8.3-8.5 tokens/s** |
| Primeira leitura de 1.898 tokens de entrada (prefill) | **87.1 s (cerca de 22 tokens/s)** |
| Pedido curto, 41 tokens de entrada | 3,3 s de prompt, 8,5 tokens/s |
| Pedido sobre um prefixo de 32.280 tokens já lido (cache KV) | **0.34-0.37 s** |
| Saída no meio de uma conversa longa | **3.1 tokens/s** |
| Primeira leitura de 32.000 tokens de entrada (nada em cache) | **cerca de 4.640 s (77 minutos)** |

Este fork não deixa o Strata rápido: faz com que ele rode em hardware que os guias do original tratam como não suportado ([docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)). As velocidades do original em RTX 5070 / RTX 3060 estão em [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) e [docs/CAPABILITIES.md](docs/CAPABILITIES.md) - uma ordem de magnitude de diferença.

<a id="run-it"></a>

## Executar

```bash
# Uma vez: construa o llama.cpp com CUDA para sm_61
git clone https://github.com/ggml-org/llama.cpp ~/llama_cpp
cmake -S ~/llama_cpp -B ~/llama_cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build ~/llama_cpp/build -j4

# Construa este repositório para o P2200
cmake -S . -B build-local-p2200 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GEMMA4_GGML_DIR=$HOME/llama_cpp -DSTRATA_BUILD_TESTS=OFF
cmake --build build-local-p2200 -j4

./gemma4-server.sh        # MODEL= STRATA= PORT= MAX_CONTEXT= são valores opcionais
```

| Endereço | O que é |
| --- | --- |
| `http://localhost:8088/` | Página de chat |
| `http://localhost:8088/monitor` | Página ao vivo com GPU, pedido em execução, velocidade e histórico (`/status` dá o mesmo em JSON) |
| `http://localhost:8088/v1/chat/completions` | API compatível com OpenAI (`stream`, `tools`); `/v1/models` responde `gemma-4-26b-a4b` |

**Não há autenticação** e o servidor escuta em `0.0.0.0`, então use só em uma rede confiável; para um servidor apenas local, execute `python3 tools/gemma4_server.py` com `--host 127.0.0.1`. Os pedidos são atendidos um por vez (um único cache KV). Imagens funcionam quando `mmproj-gemma-4-26B*.gguf` fica ao lado do arquivo do modelo.

<a id="vs-code"></a>

## VS Code

Esta é a entrada que esta máquina realmente usa em `chatLanguageModels.json` (Configuração: **Language Models: Chat Language Models** -> *Edit in Settings JSON*). `maxInputTokens` está em **122880**, mas ler 32k a frio leva mais de uma hora: **comece com 12288**. `maxOutputTokens: 8192` pode significar 15 minutos para uma resposta. As outras máquinas entram no mesmo arquivo como `customendpoint` com `apiType: messages` (`http://<IP da máquina>:<porta>/v1`). A API é compatível com OpenAI, então Claude Code, Codex CLI, OpenCode e ferramentas parecidas também funcionam (original: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md), [docs/AI_SETUP.md](docs/AI_SETUP.md)).

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

<a id="fork-only-files"></a>

## Arquivos só deste fork

[`gemma4-server.sh`](gemma4-server.sh) - o lançador usado nesta máquina (raiz do repositório).
[`tools/gemma4_server.py`](tools/gemma4_server.py) - o servidor HTTP do Gemma 4: API OpenAI, página de monitor, tokenizador, template de chat. Não existe no original.
[`docs/P2200_GEMMA4.md`](docs/P2200_GEMMA4.md) - o guia em japonês desta máquina: ambiente, flags de compilação, inicialização, solução de problemas.
Código do Gemma 4 em `src/gemma4/` e `include/strata/gemma4/`, além de `tools/gemma4_*` e `data/dsv41/`.
`build-local-p2200/` (no gitignore) e os cerca de 15 GB de pesos em `~/models/` (`/models/` está no gitignore) - nunca commitado.

<a id="upstream-strata"></a>

## Strata original (app, Terminal, outros modelos)

O código, a documentação e os testes do original continuam nesta árvore. O aplicativo de desktop Strata, o Terminal e as outras famílias de modelos (DeepSeek V4.1-Flash, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B) são publicados pelo original, não por este fork.

- Guias: [docs/README.md](docs/README.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/CAPABILITIES.md](docs/CAPABILITIES.md) · [docs/INSTALL.md](docs/INSTALL.md) · [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) · [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)
- A linha de instalação do Terminal do original (instala a versão original, não esta compilação):

```bash
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

<sub>Instaladores do original: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg` ([releases/latest](https://github.com/Niko1221/Strata/releases/latest))</sub>

Patrocínio e contato **com o autor do original** acontecem em [Niko1221/Strata](https://github.com/Niko1221/Strata) ([GitHub Sponsors](https://github.com/sponsors/Niko1221), [Buy Me a Coffee](https://buymeacoffee.com/strataengine)); este fork não aceita apoio. Privacidade: [docs/SAFETY.md](docs/SAFETY.md).

Original: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) · Esta bifurcação: [github.com/takuya-sawa/Strata_QUADROP2200_Gemma4](https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4)
