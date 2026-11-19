> [!NOTE]
> **Este repositorio es una bifurcación (fork) de [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT), no el proyecto original.** Es la
> versión construida para un único equipo antiguo: **Intel Core i5-3470 (4 núcleos, sin AVX2) · 16 GB de RAM · NVIDIA Quadro P2200 5 GB** (sm_61,
> con enlace PCIe Gen1 x16), que sirve **Gemma 4 26B-A4B** como servidor compatible con OpenAI en `http://localhost:8088`. Los modelos, los
> instaladores, las guías de hardware y el soporte del original están en **[Niko1221/Strata](https://github.com/Niko1221/Strata)**. Guía completa
> (japonés): **[docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md)**. Esta página solo describe la máquina de esta bifurcación; las guías originales
> siguen en el árbol: [docs/README.md](docs/README.md), [docs/MODELS.md](docs/MODELS.md), [docs/CAPABILITIES.md](docs/CAPABILITIES.md), [docs/INSTALL.md](docs/INSTALL.md), [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Versión de esta bifurcación: Gemma 4 26B-A4B en una Quadro P2200 de 5 GB</h3>
<p align="center">Un PC débil, un servidor local de modelo.</p>

<p align="center">
  <a href="#this-machine"><b>La máquina</b></a> ·
  <a href="#measured-speed"><b>Velocidad medida</b></a> ·
  <a href="#run-it"><b>Arrancarlo</b></a> ·
  <a href="#vs-code"><b>VS Code</b></a> ·
  <a href="#fork-only-files"><b>Solo en el fork</b></a> ·
  <a href="#upstream-strata"><b>Strata original</b></a>
</p>

<p align="center">
  <a href="https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4"><img src="https://img.shields.io/badge/fork-Niko1221%2FStrata-0ab887" alt="forked from Niko1221/Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=upstream+release&color=0ab887" alt="upstream Strata release"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <b>Español</b> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

<a id="this-machine"></a>

## La máquina (lo floja que es)

| Componente | Detalle |
| --- | --- |
| CPU | Intel Core i5-3470 @ 3.20 GHz · 4 núcleos / 4 subprocesos (Ivy Bridge) · **AVX sí, AVX2 no** |
| RAM | 16 GB (15.06 GiB) + 4 GB swap, unos 3 GB siempre en uso |
| Disco | 440 GB SATA SSD (TOSHIBA Q300), 336 GB libres |
| GPU | **NVIDIA Quadro P2200** (GP106GL) · 5,120 MiB VRAM · compute capability 6.1 (sm_61) · 75 W · **el enlace funciona a PCIe Gen1 x16** |
| Sistema | Ubuntu 24.04.4 LTS · NVIDIA driver 580.178.04 · CUDA 12.2 · gcc-12 · CMake 3.28.3 · Python 3.12.3 |
| Motor | Strata 0.1.59.1 compilado aquí (`build-local-p2200/strata`) |
| Pesos | `gemma-4-26B_q4_0-it.gguf` 14.4 GB + `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` 0.81 GB en `~/models/` (nunca commiteado) |

El modelo **no cabe en 5 GB de VRAM ni en 16 GB de RAM**. Funciona porque los expertos MoE se quedan en la CPU (nativos / cuantizados IQ) y solo las capas de atención van a la VRAM (unos 4,7 GB). `tools/gemma4_server.py` inicia `strata` con `--native`; `--no-gpu` va más lento. Uso real: GPU **4.696 / 5.120 MiB** a 34 °C, 11.712 MiB de RAM disponible.

<a id="measured-speed"></a>

## Velocidad medida (esta máquina)

Medido enviando peticiones reales a `http://localhost:8088` (2026-11-18, Strata 0.1.59.1).

| Situación | Medido |
| --- | --- |
| Salida (decode) | **8.3-8.5 tokens/s** |
| Primera lectura de 1.898 tokens de entrada (prefill) | **87.1 s (unos 22 tokens/s)** |
| Petición corta, 41 tokens de entrada | 3,3 s de prompt, 8,5 tokens/s |
| Petición sobre un prefijo de 32.280 tokens ya leído (cache KV) | **0.34-0.37 s** |
| Salida a mitad de una conversación larga | **3.1 tokens/s** |
| Primera lectura de 32.000 tokens de entrada (sin cache) | **unos 4.640 s (77 minutos)** |

Esta bifurcación no hace Strata rápido: hace que funcione en hardware que las guías originales dan por no soportado ([docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)). Las velocidades originales de RTX 5070 / RTX 3060 están en [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) y [docs/CAPABILITIES.md](docs/CAPABILITIES.md), a un orden de magnitud de distancia.

<a id="run-it"></a>

## Arrancarlo

```bash
# Una vez: construir llama.cpp con CUDA para sm_61
git clone https://github.com/ggml-org/llama.cpp ~/llama_cpp
cmake -S ~/llama_cpp -B ~/llama_cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build ~/llama_cpp/build -j4

# Construir este repositorio para el P2200
cmake -S . -B build-local-p2200 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GEMMA4_GGML_DIR=$HOME/llama_cpp -DSTRATA_BUILD_TESTS=OFF
cmake --build build-local-p2200 -j4

./gemma4-server.sh        # MODEL= STRATA= PORT= MAX_CONTEXT= son ajustes opcionales
```

| Dirección | Qué es |
| --- | --- |
| `http://localhost:8088/` | Página de chat |
| `http://localhost:8088/monitor` | Página en vivo con GPU, petición activa, velocidad e historial (`/status` da lo mismo en JSON) |
| `http://localhost:8088/v1/chat/completions` | API compatible con OpenAI (`stream`, `tools`); `/v1/models` responde `gemma-4-26b-a4b` |

**No hay autenticación** y el servidor escucha en `0.0.0.0`, así que úselo solo en una red de confianza; para un servidor solo local arranque `python3 tools/gemma4_server.py` con `--host 127.0.0.1`. Las peticiones se atienden de una en una (un solo cache KV). Las imágenes funcionan cuando `mmproj-gemma-4-26B*.gguf` está junto al archivo del modelo.

<a id="vs-code"></a>

## VS Code

Esta es la entrada que esta máquina usa de verdad en `chatLanguageModels.json` (Ajuste: **Language Models: Chat Language Models** -> *Edit in Settings JSON*). `maxInputTokens` está en **122880**, pero un prompt de 32k en frío tarda más de una hora: **empiece en 12288**. `maxOutputTokens: 8192` puede suponer 15 minutos por respuesta. Las demás máquinas se añaden al mismo archivo como `customendpoint` con `apiType: messages` (`http://<IP de esa máquina>:<puerto>/v1`). La API es compatible con OpenAI, así que también sirven Claude Code, Codex CLI, OpenCode y herramientas parecidas (original: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md), [docs/AI_SETUP.md](docs/AI_SETUP.md)).

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

## Archivos solo de esta bifurcación

[`gemma4-server.sh`](gemma4-server.sh) - el lanzador que usa este equipo (raíz del repositorio).
[`tools/gemma4_server.py`](tools/gemma4_server.py) - el servidor HTTP de Gemma 4: API OpenAI, página de monitor, tokenizador, plantilla de chat. No está en el original.
[`docs/P2200_GEMMA4.md`](docs/P2200_GEMMA4.md) - la guía en japonés de este equipo: entorno, flags de compilación, arranque, solución de problemas.
Código de Gemma 4 en `src/gemma4/` y `include/strata/gemma4/`, más `tools/gemma4_*` y `data/dsv41/`.
`build-local-p2200/` (en gitignore) y los unos 15 GB de pesos en `~/models/` (`/models/` está en gitignore) - nunca commiteado.

<a id="upstream-strata"></a>

## Strata original (app, Terminal, otros modelos)

El código, la documentación y las pruebas del original siguen en este árbol. La aplicación de escritorio Strata, el Terminal y las otras familias de modelos (DeepSeek V4.1-Flash, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B) las publica el original, no esta bifurcación.

- Guías: [docs/README.md](docs/README.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/CAPABILITIES.md](docs/CAPABILITIES.md) · [docs/INSTALL.md](docs/INSTALL.md) · [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) · [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)
- La línea de una sola orden del Terminal original (instala la versión original, no esta compilación):

```bash
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

<sub>Instaladores del original: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg` ([releases/latest](https://github.com/Niko1221/Strata/releases/latest))</sub>

El patrocinio y el contacto **con el autor original** van por [Niko1221/Strata](https://github.com/Niko1221/Strata) ([GitHub Sponsors](https://github.com/sponsors/Niko1221), [Buy Me a Coffee](https://buymeacoffee.com/strataengine)); esta bifurcación no acepta apoyo. Privacidad: [docs/SAFETY.md](docs/SAFETY.md).

Original: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) · Esta bifurcación: [github.com/takuya-sawa/Strata_QUADROP2200_Gemma4](https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4)
