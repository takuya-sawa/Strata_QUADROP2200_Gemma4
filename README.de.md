> [!NOTE]
> **Dieses Repository ist eine Abspaltung (Fork) von [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT), nicht das Projekt selbst.** Es ist
> der Build eines einzelnen alten Desktops: **Intel Core i5-3470 (4 Kerne, kein AVX2) · 16 GB RAM · NVIDIA Quadro P2200 5 GB** (sm_61, Link läuft
> mit PCIe Gen1 x16), der **Gemma 4 26B-A4B** als OpenAI-kompatiblen Server auf `http://localhost:8088` bereitstellt. Modelle, Installateure,
> Hardware-Guides und Support des Originals liegen bei **[Niko1221/Strata](https://github.com/Niko1221/Strata)**. Ausführlicher Leitfaden (Japanisch):
> **[docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md)**. Diese Seite beschreibt nur den Rechner dieses Forks; die Original-Guides liegen weiter im Baum:
> [docs/README.md](docs/README.md), [docs/MODELS.md](docs/MODELS.md), [docs/CAPABILITIES.md](docs/CAPABILITIES.md), [docs/INSTALL.md](docs/INSTALL.md), [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Build dieses Forks: Gemma 4 26B-A4B auf einer Quadro P2200 mit 5 GB</h3>
<p align="center">Ein schwacher PC, ein lokaler Modellserver.</p>

<p align="center">
  <a href="#this-machine"><b>Der Rechner</b></a> ·
  <a href="#measured-speed"><b>Gemessene Geschwindigkeit</b></a> ·
  <a href="#run-it"><b>Starten</b></a> ·
  <a href="#vs-code"><b>VS Code</b></a> ·
  <a href="#fork-only-files"><b>Nur im Fork</b></a> ·
  <a href="#upstream-strata"><b>Original-Strata</b></a>
</p>

<p align="center">
  <a href="https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4"><img src="https://img.shields.io/badge/fork-Niko1221%2FStrata-0ab887" alt="forked from Niko1221/Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=upstream+release&color=0ab887" alt="upstream Strata release"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <b>Deutsch</b> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

<a id="this-machine"></a>

## Der Rechner (wie schwach er ist)

| Bauteil | Angabe |
| --- | --- |
| CPU | Intel Core i5-3470 @ 3.20 GHz · 4 Kerne / 4 Threads (Ivy Bridge) · **AVX ja, AVX2 nein** |
| RAM | 16 GB (15.06 GiB) + 4 GB swap, davon ständig etwa 3 GB in Gebrauch |
| Festplatte | 440 GB SATA SSD (TOSHIBA Q300), 336 GB frei |
| GPU | **NVIDIA Quadro P2200** (GP106GL) · 5,120 MiB VRAM · compute capability 6.1 (sm_61) · 75 W · **Link läuft mit PCIe Gen1 x16** |
| Betriebssystem | Ubuntu 24.04.4 LTS · NVIDIA driver 580.178.04 · CUDA 12.2 · gcc-12 · CMake 3.28.3 · Python 3.12.3 |
| Engine | Strata 0.1.59.1 hier gebaut (`build-local-p2200/strata`) |
| Gewichte | `gemma-4-26B_q4_0-it.gguf` 14.4 GB + `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` 0.81 GB in `~/models/` (nie eingecheckt) |

Das Modell **passt weder in 5 GB VRAM noch in 16 GB RAM**. Es läuft, weil die MoE-Experten auf der CPU bleiben (nativ / IQ-quantisiert) und nur die Attention-Ebenen im VRAM liegen (etwa 4,7 GB). `tools/gemma4_server.py` startet `strata` mit `--native`; `--no-gpu` ist langsamer. Im Betrieb: GPU **4.696 / 5.120 MiB** bei 34 °C, 11.712 MiB RAM frei.

<a id="measured-speed"></a>

## Gemessene Geschwindigkeit (dieser Rechner)

Gemessen mit echten Anfragen an `http://localhost:8088` (2026-11-18, Strata 0.1.59.1).

| Situation | Gemessen |
| --- | --- |
| Ausgabe (decode) | **8.3-8.5 tokens/s** |
| Erstes Lesen von 1.898 Eingabe-Tokens (prefill) | **87.1 s (etwa 22 Token/s)** |
| Kurze Anfrage, 41 Eingabe-Tokens | 3,3 s Prompt, 8,5 Token/s |
| Folgeanfrage über einen schon gelesenen 32.280-Token-Präfix (KV-Cache) | **0.34-0.37 s** |
| Ausgabe mitten in einer langen Unterhaltung | **3.1 tokens/s** |
| Erstes Lesen von 32.000 Eingabe-Tokens (nichts gecacht) | **etwa 4.640 s (77 Minuten)** |

Dieser Fork macht Strata also nicht schnell. Er bringt es auf Hardware zum Laufen, die die Original-Guides als nicht unterstützt behandeln ([docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)). Original-Zahlen für RTX 5070 / RTX 3060 stehen in [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) und [docs/CAPABILITIES.md](docs/CAPABILITIES.md) - Größenordnungen entfernt.

<a id="run-it"></a>

## Starten

```bash
# Einmal: llama.cpp mit CUDA für sm_61 bauen
git clone https://github.com/ggml-org/llama.cpp ~/llama_cpp
cmake -S ~/llama_cpp -B ~/llama_cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build ~/llama_cpp/build -j4

# Dieses Repository für den P2200 bauen
cmake -S . -B build-local-p2200 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GEMMA4_GGML_DIR=$HOME/llama_cpp -DSTRATA_BUILD_TESTS=OFF
cmake --build build-local-p2200 -j4

./gemma4-server.sh        # MODEL= STRATA= PORT= MAX_CONTEXT= sind optionale Werte
```

| Adresse | Was es ist |
| --- | --- |
| `http://localhost:8088/` | Chat-Seite |
| `http://localhost:8088/monitor` | Live-Seite mit GPU, laufender Anfrage, Geschwindigkeit und Verlauf (`/status` liefert dasselbe als JSON) |
| `http://localhost:8088/v1/chat/completions` | OpenAI-kompatible API (`stream`, `tools`); `/v1/models` antwortet `gemma-4-26b-a4b` |

**Es gibt keine Authentifizierung**, der Server lauscht auf `0.0.0.0` - nur in einem vertrauenswürdigen Netz benutzen; für einen rein lokalen Server `python3 tools/gemma4_server.py` mit `--host 127.0.0.1` starten. Anfragen werden nacheinander behandelt (ein einziger KV-Cache). Bilder gehen nur, wenn `mmproj-gemma-4-26B*.gguf` neben der Modelldatei liegt.

<a id="vs-code"></a>

## VS Code

Diesen Eintrag benutzt dieser Rechner wirklich in `chatLanguageModels.json` (Einstellung: **Language Models: Chat Language Models** -> *Edit in Settings JSON*). `maxInputTokens` ist hier auf **122880** gestellt; ein kalter 32k-Prompt dauert über eine Stunde, **fangen Sie also mit 12288 an**. `maxOutputTokens: 8192` kann 15 Minuten für eine Antwort bedeuten. Andere Rechner kommen als `customendpoint` mit `apiType: messages` in dieselbe Datei (`http://<IP des Rechners>:<Port>/v1`). Die API ist OpenAI-kompatibel, daher funktionieren auch Claude Code, Codex CLI, OpenCode und ähnliche Werkzeuge (Original: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md), [docs/AI_SETUP.md](docs/AI_SETUP.md)).

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

## Nur in diesem Fork

[`gemma4-server.sh`](gemma4-server.sh) - der Startbefehl dieses Rechners (Stammverzeichnis des Repositories).
[`tools/gemma4_server.py`](tools/gemma4_server.py) - der HTTP-Server für Gemma 4: OpenAI-API, Live-Seite, Tokenizer, Chat-Vorlage. Nicht im Original.
[`docs/P2200_GEMMA4.md`](docs/P2200_GEMMA4.md) - die japanische Anleitung für diesen Rechner: Umgebung, Bau-Flags, Start, Fehlerbehebung.
Gemma-4-Code in `src/gemma4/` und `include/strata/gemma4/`, dazu `tools/gemma4_*` und `data/dsv41/`.
`build-local-p2200/` (gitignored) und die rund 15 GB Gewichte in `~/models/` (`/models/` ist gitignored) - nie eingecheckt.

<a id="upstream-strata"></a>

## Original-Strata (App, Terminal, andere Modelle)

Der Code, die Dokumente und die Tests des Originals liegen weiter in diesem Baum. Die Strata-Desktop-App, das Terminal und die anderen Modellfamilien (DeepSeek V4.1-Flash, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B) werden vom Original veröffentlicht, nicht von diesem Fork.

- Guides: [docs/README.md](docs/README.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/CAPABILITIES.md](docs/CAPABILITIES.md) · [docs/INSTALL.md](docs/INSTALL.md) · [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) · [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)
- Die Ein-Zeile des Original-Terminals (installiert die Original-Version, nicht diesen Build):

```bash
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

<sub>Installateure des Originals: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg` ([releases/latest](https://github.com/Niko1221/Strata/releases/latest))</sub>

Sponsorings und Kontakt **des Originalautors** laufen über [Niko1221/Strata](https://github.com/Niko1221/Strata) ([GitHub Sponsors](https://github.com/sponsors/Niko1221), [Buy Me a Coffee](https://buymeacoffee.com/strataengine)); dieser Fork nimmt keine Unterstützung an. Datenschutz: [docs/SAFETY.md](docs/SAFETY.md).

Original: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) · Dieser Fork: [github.com/takuya-sawa/Strata_QUADROP2200_Gemma4](https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4)
