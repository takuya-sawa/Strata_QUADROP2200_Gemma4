> [!NOTE]
> **To repozytorium jest forkiem [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT), a nie samym projektem.** To kompilacja Straty zrobiona
> na jednym starym desktopie: **Intel Core i5-3470 (4 rdzenie, brak AVX2) · 16 GB RAM · NVIDIA Quadro P2200 5 GB** (sm_61, łącze działa jako PCIe
> Gen1 x16), która podaje **Gemma 4 26B-A4B** jako serwer zgodny z OpenAI na `http://localhost:8089`. Modele, instalatory, opisy sprzętu i wsparcie
> oryginału są w **[Niko1221/Strata](https://github.com/Niko1221/Strata)**. Pełny opis (po japońsku): **[docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md)**.
> Ta strona opisuje tylko maszynę tego forka; opisy oryginału nadal są w tym drzewie: [docs/README.md](docs/README.md), [docs/MODELS.md](docs/MODELS.md), [docs/CAPABILITIES.md](docs/CAPABILITIES.md), [docs/INSTALL.md](docs/INSTALL.md), [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Wersja tego forka: Gemma 4 26B-A4B na Quadro P2200 z 5 GB</h3>
<p align="center">Jeden słaby komputer, jeden lokalny serwer modeli.</p>

<p align="center">
  <a href="#this-machine"><b>Maszyna</b></a> ·
  <a href="#measured-speed"><b>Zmierzona prędkość</b></a> ·
  <a href="#run-it"><b>Uruchomienie</b></a> ·
  <a href="#vs-code"><b>VS Code</b></a> ·
  <a href="#fork-only-files"><b>Tylko w forku</b></a> ·
  <a href="#upstream-strata"><b>Oryginał Strata</b></a>
</p>

<p align="center">
  <a href="https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4"><img src="https://img.shields.io/badge/fork-Niko1221%2FStrata-0ab887" alt="forked from Niko1221/Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=upstream+release&color=0ab887" alt="upstream Strata release"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <b>Polski</b></sub></p>

<a id="this-machine"></a>

## Maszyna (jak jest słaba)

| Element | Parametr |
| --- | --- |
| CPU | Intel Core i5-3470 @ 3.20 GHz · 4 rdzenie / 4 wątki (Ivy Bridge) · **AVX tak, AVX2 nie** |
| RAM | 16 GB (15.06 GiB) + 4 GB swap, ok. 3 GB stale zajęte |
| Dysk | 440 GB SATA SSD (TOSHIBA Q300), 336 GB wolne |
| GPU | **NVIDIA Quadro P2200** (GP106GL) · 5,120 MiB VRAM · compute capability 6.1 (sm_61) · 75 W · **łącze działa jako PCIe Gen1 x16** |
| System | Ubuntu 24.04.4 LTS · NVIDIA driver 580.178.04 · CUDA 12.2 · gcc-12 · CMake 3.28.3 · Python 3.12.3 |
| Silnik | Strata 0.1.59.1 zbudowany tutaj (`build-local-p2200/strata`) |
| Wagi | `gemma-4-26B_q4_0-it.gguf` 14.4 GB + `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` 0.81 GB w `~/models/` (nigdy nie commitowany) |

Model **nie mieści się ani w 5 GB VRAM, ani w 16 GB RAM**. Działa, ponieważ eksperci MoE pozostają po stronie CPU (natywni / kwantyzowani IQ), a warstwy uwagi zajmują VRAM (około 4,7 GB). `tools/gemma4_server.py` uruchamia `strata` z `--native`; `--no-gpu` jest jeszcze wolniejsze. W trakcie pracy: GPU **4696 / 5120 MiB** w 34 °C, 11 712 MiB wolnego RAM.

<a id="measured-speed"></a>

## Zmierzona prędkość (ta maszyna)

Zmierzone przez wysyłanie prawdziwych próśb do `http://localhost:8089` (2026-11-18, Strata 0.1.59.1).

| Sytuacja | Zmierzone |
| --- | --- |
| Wyjście (decode) | **8.3-8.5 tokens/s** |
| Pierwsze wczytanie 1 898 tokenów wejściowych (prefill) | **87.1 s (około 22 tokenów/s)** |
| Krótka prośba, 41 tokeny wejściowe | prompt 3,3 s, 8,5 tokena/s |
| Kolejna prośba na już wczytanym prefiksie 32 280 tokenów (pamięć KV) | **0.34-0.37 s** |
| Wyjście w trakcie długiej rozmowy | **3.1 tokens/s** |
| Pierwsze wczytanie 32 000 tokenów wejściowych (bez pamięci) | **około 4 640 s (77 minut)** |

Ten fork nie czyni Straty szybkiej - sprawia, że działa ona na sprzęcie, który opisy oryginału uznają za nieobsługiwany ([docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)). Prędkości oryginału na RTX 5070 / RTX 3060 są w [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) i [docs/CAPABILITIES.md](docs/CAPABILITIES.md) - różnią się o rząd wielkości.

<a id="run-it"></a>

## Uruchomienie

```bash
# Raz: zbuduj llama.cpp z CUDA dla sm_61
git clone https://github.com/ggml-org/llama.cpp ~/llama_cpp
cmake -S ~/llama_cpp -B ~/llama_cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build ~/llama_cpp/build -j4

# Zbuduj to repozytorium pod P2200
cmake -S . -B build-local-p2200 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GEMMA4_GGML_DIR=$HOME/llama_cpp -DSTRATA_BUILD_TESTS=OFF
cmake --build build-local-p2200 -j4

./gemma4-server.sh        # MODEL= STRATA= PORT= MAX_CONTEXT= to opcjonalne wartości
```

| Adres | Co to jest |
| --- | --- |
| `http://localhost:8089/` | Strona czatu |
| `http://127.0.0.1:8088/#/monitor` | Monitor interfejsu webowego (`./strata-web.sh`); metryki silnika: `http://localhost:8089/metrics` |
| `http://localhost:8089/v1/chat/completions` | API zgodne z OpenAI (`stream`, `tools`); `/v1/models` odpowiada `gemma-4-26b-a4b` |

**Nie ma uwierzytelniania**, serwer nasłuchuje na `0.0.0.0` - używaj tylko w zaufanej sieci; dla serwera lokalnego uruchom `python3 tools/gemma4_server.py` z `--host 127.0.0.1`. Prośby są obsługiwane jedna po drugiej (pojedyncza pamięć KV). Obrazy działają, gdy `mmproj-gemma-4-26B*.gguf` leży obok pliku modelu.

<a id="vs-code"></a>

## VS Code

Tego wpisu ta maszyna naprawdę używa w `chatLanguageModels.json` (ustawienie: **Language Models: Chat Language Models** -> *Edit in Settings JSON*). `maxInputTokens` jest tu ustawione na **122880**, ale zimne wczytanie 32k trwa ponad godzinę, więc **zacznij od 12288**. `maxOutputTokens: 8192` może oznaczać 15 minut na jedną odpowiedź. Pozostałe komputery są dopisane w tym samym pliku jako `customendpoint` z `apiType: messages` (`http://<adres komputera>:<port>/v1`). API jest zgodne z OpenAI, więc działają też Claude Code, Codex CLI, OpenCode i podobne narzędzia (oryginał: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md), [docs/AI_SETUP.md](docs/AI_SETUP.md)).

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

## Pliki tylko tego forka

[`gemma4-server.sh`](gemma4-server.sh) - skrypt startowy używany na tej maszynie (korzeń repozytorium).
[`tools/gemma4_server.py`](tools/gemma4_server.py) - serwer HTTP dla Gemma 4: API OpenAI, strona monitora, tokenizator, szablon czatu. Nie ma go w oryginale.
[`docs/P2200_GEMMA4.md`](docs/P2200_GEMMA4.md) - japoński przewodnik po tej maszynie: środowisko, flagi kompilacji, start, rozwiązywanie problemów.
Kod Gemma 4 w `src/gemma4/` i `include/strata/gemma4/`, ponadto `tools/gemma4_*` i `data/dsv41/`.
`build-local-p2200/` (w gitignore) i około 15 GB wagów w `~/models/` (`/models/` jest w gitignore) - nigdy nie commitowany.

<a id="upstream-strata"></a>

## Oryginał Strata (aplikacja, Terminal, inne modele)

Kod, dokumentacja i testy oryginału nadal są w tym drzewie. Aplikację desktopową Strata, Terminal i pozostałe rodziny modeli (DeepSeek V4.1-Flash, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B) publikuje oryginał, nie ten fork.

- Opisy: [docs/README.md](docs/README.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/CAPABILITIES.md](docs/CAPABILITIES.md) · [docs/INSTALL.md](docs/INSTALL.md) · [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) · [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)
- Jednowierszowa instalacja Terminala oryginału (instaluje wydanie oryginału, nie tę kompilację):

```bash
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

<sub>Instalatory oryginału: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg` ([releases/latest](https://github.com/Niko1221/Strata/releases/latest))</sub>

Wspieranie i kontakt **z autorem oryginału** odbywają się przez [Niko1221/Strata](https://github.com/Niko1221/Strata) ([GitHub Sponsors](https://github.com/sponsors/Niko1221), [Buy Me a Coffee](https://buymeacoffee.com/strataengine)); ten fork nie przyjmuje wsparcia. Prywatność: [docs/SAFETY.md](docs/SAFETY.md).

Oryginał: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) · Ten fork: [github.com/takuya-sawa/Strata_QUADROP2200_Gemma4](https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4)
