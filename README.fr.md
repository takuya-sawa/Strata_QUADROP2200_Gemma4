> [!NOTE]
> **Ce dépôt est un fork de [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT), ce n'est pas le projet original.** C'est la version
> construite pour un seul vieux PC de bureau : **Intel Core i5-3470 (4 cœurs, pas d'AVX2) · 16 Go de RAM · NVIDIA Quadro P2200 5 Go** (sm_61, lien
> en PCIe Gen1 x16), qui sert **Gemma 4 26B-A4B** comme serveur compatible OpenAI sur `http://localhost:8089`. Les modèles, les installateurs, les
> guides matériels et le support de l'original sont chez **[Niko1221/Strata](https://github.com/Niko1221/Strata)**. Guide complet (japonais) :
> **[docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md)**. Cette page ne décrit que la machine de ce fork ; les guides d'origine restent dans l'arborescence :
> [docs/README.md](docs/README.md), [docs/MODELS.md](docs/MODELS.md), [docs/CAPABILITIES.md](docs/CAPABILITIES.md), [docs/INSTALL.md](docs/INSTALL.md), [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Version de ce fork : Gemma 4 26B-A4B sur une Quadro P2200 de 5 Go</h3>
<p align="center">Un PC faible, un serveur de modèle local.</p>

<p align="center">
  <a href="#this-machine"><b>La machine</b></a> ·
  <a href="#measured-speed"><b>Vitesse mesurée</b></a> ·
  <a href="#run-it"><b>Lancer</b></a> ·
  <a href="#vs-code"><b>VS Code</b></a> ·
  <a href="#fork-only-files"><b>Propre au fork</b></a> ·
  <a href="#upstream-strata"><b>Strata original</b></a>
</p>

<p align="center">
  <a href="https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4"><img src="https://img.shields.io/badge/fork-Niko1221%2FStrata-0ab887" alt="forked from Niko1221/Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=upstream+release&color=0ab887" alt="upstream Strata release"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <b>Français</b> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

<a id="this-machine"></a>

## La machine (à quel point elle est faible)

| Composant | Détail |
| --- | --- |
| CPU | Intel Core i5-3470 @ 3.20 GHz · 4 cœurs / 4 threads (Ivy Bridge) · **AVX oui, AVX2 non** |
| RAM | 16 GB (15.06 GiB) + 4 GB swap, environ 3 GB utilisés en permanence |
| Disque | 440 GB SATA SSD (TOSHIBA Q300), 336 GB libres |
| GPU | **NVIDIA Quadro P2200** (GP106GL) · 5,120 MiB VRAM · compute capability 6.1 (sm_61) · 75 W · **le lien tourne en PCIe Gen1 x16** |
| Système | Ubuntu 24.04.4 LTS · NVIDIA driver 580.178.04 · CUDA 12.2 · gcc-12 · CMake 3.28.3 · Python 3.12.3 |
| Moteur | Strata 0.1.59.1 construé ici (`build-local-p2200/strata`) |
| Poids | `gemma-4-26B_q4_0-it.gguf` 14.4 GB + `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` 0.81 GB dans `~/models/` (jamais commité) |

Le modèle **ne tient ni dans 5 Go de VRAM ni dans 16 Go de RAM**. Il fonctionne quand même parce que les experts MoE restent sur le CPU (natifs / quantifiés IQ) et que seules les couches d'attention vont dans la VRAM (environ 4,7 Go). `tools/gemma4_server.py` démarre `strata` avec `--native` ; `--no-gpu` est plus lent. En conditions réelles : GPU **4 696 / 5 120 MiB** à 34 °C, 11 712 MiB de RAM disponibles.

<a id="measured-speed"></a>

## Vitesse mesurée (cette machine)

Mesuré en envoyant de vraies requêtes à `http://localhost:8089` (2026-11-18, Strata 0.1.59.1).

| Situation | Mesure |
| --- | --- |
| Sortie (decode) | **8.3-8.5 tokens/s** |
| Première lecture de 1 898 jetons d'entrée (prefill) | **87.1 s (environ 22 tokens/s)** |
| Requête courte, 41 jetons d'entrée | 3,3 s de prompt, 8,5 tokens/s |
| Requête sur un préfixe de 32 280 jetons déjà lu (cache KV) | **0.34-0.37 s** |
| Sortie en milieu de conversation longue | **3.1 tokens/s** |
| Première lecture de 32 000 jetons d'entrée (rien en cache) | **environ 4 640 s (77 minutes)** |

Ce fork ne rend pas Strata rapide : il le fait fonctionner sur un matériel que les guides d'origine considèrent hors support ([docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)). Les vitesses d'origine sur RTX 5070 / RTX 3060 sont dans [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) et [docs/CAPABILITIES.md](docs/CAPABILITIES.md) - un ordre de grandeur d'écart.

<a id="run-it"></a>

## Lancer

```bash
# Une fois : construire llama.cpp avec CUDA pour sm_61
git clone https://github.com/ggml-org/llama.cpp ~/llama_cpp
cmake -S ~/llama_cpp -B ~/llama_cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build ~/llama_cpp/build -j4

# Construire ce dépôt pour le P2200
cmake -S . -B build-local-p2200 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GEMMA4_GGML_DIR=$HOME/llama_cpp -DSTRATA_BUILD_TESTS=OFF
cmake --build build-local-p2200 -j4

./gemma4-server.sh        # MODEL= STRATA= PORT= MAX_CONTEXT= sont des valeurs optionnelles
```

| Adresse | Ce que c'est |
| --- | --- |
| `http://localhost:8089/` | Page de chat |
| `http://127.0.0.1:8088/#/monitor` | Moniteur de l'interface web (`./strata-web.sh`) ; métriques du moteur : `http://localhost:8089/metrics` |
| `http://localhost:8089/v1/chat/completions` | API compatible OpenAI (`stream`, `tools`) ; `/v1/models` répond `gemma-4-26b-a4b` |

**Il n'y a pas d'authentification** et le serveur écoute sur `0.0.0.0` : ne l'utilisez que sur un réseau de confiance ; pour un serveur purement local, lancez `python3 tools/gemma4_server.py` avec `--host 127.0.0.1`. Les requêtes sont traitées une par une (un seul cache KV). Les images marchent quand `mmproj-gemma-4-26B*.gguf` est à côté du fichier du modèle.

<a id="vs-code"></a>

## VS Code

C'est l'entrée réellement utilisée sur cette machine dans `chatLanguageModels.json` (Réglage : **Language Models: Chat Language Models** -> *Edit in Settings JSON*). `maxInputTokens` y est réglé sur **122880**, mais un prompt de 32k à froid dépasse l'heure : **commencez à 12288**. `maxOutputTokens: 8192` peut demander 15 minutes pour une réponse. Les autres machines sont ajoutées dans le même fichier en `customendpoint` avec `apiType: messages` (`http://<IP de la machine>:<port>/v1`). L'API est compatible OpenAI, donc Claude Code, Codex CLI, OpenCode et outils similaires fonctionnent aussi (original : [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md), [docs/AI_SETUP.md](docs/AI_SETUP.md)).

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

## Fichiers propres à ce fork

[`gemma4-server.sh`](gemma4-server.sh) - le lanceur utilisé par cette machine (racine du dépôt).
[`tools/gemma4_server.py`](tools/gemma4_server.py) - le serveur HTTP de Gemma 4 : API OpenAI, page de suivi, tokenizer, template de chat. Absent de l'original.
[`docs/P2200_GEMMA4.md`](docs/P2200_GEMMA4.md) - le guide en japonais de cette machine : environnement, flags de build, démarrage, dépannage.
Le code Gemma 4 dans `src/gemma4/` et `include/strata/gemma4/`, plus `tools/gemma4_*` et `data/dsv41/`.
`build-local-p2200/` (gitignored) et les environ 15 GB de poids dans `~/models/` (`/models/` est gitignored) - jamais commité.

<a id="upstream-strata"></a>

## Strata original (application, Terminal, autres modèles)

Le code, la documentation et les tests de l'original sont toujours dans cette arborescence. L'application de bureau Strata, le Terminal et les autres familles de modèles (DeepSeek V4.1-Flash, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B) sont publiés par l'original, pas par ce fork.

- Guides: [docs/README.md](docs/README.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/CAPABILITIES.md](docs/CAPABILITIES.md) · [docs/INSTALL.md](docs/INSTALL.md) · [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) · [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)
- La ligne d'installation du Terminal d'origine (installe la version d'origine, pas cette compilation) :

```bash
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

<sub>Installateurs de l'original: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg` ([releases/latest](https://github.com/Niko1221/Strata/releases/latest))</sub>

Le parrainage et le contact **avec l'auteur original** passent par [Niko1221/Strata](https://github.com/Niko1221/Strata) ([GitHub Sponsors](https://github.com/sponsors/Niko1221), [Buy Me a Coffee](https://buymeacoffee.com/strataengine)) ; ce fork n'accepte pas de soutien. Confidentialité : [docs/SAFETY.md](docs/SAFETY.md).

Original : [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) · Ce fork : [github.com/takuya-sawa/Strata_QUADROP2200_Gemma4](https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4)
