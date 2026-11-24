> [!NOTE]
> **このリポジトリは [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT) のフォークであり、本家そのものではありません。** 次の古いデスクトップ
> 1 台向けに構築した Strata です: **Intel Core i5-3470 (4 コア・AVX2 なし)・メモリ 16 GB・NVIDIA Quadro P2200 5 GB (sm_61・実リンク PCIe Gen1 x16)** に、
> **Gemma 4 26B-A4B** を OpenAI 互換サーバー `http://localhost:8088` として動かします。本家 Strata のモデル・インストーラー・対応ハードウェア・サポートは
> **[Niko1221/Strata](https://github.com/Niko1221/Strata)** のものです。詳しくは **[docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md)**。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">このフォークの構成: Quadro P2200 (5 GB) で Gemma 4 26B-A4B</h3>
<p align="center">貧弱な PC 1 台・ローカルなモデルサーバー 1 本。One weak PC, one local model server.</p>

<p align="center">
  <a href="#this-machine"><b>このマシン</b></a> ·
  <a href="#measured-speed"><b>実測した速度</b></a> ·
  <a href="#run-it"><b>起動手順</b></a> ·
  <a href="#vs-code"><b>VS Code</b></a> ·
  <a href="#fork-only-files">このフォークだけのファイル</a> ·
  <a href="#upstream-strata">本家 Strata</a>
</p>

<p align="center">
  <a href="https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4"><img src="https://img.shields.io/badge/fork-Niko1221%2FStrata-0ab887" alt="forked from Niko1221/Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT ライセンス"></a>
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=upstream+release&color=0ab887" alt="本家 Strata のリリース"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <b>日本語</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**このページが書くのは「このフォークの実環境」だけです。** 本家のドキュメントはこのツリーにも残っています: [docs/README.md](docs/README.md) (ガイド一覧)、
[docs/MODELS.md](docs/MODELS.md)、[docs/CAPABILITIES.md](docs/CAPABILITIES.md)、[docs/INSTALL.md](docs/INSTALL.md)、[docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)。

<a id="this-machine"></a>

## このマシン (どんな貧弱な環境か)

| 項目 | このマシン |
| --- | --- |
| CPU | Intel Core i5-3470 @ 3.20 GHz・4 コア / 4 スレッド (Ivy Bridge)・**AVX あり・AVX2 なし** |
| メモリ | 16 GB (15.06 GiB) + スワップ 4 GB (常時約 3 GB 使用中) |
| ディスク | SATA SSD 440 GB (TOSHIBA Q300)・`/home` に 336 GB 空き |
| GPU | **NVIDIA Quadro P2200** (GP106GL)・5,120 MiB VRAM・compute capability 6.1 (sm_61)・75 W・**実リンク PCIe Gen1 x16** |
| OS | Ubuntu 24.04.4 LTS・NVIDIA driver 580.178.04・CUDA 12.2・gcc-12・CMake 3.28.3・Python 3.12.3 |
| エンジン | Strata 0.1.59.1 (このツリーの `build-local-p2200/strata`) |
| モデル | `gemma-4-26B_q4_0-it.gguf` 14.4 GB + `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` 0.81 GB を `~/models/` に (コミットしません) |

モデルは **VRAM 5 GB にも RAM 16 GB にも入りきりません。** 動くのは、MoE のエキスパートを CPU 側 (ネイティブ / IQ 量子化) に置き、attention 部分だけを VRAM (約 4.7 GB) に載せているからです。`tools/gemma4_server.py` は `strata --native` で起動します (`--no-gpu` にすると完全に CPU だけになり、さらに遅くなります)。稼働中の実測: GPU **4,696 / 5,120 MiB**・34 °C・利用可能メモリ 11,712 MiB・リクエスト 11 件・エラー 0 件。

<a id="measured-speed"></a>

## 実測した速度 (このマシン)

`http://localhost:8088` に実際に投げて測った値です (2026-11-18、Strata 0.1.59.1)。

| 状況 | 実測 |
| --- | --- |
| 出力 (decode) | **8.3〜8.5 token/s** |
| 入力 1,898 token を初めて読む (prefill) | **87.1 秒 (約 22 token/s)** |
| 入力 41 token の短いリクエスト | プロンプト 3.3 秒・出力 8.5 token/s |
| 一度読んだ 32,280 token を含む後続リクエスト (KV キャッシュ再利用) | **0.34〜0.37 秒** |
| 長い会話の途中での出力 | **3.1 token/s** |
| 入力 32,000 token を**初めて**読む (キャッシュなし) | **約 4,640 秒 (77 分)** |

つまりこのフォークは Strata を **速くするものではなく、本家のガイドが「対応外」と扱う古いカードで (→ [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md))、動くようにするもの**です。本家の RTX 5070 / RTX 3060 の速度は [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) と [docs/CAPABILITIES.md](docs/CAPABILITIES.md) にあります。桁で違います。

<a id="run-it"></a>

## 起動手順

```bash
# 1 回目だけ: llama.cpp を GGML CUDA 付きでビルド (sm_61 向け。Gemma 4 エンジンは ggml-cuda 経由で VRAM を使い、画像用の補助プログラムもここから作ります)
git clone https://github.com/ggml-org/llama.cpp ~/llama_cpp
cmake -S ~/llama_cpp -B ~/llama_cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build ~/llama_cpp/build -j4

# このリポジトリを P2200 (sm_61) 向けにビルド
cmake -S . -B build-local-p2200 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GEMMA4_GGML_DIR=$HOME/llama_cpp -DSTRATA_BUILD_TESTS=OFF
cmake --build build-local-p2200 -j4

./gemma4-server.sh        # MODEL= STRATA= PORT= MAX_CONTEXT= で上書きできます
```

| 場所 | 内容 |
| --- | --- |
| `http://localhost:8088/` | チャットページ |
| `http://localhost:8088/monitor` | GPU・実行中リクエスト・速度・履歴のライブページ (`/status` は同じ内容を JSON で) |
| `http://localhost:8088/v1/chat/completions` | OpenAI 互換 API (`stream`・`tools` 対応)・`/v1/models` は `gemma-4-26b-a4b` |

画像は `mmproj-gemma-4-26B*.gguf` がモデルと同じ場所にあるときだけ動きます (ないと画像リクエストは 400)。**認証はありません** (`0.0.0.0` で待ち受けます)。信頼できるネットワークだけで使い、ローカルだけにしたいときは `python3 tools/gemma4_server.py` に `--host 127.0.0.1` を付けて起動してください。リクエストは KV キャッシュが 1 個なので 1 件ずつ処理されます。

サーバーは他に `/v1/status` (`"service":"strata"` — 本家の Web 画面がエンジンを識別するために使う) と `/health` も返します。`max_tokens` を入れないリクエストの出力は 4096 まで伸びます。

<a id="web-ui"></a>

## 本家の Web 画面 (アプリの UI)

本家 Strata の Web 画面 (`apps/host/`) を、上の `:8088` のエンジンに繋いで使えます。

```bash
./strata-web.sh          # → http://127.0.0.1:8791/  (Node 24 と dist が無いとき自動で用意します)
```

セットアップのモーダルは **「すぐにアプリを使う」** で閉じ、会話のモデルピッカーの API models から **「Gemma 4 26B-A4B」** を選んでください (会話ごとに選ぶ)。コンソールに `inference · Strata engine` と出るのが正常な状態で、トークン数などの詳細な統計は `http://localhost:8088/monitor` に見えます。loopback 限定なので、他の PC からは `ssh -L 8791:localhost:8791 <このPC>` で開きます。手順と注意は [docs/P2200_GEMMA4.md](docs/P2200_GEMMA4.md#web-ui)。

<a id="vs-code"></a>

## VS Code

このマシンで実際に使っている `chatLanguageModels.json` のエントリです (Setting: **Language Models: Chat Language Models** → *Edit in Settings JSON*)。

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

`maxInputTokens` は実際に **122880** に設定しています。ただし冷えた状態で 32k を読むと 1 時間以上かかるので、**まずは 12288 から始める**ことをおすすめします (先頭のメッセージが同じ続きのリクエストだけが KV キャッシュで速くなります)。`maxOutputTokens: 8192` は 1 回答に 15 分以上かかることがあります。他のマシン (RTX / GTX のサーバー) は同じファイルに `customendpoint`・`"apiType": "messages"` で `http://<そのマシンの IP>:<port>/v1` として追加しています。OpenAI 互換なので Claude Code・Codex CLI・OpenCode などからも使えます (本家: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md)、[docs/AI_SETUP.md](docs/AI_SETUP.md))。

<a id="fork-only-files"></a>

## このフォークだけのファイル

- [`gemma4-server.sh`](gemma4-server.sh) — この環境で使っている起動スクリプト (リポジトリ直下)。
- [`strata-web.sh`](strata-web.sh) — 本家の Web 画面 (`apps/host/`) を上のエンジンに繋いで起動するスクリプト (Node 24・`dist`・接続設定をまとめて用意します)。
- [`tools/gemma4_server.py`](tools/gemma4_server.py) — Gemma 4 用 HTTP サーバー (OpenAI 互換 API・モニタページ・トークナイザ・チャットテンプレート)。本家にはありません。
- [`docs/P2200_GEMMA4.md`](docs/P2200_GEMMA4.md) — このマシンのガイド (動作環境・ビルドフラグ・起動・つまずいたとき)。
- `src/gemma4/` と `include/strata/gemma4/` の Gemma 4 用のコード、`tools/gemma4_*` の補助ツール、`data/dsv41/`。
- `build-local-p2200/` (gitignore 済み) と `~/models/` の約 15 GB のウェイト (`/models/` は gitignore 済み)。コミットしません。

<a id="upstream-strata"></a>

## 本家 Strata (アプリ・Terminal・他のモデル)

本家のコード・ドキュメント・テストはこのツリーに残っています。Strata デスクトップアプリ、Terminal、他のモデル (DeepSeek V4.1-Flash、Qwen3.8-Flash-Next、Qwen3.6-35B-A3B) はこのフォークではなく本家が出しています。

- ガイド: [docs/README.md](docs/README.md) · [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) · [docs/DETAILS.md](docs/DETAILS.md) · [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) · [docs/OLDER_GPUS.md](docs/OLDER_GPUS.md)
- 本家リリースの Terminal を入れる場合 (このフォークの実環境用ではなく、本家 0.2.13 を使う手順です):

```bash
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

<sub>本家のインストーラー: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg` ([releases/latest](https://github.com/Niko1221/Strata/releases/latest))</sub>

本家作者への **支援・スポンサー・お問い合わせは [Niko1221/Strata](https://github.com/Niko1221/Strata) 側** です ([GitHub Sponsors](https://github.com/sponsors/Niko1221)、[Buy Me a Coffee](https://buymeacoffee.com/strataengine))。このフォークは支援を受け付けていません。MIT ライセンスの、読んで変更できるコードです。プライバシー: [docs/SAFETY.md](docs/SAFETY.md)。本家: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) · このフォーク: [github.com/takuya-sawa/Strata_QUADROP2200_Gemma4](https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4)。
