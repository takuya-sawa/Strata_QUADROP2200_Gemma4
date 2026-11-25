# Quadro P2200 で Gemma 4 26B-A4B を動かす (このフォークの実環境)

[Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT) を、次の古いデスクトップマシン用に構築したフォークです。
Strata 本家の使い方・対応環境は [README](../README.md) と [ガイド一覧](README.md) を、古い NVIDIA カードに関する本家の説明は [OLDER_GPUS.md](OLDER_GPUS.md) を参照してください。このページは **このフォークが実際に動いている環境と、その構築・起動手順** だけを書きます。

## 動作環境 (どんな貧弱なマシンか)

| 項目 | このマシン |
| --- | --- |
| CPU | Intel Core i5-3470 @ 3.20 GHz (Ivy Bridge、4 コア / 4 スレッド) |
| CPU 命令セット | AVX あり、**AVX2 なし** (`/proc/cpuinfo` に `avx2` フラグなし) |
| メモリ | 16 GB (15.06 GiB)。稼働中に利用可能約 12 GiB |
| スワップ | 4 GB。常時約 3 GB 使用 (メモリが足りないので常に不足気味) |
| ディスク | SATA SSD 440 GB (TOSHIBA Q300)・`/home` に 336 GB 空き |
| GPU | **NVIDIA Quadro P2200** (GP106GL)・5,120 MiB VRAM・compute capability 6.1 (sm_61)・75 W |
| PCIe | マザーボードは PCIe 3.0 対応だが、**実リンクは Gen1 x16** (カードが Gen1 で動作) |
| OS | Ubuntu 24.04.4 LTS・Linux 7.0.0-42-generic・x86_64 |
| ドライバ | NVIDIA 580.178.04 (sm_61 をサポートする最後の系統。本家は [CUDA 12 エンジンの最適化](OLDER_GPUS.md) を参照) |
| ツールチェーン | CUDA 12.2 (nvcc 12.2.140)・gcc-12 12.4.0・CMake 3.28.3・Python 3.12.3・GNU make (ninja 未導入) |
| Strata | 0.1.59.1 (このツリーの `build-local-p2200/strata`)。本家のリリースは [v0.2.13](https://github.com/Niko1221/Strata/releases/latest) |

モデル本体が **14.4 GB**、画像用の projector が **0.81 GB** で、VRAM は 5 GB、メモリは 16 GB です。
つまり **モデルはメモリにも VRAM にも入りきりません。** この組み合わせで動くのは、MoE のエキスパートを CPU 側 (ネイティブ / IQ 量子化) に置き、attention 部分だけを VRAM に載せる配置だからです。
`strata` は `--native` 付きで起動します (`tools/gemma4_server.py` が付けます)。`--no-gpu` にすると完全に CPU だけになり、さらに遅くなります。

実際の使用量 (稼働中の `/status` から):

- GPU: **4,696 / 5,120 MiB**・温度 34 °C・待機リクエスト 0
- メモリ: 利用可能 **11,712 MiB**・スワップ約 3 GB 使用中
- 累積: リクエスト 11 件・エラー 0 件・入力 195,864 token・出力 572 token・コンテキスト 1,938 / 131,072

## 速度 (このマシンで実測)

当時のエンジン API (`http://localhost:8088`) に実際に投げて測った値です (2026-11-18、Strata 0.1.59.1)。現在のエンジン API は既定で `:8089` です。

| 状況 | 実測 |
| --- | --- |
| 入力 1,898 token (初めて読むプロンプト) の prefill | **87.1 秒 ≈ 22 token/s** |
| 出力 (decode) | **8.3〜8.5 token/s** |
| 入力 41 token の短いリクエスト | プロンプト 3.3 秒・出力 8.5 token/s |
| 一度読んだ 32,280 token を含む後続リクエスト (KV キャッシュ再利用) | プロンプト **0.34〜0.37 秒** |
| 長い会話 continued 時の出力 | **3.1 token/s** |
| 入力 32,000 token を**初めて**読む (キャッシュなし) | **約 4,640 秒 (77 分)** |

からわかる実用上の注意:

- **長いプロンプトを初めて読むのが最大のボトルネック**です。32k を冷えた状態で読むと 1 時間以上かかるため、VS Code などのツールには `maxInputTokens` を **12,288 程度に絞って**渡すのが安全です (下の例はこのマシンで実際に動かしている値が 122,880 で、12,288 を目安としておすすめしています)。
- 同じ会話の続き (先頭のメッセージが同じリクエスト) は KV キャッシュを使い、数十万 token の読み直しでも 1 秒以内になります。
- 同時リクエストは処理できません (KV キャッシュが 1 個)。サーバーは 1 件ずつ処理し、2 件目は 1 件目の終わりを待ちます。他のアプリから同時に叩かないでください。
- 出力 8 token/s は「読んで納得する速度」ではありません。コード生成など長い出力を待つ用途では、体感ではかなり遅いです。

本家の速度表 (RTX 5070 / RTX 3060 など) は [DEEPSEEK41.md](DEEPSEEK41.md) や [CAPABILITIES.md](CAPABILITIES.md) にあります。比較すると、このマシンは **桁で遅い**です。あくまで「動く・試せる・ローカルで完結する」ための構成です。

## 1. ビルド

前提として CUDA 12.2 と、GPU 転送用の llama.cpp が必要になります (Strata の Gemma 4 エンジンは ggml-cuda を通じて VRAM を使い、画像用の補助プログラムもここから作られるため)。

```bash
# llama.cpp を GGML CUDA 付きで 1 度ビルドしておく (sm_61 向け)
git clone https://github.com/ggml-org/llama.cpp ~/llama_cpp
cmake -S ~/llama_cpp -B ~/llama_cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build ~/llama_cpp/build -j4

# このリポジトリを P2200 (sm_61) 向けにビルド
cmake -S . -B build-local-p2200 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GEMMA4_GGML_DIR=$HOME/llama_cpp \
  -DSTRATA_BUILD_TESTS=OFF
cmake --build build-local-p2200 -j4
```

`STRATA_GEMMA4_GGML_DIR` を指定しない場合、Gemma 4 の推論は CPU だけになり、画像入力の `strata-gemma4-vision` も作されません (`cmake/strata_gpu_engine.cmake`)。
ビルドは 4 スレッドではかなり時間がかかります。生成物は `build-local-p2200/strata` (約 97 MB) と `build-local-p2200/strata-gemma4-vision` です。`build*/` は `.gitignore` されているのでコミットしません。

本家のビルドガイド: [SOURCE_INSTALL.md](SOURCE_INSTALL.md)、古いカード向けのフラットの説明は [OLDER_GPUS.md](OLDER_GPUS.md#building-it-yourself)。

## 2. モデルファイル

`~/models/` に置いています (リポジトリには入れません。`/models/` は gitignore 済み)。

| ファイル | 大きさ |
| --- | --- |
| `gemma-4-26B_q4_0-it.gguf` | 14.4 GB |
| `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` | 0.81 GB |

`mmproj-*` はモデルと同じディレクトリにあるのをサーバーが自動で見つけます (`--mmproj` で明示もできます)。見つからない場合画像入力は off になり、画像を送ると 400 が返ります。

## 3. 起動

```bash
./gemma4-server.sh
```

リポジトリ直下の [`gemma4-server.sh`](../gemma4-server.sh) が `tools/gemma4_server.py` を起動します。`build-local-p2200/strata` → `build/strata` → `engine/strata` の順で見つけます。環境変数で変更できます:

```bash
MODEL=~/models/gemma-4-26B_q4_0-it.gguf \
STRATA=build-local-p2200/strata \
PORT=8089 \
MAX_CONTEXT=131072 \
./gemma4-server.sh
```

サーバーが起動すると:

| 場所 | 内容 |
| --- | --- |
| `http://localhost:8089/` | チャットページ |
| `http://127.0.0.1:8088/#/monitor` | Web 画面の Monitor (別途 `./strata-web.sh` を起動) |
| `http://localhost:8089/metrics` | Web 画面の Monitor 用 JSON |
| `http://localhost:8089/status` | サーバーの状態を JSON で |
| `http://localhost:8089/v1/models` | `gemma-4-26b-a4b` (`modalities.input` に `text` と `image` を公言) |
| `http://localhost:8089/v1/chat/completions` | OpenAI 互換 API (`stream`・`tools` 対応) |
| `http://localhost:8089/v1/status` | Strata の Web 画面がエンジンを識別するために使う JSON (`"service": "strata"`・コンテキスト・vision・エンジン版)。本家の `strata` 引擎と同じ形 |
| `http://localhost:8089/health` | 生きているかの確認 (`{"status":"ok","service":"strata", ...}`) |

リクエストに `max_tokens` (または `max_completion_tokens`) を入れない場合、出力は `DEFAULT_MAX_TOKENS = 4096` まで伸びます (古い 512 固定だと Web 画面の回答が途中で切れていました)。空きが足りないときは `max_context - 入力` に自動で絞られます。

**認証はありません。** `0.0.0.0` で待ち受けするので、LAN 上の誰でもこの PC の GPU を使えます。信頼できないネットワーク (グローバル IP、公開クラウド) では起動しないでください。不要なときは `--host 127.0.0.1` を付けてください:

```bash
python3 tools/gemma4_server.py ~/models/gemma-4-26B_q4_0-it.gguf \
  --strata build-local-p2200/strata --host 127.0.0.1 --port 8089
```

`gemma4_server.py` が知らない引数はエンジンにそのまま渡ります (`--no-gpu`、`--kv-type` など)。

## 4. VS Code / ツールからつながぐ

VS Code では `chatLanguageModels.json` (Setting: **Language Models: Chat Language Models** → Edit in Settings JSON) に追加します。`apiKey` は何らかの文字列なら受理されます。

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

このマシンでは `maxInputTokens` を **12288 から始める**ことをおすすめします (上の速度表の通り、長いプロンプトの初回読みが非常に高い)。`maxOutputTokens` も 8192 まで広げると 1 回答に 15 分以上かかることがあります。

OpenAI 互換なので、Claude Code・Codex CLI・OpenCode などからも使えます (本家: [TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md)、[AI_SETUP.md](AI_SETUP.md))。

<a id="web-ui"></a>

## 5. 本家の Web 画面 (Strata アプリの UI)

本家 Strata の Web 画面 [`apps/host/`](../apps/host) を、このフォークの実エンジン (上の `:8089`) に接続して使えます。起動はリポジトリ直下の [`strata-web.sh`](../strata-web.sh) 一つです:

```bash
./strata-web.sh          # → http://127.0.0.1:8088/
```

`PORT=` `ENGINE_PORT=` `MODEL=` `MAX_CONTEXT=` で上書きできます。スクリプトがやっていること:

1. **Node 24 を用意する** — `apps/host` は Node >= 24.12 を要求します。リポジトリに pinned 版 ([`.node/node-v24.22.3-linux-x64.tar.xz`](../.node)) が入っているので、`~/.local/node/` に展開して使います (システムの Node が 24 以上ならそれを使います)。
2. **`dist` (UI のビルド成果物) が無いとき `npm ci && npm run build`** — `node_modules` が無いときも `npm ci`。`apps/host/dist/` と `node_modules/` は gitignore 済みなので、クローン後はここで生成するか、ビルド済みのコピーを置いてください。
3. **`:8089` が空いていれば `gemma4-server.sh` を裏で起動** (起動完了まで最大 240 秒待ち、終了時に一緒に止めます)。使用中の場合は既存のサーバーを使い、`GET /v1/status` が無い古いコードなら警告を出します。
4. **エンジンを「Strata のエンジン」として認識させる** — `STRATA_REAL_ENGINE=1` と `STRATA_SERVER_URL=http://127.0.0.1:8089/v1` を渡します。Web 画面は起動時に `GET /v1/status` を叩き、`"service": "strata"` が返ることでエンジンを識別します (ここが本家 UI とこのフォークのエンジンを繋ぐ要です)。コンソールに `inference · Strata engine` と出れば成功で、`simulated` ならエンジンを見ていません。
5. **会話用のモデル接続を作成** — `~/.strata/data/model-providers.json` (0600) に `custom` / `openai-chat` の接続 `p2200-gemma4` → `http://127.0.0.1:8089/v1` の `gemma-4-26b-a4b` を入れます。既存の旧既定接続 (`:8088/v1`) は移行し、それ以外の接続は変更しません。会話はこの接続経由で `:8089` に届きます。

**初めての起動でやること** (セットアップのウィザードは、この構成ではモデルのダウンロードに進む必要がありません):

- 最初に出る「Strataをセットアップ」のモーダルは **"すぐにアプリを使う"** を押して閉じます ("はじめましょう" は本家のダウンロード流程へ進むボタンです)。
- 会話画面の上部の **モデルピッカー** (「モデルの読み込みなし」) を開き、**API models** の中から **"Gemma 4 26B-A4B"** を選びます。選ぶと "No model is loaded" の案内は消えます。この選択は会話ごとに必要です (新しい会話ではもう一度選ぶ)。
- 会話の本文・履歴はこの PC の `~/.strata/data/chats/` に保存されます (ブラウザ側には保存されないので、再読み込みでも残ります)。

**注意:**

- Web 画面は既定では **loopback 専用**です (`STRATA_BIND=127.0.0.1`)。他の PC (Windows など) から安全に開くには SSH ポートフォワードを使います:
  ```bash
  ssh -L 8088:localhost:8088 owner@192.168.1.10   # ブラウザで http://localhost:8088/
  ```
- LAN に直接公開する場合は、例えば `STRATA_BIND=192.168.1.10 STRATA_WEB_LAN_ORIGIN=http://192.168.1.10:8088 ./strata-web.sh` で起動し、指定した URL から開きます。**認証はなく、LAN 上の利用者にチャット履歴やホストの操作機能が公開されるため、信頼できるネットワークでのみ使用してください。** IP アドレスやポートを変える場合は両方の設定を合わせて変更してください。
- モデルピッカーの "API models" を使わず **ローカルモデル (ダウンロードして読み込む)** を選まないでください。ウェイトのダウンロード (66 GB) だけでなく、本家の読み込み経路は `llama-server` を起動しようとしてこのビルドでは失敗します。エンジンには必ず `:8089` の接続 (上の接続) 経由でつないでください。
- GPU・速度・トークン数・履歴は Web 画面の **Monitor** タブ (`http://127.0.0.1:8088/#/monitor`) で確認できます。エンジンが報告しない項目は `—` で表示されます。
- Web 画面が持つ OpenAI 互換 API (`/v1/...`) は Strata デスクトップの「他のデバイスに公開」設定が必要なため、外部ツール (VS Code など) は引き続き **`:8089` を直接**使ってください (上の「4.」)。
- 本家の Web 画面に関する詳しい説明は [本家のガイド一覧](README.md) (`apps/host` の使い方) を参照してください。このセクションはこのマシン固有の接続方法だけを書きます。

## つまずいたとき

- `strata が見つかりません` → 上の「1. ビルド」を実行するか、`STRATA=` で既存のビルドを指定します。
- `MODEL=/path/to/... ./gemma4-server.sh` → 相対パス `$HOME` の展開されない形で渡しています。絶対パスで書いてください。
- エンジ子が起動しない (`the engine did not start`) → VRAM またはメモリが尽きています。他の GPU 利用プロセス (Strata デスクトップ、別のチャット) を閉じてください。`MAX_CONTEXT=32768` で下げると楽になります。
- 画像が 400 になる → `mmproj-gemma-4-26B*.gguf` がモデルと同じ場所にあるか、`build-local-p2200/strata-gemma4-vision` があるか確認してください。
- 全体的に遅い / スワップが増えていく → まずメモリ不足を疑ってください。`MAX_CONTEXT` を下げ、ブラウザなどを閉じ、`--no-gpu` は使わないでください。
- Web 画面で `inference · simulated` となる → エンジンが見てもらえていません。`curl http://127.0.0.1:8089/v1/status` が `"service":"strata"` を返すか確認し、返らないなら (古いコードで動いているサーバーを止めて) `./strata-web.sh` をもう一度実行してください。モデルピッカーで API の "Gemma 4 26B-A4B" が選ばれているかも確認してください。Monitor 上部の「モデルの読み込みなし」はアプリ管理のローカルモデルに関する表示で、API 接続のエンジン稼働を否定するものではありません。
- Ctrl+C 後に GPU が空かない / `:8089` が塞がったまま → 古い `strata-gemma4` が残っています。`ps -C strata-gemma4` で PID を見て `kill <PID>` してください (GPU を掴んでいるかは `nvidia-smi` で確認できます)。
- Web 画面の `/api/...` が 403 になる → localhost 以外から開く場合は `STRATA_WEB_LAN_ORIGIN` がブラウザの URL のオリジンと一致するか確認してください。既定の安全な設定では loopback または `ssh -L` で開きます。
- トラブルの一般手順: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## このリポジトリにしかないもの

- `tools/gemma4_server.py` — Gemma 4 用 HTTP サーバー (OpenAI 互換 API・モニター用メトリクス・トークナイザ・チャットテンプレート)。本家にはありません。
- `gemma4-server.sh` — この環境の起動スクリプト。
- `strata-web.sh` — 本家の Web 画面 (`apps/host/`) を上のエンジンに繋いで起動するスクリプト (上の「5.」)。
- `apps/host/` — 本家から持ち込んだ Web 画面 (dist は gitignore 済みなので `npm ci && npm run build` で生成するか、ビルド済みのコピーを置いてください)。
- `build-local-p2200/` (gitignore 済み) — P2200 用のビルド。
- [README.md](../README.md) / [README.ja.md](../README.ja.md) — 上の動作環境・実測速度を載せたこのフォーク専用ページ (本家の README は削除しました)。

## 関連ガイド

- [OLDER_GPUS.md](OLDER_GPUS.md) — 古い NVIDIA/AMD カードの本家公式ガイド (sm_61・CUDA 12 エンジン)
- [SOURCE_INSTALL.md](SOURCE_INSTALL.md) — ソースからのビルド
- [TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md) — CLI とアプリ接続 (`strata connect`)
- [MODELS.md](MODELS.md)・[CAPABILITIES.md](CAPABILITIES.md) — モデルと対応機能
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)・[HOW_IT_WORKS.md](HOW_IT_WORKS.md)
