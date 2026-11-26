# 本家 Strata の Web UI を別の PC に移す

この文書は、別の Linux PC でも本家 Strata の Web UI、チャット、Monitor を使うための手順と、このフォークで行った変更の記録です。2026-11-25 時点の情報です。

**最初に:** このフォークには、本家由来の Web UI のソースと今回の修正がすでに入っています。同じ Gemma 4 構成なら、もう一度本家からコピーする必要はありません。別のモデルやエンジンに移す場合は、そのエンジンの API とハードウェアに合わせた調整・実機確認が必要です。別の PC での動作はまだ確認していません。

## 1. ソースの入手先とバージョン

| 入手先 | 用途 |
| --- | --- |
| [このフォーク](https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4) | 今回の変更を含む。**同じ構成の再現はこちらを推奨** |
| [本家 Strata](https://github.com/Niko1221/Strata) | 本家のソースと最新情報 |
| [本家 v0.2.13 の Web UI](https://github.com/Niko1221/Strata/tree/v0.2.13/apps/host) | 今回使った UI と同じ `0.2.13` 系列のソース |

本家の `v0.2.13` タグには `apps/host/` があります。現在の本家 `main` はディレクトリ構成が異なるため、最新ソースをそのまま同じ手順で上書きしないでください。このフォークの初期取り込みと本家タグは、ファイルによっては差分があります。バージョン番号が同じでも完全に同一とは限りません。

本家の指定バージョンを**参考用に別ディレクトリへ**取得する場合:

```bash
git clone --branch v0.2.13 --single-branch https://github.com/Niko1221/Strata.git ~/Strata-upstream-reference
```

本家のライセンスと配布条件は取得したリポジトリの `LICENSE` を確認してください。本家を取得するだけでは、このフォークのエンジン接続・Monitor 対応は入りません。

## 2. 同じ Gemma 4 構成を新しい Linux PC で使う

この方法は Ubuntu/Linux x86-64、Gemma 4 26B-A4B と、このフォークのエンジンを想定しています。GPU や CPU が異なる場合はビルド設定が変わります。P2200 向けの具体的なエンジンビルドは [P2200_GEMMA4.md](P2200_GEMMA4.md#1-ビルド) を参照してください。`build-local-p2200/strata` を別の CPU/GPU にそのままコピーして動くとは限りません。

### 2.1 取得と準備

```bash
git clone https://github.com/takuya-sawa/Strata_QUADROP2200_Gemma4.git ~/Strata-QUADRO
cd ~/Strata-QUADRO
git log -1 --oneline
```

この手順は修正コミット `a138398` を含む版を前提とします。既存の別リポジトリには上書きせず、新しいディレクトリに取得してください。以下が成功することを確認します。

```bash
git merge-base --is-ancestor a138398 HEAD && echo "Web UI 修正あり"
```

- Node.js **24.12 以上**、npm、Python 3、Git、curl が必要です。Linux x86-64 用 Node 24.22.3 のアーカイブはフォークの `.node/` に含まれ、起動スクリプトが必要に応じて展開します。別の OS / CPU では自分で対応する Node を用意してください。
- モデル GGUF と、画像入力を使う場合の mmproj GGUF は**別途用意**します。Git には含まれません。既定のモデルは `~/models/gemma-4-26B_q4_0-it.gguf` です。
- 対応する GPU ドライバー、エンジンのビルド、必要なら画像用ヘルパーをその PC に用意します。エンジンの探す順序は `build-local-p2200/strata` → `build/strata` → `engine/strata` です。別の場所なら環境変数 `STRATA` に実行ファイルを指定します。
- `apps/host/dist/` と `node_modules/` は Git に含まれません。初回起動時、`strata-web.sh` が必要に応じて `npm ci` と `npm run build` を行います。

### 2.2 起動

```bash
cd ~/Strata-QUADRO
./strata-web.sh
```

別のモデルの**ファイルパス**またはエンジンの場所を指定する例:

```bash
MODEL="$HOME/models/gemma-4-26B_q4_0-it.gguf" \
STRATA="$HOME/Strata-QUADRO/build-local-p2200/strata" \
./strata-web.sh
```

このスクリプトと `gemma4_server.py` は Gemma 4 向けです。`MODEL` を別モデルの GGUF に変えるだけで、別のモデルファミリーに対応するわけではありません。

| 用途 | 既定アドレス |
| --- | --- |
| 本家 Web UI | `http://127.0.0.1:8088/` |
| Monitor | `http://127.0.0.1:8088/#/monitor` |
| Gemma の OpenAI 互換 API | `http://127.0.0.1:8089/v1` |
| エンジン確認 | `http://127.0.0.1:8089/health`、`/v1/status`、`/metrics` |

Web UI とエンジンは別のポートが必要です。`PORT` は Web、`ENGINE_PORT` はエンジン、`MAX_CONTEXT` はコンテキストを変更します。`strata-web.sh` は既に `ENGINE_PORT` の `/health` が応答する場合、そのサーバーを再利用するので、**先にそのポートで正しいエンジンが動いていることを確認**してください。

初回はセットアップで「すぐにアプリを使う」を選び、チャットのモデルピッカーで API モデル **Gemma 4 26B-A4B** を選びます。`~/.strata/data/model-providers.json` がない場合、スクリプトが接続を作成します。既存ファイルがある場合は他の接続を上書きしません。Gemma 接続がない、あるいはポートを変えた場合は Web UI の **設定 → モデル → API モデル** で接続先を確認・追加・修正してください。既存ファイルの内容や API キーを公開・コミットしないでください。

`model-providers.json` に接続が既にある場合、旧ランチャーの既定接続（同じ接続 ID で `localhost` または `127.0.0.1` の `8088/v1`）だけは新しいエンジンポートへ移行します。その他の接続は自動変更されません。

### 2.3 動作確認

```bash
curl -fsS http://127.0.0.1:8088/ -o /dev/null && echo "Web UI OK"
curl -fsS http://127.0.0.1:8089/health
curl -fsS http://127.0.0.1:8089/v1/models
curl -fsS http://127.0.0.1:8089/metrics
```

ブラウザーから短いチャットを送信し、**Monitor** の GPU 情報とリクエスト履歴、**モデル** の Gemma 名、**配信** の API URL とモデル ID を確認します。エンジンが報告しない値は `—` になります。`Ctrl+C` でスクリプトが自分で起動したエンジンを停止します。起動前からあったエンジンは停止しません。

### 2.4 別の PC から Web UI にアクセスする

既定では Web UI は loopback 限定です。SSH サーバーが利用できるなら、アクセスする側で以下を実行し、その側のブラウザーで `http://localhost:8088/` を開きます。`user` と IP は移設先の値に変更します。

```bash
ssh -L 8088:localhost:8088 user@192.168.1.10
```

SSH サーバーが動いていなければ、別途セットアップが必要です。

**LAN に直接公開する場合は認証なしです。** LAN 内の利用者に Web UI のチャット履歴やホストの操作機能も公開されます。信頼できるネットワーク内だけで、リスクを理解したうえで有効にしてください。公開インターネットには出さないでください。

```bash
cd ~/Strata-QUADRO
STRATA_BIND=192.168.1.10 \
STRATA_WEB_LAN_ORIGIN=http://192.168.1.10:8088 \
./strata-web.sh
```

IP は移設先のアドレスに置き換えます。オリジンは実際にブラウザーで開く URL の `http://IP:ポート` と一致させます。ファイアウォールの設定によっては別 PC から届かない場合があります。**エンジン API も既定では `0.0.0.0:8089`、認証なし**です。Web UI を loopback 限定にしても、API は別途ネットワークから到達できる場合があります。

別 PC の OpenAI Chat Completions クライアントからは、Web UI の `8088` ではなくエンジンの `http://<移設先IP>:8089/v1` を使います。モデル ID は `/v1/models` で確認してください。空でないダミー API キーを要求するクライアントもありますが、それはサーバー側の認証にはなりません。

## 3. 別モデル・別エンジンに組み込む場合

本家 Web UI を表示することと、その PC のモデルや Monitor が実際に動くことは別です。このフォークの `gemma4-server.sh` や GGUF を無条件に使わず、**その PC ですでに動いているエンジンを残し**、次を確認してください。

1. エンジンのモデル ID、OpenAI Chat Completions の base URL、認証、対応プロトコルを確認する。まずそのエンジン自身の手順で API を検証する。
2. 既存のプロジェクトをバックアップし、このフォークを別ディレクトリに clone して比較する。`apps/host/` 全体の単純な上書きや、バージョンの異なる本家へのパッチの無検証適用はしない。
3. このフォークの `apps/host/` は Node 24.12 以上で `npm ci`、`npm run build` により `dist/` を生成できる。テスト用プロファイルを使う場合はデータの保存先にも注意する。
4. Web UI を自分のエンジンへ向け、**設定 → モデル**で API 接続を登録する。`STRATA_SERVER_URL` はホストがアクセスできるエンジンの `/v1` URL、`STRATA_REAL_ENGINE=1` は実エンジンを使う指定。UI の API 接続も同じエンジンを指す必要がある。設定後、チャットから対象 API モデルを選ぶ。
5. Strata エンジンとして認識させるには `GET /v1/status` の `service: "strata"` 等が必要。OpenAI 互換チャットだけのサーバーでは、それだけで Strata の状態や Monitor が自動的に動くわけではない。
6. Monitor には本家互換 `GET /metrics` の `engine`、`live`、`requests` 等が必要。速度、トークン、GPU、履歴の各値は実エンジンの報告を使い、未報告値を推測で埋めない。エンジンによって項目やプロトコルを個別に実装する必要がある。
7. このフォークで追加した外部モデル用のモデル画面・左の表示・配信画面には、`STRATA_SERVER_URL` と `STRATA_TASK_REMOTE_MODEL=<保存した接続ID>:<モデルID>`、および互換 `/metrics` が必要。配信画面のアドレスと認証状態には、このフォークの `/metrics` が返す `api` 情報も使う。別エンジンで同じ情報が報告されなければ `Unavailable` と表示される。
8. 実チャット、Monitor の生成中・完了後、別 PC からのアクセス、停止・再起動を、その PC の実環境で確認する。画像入力やツール呼び出しも、それぞれエンジンが対応している場合に限り個別に確認する。

### 別エンジンで Web ホストだけを起動する例

この例では、別途起動済みのエンジンが **同じ PC の** `http://127.0.0.1:8089/v1` にあるものとします。Node 24.12 以上と npm を PATH 上に用意し、アドレスは実際の環境に変更してください。未設定なら、Linux x86-64 で「2.2 起動」を一度実施した後は `export PATH="$HOME/.local/node/node-v24.22.3-linux-x64/bin:$PATH"` でも使用できます。

```bash
cd ~/Strata-QUADRO/apps/host
node --version
npm ci
npm run build
STRATA_REAL_ENGINE=1 \
STRATA_SERVER_URL=http://127.0.0.1:8089/v1 \
STRATA_BIND=127.0.0.1 STRATA_PORT=8088 \
npm run serve
```

`http://127.0.0.1:8088/` を開き、上記のとおり API 接続を登録します。このコマンドはエンジンを起動・停止しません。いったん `Ctrl+C` で Web ホストを止め、接続 ID とモデル ID を確認します。次のコマンドは ID と名前のみ表示し、API キーを表示しません。

```bash
python3 -c 'import json, pathlib; p=pathlib.Path.home()/".strata/data/model-providers.json"; cs=json.loads(p.read_text()); [print("接続ID:", c["id"], "名前:", c["name"], "モデルID:", ", ".join(m["id"] for m in c.get("models", []))) for c in cs]'
```

次回からは対象の接続 ID とモデル ID を以下に代入して Web ホストを起動します。チャット自体は互換 API で動く場合がありますが、実モデルの状態や Monitor を正しく表示するには前述の追加対応が必要です。

```bash
STRATA_REAL_ENGINE=1 \
STRATA_SERVER_URL=http://127.0.0.1:8089/v1 \
STRATA_TASK_REMOTE_MODEL='<接続ID>:<モデルID>' \
STRATA_BIND=127.0.0.1 STRATA_PORT=8088 \
npm run serve
```

## 4. 更新と困ったとき

- **コードを更新したが画面が変わらない:** `dist/` があるとランチャーは自動で再ビルドしません。Node を PATH に通し、`cd ~/Strata-QUADRO/apps/host && npm run build` を実行してから Web を再起動します。依存が未導入または lockfile が変わった場合は先に `npm ci` を実行します。
- **別のサービスを再利用してしまう:** `curl -fsS http://127.0.0.1:8089/v1/status` と `/v1/models` でサービス・モデルを確認します。意図しないサービスなら、その管理者と確認して適切に停止するか、空いている `ENGINE_PORT` を使います。
- **チャットのモデルが見つからない:** 設定の API 接続の URL・モデル ID と、エンジンの `/v1/models` を照合します。既存の設定ファイルがあると、ランチャーは通常、新しい接続を追加しません。
- **Monitor が使えない:** `curl -fsS http://127.0.0.1:8089/metrics` の結果とホストのログを確認します。互換形式の `/metrics` がなければ別エンジン側の実装が必要です。
- **別 PC から開けない:** Web の bind、ブラウザーの URL と `STRATA_WEB_LAN_ORIGIN`、ファイアウォールを確認します。8088 は Web、8089 は API です。認証はどちらにもありません。

## 5. 今回の変更箇所

このフォークで実装済みの内容を調べるときの主なファイルです。

- [strata-web.sh](../strata-web.sh): Web とエンジンの起動、ポート、モデル接続の作成。
- [tools/gemma4_server.py](../tools/gemma4_server.py): Gemma 用 `/v1/status`、`/metrics`、API と生成統計。
- [apps/host/server/monitor.mjs](../apps/host/server/monitor.mjs)、[monitor-dashboard.mjs](../apps/host/server/monitor-dashboard.mjs): 外部メトリクスの取得と本家 Monitor への変換。
- [apps/host/src/screens/ModelRun.tsx](../apps/host/src/screens/ModelRun.tsx)、[Serving.tsx](../apps/host/src/screens/Serving.tsx)、[Sidebar.tsx](../apps/host/src/app/Sidebar.tsx): 実モデルと API の表示。
- [apps/host/server/index.mjs](../apps/host/server/index.mjs): LAN 公開を明示した場合のオリジン設定。既定は loopback。
- [apps/host/tests/external-monitor.test.mjs](../apps/host/tests/external-monitor.test.mjs)、[tools/test_gemma4_server.py](../tools/test_gemma4_server.py): 外部 Monitor と Gemma API のテスト。

変更履歴はフォーク内で確認できます。`3f8d3b4` はこのフォークの初期取り込みであり、本家 `v0.2.13` そのものではありません。

```bash
git log --oneline -- apps/host tools/gemma4_server.py strata-web.sh
git diff 3f8d3b4 a138398 -- apps/host tools/gemma4_server.py strata-web.sh gemma4-server.sh
```

主なコミットは `eb55a5d`（エンジン接続）、`97128ac`（Monitor とポート）、`e9bb488`（モデル表示と LAN）、`a138398`（配信画面と左のモデル表示）です。これらを別の構成に適用するときは差分を読み、衝突と依存関係を確認してから移植してください。
