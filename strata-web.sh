#!/usr/bin/env bash
# Strata の Web 画面 (apps/host): http://127.0.0.1:8088/ (認証なし・loopback 限定)。
# 会話のエンジンは :8089 で動きます。
# 設定は環境変数で上書きできる:
#   PORT=8088 ENGINE_PORT=8089 MAX_CONTEXT=131072 STRATA_BIND=127.0.0.1 ./strata-web.sh
# Windows などからは SSH ポートフォワードで: ssh -L 8088:localhost:8088 <このPC>
cd "$(dirname "$0")" || exit 1

SCRIPT_DIR=$PWD
HOST_DIR=$SCRIPT_DIR/apps/host
PORT=${PORT:-8088}
BIND=${STRATA_BIND:-127.0.0.1}
ENGINE_PORT=${ENGINE_PORT:-8089}
MAX_CONTEXT=${MAX_CONTEXT:-131072}
MODEL=${MODEL:-$HOME/models/gemma-4-26B_q4_0-it.gguf}
if [ "$PORT" = "$ENGINE_PORT" ]; then
  echo "Web 画面とエンジンには別々のポートが必要です。" >&2
  exit 1
fi
NODE_VERSION=node-v24.22.3-linux-x64
NODE_HOME=${NODE_HOME:-$HOME/.local/node}

# --- Node.js (apps/host は Node >= 24.12 が必要。リポジトリの .node/ に pinned 版がある) ---
NODE_BIN=$NODE_HOME/$NODE_VERSION/bin/node
if [ ! -x "$NODE_BIN" ]; then
  if command -v node >/dev/null 2>&1 && node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 24 ? 0 : 1)'; then
    NODE_BIN=$(command -v node)
  elif [ -f "$SCRIPT_DIR/.node/$NODE_VERSION.tar.xz" ]; then
    echo "Node.js を展開します ($NODE_VERSION → $NODE_HOME)。"
    mkdir -p "$NODE_HOME" && tar -xJf "$SCRIPT_DIR/.node/$NODE_VERSION.tar.xz" -C "$NODE_HOME" || exit 1
    NODE_BIN=$NODE_HOME/$NODE_VERSION/bin/node
  else
    echo "Node.js 24 が見つかりません。.node/$NODE_VERSION.tar.xz を置くか、Node 24 をインストールしてください。"
    exit 1
  fi
fi
export PATH="$(dirname "$NODE_BIN"):$PATH"

if [ ! -d "$HOST_DIR/dist" ]; then
  echo "Web 画面のビルドが見つかりません。npm run build で作成します。"
  ( cd "$HOST_DIR" && npm ci && npm run build ) || exit 1
fi
if [ ! -d "$HOST_DIR/node_modules" ]; then
  echo "依存パッケージをインストールします (npm ci)。"
  ( cd "$HOST_DIR" && npm ci ) || exit 1
fi

# --- 会話のエンジン (:8089) ---
engine_up() { curl -fsS --max-time 5 "http://127.0.0.1:$ENGINE_PORT/health" >/dev/null 2>&1; }
ENGINE_PID=""
if ! engine_up; then
  echo "Gemma 4 サーバーを起動します (GPU を使うので、他の Gemma 4 チャットは閉じてください)。"
  MODEL="$MODEL" PORT="$ENGINE_PORT" MAX_CONTEXT="$MAX_CONTEXT" "$SCRIPT_DIR/gemma4-server.sh" >"$SCRIPT_DIR/strata-engine.log" 2>&1 &
  ENGINE_PID=$!
  for _ in $(seq 1 120); do engine_up && break; sleep 2; done
  if ! engine_up; then
    echo "Gemma 4 サーバーが起動しませんでした: $SCRIPT_DIR/strata-engine.log"
    kill "$ENGINE_PID" 2>/dev/null
    exit 1
  fi
elif ! curl -fsS --max-time 5 "http://127.0.0.1:$ENGINE_PORT/v1/status" >/dev/null 2>&1; then
  echo "注意: :$ENGINE_PORT のサーバーが古いコードで動いています (GET /v1/status がありません)。"
  echo "Web 画面がエンジンを識別できないときは、そのサーバーを Ctrl+C で止めてこのスクリプトを再び実行してください。"
fi

# --- Web 画面がエンジンを「Strata のエンジン」として認識するための設定 ---
export STRATA_REAL_ENGINE=1
export STRATA_SERVER_URL=http://127.0.0.1:$ENGINE_PORT/v1
export STRATA_NATIVE_REPO=$SCRIPT_DIR
export STRATA_PORT=$PORT
export STRATA_BIND=$BIND
CONNECTION_ID=${CONNECTION_ID:-p2200-gemma4}
MODEL_ID=${MODEL_ID:-gemma-4-26b-a4b}
export STRATA_TASK_REMOTE_MODEL=$CONNECTION_ID:$MODEL_ID

# 会話に選ぶモデルの接続 (~/.strata/data/model-providers.json)。無いときだけ作成します。
PROVIDERS_FILE=$HOME/.strata/data/model-providers.json
if [ ! -f "$PROVIDERS_FILE" ]; then
  mkdir -p "$HOME/.strata/data"
  umask 077
  cat >"$PROVIDERS_FILE" <<JSON
[
  {
    "id": "$CONNECTION_ID",
    "kind": "custom",
    "provider": null,
    "name": "Gemma 4 26B-A4B (Quadro P2200)",
    "baseUrl": "http://127.0.0.1:$ENGINE_PORT/v1",
    "protocol": "openai-chat",
    "apiKey": "",
    "models": [
      { "id": "$MODEL_ID", "name": "Gemma 4 26B-A4B", "contextMax": $MAX_CONTEXT, "maxOutput": 4096, "vision": true, "reasoning": false }
    ]
  }
]
JSON
  chmod 600 "$PROVIDERS_FILE"
  echo "接続を作成しました: $PROVIDERS_FILE (会話のモデルピッカーに '$MODEL_ID' が出ます)"
else
  # 旧ランチャーの既定接続だけを移行し、他の接続や変更済み URL はそのまま残す。
  node - "$PROVIDERS_FILE" "$CONNECTION_ID" "$ENGINE_PORT" <<'NODE' || exit 1
const fs = require('node:fs');
const [file, id, port] = process.argv.slice(2);
const connections = JSON.parse(fs.readFileSync(file, 'utf8'));
if (!Array.isArray(connections)) throw new Error('Invalid model providers file');
const connection = connections.find(c => c.id === id && c.kind === 'custom' &&
  c.protocol === 'openai-chat' && /^http:\/\/(127\.0\.0\.1|localhost):8088\/v1\/?$/.test(c.baseUrl));
if (connection) {
  connection.baseUrl = `http://127.0.0.1:${port}/v1`;
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(connections, null, 2) + '\n', { mode: fs.statSync(file).mode & 0o777, flag: 'wx' });
  fs.renameSync(tmp, file);
  console.log(`旧 Gemma 接続をエンジンのポート ${port} に移行しました。`);
}
NODE
fi

# 終了ときはエンジンに SIGINT (サーバー自身がエンジン子プロセスを片付ける) → 効かなければ SIGTERM/KILL。
# エンジンの子 (strata-gemma4) が GPU を掴んだまま残らないように、残っていたら個別に止めます。
cleanup() {
  [ -z "$ENGINE_PID" ] && return
  kill -INT "$ENGINE_PID" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$ENGINE_PID" 2>/dev/null || break; sleep 0.5; done
  kill -TERM "$ENGINE_PID" 2>/dev/null
  for _ in $(seq 1 10); do kill -0 "$ENGINE_PID" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$ENGINE_PID" 2>/dev/null
  for child in $(ps -o pid= --ppid "$ENGINE_PID" 2>/dev/null); do kill "$child" 2>/dev/null; done
}
trap cleanup EXIT INT TERM

# exec だと trap が消えるので、 node はフォアグラウンドで走らせてCtrl+C を node にそのまま届け、
# 終わったら (EXIT の trap で) 自分が起動したエンジンだけを片付ける。
echo "Strata Web 画面: http://$BIND:$PORT/  (この PC 上のみ。他の PC からは ssh -L $PORT:localhost:$PORT <このPC>)"
echo "会話のモデル:    http://127.0.0.1:$ENGINE_PORT/v1 の $MODEL_ID  (終了は Ctrl+C)"
cd "$HOST_DIR" || exit 1
node server/index.mjs --dist dist
