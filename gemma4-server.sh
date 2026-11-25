#!/usr/bin/env bash
# Gemma 4 26B-A4B を OpenAI 互換サーバーとして LAN に公開する (http://<このPCのIP>:8089/ ・ 認証なし)。
# tools/gemma4_server.py を、このリポジトリのビルドに対して起動する。設定は環境変数で上書きできる:
#   MODEL=~/models/model.gguf STRATA=build/strata PORT=8089 MAX_CONTEXT=131072 ./gemma4-server.sh
# 画像入力 (mmproj) はモデルと同じ場所の mmproj-gemma-4-26B*.gguf を自動で見つけます。詳しくは docs/P2200_GEMMA4.md。
cd "$(dirname "$0")" || exit 1

SCRIPT_DIR=$PWD
MODEL=${MODEL:-$HOME/models/gemma-4-26B_q4_0-it.gguf}
PORT=${PORT:-8089}
MAX_CONTEXT=${MAX_CONTEXT:-131072}

# STRATA を未指定なら、このツリーにあるビルド探す (P2200 のローカルビルド → 通常の build/ → setup.py の engine/)
if [ -z "$STRATA" ]; then
  for candidate in "$SCRIPT_DIR/build-local-p2200/strata" "$SCRIPT_DIR/build/strata" "$SCRIPT_DIR/engine/strata"; do
    if [ -x "$candidate" ]; then STRATA=$candidate; break; fi
  done
fi

if [ ! -f "$MODEL" ]; then
  echo "モデルが見つかりません: $MODEL"
  echo "MODEL=/path/to/gemma-4-26B_q4_0-it.gguf ./gemma4-server.sh のように指定してください。"
  exit 1
fi
if [ ! -x "$STRATA" ]; then
  echo "strata が見つかりません (ビルドしていません)。ビルド手順は docs/P2200_GEMMA4.md。"
  exit 1
fi
if [ ! -f "$SCRIPT_DIR/tools/gemma4_server.py" ]; then
  echo "tools/gemma4_server.py が見つかりません。"
  exit 1
fi

echo "Gemma 4 サーバーを起動します (GPU を使うので、他の Gemma 4 チャットは閉じてください)。"
echo "strata:  $STRATA"
echo "モデル:  $MODEL"
echo "アクセス: $(hostname -I 2>/dev/null | awk '{print "http://"$1":'"$PORT"'/"}')  モニターは Web 画面の Monitor タブ。終了は Ctrl+C。"
exec python3 "$SCRIPT_DIR/tools/gemma4_server.py" "$MODEL" --strata "$STRATA" --port "$PORT" --max-context "$MAX_CONTEXT"
