#!/bin/bash
# engine-hy4/run.sh DATA [--port 8080] [--host 127.0.0.1] [--ctx 65536] [--slots 7700]: the Hy4 flash engine on a Unix socket in
# DATA and the OpenAI-compatible wrapper on http://HOST:PORT/v1 (chat completions with stream, tools and reasoning_content,
# /health, /v1/models). Ctrl+C or SIGTERM stops both. Loading takes ~4 minutes (53 GiB of experts into page-locked RAM), then the
# Engram table is read once into the page cache (~2 minutes; HY4_PREREAD=0 skips it). Tested: --ctx 65536 with --slots 7700 and --ctx 32768
# with --slots 8100 (more slots, fewer copies over PCIe). Engine switches (HY4_*) are in docs/HY4.md.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: bash engine-hy4/run.sh DATA [--port N] [--host H] [--ctx C] [--slots S]" >&2; exit 1; }
DATA=$(cd "$1" && pwd)
shift
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=8080 HOST=127.0.0.1 CTX=65536 SLOTS=7700
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --ctx) CTX=$2; shift 2 ;;
    --slots) SLOTS=$2; shift 2 ;;
    *) echo "run.sh: unknown argument $1" >&2; exit 1 ;;
  esac
done
[ -f "$DATA/hy4.env" ] || { echo "run.sh: no $DATA/hy4.env, run setup.sh $DATA first" >&2; exit 1; }
set -a
. "$DATA/hy4.env"
set +a
BIN=$HERE/build/hy4_v1
[ -x "$BIN" ] || { echo "run.sh: no $BIN, run engine-hy4/build.sh first" >&2; exit 1; }
SHARD2=$DATA/Hy4-flash-IQ3_XXS-00002-of-00002.gguf
[ -e "$(readlink -f "$SHARD2")" ] || { echo "run.sh: the Engram table $SHARD2 is not there" >&2; exit 1; }
SOCK=$DATA/hy4-$PORT.sock
rm -f "$SOCK"   # an old socket would let the wrapper start before the engine is loaded
echo "run.sh: $(date '+%F %T') engine $BIN, context $CTX, $SLOTS slots a card"
"$BIN" --model "$HY4_GGUF" --slots "$SLOTS" --ctx "$CTX" --serve "$SOCK" &
ENGINE=$!
WRAP=
stop() {
  [ -n "$WRAP" ] && kill "$WRAP" 2>/dev/null
  kill "$ENGINE" 2>/dev/null
  wait 2>/dev/null
  exit 0
}
trap stop TERM INT
for _ in $(seq 900); do
  [ -S "$SOCK" ] && break
  kill -0 "$ENGINE" 2>/dev/null || { echo "run.sh: the engine ended while loading" >&2; exit 1; }
  sleep 1
done
[ -S "$SOCK" ] || { echo "run.sh: the engine did not open $SOCK in 15 minutes" >&2; stop; }
if [ "${HY4_PREREAD:-1}" != 0 ]; then
  t0=$(date +%s)
  cat "$(readlink -f "$SHARD2")" > /dev/null
  echo "run.sh: Engram table read into the page cache in $(( $(date +%s) - t0 )) s"
fi
"$HY4_PYTHON" "$HERE/serve/hy4_openai.py" "$PORT" "$SOCK" "$HOST" &
WRAP=$!
echo "run.sh: OpenAI-compatible API on http://$HOST:$PORT/v1 (model hy4-flash-iq3xxs)"
wait -n "$ENGINE" "$WRAP" || true
stop
