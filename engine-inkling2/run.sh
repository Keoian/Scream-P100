#!/bin/bash
# engine-inkling2/run.sh DATA [CTX=32768] [PORT=8080]: the engine on both P100 (the model, the DSpark drafter and its draft
# vocabulary), then the OpenAI-compatible server on http://127.0.0.1:PORT/v1. CTX up to 262144 with the drafter (15.2 GiB a
# card; 32768: 11.8 GiB). Ready in ~30-70 s. Ctrl-C or SIGTERM stops both. INK2_* switches pass through to the engine
# (docs/INKLING2.md); INK2_OA_MTP / INK2_OA_LOOKUP set the server's default drafts (3 / 15).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); B=$H/build
D=${1:?usage: run.sh DATA [CTX] [PORT]}; CTX=${2:-32768}; PORT=${3:-8080}
SOCK=$D/ink2-$PORT.sock
rm -f "$SOCK"
"$B/ink2_v1" --model "$D/Inkling-2-Mini-NVFP4-Q8_0.gguf" --mtp "$D/dspark-Inkling-2-Mini-Q8_0.gguf" \
  --draft-vocab "$H/data/draft_vocab_32768.bin" --serve "$SOCK" --ctx "$CTX" &
EP=$!
trap 'kill $EP 2>/dev/null' EXIT
for i in $(seq 180); do [ -S "$SOCK" ] && break; kill -0 $EP 2>/dev/null || { echo "the engine ended while loading" >&2; exit 1; }; sleep 1; done
[ -S "$SOCK" ] || { echo "no engine socket after 180 s" >&2; exit 1; }
export INK2_ENGINE_PID=$EP
"$D/venv/bin/python" "$H/serve/ink2_server.py" --port "$PORT" --sock "$SOCK" --ctx "$CTX" --model "$D/Inkling-2-Mini-NVFP4-Q8_0.gguf" \
  --tokenizer "$B/llama.cpp/build/bin/llama-tokenize" --gguf-py "$B/llama.cpp/gguf-py"
