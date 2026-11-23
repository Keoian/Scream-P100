#!/bin/bash
# engine-inkling2/run.sh DATA [CTX=32768] [PORT=8080]: the engine on both P100 (the model, the DSpark drafter and its draft
# vocabulary), then the OpenAI-compatible server on http://127.0.0.1:PORT/v1. CTX up to 524288 with the drafter, two choices
# for a long context (names accepted):
#   256k = 262144  the KV cache all in f16, 15.2 GiB a card
#   512k = 524288  the full layers' KV from row 32,768 on in int8 (the rows below in f16, bit for bit as 256k), 15.24 GiB a card
# Any CTX above 262144 takes the int8 mode by itself (INK2_KV8=0/1 forces it). 32768: 11.8 GiB a card. Pictures when DATA holds
# the vision projector (setup.sh) and build/ink2_vis exists, INK2_PICTURES=0 turns them off. Ready in ~30-70 s. Ctrl-C or
# SIGTERM stops both. INK2_* switches pass through to the engine (docs/INKLING2.md); INK2_OA_MTP / INK2_OA_LOOKUP set the
# server's default drafts (3 / 15), INK2_LOOP_TEXT / INK2_LOG_EVERY its loop limit for answers and tool calls (16000
# characters) and its progress line while a request runs (60 s). The engine keeps conversations' checkpoints and parked conversations in RAM (up to
# INK2_CKPT_MB = 24576 and INK2_PARK_MB = 32768 MiB, taken as they are needed; set them lower on a machine with less RAM).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); B=$H/build
D=${1:?usage: run.sh DATA [CTX|256k|512k] [PORT]}; CTX=${2:-32768}; PORT=${3:-8080}
case "$CTX" in 256k|256K) CTX=262144 ;; 512k|512K) CTX=524288 ;; esac
[[ "$CTX" =~ ^[0-9]+$ ]] && [ "$CTX" -le 524288 ] || { echo "CTX: a number up to 524288, 256k or 512k" >&2; exit 2; }
MMPROJ=$D/mmproj-Inkling-2-Mini-BF16.gguf; VIS=()
if [ "${INK2_PICTURES:-1}" != 0 ] && [ -f "$MMPROJ" ] && [ -x "$B/ink2_vis" ]; then VIS=(--mmproj "$MMPROJ" --vis "$B/ink2_vis"); fi
echo "run.sh: context $CTX ($([ "$CTX" -gt 262144 ] && echo "KV int8 from row 32768" || echo "KV f16")), pictures $([ ${#VIS[@]} -gt 0 ] && echo on || echo off)" >&2
SOCK=$D/ink2-$PORT.sock
rm -f "$SOCK"
"$B/ink2_v1" --model "$D/Inkling-2-Mini-NVFP4-Q8_0.gguf" --mtp "$D/dspark-Inkling-2-Mini-Q8_0.gguf" \
  --draft-vocab "$H/data/draft_vocab_32768.bin" --serve "$SOCK" --ctx "$CTX" &
EP=$!
trap 'kill $EP 2>/dev/null' EXIT
for i in $(seq 180); do [ -S "$SOCK" ] && break; kill -0 $EP 2>/dev/null || { echo "the engine ended while loading" >&2; exit 1; }; sleep 1; done
[ -S "$SOCK" ] || { echo "no engine socket after 180 s" >&2; exit 1; }
export INK2_ENGINE_PID=$EP
TOK=(); [ -x "$B/ink2_tok" ] && TOK=(--tok "$B/ink2_tok")   # the tokenizer that stays up (a build from before it: llama-tokenize a request)
"$D/venv/bin/python" "$H/serve/ink2_server.py" --port "$PORT" --sock "$SOCK" --ctx "$CTX" --model "$D/Inkling-2-Mini-NVFP4-Q8_0.gguf" \
  --tokenizer "$B/llama.cpp/build/bin/llama-tokenize" --gguf-py "$B/llama.cpp/gguf-py" ${VIS[@]+"${VIS[@]}"} ${TOK[@]+"${TOK[@]}"}
