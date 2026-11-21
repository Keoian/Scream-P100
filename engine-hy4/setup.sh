#!/bin/bash
# engine-hy4/setup.sh DATA: everything the Hy4 flash engine needs besides its own build, in the folder DATA (~101 GB free):
#   1. the model, from Xylolabs/Hy4-flash-GGUF at a pinned revision, size and SHA-256 checked:
#      the IQ3_XXS trunk (shard 1, 64.9 GB) and the IQ4_NL Engram table (shard 2 of the Q3_K_M-EIQ4_NL pair, 35.6 GB), which the
#      engine opens under the trunk's shard-2 name (a symbolic link)
#   2. a Python venv with jinja2 and gguf, and the GGUF's chat template written out for the OpenAI wrapper
#   3. llama.cpp's llama-tokenize from the commit that adds Hy4 (llama.cpp PR #32517), built for the CPU only
# Files you already have: HY4_TRUNK=/path/Hy4-flash-IQ3_XXS-00001-of-00002.gguf, HY4_ENGRAM=/path/Hy4-flash-Q3_K_M-EIQ4_NL-00002-of-00002.gguf,
# HY4_TOKENIZE=/path/llama-tokenize (a llama.cpp build that knows Hy4). HF_TOKEN is sent if set. Run it again to resume.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: bash engine-hy4/setup.sh DATA" >&2; exit 1; }
mkdir -p "$1"
DATA=$(cd "$1" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=Xylolabs/Hy4-flash-GGUF
REV=dc1e22e7a3c13c4e6aadc9c238d4a49af13610ce
TRUNK=Hy4-flash-IQ3_XXS-00001-of-00002.gguf
TRUNK_SIZE=64886586208
TRUNK_SHA=591f8f2f70dd577340b44a2da0797fc9004429da9fa385c92fe90206110a5f3f
ENGRAM=Hy4-flash-Q3_K_M-EIQ4_NL-00002-of-00002.gguf
ENGRAM_SIZE=35640346656
ENGRAM_SHA=0ac3ea9e6a73a77efd270cb2fbc6453657eb12fb27718a604503c2636d561c95
LLAMA_COMMIT=0a054d0b02454c998e068381d15f3f6615bdd987   # "model : add hy_v4_flash support" (PR #32517, not merged as of 21 Nov 2026)
say() { echo "setup.sh: $*"; }

# 0. the machine
mem=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
[ "$mem" -ge 90 ] || say "WARNING: ${mem} GiB of RAM. The engine pins 53 GiB of experts and the Engram table wants another 33 GiB in the page cache; we ran it with 99 GiB"
if command -v nvidia-smi >/dev/null; then
  say "cards: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
fi

# 1. the model
fetch() {   # fetch NAME SIZE SHA [EXISTING]
  local name=$1 size=$2 sha=$3 have=${4:-} out="$DATA/$1"
  if [ -n "$have" ]; then
    [ -e "$out" ] || ln -s "$(readlink -f "$have")" "$out"
  fi
  if [ "$(stat -Lc %s "$out" 2>/dev/null || echo 0)" -lt "$size" ]; then
    say "downloading $name ($((size / 1000000000)) GB) from $REPO"
    local auth=()
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
    local n=0
    while [ "$(stat -Lc %s "$out" 2>/dev/null || echo 0)" -lt "$size" ]; do
      curl -fL --retry 5 --retry-delay 5 --connect-timeout 20 --speed-limit 100000 --speed-time 60 -C - "${auth[@]}" \
        -o "$out" "https://huggingface.co/$REPO/resolve/$REV/$name" || { n=$((n + 1)); [ $n -gt 50 ] && exit 1; sleep 10; }
    done
  fi
  [ "$(stat -Lc %s "$out")" -eq "$size" ] || { say "$name: size $(stat -Lc %s "$out"), expected $size"; exit 1; }
  if [ ! -e "$out.sha256-ok" ]; then
    say "checking $name's SHA-256 (reads the whole file: minutes)"
    [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ] || { say "$name: SHA-256 differs from $sha"; exit 1; }
    touch "$out.sha256-ok"
  fi
}
fetch "$TRUNK" "$TRUNK_SIZE" "$TRUNK_SHA" "${HY4_TRUNK:-}"
fetch "$ENGRAM" "$ENGRAM_SIZE" "$ENGRAM_SHA" "${HY4_ENGRAM:-}"
ln -sfn "$ENGRAM" "$DATA/Hy4-flash-IQ3_XXS-00002-of-00002.gguf"   # the engine opens the trunk's shard 2 by name

# 2. Python and the chat template
if [ ! -x "$DATA/venv/bin/python" ]; then
  say "Python venv"
  python3 -m venv "$DATA/venv"
  "$DATA/venv/bin/pip" install -q "jinja2==3.1.6" "gguf==0.19.0"
fi
"$DATA/venv/bin/python" - "$DATA/$TRUNK" "$DATA/hy4_template.jinja" <<'EOF'
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields["tokenizer.chat_template"]
open(sys.argv[2], "w", encoding="utf-8").write(bytes(f.parts[f.data[0]]).decode("utf-8"))
EOF
say "chat template: $DATA/hy4_template.jinja ($(wc -c < "$DATA/hy4_template.jinja") bytes)"

# 3. llama-tokenize
TOK=${HY4_TOKENIZE:-$DATA/llama.cpp/build/bin/llama-tokenize}
if [ ! -x "$TOK" ]; then
  say "llama.cpp $LLAMA_COMMIT (llama-tokenize only, CPU)"
  L=$DATA/llama.cpp
  if [ ! -d "$L/.git" ]; then
    git init -q "$L"
    git -C "$L" remote add origin https://github.com/ggml-org/llama.cpp
  fi
  git -C "$L" fetch -q --depth 1 origin "$LLAMA_COMMIT" || git -C "$L" fetch -q origin pull/32517/head
  git -C "$L" checkout -q "$LLAMA_COMMIT"
  cmake -S "$L" -B "$L/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_BUILD_TYPE=Release > "$L/cmake.log" 2>&1 || { say "cmake failed: $L/cmake.log"; exit 1; }
  cmake --build "$L/build" --target llama-tokenize -j "$(nproc)" > "$L/build.log" 2>&1 || { say "build failed: $L/build.log"; exit 1; }
fi
[ -x "$TOK" ] || { say "no llama-tokenize at $TOK"; exit 1; }

cat > "$DATA/hy4.env" <<EOF
HY4_GGUF=$DATA/$TRUNK
HY4_TOKENIZE=$TOK
HY4_TEMPLATE=$DATA/hy4_template.jinja
HY4_PYTHON=$DATA/venv/bin/python
EOF
say "done: $DATA/hy4.env. Next: bash engine-hy4/run.sh $1"
