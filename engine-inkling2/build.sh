#!/bin/bash
# engine-inkling2/build.sh: builds the Inkling 2 Mini engine (build/ink2_v1, one CUDA program for two Tesla P100, sm_60) and,
# for the server's tokenizer, llama.cpp's llama-tokenize at commit 690b87d on the CPU (no CUDA in that build), plus its gguf-py.
# Needs CUDA 12.x (nvcc, cuBLAS), a host compiler nvcc accepts (CUDA 12.4: g++ <= 13; set HOST_CXX=g++-13 if your default is
# newer), cmake, curl. About 1 minute for the engine and 2-3 for llama-tokenize (21 Nov: 2 min 55 s in all on our server).
set -euo pipefail
cd "$(dirname "$0")"
H=$(pwd); B=$H/build; mkdir -p "$B"
HOST_CXX=${HOST_CXX:-g++}
echo "== ink2_v1 (nvcc, host compiler $HOST_CXX)"
(cd src && nvcc -O2 -std=c++17 -arch=sm_60 -ccbin "$HOST_CXX" -o "$B/ink2_v1" ink2_v1.cu gguf_min.cpp -lcublas)
LC=690b87d
if [ ! -x "$B/llama.cpp/build/bin/llama-tokenize" ]; then
  echo "== llama.cpp $LC: llama-tokenize (CPU) and gguf-py"
  rm -rf "$B/llama.cpp"; mkdir -p "$B/llama.cpp"
  curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/$LC.tar.gz" | tar xz -C "$B/llama.cpp" --strip-components=1
  cmake -S "$B/llama.cpp" -B "$B/llama.cpp/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > "$B/llama.cpp.cmake.log"
  cmake --build "$B/llama.cpp/build" --target llama-tokenize -j "$(nproc)" > "$B/llama.cpp.build.log"
fi
echo "done: $B/ink2_v1, $B/llama.cpp/build/bin/llama-tokenize"
