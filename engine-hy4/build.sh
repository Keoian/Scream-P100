#!/bin/bash
# engine-hy4/build.sh: the Hy4 flash engine (hy4_v1) and its three checks for two Tesla P100 (sm_60), into engine-hy4/build/.
# About two minutes. NVCC (nvcc) and CCBIN (the first of g++-13, g++-12, g++) can be set. CUDA 12.4 with g++-13 is what we used.
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
if [ -z "${CCBIN:-}" ]; then
  for c in g++-13 g++-12 g++; do
    if command -v "$c" >/dev/null; then CCBIN=$c; break; fi
  done
fi
command -v "$NVCC" >/dev/null || { echo "build.sh: no $NVCC (CUDA 12 toolkit)" >&2; exit 1; }
mkdir -p build
FLAGS=(-O2 -std=c++17 -arch=sm_60 -ccbin "$CCBIN" -Isrc)
echo "build.sh: $("$NVCC" --version | tail -1), host compiler $CCBIN"
"$NVCC" "${FLAGS[@]}" -o build/hy4_v1 src/hy4_v1.cu src/gguf_min.cpp -lcublas
"$NVCC" "${FLAGS[@]}" -o build/bench_dec tools/bench_dec.cu src/gguf_min.cpp -lcublas
"$NVCC" "${FLAGS[@]}" -o build/bench_bw tools/bench_bw.cu
"$NVCC" "${FLAGS[@]}" -o build/gemm_bench tools/gemm_bench.cu -lcublas
sha256sum build/hy4_v1 | cut -c1-16 | sed 's/^/build.sh: build\/hy4_v1 /'
