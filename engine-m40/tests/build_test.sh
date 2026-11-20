#!/bin/bash
# build_test.sh [m40_rows_test|m40_pf_test]: builds one of the M40 kernel checks in this folder with the engine's flags for sm_52
# (as our mikro/m40_test_gradnja.sh did on the M40 server). Run build.sh --m40 first (it fetches the pinned llama.cpp headers).
# Then ./m40_rows_test or ./m40_pf_test on idle cards (no engine loaded): each prints every M40 expert type's worst relative error
# against a CPU reference (ggml's dequantize_row_* formulas, sums in double) and the kernels' times.
#   m40_rows_test  the decode window's kernel (m40_moe_rows.cuh) for windows of 1 to 16 rows, and the prompt path's k_gemm4_job
#   m40_pf_test    the prompt path's k_gemm4_job against k_m40_pf (m40_pf_gemm.cuh, FNX_M40_PF=1) on jobs of 1 to 96 tokens
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
N=${1:-m40_rows_test}
say() { echo "build_test.sh: $*"; }
die() { echo "build_test.sh: $*" >&2; exit 1; }
. "$ROOT/scripts/toolchain.sh"
pick_toolchain
G=${GG:-$ROOT/third_party/llama.cpp}/ggml
[ -f "$G/src/ggml-cuda/vecdotq.cuh" ] || die "no llama.cpp headers in $G: run 'bash build.sh --m40' first"
S=$ROOT/engine-m40
cd "$HERE"
nice -n 10 "$NVCC" -std=c++17 -O3 -arch=sm_52 -ccbin "$HOSTCXX" --use_fast_math --expt-relaxed-constexpr -Wno-deprecated-gpu-targets \
  -include "$S/p100_compat.h" -DFNX_NCARD=2 -DFNX_ARCH=52 -DFNX_27B=0 -DFNX_HOSTRING=1 -DFNX_PF_JOB=96 -DFNX_DIR="\"$S\"" \
  -I"$S/src" -I"$S/tools" -I"$G/src/ggml-cuda" -I"$G/src" -I"$G/include" "$N.cu" -o "$N" -lcuda -lcublas -lpthread
say "built $HERE/$N"
