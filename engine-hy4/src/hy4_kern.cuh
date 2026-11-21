// hy4_kern.cuh (2026-11-20): v0's constants, Mat and kernels, shared by hy4_v1.cu (generated from hy4_v0.cu lines 1-55 and 82-712)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <thread>
#include <vector>

#include "gguf_min.hpp"
#include "hy4_tables.h"

namespace hy4 {

constexpr int N = 3072, HC = 4, D = N * HC, NL = 42, NEXP = 512, TOPK = 16, FFE = 1536, LAT = 1536;
constexpr int QLR = 1024, KVL = 512, ROT = 64, NOPE = 192, HDQ = 256, HDV = 256, KROW = KVL + ROT;
constexpr int IDXH = 16, IDXD = 128, KPOOL = 4, TOPPOOL = 2048 / KPOOL - 1;
constexpr int MAXT = 16, SWA = 512, RING = 1024, FFD = 13824;
// v1v: the indexer's value|gate rows live in a ring of IVR rows (a group needs only its own 4; a prompt chunk of 2048 plus the 3 before it)
constexpr int IVR = 2056;
constexpr int E_DIM = 96, E_COLS = 16, E_W = E_DIM * E_COLS, E_NGRAM = 3, E_HPN = 8;
constexpr float ENGRAM_EPS = 1.1920929e-07f;
constexpr int MAXSEL = TOPPOOL * KPOOL + KPOOL;   // positions a sparse layer reads past 511 pools
constexpr size_t EXP_BYTES = (size_t) FFE * (LAT / 256) * 98;   // one expert matrix: 1536 rows of 6 IQ3_XXS blocks

static void ck(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s at %d: %s\n", what, line, cudaGetErrorString(e));
        std::exit(2);
    }
}
#define CK(x) ck((x), #x, __LINE__)

// ---------------------------------------------------------------------------------------------------------------- weights
// type 8 (Q8_0) is kept split as in the split engine's Q8S: q = int8 values [rows][cols], d = f16 scales [rows][cols / 32]
struct Mat {
    uint32_t type = 0;
    int rows = 0, cols = 0;
    const void* d = nullptr;   // device: F32 / F16 rows, or the Q8S scales
    const int8_t* q = nullptr; // Q8S values
};

// ------------------------------------------------------------------------------------------------------------- kernels
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = 0;
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        s = warp_sum(s);
        if (threadIdx.x == 0) red[0] = s;
    }
    __syncthreads();
    s = red[0];
    __syncthreads();
    return s;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

// y[t][r] = W[r] . x[t + group(r)]: one warp per row, the lanes over each block's 32 values. Rows come in groups of rg
// rows whose input starts xg floats further on (per-head matrices: k_b, v_b); rg = rows: one input for all
__global__ void k_gemv(uint32_t type, const void* W, int rows, int cols, const float* x, int ldx, int rg, int xg, float* y, int ldy, int T) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (r >= rows) return;
    const float* xr = x + (size_t) (r / rg) * xg;
    float acc[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) acc[t] = 0.0f;
    if (type == 8) {   // Q8_0: 34 bytes a block
        const uint8_t* row = (const uint8_t*) W + (size_t) r * (cols / 32) * 34;
        for (int b = 0; b < cols / 32; ++b) {
            const uint8_t* blk = row + (size_t) b * 34;
            const float d = __half2float(*(const __half*) blk);
            const float w = d * (float) (int8_t) blk[2 + l];
            const int c = b * 32 + l;
#pragma unroll
            for (int t = 0; t < MAXT; ++t)
                if (t < T) acc[t] += w * xr[(size_t) t * ldx + c];
        }
    } else if (type == 0) {
        const float* row = (const float*) W + (size_t) r * cols;
        for (int c = l; c < cols; c += 32) {
            const float w = row[c];
#pragma unroll
            for (int t = 0; t < MAXT; ++t)
                if (t < T) acc[t] += w * xr[(size_t) t * ldx + c];
        }
    } else {   // F16
        const __half* row = (const __half*) W + (size_t) r * cols;
        for (int c = l; c < cols; c += 32) {
            const float w = __half2float(row[c]);
#pragma unroll
            for (int t = 0; t < MAXT; ++t)
                if (t < T) acc[t] += w * xr[(size_t) t * ldx + c];
        }
    }
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < T) {
            const float s = warp_sum(acc[t]);
            if (l == 0) y[(size_t) t * ldy + r] = s;
        }
    }
}

// The bandwidth kernels: a block of 8 warps holds GW_R rows a warp; the input goes through shared memory in chunks of 512 (Q8S) or
// 256 (F16) columns, stored so that lane l's j-th value sits at [j][l] (no bank conflicts); each lane loads 16 bytes of each row a
// chunk (16 int8 or 8 f16) in one load. Rows of a block share one input group (rg a multiple of the block's rows)
constexpr int GW_R = 2, GW_BR = 8 * GW_R;
__global__ void __launch_bounds__(256) k_gemv8(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols, const float* x,
                                                int ldx, int rg, int xg, float* y, int ldy, int T) {
    extern __shared__ float xs[];   // [T][16][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * GW_BR, r0 = rb + warp * GW_R;
    const float* xb = x + (size_t) (rb / rg) * xg;
    float acc[GW_R][MAXT];
#pragma unroll
    for (int a = 0; a < GW_R; ++a)
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[a][t] = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += 512) {
        __syncthreads();
        for (int i = threadIdx.x; i < T * 512; i += 256) {
            const int t = i >> 9, c = i & 511;
            xs[t * 512 + (c & 15) * 32 + (c >> 4)] = c0 + c < cols ? xb[(size_t) t * ldx + c0 + c] : 0.0f;
        }
        __syncthreads();
        const int c = c0 + lane * 16;
        if (c >= cols) continue;
#pragma unroll
        for (int a = 0; a < GW_R; ++a) {
            const int r = r0 + a;
            if (r >= rows) break;
            const int4 w4 = __ldg((const int4*) (q + (size_t) r * cols + c));
            const float sc = __half2float(d[(size_t) r * (cols / 32) + c / 32]);
            const int8_t* w = (const int8_t*) &w4;
            float wf[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) wf[j] = (float) w[j];
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t >= T) break;
                const float* xt = xs + t * 512 + lane;
                float s = 0;
#pragma unroll
                for (int j = 0; j < 16; ++j) s += wf[j] * xt[j * 32];
                acc[a][t] += sc * s;
            }
        }
    }
#pragma unroll
    for (int a = 0; a < GW_R; ++a) {
        const int r = r0 + a;
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const float s = warp_sum(acc[a][t]);
            if (lane == 0 && r < rows) y[(size_t) t * ldy + r] = s;
        }
    }
}
__global__ void __launch_bounds__(256) k_gemv16(const __half* __restrict__ W, int rows, int cols, const float* x, int ldx, float* y, int ldy, int T) {
    extern __shared__ float xs[];   // [T][8][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * GW_BR, r0 = rb + warp * GW_R;
    float acc[GW_R][MAXT];
#pragma unroll
    for (int a = 0; a < GW_R; ++a)
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[a][t] = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += 256) {
        __syncthreads();
        for (int i = threadIdx.x; i < T * 256; i += 256) {
            const int t = i >> 8, c = i & 255;
            xs[t * 256 + (c & 7) * 32 + (c >> 3)] = c0 + c < cols ? x[(size_t) t * ldx + c0 + c] : 0.0f;
        }
        __syncthreads();
        const int c = c0 + lane * 8;
        if (c >= cols) continue;
#pragma unroll
        for (int a = 0; a < GW_R; ++a) {
            const int r = r0 + a;
            if (r >= rows) break;
            const int4 w4 = __ldg((const int4*) (W + (size_t) r * cols + c));
            const __half* w = (const __half*) &w4;
            float wf[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) wf[j] = __half2float(w[j]);
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t >= T) break;
                const float* xt = xs + t * 256 + lane;
                float s = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) s += wf[j] * xt[j * 32];
                acc[a][t] += s;
            }
        }
    }
#pragma unroll
    for (int a = 0; a < GW_R; ++a) {
        const int r = r0 + a;
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const float s = warp_sum(acc[a][t]);
            if (lane == 0 && r < rows) y[(size_t) t * ldy + r] = s;
        }
    }
}

// F32 rows (the router): 4 values a lane a chunk of 128 columns, the same shared-memory layout
__global__ void __launch_bounds__(256) k_gemv32(const float* __restrict__ W, int rows, int cols, const float* x, int ldx, float* y, int ldy, int T) {
    extern __shared__ float xs[];   // [T][4][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * GW_BR, r0 = rb + warp * GW_R;
    float acc[GW_R][MAXT];
#pragma unroll
    for (int a = 0; a < GW_R; ++a)
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[a][t] = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += 128) {
        __syncthreads();
        for (int i = threadIdx.x; i < T * 128; i += 256) {
            const int t = i >> 7, c = i & 127;
            xs[t * 128 + (c & 3) * 32 + (c >> 2)] = c0 + c < cols ? x[(size_t) t * ldx + c0 + c] : 0.0f;
        }
        __syncthreads();
        const int c = c0 + lane * 4;
        if (c >= cols) continue;
#pragma unroll
        for (int a = 0; a < GW_R; ++a) {
            const int r = r0 + a;
            if (r >= rows) break;
            const float4 w = __ldg((const float4*) (W + (size_t) r * cols + c));
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t >= T) break;
                const float* xt = xs + t * 128 + lane;
                acc[a][t] += w.x * xt[0] + w.y * xt[32] + w.z * xt[64] + w.w * xt[96];
            }
        }
    }
#pragma unroll
    for (int a = 0; a < GW_R; ++a) {
        const int r = r0 + a;
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const float s = warp_sum(acc[a][t]);
            if (lane == 0 && r < rows) y[(size_t) t * ldy + r] = s;
        }
    }
}

// Q8S at T = 1 (writing): no shared memory and no barriers, 4 rows a warp, the lanes read x through L1; the columns split into S
// parts (blocks.y) when the rows alone give too few blocks: the parts go to g_part [S][rows] and k_sum_parts adds them in a fixed
// order, so every card gets the same bits for the same input (the cards' copies of the replicated matrices must agree)
constexpr int G1_R = 4, G1_BR = 8 * G1_R, G1_PART = 65536;
__device__ float g_part[G1_PART];
__device__ unsigned g_g1_cnt[1024];   // v1j: k_gemv8_t1's arrivals by row block, back to 0 after each matrix
static bool g_g1_fused = false;       // v1j: set by the window path (the engine's dec_on)
__global__ void __launch_bounds__(256) k_gemv8_t1(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols,
                                                   const float* __restrict__ x, int rg, int xg, float* y, int S, int fused) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * G1_BR, r0 = rb + warp * G1_R;
    const float* xb = x + (size_t) (rb / rg) * xg;
    const int nch = (cols + 511) / 512, s = blockIdx.y, ch0 = s * nch / S, ch1 = (s + 1) * nch / S;
    float acc[G1_R] = {};
    for (int ch = ch0; ch < ch1; ++ch) {
        const int c = ch * 512 + lane * 16;
        if (c >= cols) continue;
        const float4 x0 = __ldg((const float4*) (xb + c)), x1 = __ldg((const float4*) (xb + c + 4)), x2 = __ldg((const float4*) (xb + c + 8)),
                     x3 = __ldg((const float4*) (xb + c + 12));
#pragma unroll
        for (int a = 0; a < G1_R; ++a) {
            const int r = r0 + a;
            if (r >= rows) break;
            const int4 w4 = __ldg((const int4*) (q + (size_t) r * cols + c));
            const float sc = __half2float(d[(size_t) r * (cols / 32) + c / 32]);
            const int8_t* w = (const int8_t*) &w4;
            const float v = w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w +
                            w[8] * x2.x + w[9] * x2.y + w[10] * x2.z + w[11] * x2.w + w[12] * x3.x + w[13] * x3.y + w[14] * x3.z + w[15] * x3.w;
            acc[a] += sc * v;
        }
    }
#pragma unroll
    for (int a = 0; a < G1_R; ++a) {
        const float v = warp_sum(acc[a]);
        const int r = r0 + a;
        if (lane == 0 && r < rows) {
            if (S == 1) y[r] = v;
            else g_part[(size_t) s * rows + r] = v;
        }
    }
    // v1j (fused): the last of a row block's S parts to finish adds them in k_sum_parts' order (one launch less a matrix)
    if (S == 1 || !fused) return;
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&g_g1_cnt[blockIdx.x], 1u) == (unsigned) S - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < G1_BR && rb + (int) threadIdx.x < rows) {
        const int r = rb + threadIdx.x;
        float v = 0;
        for (int k = 0; k < S; ++k) v += ((volatile float*) g_part)[(size_t) k * rows + r];
        y[r] = v;
    }
    if (threadIdx.x == 0) g_g1_cnt[blockIdx.x] = 0;
}
__global__ void k_sum_parts(float* y, int rows, int S) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float v = 0;
    for (int s = 0; s < S; ++s) v += g_part[(size_t) s * rows + r];
    y[r] = v;
}

static void gemv(const Mat& W, const float* x, int ldx, float* y, int ldy, int T, cudaStream_t st, int rg = 0, int xg = 0) {
    static const bool g1_off = getenv("HY4_G1_OFF") != nullptr;
    if (W.type == 8 && T == 1 && W.cols % 16 == 0 && (!rg || rg % G1_BR == 0) && !g1_off) {
        const int rb = (W.rows + G1_BR - 1) / G1_BR, nch = (W.cols + 511) / 512;
        int S = std::max(1, std::min(nch, 448 / rb));
        while (S > 1 && (size_t) S * W.rows > G1_PART) --S;
        const int fused = g_g1_fused && S > 1 && rb <= 1024;
        k_gemv8_t1<<<dim3(rb, S), 256, 0, st>>>(W.q, (const __half*) W.d, W.rows, W.cols, x, rg ? rg : W.rows, xg, y, S, fused);
        if (S > 1 && !fused) k_sum_parts<<<(W.rows + 255) / 256, 256, 0, st>>>(y, W.rows, S);
        return;
    }
    if (W.type == 0 && !rg && W.cols % 4 == 0 && W.cols >= 512) {
        k_gemv32<<<(W.rows + GW_BR - 1) / GW_BR, 256, (size_t) T * 128 * 4, st>>>((const float*) W.d, W.rows, W.cols, x, ldx, y, ldy, T);
        return;
    }
    if (W.type == 8) {
        if (rg && rg % GW_BR) throw std::runtime_error("gemv: row group not a multiple of the block's rows");
        k_gemv8<<<(W.rows + GW_BR - 1) / GW_BR, 256, (size_t) T * 512 * 4, st>>>(W.q, (const __half*) W.d, W.rows, W.cols, x, ldx,
                                                                                  rg ? rg : W.rows, xg, y, ldy, T);
    } else if (W.type == 1 && !rg && W.cols % 8 == 0) {
        k_gemv16<<<(W.rows + GW_BR - 1) / GW_BR, 256, (size_t) T * 256 * 4, st>>>((const __half*) W.d, W.rows, W.cols, x, ldx, y, ldy, T);
    } else {
        const int warps = 8;
        k_gemv<<<(W.rows + warps - 1) / warps, warps * 32, 0, st>>>(W.type, W.d, W.rows, W.cols, x, ldx, rg ? rg : W.rows, xg, y, ldy, T);
    }
}

// y = x / rms(x) * w over rows of n (w may be null); eps given
__global__ void k_rms(const float* x, int ldx, const float* w, float* y, int ldy, int n, float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = y + (size_t) blockIdx.x * ldy;
    float s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, red);
    const float k = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * k * (w ? w[i] : 1.0f);
}
static void rms(const float* x, int ldx, const float* w, float* y, int ldy, int rows, int n, float eps, cudaStream_t st) {
    k_rms<<<rows, 256, 0, st>>>(x, ldx, w, y, ldy, n, eps);
}

// iHC read: mixes = fn . rms(R_t) (2*HC of them, fn [2HC][D]); pre = sigmoid(mix * s0 + b) + eps, post = sigmoid(mix * s1 + b)
// * mag + eps; x = sum_c pre_c R_c. nmix = 2 * HC (a layer's half) or HC (the head: pre only, scale[0] for all)
__global__ void k_hc_pre(const float* R, const float* fn, const float* scale, const float* base, int nmix, float eps_rms, float eps_hc,
                         float mag, float* x, float* post) {
    __shared__ float red[32];
    __shared__ float mix[2 * HC], pre[HC];
    const float* r = R + (size_t) blockIdx.x * D;
    float s = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += r[i] * r[i];
    s = block_sum(s, red);
    const float k = rsqrtf(s / D + eps_rms);
    for (int j = 0; j < nmix; ++j) {
        float a = 0;
        for (int i = threadIdx.x; i < D; i += blockDim.x) a += fn[(size_t) j * D + i] * r[i];
        a = block_sum(a, red);
        if (threadIdx.x == 0) mix[j] = a * k;
    }
    __syncthreads();
    if (threadIdx.x < HC) {
        pre[threadIdx.x] = sigm(mix[threadIdx.x] * scale[0] + base[threadIdx.x]) + eps_hc;
        if (nmix == 2 * HC)
            post[(size_t) blockIdx.x * HC + threadIdx.x] = sigm(mix[HC + threadIdx.x] * scale[1] + base[HC + threadIdx.x]) * mag + eps_hc;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = 0;
#pragma unroll
        for (int c = 0; c < HC; ++c) v += pre[c] * r[c * N + i];
        x[(size_t) blockIdx.x * N + i] = v;
    }
}
// v1: the same read on HCB blocks of 256 columns a token (k_hc_pre is one block a token: 183 us at T = 1), then one block a token
// for the finish. Each device has its own g_hc_part (module globals are per device), used in stream order
constexpr int HCB = D / 256;
__device__ float g_hc_part[2048 * HCB * 9];   // HY4_PFC rows (hy4_pf.cuh)
// v1q: the 1 + nmix sums in one pass (v1: one block_sum after another, 18 barriers: 3.5 ms a 2048-token chunk, 11 us a token):
// each value goes through block_sum's own steps (warp butterfly, the warps' partials, their butterfly), so the bits are the same
__global__ void __launch_bounds__(256) k_hc_mix(const float* R, const float* fn, int nmix) {
    __shared__ float red[8][9];
    const int t = blockIdx.x, b = blockIdx.y, i = b * 256 + threadIdx.x, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float r = R[(size_t) t * D + i];
    float v[9];
    v[0] = r * r;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[1 + j] = j < nmix ? fn[(size_t) j * D + i] * r : 0.0f;
#pragma unroll
    for (int j = 0; j < 9; ++j) v[j] = warp_sum(v[j]);
    if (l == 0)
#pragma unroll
        for (int j = 0; j < 9; ++j) red[w][j] = v[j];
    __syncthreads();
    if (threadIdx.x < 32) {
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            float s = threadIdx.x < 8 ? red[threadIdx.x][j] : 0.0f;
            s = warp_sum(s);
            if (threadIdx.x == 0 && j <= nmix) g_hc_part[(t * HCB + b) * 9 + j] = s;
        }
    }
}
__global__ void k_hc_fin(const float* R, const float* scale, const float* base, int nmix, float eps_rms, float eps_hc, float mag, float* x,
                         float* post) {
    __shared__ float tot[9], pre[HC];
    const int t = blockIdx.x;
    if (threadIdx.x <= nmix) {
        float s = 0;
        for (int b = 0; b < HCB; ++b) s += g_hc_part[(t * HCB + b) * 9 + threadIdx.x];
        tot[threadIdx.x] = s;
    }
    __syncthreads();
    const float k = rsqrtf(tot[0] / D + eps_rms);
    if (threadIdx.x < HC) {
        pre[threadIdx.x] = sigm(tot[1 + threadIdx.x] * k * scale[0] + base[threadIdx.x]) + eps_hc;
        if (nmix == 2 * HC)
            post[t * HC + threadIdx.x] = sigm(tot[1 + HC + threadIdx.x] * k * scale[1] + base[HC + threadIdx.x]) * mag + eps_hc;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = 0;
#pragma unroll
        for (int c = 0; c < HC; ++c) v += pre[c] * R[(size_t) t * D + c * N + i];
        x[(size_t) t * N + i] = v;
    }
}
static void hc_pre(const float* R, const float* fn, const float* scale, const float* base, int nmix, float eps_rms, float eps_hc, float mag,
                   float* x, float* post, int T, cudaStream_t st) {
    k_hc_mix<<<dim3(T, HCB), 256, 0, st>>>(R, fn, nmix);
    k_hc_fin<<<T, 256, 0, st>>>(R, scale, base, nmix, eps_rms, eps_hc, mag, x, post);
}
// R_c += post_c * x; grid (T, HCB)
__global__ void k_hc_post(float* R, const float* x, const float* post) {
    const int t = blockIdx.x, i = blockIdx.y * 256 + threadIdx.x;
    R[(size_t) t * D + i] += post[t * HC + i / N] * x[(size_t) t * N + i % N];
}

// LLAMA_ROPE_TYPE_NORM on n_rot values at x[t * ldx + h * hs + off]: pairs (2i, 2i+1), theta = pos * base^(-2i / n_rot)
__global__ void k_rope(float* x, int ldx, int heads, int hs, int off, const int* pos, float base) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;   // i < ROT / 2
    float* p = x + (size_t) t * ldx + (size_t) h * hs + off;
    const float th = (float) pos[t] * powf(base, -2.0f * i / ROT);
    float c, s;
    sincosf(th, &s, &c);
    const float a = p[2 * i], b = p[2 * i + 1];
    p[2 * i] = a * c - b * s;
    p[2 * i + 1] = a * s + b * c;
}

// the latent row of a token: [rms(c) * w (512) | k_pe roped (64)] into the cache row of its position
__global__ void k_kv_write(const float* kv, const float* w, float eps, __half* cache, const int* row) {   // v1: the latent cache in f16
    __shared__ float red[32];
    const float* r = kv + (size_t) blockIdx.x * KROW;
    __half* o = cache + (size_t) row[blockIdx.x] * KROW;
    float s = 0;
    for (int i = threadIdx.x; i < KVL; i += blockDim.x) s += r[i] * r[i];
    s = block_sum(s, red);
    const float k = rsqrtf(s / KVL + eps);
    for (int i = threadIdx.x; i < KVL; i += blockDim.x) o[i] = __float2half(r[i] * k * w[i]);
    for (int i = threadIdx.x; i < ROT; i += blockDim.x) o[KVL + i] = __float2half(r[KVL + i]);
}

// q for the latent attention: qa [T][H][KROW] = [q_nope . k_b (qabs, 512) | q_pe (64, from q [T][H][256])]
__global__ void k_q_assemble(const float* q, const float* qabs, float* qa, int H) {
    const int t = blockIdx.x, h = blockIdx.y;
    const size_t th = (size_t) t * H + h;
    for (int i = threadIdx.x; i < KROW; i += blockDim.x)
        qa[th * KROW + i] = i < KVL ? qabs[th * KVL + i] : q[th * HDQ + NOPE + (i - KVL)];
}
__global__ void k_scale(float* a, size_t n, float k) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) a[i] *= k;
}

// one (head, token): scores over its positions, softmax with the head's sink, o = sum p * c (512). Positions: sel[t] (count
// nsel[t]) or, when nsel[t] < 0, [lo, pos]; the cache row of position p is p (sparse) or p % RING (window)
__global__ void k_mla_attn(const float* qa, const __half* cache, const int* pos, int window, const int* sel, const int* nsel,
                           const float* sinks, float scale, float* o, int H) {
    extern __shared__ float sm[];
    float* q = sm;                 // KROW
    float* sc = sm + KROW;         // up to MAXSEL
    __shared__ float red[32];
    const int h = blockIdx.x, t = blockIdx.y, p = pos[t];
    for (int i = threadIdx.x; i < KROW; i += blockDim.x) q[i] = qa[((size_t) t * H + h) * KROW + i];
    int n, lo = 0;
    const bool list = nsel && nsel[t] >= 0;
    if (list) n = nsel[t];
    else {
        lo = window ? max(0, p - SWA + 1) : 0;
        n = p - lo + 1;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int pp = list ? sel[(size_t) t * MAXSEL + j] : lo + j;
        const __half* r = cache + (size_t) (window ? pp % RING : pp) * KROW;
        float a = 0;
        for (int i = 0; i < KROW; ++i) a += q[i] * __half2float(r[i]);
        a *= scale;
        sc[j] = a;
        m = fmaxf(m, a);
    }
    // block max
    for (int k = 16; k; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, k));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mm = sinks[h];
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) mm = fmaxf(mm, red[w]);
        red[0] = mm;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float s = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float e = __expf(sc[j] - m);
        sc[j] = e;
        s += e;
    }
    s = block_sum(s, red);
    const float den = s + __expf(sinks[h] - m);
    for (int i = threadIdx.x; i < KVL; i += blockDim.x) {
        float a = 0;
        for (int j = 0; j < n; ++j) {
            const int pp = list ? sel[(size_t) t * MAXSEL + j] : lo + j;
            a += sc[j] * __half2float(cache[(size_t) (window ? pp % RING : pp) * KROW + i]);
        }
        o[((size_t) t * H + h) * KVL + i] = a / den;
    }
}

__global__ void k_sigmul(float* a, const float* g, int n) {   // a *= sigmoid(g), rows of n
    for (size_t k = (size_t) blockIdx.y * n + blockIdx.x * blockDim.x + threadIdx.x; k < (size_t) (blockIdx.y + 1) * n;
         k += (size_t) gridDim.x * blockDim.x)
        a[k] *= sigm(g[k]);
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int n, float clamp) {   // clamp 0: plain
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float a = g[(size_t) t * n + i], b = u[(size_t) t * n + i];
        if (clamp > 0) {
            a = fminf(a, clamp);
            b = fminf(fmaxf(b, -clamp), clamp);
        }
        h[(size_t) t * n + i] = a * sigm(a) * b;
    }
}
__global__ void k_add(float* y, const float* x, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) y[i] += x[i];
}

// router: probs = sigmoid(logits), top TOPK of probs + bias, weights = probs / max(sum, 6.1e-5) * scale. One warp a token
__global__ void k_route(const float* logits, const float* bias, float wscale, int* topi, float* topw) {
    const int t = blockIdx.x, l = threadIdx.x;
    float pr[NEXP / 32], sc[NEXP / 32];
    for (int k = 0; k < NEXP / 32; ++k) {
        pr[k] = sigm(logits[(size_t) t * NEXP + k * 32 + l]);
        sc[k] = pr[k] + bias[k * 32 + l];
    }
    float wsum = 0, w[TOPK];
    int id[TOPK];
    for (int j = 0; j < TOPK; ++j) {
        float best = -INFINITY;
        int bi = 0x7fffffff;
        for (int k = 0; k < NEXP / 32; ++k)
            if (sc[k] > best) { best = sc[k]; bi = k * 32 + l; }
        for (int o = 16; o; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
        }
        id[j] = bi;
        if ((bi & 31) == l) sc[bi >> 5] = -INFINITY;
        w[j] = __shfl_sync(0xffffffffu, (bi & 31) == l ? pr[bi >> 5] : 0.0f, bi & 31);
        wsum += w[j];
    }
    if (l == 0) {
        const float k = wscale / fmaxf(wsum, 6.103515625e-5f);
        for (int j = 0; j < TOPK; ++j) {
            topi[t * TOPK + j] = id[j];
            topw[t * TOPK + j] = w[j] * k;
        }
    }
}

// IQ3_XXS rows times a 1536-wide input, one warp a row. Jobs: (input row, matrix in VRAM, output row); grid.y = job
struct EJob {
    const uint8_t* m;
    int in, out;
};
__global__ void k_iq3_gemv(const EJob* jobs, const float* x, int ldx, float* y, int ldy, int rows, int cols) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), l = threadIdx.x & 31;
    const EJob J = jobs[blockIdx.y];
    if (r >= rows) return;
    const uint8_t* row = J.m + (size_t) r * (cols / 256) * 98;
    const float* xr = x + (size_t) J.in * ldx;
    const int ib = l >> 2, l4 = l & 3;
    float acc = 0;
    for (int b = 0; b < cols / 256; ++b) {
        const uint8_t* blk = row + (size_t) b * 98;
        const float d = __half2float(*(const __half*) blk);
        const uint8_t* qs = blk + 2;
        uint32_t aux;
        memcpy(&aux, qs + 64 + 4 * ib, 4);
        const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
        const uint8_t signs = d_ksigns_iq2xs[(aux >> (7 * l4)) & 127];
        const uint32_t g1 = d_iq3xxs_grid[qs[8 * ib + 2 * l4]], g2 = d_iq3xxs_grid[qs[8 * ib + 2 * l4 + 1]];
        const float* xx = xr + b * 256 + ib * 32 + l4 * 8;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += db * (float) ((g1 >> (8 * j)) & 255) * ((signs >> j) & 1 ? -1.0f : 1.0f) * xx[j];
            acc += db * (float) ((g2 >> (8 * j)) & 255) * ((signs >> (j + 4)) & 1 ? -1.0f : 1.0f) * xx[4 + j];
        }
    }
    acc = warp_sum(acc);
    if (l == 0) y[(size_t) J.out * ldy + r] = acc;
}

// v1's IQ3_XXS kernel: a block of 8 warps holds 2 rows a warp of one job (grid.y); the job's input, the grid and the sign table
// sit in shared memory; lane l's two grid indices are the 16-bit word at 2 + 2l of each 98-byte block, its sub-block's scale and
// signs the 32-bit word at 66 + 4 (l / 4) (read as two 16-bit halves: blocks are 2-byte aligned only)
constexpr int IQ_R = 2, IQ_BR = 8 * IQ_R, IQ_MAXC = 1536;
__global__ void __launch_bounds__(256) k_iq3_rows(const EJob* jobs, const float* x, int ldx, float* y, int ldy, int rows, int cols) {
    __shared__ __align__(16) float xs[IQ_MAXC];
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    const EJob J = jobs[blockIdx.y];
    const float* xr = x + (size_t) J.in * ldx;
    for (int i = threadIdx.x; i < cols; i += 256) xs[i] = xr[i];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    const int r0 = blockIdx.x * IQ_BR + warp * IQ_R, nb = cols / 256;
    float acc[IQ_R];
#pragma unroll
    for (int a = 0; a < IQ_R; ++a) acc[a] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float4 xa = *(const float4*) &xs[b * 256 + l * 8], xb = *(const float4*) &xs[b * 256 + l * 8 + 4];
#pragma unroll
        for (int a = 0; a < IQ_R; ++a) {
            const int r = r0 + a;
            if (r >= rows) break;
            const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) r * nb + b) * 98);
            const float d = __half2float(__ushort_as_half(__ldg(blk)));
            const uint16_t qq = __ldg(blk + 1 + l);
            const uint32_t aux = (uint32_t) __ldg(blk + 33 + 2 * ib) | ((uint32_t) __ldg(blk + 34 + 2 * ib) << 16);
            const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
            const uint32_t s = sg[(aux >> (7 * l4)) & 127];
            const uint32_t g1 = grid[qq & 255], g2 = grid[qq >> 8];
            float v = 0;
            v += (float) (g1 & 255) * (s & 1 ? -xa.x : xa.x);
            v += (float) ((g1 >> 8) & 255) * (s & 2 ? -xa.y : xa.y);
            v += (float) ((g1 >> 16) & 255) * (s & 4 ? -xa.z : xa.z);
            v += (float) (g1 >> 24) * (s & 8 ? -xa.w : xa.w);
            v += (float) (g2 & 255) * (s & 16 ? -xb.x : xb.x);
            v += (float) ((g2 >> 8) & 255) * (s & 32 ? -xb.y : xb.y);
            v += (float) ((g2 >> 16) & 255) * (s & 64 ? -xb.z : xb.z);
            v += (float) (g2 >> 24) * (s & 128 ? -xb.w : xb.w);
            acc[a] += db * v;
        }
    }
#pragma unroll
    for (int a = 0; a < IQ_R; ++a) {
        const float s = warp_sum(acc[a]);
        if (l == 0 && r0 + a < rows) y[(size_t) J.out * ldy + r0 + a] = s;
    }
}

// moe[t] = sum over k of topw[t][k] * y[t * TOPK + k], in k order
__global__ void k_moe_sum(const float* y, const float* topw, float* moe) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < LAT; i += blockDim.x) {
        float a = 0;
        for (int k = 0; k < TOPK; ++k) a += topw[t * TOPK + k] * y[((size_t) t * TOPK + k) * LAT + i];
        moe[(size_t) t * LAT + i] = a;
    }
}

// Engram: rows [T][16][96] -> rms per row (eps 1.19e-7) * lookup_norm [16][96] (valid 0: the token has no n-gram)
__global__ void k_engram_rows(float* rows, const float* lnorm) {
    __shared__ float red[32];
    float* r = rows + ((size_t) blockIdx.x * E_COLS + blockIdx.y) * E_DIM;
    float v = threadIdx.x < E_DIM ? r[threadIdx.x] : 0.0f;
    const float s = block_sum(v * v, red);
    const float k = rsqrtf(s / E_DIM + ENGRAM_EPS);
    if (threadIdx.x < E_DIM) r[threadIdx.x] = v * k * lnorm[blockIdx.y * E_DIM + threadIdx.x];
}
// kv [T][HC + 1][N] (+ bias) -> per stream: gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)), dot = <rms(R_c), rms(key_c) * g_c>
// / sqrt(N); R_c += rms(value) * gamma * (gate_c + hcs_c), only where valid[t]
__global__ void k_engram_apply(float* R, const float* kv, const float* kvb, const float* egate, const float* gamma, const float* hcs,
                               const int* valid) {
    __shared__ float red[32];
    __shared__ float gate[HC], kval;
    const int t = blockIdx.x;
    const float* k = kv + (size_t) t * (HC + 1) * N;
    float* r = R + (size_t) t * D;
    float sv = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float v = k[HC * N + i] + kvb[HC * N + i];
        sv += v * v;
    }
    sv = block_sum(sv, red);
    if (threadIdx.x == 0) kval = rsqrtf(sv / N + ENGRAM_EPS);
    for (int c = 0; c < HC; ++c) {
        float sr = 0, sk = 0, dt = 0;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            const float a = r[c * N + i], b = k[c * N + i] + kvb[c * N + i];
            sr += a * a;
            sk += b * b;
            dt += a * b * egate[c * N + i];
        }
        sr = block_sum(sr, red);
        sk = block_sum(sk, red);
        dt = block_sum(dt, red);
        if (threadIdx.x == 0) {
            const float dot = dt * rsqrtf(sr / N + ENGRAM_EPS) * rsqrtf(sk / N + ENGRAM_EPS) / sqrtf((float) N);
            const float mg = sqrtf(fmaxf(fabsf(dot), 1e-6f));
            gate[c] = sigm(dot < 0 ? -mg : mg);
        }
    }
    __syncthreads();
    if (!valid[t]) return;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float sig = (k[HC * N + i] + kvb[HC * N + i]) * kval * gamma[i];
#pragma unroll
        for (int c = 0; c < HC; ++c) r[c * N + i] += sig * (gate[c] + hcs[c * N + i]);
    }
}

// indexer: from ik (rms'd key [T][128]) the cache row value | gate at pos (iv, ig given as [T][128] each)
__global__ void k_idx_store(const float* iv, const float* ig, float* ivg, const int* pos) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < IDXD; i += blockDim.x) {
        ivg[(size_t) (pos[t] % IVR) * 2 * IDXD + i] = iv[(size_t) t * IDXD + i];
        ivg[(size_t) (pos[t] % IVR) * 2 * IDXD + IDXD + i] = ig[(size_t) t * IDXD + i];
    }
}
// a completed group g: pooled = sum_m softmax_m(gate_m + ape_m) * value_m per channel, rms * norm, rope of the last 64 at 4g
__global__ void k_idx_pool(const float* ivg, const float* ape, const float* nrm, float eps, float base, const int* groups, float* pooled) {
    __shared__ float red[32];
    __shared__ float v[IDXD];
    const int g = groups[blockIdx.x], i = threadIdx.x;   // blockDim = IDXD
    float lg[KPOOL], mx = -INFINITY;
    for (int m = 0; m < KPOOL; ++m) {
        lg[m] = ivg[(size_t) ((g * KPOOL + m) % IVR) * 2 * IDXD + IDXD + i] + ape[m * IDXD + i];
        mx = fmaxf(mx, lg[m]);
    }
    float s = 0, a = 0;
    for (int m = 0; m < KPOOL; ++m) {
        const float e = __expf(lg[m] - mx);
        s += e;
        a += e * ivg[(size_t) ((g * KPOOL + m) % IVR) * 2 * IDXD + i];
    }
    a /= s;
    const float ss = block_sum(a * a, red);
    v[i] = a * rsqrtf(ss / IDXD + eps) * nrm[i];
    __syncthreads();
    if (i < ROT / 2) {   // rope NORM on the last 64 values at the group's first position
        const float th = (float) (g * KPOOL) * powf(base, -2.0f * i / ROT);
        float c, sn;
        sincosf(th, &sn, &c);
        const float x0 = v[IDXD - ROT + 2 * i], x1 = v[IDXD - ROT + 2 * i + 1];
        v[IDXD - ROT + 2 * i] = x0 * c - x1 * sn;
        v[IDXD - ROT + 2 * i + 1] = x0 * sn + x1 * c;
    }
    __syncthreads();
    pooled[(size_t) g * IDXD + i] = v[i];
}
// selection for one token: complete groups before its own; <= TOPPOOL of them: every position (nsel = -1), else the top TOPPOOL
// groups by sum_h w_h relu(q_h . pooled_g), their 4 positions each, then the token's own group up to its position
__global__ void k_idx_select(const float* iq, const float* w, const float* pooled, const int* pos, int* sel, int* nsel) {
    extern __shared__ float sm[];
    const int t = blockIdx.x, p = pos[t], ng = p / KPOOL;
    if (ng <= TOPPOOL) {
        if (threadIdx.x == 0) nsel[t] = -1;
        return;
    }
    float* sc = sm;                                   // [ng]
    int* id = (int*) (sm + ng);                       // [ng]
    for (int g = threadIdx.x; g < ng; g += blockDim.x) {
        float s = 0;
        for (int h = 0; h < IDXH; ++h) {
            float d = 0;
            for (int i = 0; i < IDXD; ++i) d += iq[((size_t) t * IDXH + h) * IDXD + i] * pooled[(size_t) g * IDXD + i];
            s += w[t * IDXH + h] * fmaxf(d, 0.0f);
        }
        sc[g] = s;
        id[g] = g;
    }
    __syncthreads();
    // TOPPOOL largest by repeated selection of the threshold: sort descending (odd-even, small n in v0)
    for (int ph = 0; ph < ng; ++ph) {
        for (int k = threadIdx.x * 2 + (ph & 1); k + 1 < ng; k += blockDim.x * 2) {
            if (sc[k] < sc[k + 1] || (sc[k] == sc[k + 1] && id[k] > id[k + 1])) {
                const float a = sc[k]; sc[k] = sc[k + 1]; sc[k + 1] = a;
                const int b = id[k]; id[k] = id[k + 1]; id[k + 1] = b;
            }
        }
        __syncthreads();
    }
    for (int j = threadIdx.x; j < TOPPOOL; j += blockDim.x)
        for (int m = 0; m < KPOOL; ++m) sel[(size_t) t * MAXSEL + j * KPOOL + m] = id[j] * KPOOL + m;
    if (threadIdx.x == 0) {
        int n = TOPPOOL * KPOOL;
        for (int q = ng * KPOOL; q <= p; ++q) sel[(size_t) t * MAXSEL + n++] = q;
        nsel[t] = n;
    }
}

__global__ void k_argmax(const float* logits, int n, int* out, float* outv) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float best = -INFINITY;
    int id = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (logits[i] > best) { best = logits[i]; id = i; }
    for (int o = 16; o; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, id, o);
        if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = best; bi[threadIdx.x >> 5] = id; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w)
            if (bv[w] > best || (bv[w] == best && bi[w] < id)) { best = bv[w]; id = bi[w]; }
        if (bv[0] > best || (bv[0] == best && bi[0] < id)) { best = bv[0]; id = bi[0]; }
        *out = id;
        *outv = best;
    }
}


}  // namespace hy4
