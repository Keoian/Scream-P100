// hy4_pf2.cuh (2026-11-21): v2's prompt-path kernels. Exact (no rounding beyond the f32 path's own): the token rows expanded on the
// card, the indexer's groups on the card, the sparse layers' list attention in one pass a token. Behind HY4_PF_F16 (f16 sums, the
// arithmetic of llama.cpp's and Flash-Next's prompt paths on the P100): dense matrices through HGEMM, the experts in half2, the window
// attention on its f16 rows.
#pragma once
#include <cuda_fp16.h>

namespace hy4 {

// ------------------------------------------------------------------------------------------------------------- exact (v2a, v2b)
// v2a: a chunk's token rows (Q8_0, as in the GGUF) -> R [C][D], each value d * q into all HC streams (the host loop's bits)
__global__ void k_embed_q8(const uint8_t* rows, size_t rb, float* R) {
    const uint8_t* r = rows + (size_t) blockIdx.x * rb;
    float* y = R + (size_t) blockIdx.x * D;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const int b = i >> 5, j = i & 31;
        const float v = __half2float(__ushort_as_half(*(const uint16_t*) (r + b * 34))) * (float) (int8_t) r[b * 34 + 2 + j];
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) y[cc * N + i] = v;
    }
}
__global__ void k_iota(int* g, int g0, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) g[i] = g0 + i;
}

// v2b: the sparse layers' list mode in one pass a token (v1t: the selected rows gathered into a copy, ~2.4 MB a token and layer,
// then cuBLAS's small-N kernels at ~90 GB/s, with Q and P rounded to f16). A block a token, a warp a head; the token's selected
// cache rows in tiles of SA_R rows straight from the cache (the next tile in registers during this one's math), Q in f32, the
// scores, an online softmax with the sink as one more logit without a value, and P V all in f32
constexpr int SA_R = 8, SA_LD = KROW + 8;   // rows a tile; a row's halves in shared memory (292 words: conflict-free half2 reads)
__global__ void __launch_bounds__(512) k_sel_attn(const float* qa, const __half* cache, const int* sel, const int* nsel, int H,
                                                  const float* sinks, float scale, float* o) {
    __shared__ __align__(16) float qs[16 * KROW];
    __shared__ __align__(16) __half ks[SA_R * SA_LD];
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31, nt = blockDim.x;
    const int n = nsel[t];
    for (int i = threadIdx.x; i < H * KROW; i += nt) qs[i] = qa[(size_t) t * H * KROW + i];
    const int* sl = sel + (size_t) t * MAXSEL;
    constexpr int PER = SA_R * (KROW / 8);   // int4 a tile
    int4 pre[2];
    auto load = [&](int j0) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int idx = threadIdx.x + k * nt;
            pre[k] = make_int4(0, 0, 0, 0);
            if (idx < PER) {
                const int rr = idx / (KROW / 8), cc = idx % (KROW / 8);
                if (j0 + rr < n) pre[k] = __ldg((const int4*) (cache + (size_t) sl[j0 + rr] * KROW) + cc);
            }
        }
    };
    const int r = lane >> 2, part = lane & 3;   // scores: row r of the tile, dims 8 i + 2 part and + 1
    float m = sinks[w], l = 1.0f;
    float acc[16];   // P V: dims 64 j + 2 lane and + 1 (acc[2 j], acc[2 j + 1])
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
    const int ntile = (n + SA_R - 1) / SA_R;
    if (ntile > 0) load(0);
    const float2* qq = (const float2*) (qs + w * KROW) + part;
    for (int ti = 0; ti < ntile; ++ti) {
        const int j0 = ti * SA_R;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int idx = threadIdx.x + k * nt;
            if (idx < PER) *(int4*) (ks + (idx / (KROW / 8)) * SA_LD + (idx % (KROW / 8)) * 8) = pre[k];
        }
        __syncthreads();
        if (ti + 1 < ntile) load(j0 + SA_R);
        if (w >= H) continue;
        const __half2* kr = (const __half2*) (ks + r * SA_LD) + part;
        float s = 0.0f;
#pragma unroll 8
        for (int i = 0; i < KROW / 8; ++i) {
            const float2 kv = __half22float2(kr[4 * i]), qv = qq[4 * i];
            s = fmaf(qv.x, kv.x, s);
            s = fmaf(qv.y, kv.y, s);
        }
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        const bool ok = j0 + r < n;
        const float sc = ok ? s * scale : -INFINITY;
        float mt = sc;
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 4));
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 8));
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 16));
        const float mn = fmaxf(m, mt), corr = __expf(m - mn);
        const float p = ok ? __expf(sc - mn) : 0.0f;
        float ps = p;   // the 8 rows' p (each row's 4 lanes hold the same)
        ps += __shfl_xor_sync(0xffffffffu, ps, 4);
        ps += __shfl_xor_sync(0xffffffffu, ps, 8);
        ps += __shfl_xor_sync(0xffffffffu, ps, 16);
        l = l * corr + ps;
        m = mn;
#pragma unroll
        for (int j = 0; j < 16; ++j) acc[j] *= corr;
#pragma unroll
        for (int rr = 0; rr < SA_R; ++rr) {
            const float pr = __shfl_sync(0xffffffffu, p, rr * 4);
            const __half2* vr = (const __half2*) (ks + rr * SA_LD) + lane;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 v = __half22float2(vr[32 * j]);
                acc[2 * j] = fmaf(pr, v.x, acc[2 * j]);
                acc[2 * j + 1] = fmaf(pr, v.y, acc[2 * j + 1]);
            }
        }
    }
    if (w >= H) return;
    float2* out = (float2*) (o + ((size_t) t * H + w) * KVL) + lane;
    const float inv = 1.0f / l;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[32 * j] = make_float2(acc[2 * j] * inv, acc[2 * j + 1] * inv);
}

// v2j: k_sel_attn with Q in registers (lane l: dims 2 l + 64 i and + 1, i < 9) instead of shared memory, so a tile holds 16 rows and
// two blocks fit an SM; each row's score is a butterfly sum over the warp (every lane ends with all 16 scores: the softmax needs no
// broadcast). f32 throughout like k_sel_attn (its sums in another order)
constexpr int SA2_R = 16;
template <int MINB>   // blocks an SM: 2 (64 registers, some spill) or 1
__global__ void __launch_bounds__(512, MINB) k_sel_attn2(const float* qa, const __half* cache, const int* sel, const int* nsel, int H,
                                                      const float* sinks, float scale, float* o) {
    __shared__ __align__(16) __half ks[SA2_R * SA_LD];
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31, nt = blockDim.x;
    const int n = nsel[t];
    const int* sl = sel + (size_t) t * MAXSEL;
    float2 q[9];
    {
        const float2* qr = (const float2*) (qa + ((size_t) t * H + min(w, H - 1)) * KROW) + lane;
#pragma unroll
        for (int i = 0; i < 9; ++i) q[i] = qr[32 * i];
    }
    constexpr int PER = SA2_R * (KROW / 8);   // int4 a tile (1152), straight into shared memory (the other block of the SM hides it)
    float m = sinks[min(w, H - 1)], l = 1.0f;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
    const int ntile = (n + SA2_R - 1) / SA2_R;
    for (int ti = 0; ti < ntile; ++ti) {
        const int j0 = ti * SA2_R;
        __syncthreads();
        for (int idx = threadIdx.x; idx < PER; idx += nt) {
            const int rr = idx / (KROW / 8), cc = idx % (KROW / 8);
            *(int4*) (ks + rr * SA_LD + cc * 8) =
                j0 + rr < n ? __ldg((const int4*) (cache + (size_t) sl[j0 + rr] * KROW) + cc) : make_int4(0, 0, 0, 0);
        }
        __syncthreads();
        if (w >= H) continue;
        for (int g = 0; g < SA2_R; g += 4) {   // 4 rows at a time (16 would not stay in registers)
            float s[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const __half2* kr = (const __half2*) (ks + (g + r) * SA_LD) + lane;
                float a = 0.0f;
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const float2 kv = __half22float2(kr[32 * i]);
                    a = fmaf(q[i].x, kv.x, a);
                    a = fmaf(q[i].y, kv.y, a);
                }
                s[r] = a;
            }
#pragma unroll
            for (int off = 16; off; off >>= 1)
#pragma unroll
                for (int r = 0; r < 4; ++r) s[r] += __shfl_xor_sync(0xffffffffu, s[r], off);
            float mt = m;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                s[r] = j0 + g + r < n ? s[r] * scale : -INFINITY;
                mt = fmaxf(mt, s[r]);
            }
            const float corr = __expf(m - mt);
            float ps = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                s[r] = j0 + g + r < n ? __expf(s[r] - mt) : 0.0f;
                ps += s[r];
            }
            l = l * corr + ps;
            m = mt;
#pragma unroll
            for (int j = 0; j < 16; ++j) acc[j] *= corr;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const __half2* vr = (const __half2*) (ks + (g + r) * SA_LD) + lane;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float2 v = __half22float2(vr[32 * j]);
                    acc[2 * j] = fmaf(s[r], v.x, acc[2 * j]);
                    acc[2 * j + 1] = fmaf(s[r], v.y, acc[2 * j + 1]);
                }
            }
        }
    }
    if (w >= H) return;
    float2* out = (float2*) (o + ((size_t) t * H + w) * KVL) + lane;
    const float inv = 1.0f / l;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[32 * j] = make_float2(acc[2 * j] * inv, acc[2 * j + 1] * inv);
}

// --------------------------------------------------------------------------------------------------- f16 (HY4_PF_F16, v2c-v2e)
// v2c: dense matrices as HGEMM (f16 in, f16 sums: 13-15 TFLOPS on the P100 against SGEMM's 7-8; f16 in with f32 sums is no faster)
__global__ void k_deq8h(const int8_t* q, const __half* d, size_t n, int cols, __half* out) {   // k_deq8 into f16
    const int rows = (int) (n / (size_t) cols), nb = cols / 32;
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        const int8_t* qr = q + (size_t) r * cols;
        const __half* dr = d + (size_t) r * nb;
        __half* o = out + (size_t) r * cols;
        for (int c = threadIdx.x; c < cols; c += blockDim.x) o[c] = __float2half((float) qr[c] * __half2float(dr[c >> 5]));
    }
}
__global__ void k_f2h_2d(const float* x, int ldx, int rows, int cols, __half* out) {   // x [rows][ldx] -> out [rows][cols]
    for (int r = blockIdx.x; r < rows; r += gridDim.x)
        for (int c = threadIdx.x; c < cols; c += blockDim.x) out[(size_t) r * cols + c] = __float2half(x[(size_t) r * ldx + c]);
}
__global__ void k_h2f_2d(const __half* y, int rows, int cols, float* out, int ldo) {   // y [rows][cols] -> out [rows][ldo]
    for (int r = blockIdx.x; r < rows; r += gridDim.x)
        for (int c = threadIdx.x; c < cols; c += blockDim.x) out[(size_t) r * ldo + c] = __half2float(y[(size_t) r * cols + c]);
}
// HY4_PF_SELCMP / HY4_PF_IQCMP: the largest |a - b| and |b| over n values, as float bits (both >= 0) into out[0], out[1]
__global__ void k_cmp_max(const float* a, const float* b, size_t n, unsigned* out) {
    float d = 0, m = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        d = fmaxf(d, fabsf(a[i] - b[i]));
        m = fmaxf(m, fabsf(b[i]));
    }
    atomicMax(out, __float_as_uint(d));
    atomicMax(out + 1, __float_as_uint(m));
}
// a count of non-finite values (HY4_PF_CHECK)
__global__ void k_nonfinite(const float* x, size_t n, unsigned* cnt) {
    unsigned k = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) k += !isfinite(x[i]);
    if (k) atomicAdd(cnt, k);
}

// v2d: k_iq3_gemm3 with half2 products: a step's 32 columns are one sub-block of each row (one scale), so the weights go to shared
// memory as their signed grid values / 64 (exact in f16: at most 62 / 64), the inputs as f16, each thread sums its 8 x 4 tile over
// the step's 32 columns in half2 (2 tokens a half2) and adds sum * 64 * scale into f32. Sums of 32 terms of |w| <= 1 stay far below
// f16's range for inputs up to ~2000
template <int NB, bool SKIP = false>
__global__ void __launch_bounds__(256) k_iq3_gemm4(const PJob* jobs, const int2* lists, const float* x, int ldx, float* y, int ldy,
                                                   int rows, int in_is_pair) {
    __shared__ __align__(16) __half ws[G3K][G3M];
    __shared__ __align__(16) __half xs[G3K][G3N];
    __shared__ float dbs[G3M];
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    __shared__ int2 lt[G3N];
    const PJob J = jobs[blockIdx.y];
    if (J.n == 0) return;
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    const int r0 = blockIdx.x * G3M, ty = SKIP ? threadIdx.x & 15 : threadIdx.x >> 4, tx = SKIP ? threadIdx.x >> 4 : threadIdx.x & 15;
    const int wtok = (threadIdx.x >> 5) * 8;   // SKIP: as k_iq3_gemm3
    const int dr = threadIdx.x & (G3M - 1), dh = threadIdx.x >> 7;
    const int tok = threadIdx.x >> 2, seg = (threadIdx.x & 3) * 8;
    const uint16_t* rowp = (const uint16_t*) (J.m + (size_t) (r0 + dr) * NB * 98);
    for (int t0 = 0; t0 < J.n; t0 += G3N) {
        const int nk = min(G3N, J.n - t0);
        __syncthreads();
        if (threadIdx.x < G3N) lt[threadIdx.x] = threadIdx.x < nk ? lists[J.list + t0 + threadIdx.x] : make_int2(-1, -1);
        __syncthreads();
        const float* xr = nullptr;
        if (tok < nk) xr = x + (size_t) (in_is_pair ? lt[tok].y : lt[tok].x) * ldx + seg;
        uint16_t rd, ra0, ra1, rq[2];
        float4 v0, v1;
        auto load = [&](int ks) {
            const int b = ks >> 3, ib = ks & 7;
            const uint16_t* blk = rowp + b * 49;
            rd = __ldg(blk);
            ra0 = __ldg(blk + 33 + 2 * ib);
            ra1 = __ldg(blk + 34 + 2 * ib);
            rq[0] = __ldg(blk + 1 + 4 * ib + 2 * dh);
            rq[1] = __ldg(blk + 2 + 4 * ib + 2 * dh);
            v0 = v1 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (xr) {
                v0 = __ldg((const float4*) (xr + ks * G3K));
                v1 = __ldg((const float4*) (xr + ks * G3K + 4));
            }
        };
        load(0);
        float acc[8][4];
#pragma unroll
        for (int a = 0; a < 8; ++a)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[a][k] = 0.0f;
        for (int ks = 0; ks < NB * 8; ++ks) {
            __syncthreads();
            {
                const uint32_t aux = (uint32_t) ra0 | ((uint32_t) ra1 << 16);
                if (dh == 0) dbs[dr] = __half2float(__ushort_as_half(rd)) * (0.5f + (float) (aux >> 28)) * 0.5f * 64.0f;
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int l4 = 2 * dh + p;
                    const uint32_t s = sg[(aux >> (7 * l4)) & 127];
                    const uint32_t g1 = grid[rq[p] & 255], g2 = grid[rq[p] >> 8];
                    const int k = l4 * 8;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const __half a1 = __float2half((float) ((g1 >> (8 * j)) & 255) * (1.0f / 64.0f));
                        const __half a2 = __float2half((float) ((g2 >> (8 * j)) & 255) * (1.0f / 64.0f));
                        ws[k + j][dr] = (s >> j) & 1 ? __hneg(a1) : a1;
                        ws[k + 4 + j][dr] = (s >> (4 + j)) & 1 ? __hneg(a2) : a2;
                    }
                }
                xs[seg + 0][tok] = __float2half(v0.x); xs[seg + 1][tok] = __float2half(v0.y);
                xs[seg + 2][tok] = __float2half(v0.z); xs[seg + 3][tok] = __float2half(v0.w);
                xs[seg + 4][tok] = __float2half(v1.x); xs[seg + 5][tok] = __float2half(v1.y);
                xs[seg + 6][tok] = __float2half(v1.z); xs[seg + 7][tok] = __float2half(v1.w);
            }
            __syncthreads();
            if (ks + 1 < NB * 8) load(ks + 1);
            if (SKIP && wtok >= nk) continue;
            __half2 hacc[8][2];
#pragma unroll
            for (int a = 0; a < 8; ++a) hacc[a][0] = hacc[a][1] = __float2half2_rn(0.0f);
#pragma unroll 8
            for (int k = 0; k < G3K; ++k) {
                const uint4 wv = *(const uint4*) &ws[k][ty * 8];
                const uint2 xv = *(const uint2*) &xs[k][tx * 4];
                const __half2 w01 = *(const __half2*) &wv.x, w23 = *(const __half2*) &wv.y, w45 = *(const __half2*) &wv.z,
                              w67 = *(const __half2*) &wv.w;
                const __half2 x01 = *(const __half2*) &xv.x, x23 = *(const __half2*) &xv.y;
                const __half2 wa[8] = {__low2half2(w01), __high2half2(w01), __low2half2(w23), __high2half2(w23),
                                       __low2half2(w45), __high2half2(w45), __low2half2(w67), __high2half2(w67)};
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    hacc[a][0] = __hfma2(wa[a], x01, hacc[a][0]);
                    hacc[a][1] = __hfma2(wa[a], x23, hacc[a][1]);
                }
            }
#pragma unroll
            for (int a = 0; a < 8; ++a) {
                const float d = dbs[ty * 8 + a];
                const float2 f0 = __half22float2(hacc[a][0]), f1 = __half22float2(hacc[a][1]);
                acc[a][0] = fmaf(d, f0.x, acc[a][0]);
                acc[a][1] = fmaf(d, f0.y, acc[a][1]);
                acc[a][2] = fmaf(d, f1.x, acc[a][2]);
                acc[a][3] = fmaf(d, f1.y, acc[a][3]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int tk = tx * 4 + i;
            if (tk >= nk) continue;
            float* yr = y + (size_t) lt[tk].y * ldy + r0 + ty * 8;
#pragma unroll
            for (int a = 0; a < 8; ++a) yr[a] = acc[a][i];
        }
    }
}

// v2h: k_iq3_gemm4 with 64-column steps (two sub-blocks: half the barriers and loads a column) and the products as row pairs x
// one token (the weights' half2 as they lie in shared memory, the token broadcast: 4 conversions a column instead of 8). Each
// output's f16 sum over a sub-block and its f32 fold are those of k_iq3_gemm4 in the same order, so the bits are the same
constexpr int G5K = 64;
template <int NB, bool SKIP = true>
__global__ void __launch_bounds__(256) k_iq3_gemm5(const PJob* jobs, const int2* lists, const float* x, int ldx, float* y, int ldy,
                                                   int rows, int in_is_pair) {
    __shared__ __align__(16) __half ws[G5K][G3M];
    __shared__ __align__(16) __half xs[G5K][G3N];
    __shared__ float dbs[2][G3M];
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    __shared__ int2 lt[G3N];
    const PJob J = jobs[blockIdx.y];
    if (J.n == 0) return;
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    const int r0 = blockIdx.x * G3M, ty = SKIP ? threadIdx.x & 15 : threadIdx.x >> 4, tx = SKIP ? threadIdx.x >> 4 : threadIdx.x & 15;
    const int wtok = (threadIdx.x >> 5) * 8;
    const int dr = threadIdx.x & (G3M - 1), dh = threadIdx.x >> 7;   // the row and which of the step's two sub-blocks
    const int tok = threadIdx.x >> 2, seg = (threadIdx.x & 3) * 16;
    const uint16_t* rowp = (const uint16_t*) (J.m + (size_t) (r0 + dr) * NB * 98);
    for (int t0 = 0; t0 < J.n; t0 += G3N) {
        const int nk = min(G3N, J.n - t0);
        __syncthreads();
        if (threadIdx.x < G3N) lt[threadIdx.x] = threadIdx.x < nk ? lists[J.list + t0 + threadIdx.x] : make_int2(-1, -1);
        __syncthreads();
        const float* xr = nullptr;
        if (tok < nk) xr = x + (size_t) (in_is_pair ? lt[tok].y : lt[tok].x) * ldx + seg;
        uint16_t rd, ra0, ra1, rq[4];
        float4 v[4];
        auto load = [&](int ks) {   // step ks: block ks / 4, sub-block 2 (ks % 4) + dh
            const int b = ks >> 2, ib = 2 * (ks & 3) + dh;
            const uint16_t* blk = rowp + b * 49;
            rd = __ldg(blk);
            ra0 = __ldg(blk + 33 + 2 * ib);
            ra1 = __ldg(blk + 34 + 2 * ib);
#pragma unroll
            for (int q = 0; q < 4; ++q) rq[q] = __ldg(blk + 1 + 4 * ib + q);
#pragma unroll
            for (int q = 0; q < 4; ++q) v[q] = xr ? __ldg((const float4*) (xr + ks * G5K) + q) : make_float4(0.f, 0.f, 0.f, 0.f);
        };
        load(0);
        float acc[8][4];
#pragma unroll
        for (int a = 0; a < 8; ++a)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[a][k] = 0.0f;
        for (int ks = 0; ks < NB * 4; ++ks) {
            __syncthreads();
            {
                const uint32_t aux = (uint32_t) ra0 | ((uint32_t) ra1 << 16);
                dbs[dh][dr] = __half2float(__ushort_as_half(rd)) * (0.5f + (float) (aux >> 28)) * 0.5f * 64.0f;
#pragma unroll
                for (int l4 = 0; l4 < 4; ++l4) {
                    const uint32_t s = sg[(aux >> (7 * l4)) & 127];
                    const uint32_t g1 = grid[rq[l4] & 255], g2 = grid[rq[l4] >> 8];
                    const int k = 32 * dh + l4 * 8;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const __half a1 = __float2half((float) ((g1 >> (8 * j)) & 255) * (1.0f / 64.0f));
                        const __half a2 = __float2half((float) ((g2 >> (8 * j)) & 255) * (1.0f / 64.0f));
                        ws[k + j][dr] = (s >> j) & 1 ? __hneg(a1) : a1;
                        ws[k + 4 + j][dr] = (s >> (4 + j)) & 1 ? __hneg(a2) : a2;
                    }
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    xs[seg + 4 * q + 0][tok] = __float2half(v[q].x);
                    xs[seg + 4 * q + 1][tok] = __float2half(v[q].y);
                    xs[seg + 4 * q + 2][tok] = __float2half(v[q].z);
                    xs[seg + 4 * q + 3][tok] = __float2half(v[q].w);
                }
            }
            __syncthreads();
            if (ks + 1 < NB * 4) load(ks + 1);
            if (SKIP && wtok >= nk) continue;
#pragma unroll
            for (int sb = 0; sb < 2; ++sb) {
                __half2 hacc[4][4];   // [row pair][token]
#pragma unroll
                for (int p = 0; p < 4; ++p)
#pragma unroll
                    for (int i = 0; i < 4; ++i) hacc[p][i] = __float2half2_rn(0.0f);
#pragma unroll 8
                for (int k = 32 * sb; k < 32 * sb + 32; ++k) {
                    const uint4 wv = *(const uint4*) &ws[k][ty * 8];
                    const uint2 xv = *(const uint2*) &xs[k][tx * 4];
                    const __half2 x01 = *(const __half2*) &xv.x, x23 = *(const __half2*) &xv.y;
                    const __half2 xb[4] = {__low2half2(x01), __high2half2(x01), __low2half2(x23), __high2half2(x23)};
                    const __half2 wp[4] = {*(const __half2*) &wv.x, *(const __half2*) &wv.y, *(const __half2*) &wv.z, *(const __half2*) &wv.w};
#pragma unroll
                    for (int p = 0; p < 4; ++p)
#pragma unroll
                        for (int i = 0; i < 4; ++i) hacc[p][i] = __hfma2(wp[p], xb[i], hacc[p][i]);
                }
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float d0 = dbs[sb][ty * 8 + 2 * p], d1 = dbs[sb][ty * 8 + 2 * p + 1];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 f = __half22float2(hacc[p][i]);
                        acc[2 * p][i] = fmaf(d0, f.x, acc[2 * p][i]);
                        acc[2 * p + 1][i] = fmaf(d1, f.y, acc[2 * p + 1][i]);
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int tk = tx * 4 + i;
            if (tk >= nk) continue;
            float* yr = y + (size_t) lt[tk].y * ldy + r0 + ty * 8;
#pragma unroll
            for (int a = 0; a < 8; ++a) yr[a] = acc[a][i];
        }
    }
}

// v2e: the window attention on f16 rows: the tile rows straight from the chunk's f16 rows (zero rows before position 0, as
// k_kf_fill), S = Q K^T and O = P V as HGEMM, the softmax in f32 from S (already scaled through the GEMM's alpha) into P in f16
__global__ void k_kfh_fill(const __half* kbuf, int npre, int nrows, __half* kf) {   // k_kf_fill in f16
    const int r = blockIdx.x, src = r - (SWA - 1 - npre);
    const bool ok = src >= 0 && src < nrows;
    for (int c = threadIdx.x; c < KROW / 8; c += blockDim.x)
        ((int4*) (kf + (size_t) r * KROW))[c] = ok ? ((const int4*) (kbuf + (size_t) src * KROW))[c] : make_int4(0, 0, 0, 0);
}
__global__ void k_swa_softmax_h(__half* S, int Mq, int H, int C, int p0, int nrows_total, const float* sinks) {
    const int g = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (g >= nrows_total) return;
    const int i = g / Mq, q = g % Mq, lt = q / H, h = q % H, t = i * SWTT + lt;
    __half* s = S + (size_t) g * SWT;
    if (t >= C) {
        for (int j = lane; j < SWT; j += 32) s[j] = __float2half(0.0f);
        return;
    }
    const int pbase = p0 - (SWA - 1) + i * SWTT;
    float v[SWT / 32];
    float m = sinks[h];
#pragma unroll
    for (int k = 0; k < SWT / 32; ++k) {
        const int j = lane + 32 * k;
        const bool ok = j >= lt && j <= lt + SWA - 1 && pbase + j >= 0;
        v[k] = ok ? __half2float(s[j]) : -INFINITY;
        m = fmaxf(m, v[k]);
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0;
#pragma unroll
    for (int k = 0; k < SWT / 32; ++k) {
        v[k] = v[k] == -INFINITY ? 0.0f : __expf(v[k] - m);
        sum += v[k];
    }
    sum = warp_sum(sum);
    const float inv = 1.0f / (sum + __expf(sinks[h] - m));
#pragma unroll
    for (int k = 0; k < SWT / 32; ++k) s[lane + 32 * k] = __float2half(v[k] * inv);
}

// v2i: the cross-card sum of a chunk in one kernel: each block writes its slice to host memory, sets its flag, waits for the other
// card's flag of the same slice and adds that slice (k_xput_pf then k_xget_pf: the card-to-host writes of all slices before any
// host-to-card read; here the two directions of the link overlap). y + in as before: both cards hold the same bits
constexpr int PXB2 = 64;   // blocks (the flags: 64 a card); 1024 threads, all resident at once (2 a SM)
__global__ void __launch_bounds__(1024, 2) k_xsum_pf(float* y, int n, float* out, volatile unsigned* myflags, const volatile float* in,
                                                  const volatile unsigned* flags, unsigned seq) {
    const int per = (n + PXB2 - 1) / PXB2, b0 = blockIdx.x * per, b1 = min(n, b0 + per);
    for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) out[i] = y[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        myflags[blockIdx.x] = seq;
        while (flags[blockIdx.x] < seq) {}
    }
    __syncthreads();
    __threadfence_system();
    for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) y[i] = y[i] + in[i];
}

}  // namespace hy4
