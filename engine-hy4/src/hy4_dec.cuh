// hy4_dec.cuh (2026-11-20): v1g's writing kernels, after nvprof on 200 tokens (ms a token a card): k_iq3_rows 8.4 (~107 GB/s),
// k_mla_attn 4.0 (a block a head: 16 or 32 blocks, uncoalesced rows), k_gemv32 for the router 2.0 (32 blocks), k_route 1.3 (its
// arrays indexed at run time live in local memory). Included after hy4_kern.cuh
#include <initializer_list>
#include <utility>

namespace hy4 {

// ------------------------------------------------------------------------------------------------------------ attention
// Every head of a card reads the same latent rows, so a block takes a range of one token's positions for MHG heads: 32 rows at a
// time go to shared memory, a warp a head keeps its q in registers (18 values a lane), scores its 32 positions, and folds them
// into a running max, sum and output (16 values a lane) as an online softmax. Each block leaves its range's max, sum and
// unnormalised output per head; k_mla_comb adds the ranges with the sink. Grid (ranges, H / MHG, T)
constexpr int MHG = 8, MLA_PART = 4096;   // MLA_PART: (token, head, range) triples the partials hold
template <int MSUB>
__global__ void __launch_bounds__(256) k_mla_part(const float* qa, const __half* cache, const int* pos, int window, const int* sel,
                                                  const int* nsel, float scale, int H, int per, float* opart, float* mlpart) {
    __shared__ __align__(16) __half rows[MSUB][KROW];
    const int rg = blockIdx.x, nr = gridDim.x, t = blockIdx.z, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int h = blockIdx.y * MHG + w, p = pos[t];
    int n, lo = 0;
    const bool list = nsel && nsel[t] >= 0;
    if (list) n = nsel[t];
    else {
        lo = window ? max(0, p - SWA + 1) : 0;
        n = p - lo + 1;
    }
    float q[KROW / 32];
#pragma unroll
    for (int k = 0; k < KROW / 32; ++k) q[k] = qa[((size_t) t * H + h) * KROW + l + 32 * k] * scale;
    float m = -INFINITY, s = 0, o[KVL / 32];
#pragma unroll
    for (int k = 0; k < KVL / 32; ++k) o[k] = 0.0f;
    const int j0 = rg * per * MSUB, j1 = min(n, j0 + per * MSUB);
    for (int c0 = j0; c0 < j1; c0 += MSUB) {
        const int cnt = min(MSUB, j1 - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < cnt * (KROW / 2); i += blockDim.x) {
            const int j = i / (KROW / 2), k = i % (KROW / 2);
            const int pp = list ? sel[(size_t) t * MAXSEL + c0 + j] : lo + c0 + j;
            ((__half2*) rows[j])[k] = ((const __half2*) (cache + (size_t) (window ? pp % RING : pp) * KROW))[k];
        }
        __syncthreads();
        float mine = -INFINITY;   // lane j's score: position c0 + j
        for (int j = 0; j < cnt; ++j) {
            float a = 0;
#pragma unroll
            for (int k = 0; k < KROW / 32; ++k) a += q[k] * __half2float(rows[j][l + 32 * k]);
            a = warp_sum(a);
            if (l == j) mine = a;
        }
        float cm = mine;
        for (int k = 16; k; k >>= 1) cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, k));
        const float nm = fmaxf(m, cm), al = __expf(m - nm);   // m = -inf at first: al = 0
        const float e = l < cnt ? __expf(mine - nm) : 0.0f;
        s = s * al + warp_sum(e);
#pragma unroll
        for (int k = 0; k < KVL / 32; ++k) o[k] *= al;
        for (int j = 0; j < cnt; ++j) {
            const float pj = __shfl_sync(0xffffffffu, e, j);
#pragma unroll
            for (int k = 0; k < KVL / 32; ++k) o[k] += pj * __half2float(rows[j][l + 32 * k]);
        }
        m = nm;
    }
    const size_t th = ((size_t) t * H + h) * nr + rg;
#pragma unroll
    for (int k = 0; k < KVL / 32; ++k) opart[th * KVL + l + 32 * k] = o[k];
    if (l == 0) {
        mlpart[th * 2] = m;
        mlpart[th * 2 + 1] = s;
    }
}
__global__ void k_mla_comb(const float* opart, const float* mlpart, const float* sinks, int nr, float* o, int H) {
    const int h = blockIdx.x, t = blockIdx.y;
    const size_t th = ((size_t) t * H + h) * nr;
    float M = sinks[h];
    for (int r = 0; r < nr; ++r) M = fmaxf(M, mlpart[(th + r) * 2]);
    float den = __expf(sinks[h] - M);
    for (int r = 0; r < nr; ++r) {
        const float mr = mlpart[(th + r) * 2];
        if (mr > -INFINITY) den += mlpart[(th + r) * 2 + 1] * __expf(mr - M);
    }
    for (int i = threadIdx.x; i < KVL; i += blockDim.x) {
        float a = 0;
        for (int r = 0; r < nr; ++r) {
            const float mr = mlpart[(th + r) * 2];
            if (mr > -INFINITY) a += opart[(th + r) * KVL + i] * __expf(mr - M);
        }
        o[((size_t) t * H + h) * KVL + i] = a / den;
    }
}
// nmax: the most positions a token of the window reads. Ranges: one sub-chunk of rows each, or more when the partials (MLA_PART)
// would not hold that many
static int g_msub = 16;
template <int MSUB>
static void mla_dec_t(const float* qa, const __half* cache, const int* pos, int window, const int* sel, const int* nsel, const float* sinks,
                      float scale, float* o, int H, int T, int nmax, float* opart, float* mlpart, cudaStream_t st) {
    const int steps = std::max(1, (nmax + MSUB - 1) / MSUB);
    const int cap = std::max(1, MLA_PART / (T * H));
    int nr = std::min(steps, cap);
    const int per = (steps + nr - 1) / nr;
    nr = (steps + per - 1) / per;
    k_mla_part<MSUB><<<dim3(nr, H / MHG, T), 256, 0, st>>>(qa, cache, pos, window, sel, nsel, scale, H, per, opart, mlpart);
    k_mla_comb<<<dim3(H, T), 128, 0, st>>>(opart, mlpart, sinks, nr, o, H);
}
static void mla_dec(const float* qa, const __half* cache, const int* pos, int window, const int* sel, const int* nsel, const float* sinks,
                    float scale, float* o, int H, int T, int nmax, float* opart, float* mlpart, cudaStream_t st) {
    if (g_msub == 32) mla_dec_t<32>(qa, cache, pos, window, sel, nsel, sinks, scale, o, H, T, nmax, opart, mlpart, st);
    else if (g_msub == 8) mla_dec_t<8>(qa, cache, pos, window, sel, nsel, sinks, scale, o, H, T, nmax, opart, mlpart, st);
    else mla_dec_t<16>(qa, cache, pos, window, sel, nsel, sinks, scale, o, H, T, nmax, opart, mlpart, st);
}

// The second try (nvprof on the bench: the first one waits on its row loads, one 4-byte load and store after another): the rows
// come in 16-byte loads all issued before their stores, the scores go two positions at a time, the values as half2 (lane l holds
// columns 64k + 2l, 2l + 1: conflict-free). k_mla_comb2: a thread an output
template <int MSUB>
__global__ void __launch_bounds__(256) k_mla_part2(const float* qa, const __half* cache, const int* pos, int window, const int* sel,
                                                   const int* nsel, float scale, int H, int per, float* opart, float* mlpart) {
    constexpr int RV = KROW / 8, NU = (MSUB * RV + 255) / 256;
    __shared__ __align__(16) __half rows[MSUB][KROW];
    const int rg = blockIdx.x, nr = gridDim.x, t = blockIdx.z, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int h = blockIdx.y * MHG + w, p = pos[t];
    int n, lo = 0;
    const bool list = nsel && nsel[t] >= 0;
    if (list) n = nsel[t];
    else {
        lo = window ? max(0, p - SWA + 1) : 0;
        n = p - lo + 1;
    }
    float2 q[KROW / 64];
#pragma unroll
    for (int k = 0; k < KROW / 64; ++k) {
        const float2 v = *(const float2*) &qa[((size_t) t * H + h) * KROW + 64 * k + 2 * l];
        q[k] = make_float2(v.x * scale, v.y * scale);
    }
    float m = -INFINITY, s = 0;
    float2 o[KVL / 64];
#pragma unroll
    for (int k = 0; k < KVL / 64; ++k) o[k] = make_float2(0.0f, 0.0f);
    const int j0 = rg * per * MSUB, j1 = min(n, j0 + per * MSUB);
    for (int c0 = j0; c0 < j1; c0 += MSUB) {
        const int cnt = min(MSUB, j1 - c0), tot = cnt * RV;
        int4 buf[NU];
        __syncthreads();
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            const int i = threadIdx.x + 256 * u;
            if (i < tot) {
                const int j = i / RV, k = i - j * RV;
                const int pp = list ? sel[(size_t) t * MAXSEL + c0 + j] : lo + c0 + j;
                buf[u] = __ldg((const int4*) (cache + (size_t) (window ? pp % RING : pp) * KROW) + k);
            }
        }
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            const int i = threadIdx.x + 256 * u;
            if (i < tot) ((int4*) &rows[0][0])[i] = buf[u];
        }
        __syncthreads();
        float mine = -INFINITY;   // lane j's score: position c0 + j
        for (int j = 0; j < cnt; j += 2) {
            const int j2 = min(j + 1, cnt - 1);
            float a0 = 0, a1 = 0;
#pragma unroll
            for (int k = 0; k < KROW / 64; ++k) {
                const float2 r0 = __half22float2(*(const __half2*) &rows[j][64 * k + 2 * l]);
                const float2 r1 = __half22float2(*(const __half2*) &rows[j2][64 * k + 2 * l]);
                a0 += q[k].x * r0.x + q[k].y * r0.y;
                a1 += q[k].x * r1.x + q[k].y * r1.y;
            }
            for (int o2 = 16; o2; o2 >>= 1) {
                a0 += __shfl_xor_sync(0xffffffffu, a0, o2);
                a1 += __shfl_xor_sync(0xffffffffu, a1, o2);
            }
            if (l == j) mine = a0;
            if (l == j + 1 && j + 1 < cnt) mine = a1;
        }
        float cm = mine;
        for (int k = 16; k; k >>= 1) cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, k));
        const float nm = fmaxf(m, cm), al = __expf(m - nm);
        const float e = l < cnt ? __expf(mine - nm) : 0.0f;
        s = s * al + warp_sum(e);
#pragma unroll
        for (int k = 0; k < KVL / 64; ++k) {
            o[k].x *= al;
            o[k].y *= al;
        }
        for (int j = 0; j < cnt; ++j) {
            const float pj = __shfl_sync(0xffffffffu, e, j);
#pragma unroll
            for (int k = 0; k < KVL / 64; ++k) {
                const float2 r = __half22float2(*(const __half2*) &rows[j][64 * k + 2 * l]);
                o[k].x += pj * r.x;
                o[k].y += pj * r.y;
            }
        }
        m = nm;
    }
    const size_t th = ((size_t) t * H + h) * nr + rg;
#pragma unroll
    for (int k = 0; k < KVL / 64; ++k) *(float2*) &opart[th * KVL + 64 * k + 2 * l] = o[k];
    if (l == 0) {
        mlpart[th * 2] = m;
        mlpart[th * 2 + 1] = s;
    }
}
__global__ void k_mla_comb2(const float* opart, const float* mlpart, const float* sinks, int nr, float* o, int H) {
    const int h = blockIdx.x, t = blockIdx.y, i = blockIdx.z * blockDim.x + threadIdx.x;
    const size_t th = ((size_t) t * H + h) * nr;
    float M = sinks[h];
    for (int r = 0; r < nr; ++r) M = fmaxf(M, mlpart[(th + r) * 2]);
    float den = __expf(sinks[h] - M), a = 0;
    for (int r = 0; r < nr; ++r) {
        const float mr = mlpart[(th + r) * 2];
        if (mr > -INFINITY) {
            const float f = __expf(mr - M);
            den += mlpart[(th + r) * 2 + 1] * f;
            a += opart[(th + r) * KVL + i] * f;
        }
    }
    o[((size_t) t * H + h) * KVL + i] = a / den;
}
static int g_mla_target = 168;   // blocks k_mla_part2 aims at (3 a card's SM)
template <int MSUB>
static void mla_dec2_t(const float* qa, const __half* cache, const int* pos, int window, const int* sel, const int* nsel, const float* sinks,
                       float scale, float* o, int H, int T, int nmax, float* opart, float* mlpart, cudaStream_t st) {
    const int steps = std::max(1, (nmax + MSUB - 1) / MSUB), groups = H / MHG;
    const int cap = std::max(1, MLA_PART / (T * H));
    int nr = std::max(1, std::min({steps, cap, (g_mla_target + groups * T - 1) / (groups * T)}));
    const int per = (steps + nr - 1) / nr;
    nr = (steps + per - 1) / per;
    k_mla_part2<MSUB><<<dim3(nr, groups, T), 256, 0, st>>>(qa, cache, pos, window, sel, nsel, scale, H, per, opart, mlpart);
    k_mla_comb2<<<dim3(H, T, KVL / 128), 128, 0, st>>>(opart, mlpart, sinks, nr, o, H);
}
static void mla_dec2(const float* qa, const __half* cache, const int* pos, int window, const int* sel, const int* nsel, const float* sinks,
                     float scale, float* o, int H, int T, int nmax, float* opart, float* mlpart, cudaStream_t st) {
    if (g_msub == 32) mla_dec2_t<32>(qa, cache, pos, window, sel, nsel, sinks, scale, o, H, T, nmax, opart, mlpart, st);
    else mla_dec2_t<16>(qa, cache, pos, window, sel, nsel, sinks, scale, o, H, T, nmax, opart, mlpart, st);
}

// --------------------------------------------------------------------------------------------------------------- experts
// IQ3_XXS rows: v1's kernel with 8 rows a warp (64 a block) and the input through L1 instead of shared memory, the 8 rows' loads
// issued before their sums
constexpr int IQ2_R = 8, IQ2_BR = 8 * IQ2_R;
__global__ void __launch_bounds__(256) k_iq3_rows2(const EJob* jobs, const float* __restrict__ x, int ldx, float* y, int ldy, int rows,
                                                   int cols) {
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    const EJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    __syncthreads();
    const float* xr = x + (size_t) J.in * ldx;
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    const int r0 = blockIdx.x * IQ2_BR + warp * IQ2_R, nb = cols / 256;
    float acc[IQ2_R];
#pragma unroll
    for (int a = 0; a < IQ2_R; ++a) acc[a] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float4 xa = __ldg((const float4*) (xr + b * 256 + l * 8)), xb = __ldg((const float4*) (xr + b * 256 + l * 8 + 4));
        uint16_t dq[IQ2_R], qq[IQ2_R], a0[IQ2_R], a1[IQ2_R];
#pragma unroll
        for (int a = 0; a < IQ2_R; ++a) {
            const int r = min(r0 + a, rows - 1);
            const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) r * nb + b) * 98);
            dq[a] = __ldg(blk);
            qq[a] = __ldg(blk + 1 + l);
            a0[a] = __ldg(blk + 33 + 2 * ib);
            a1[a] = __ldg(blk + 34 + 2 * ib);
        }
#pragma unroll
        for (int a = 0; a < IQ2_R; ++a) {
            const uint32_t aux = (uint32_t) a0[a] | ((uint32_t) a1[a] << 16);
            const float db = __half2float(__ushort_as_half(dq[a])) * (0.5f + (float) (aux >> 28)) * 0.5f;
            const uint32_t s = sg[(aux >> (7 * l4)) & 127];
            const uint32_t g1 = grid[qq[a] & 255], g2 = grid[qq[a] >> 8];
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
    for (int a = 0; a < IQ2_R; ++a) {
        const float s = warp_sum(acc[a]);
        if (l == 0 && r0 + a < rows) y[(size_t) J.out * ldy + r0 + a] = s;
    }
}

// v1g's second try: the grid as floats and the sign bytes as eight +-1 floats, both in shared memory (no int-to-float conversion,
// which runs at a quarter of the FMA rate on sm_60): 8 values cost 4 shared loads, 8 multiplies and 8 FMAs. R rows a warp
template <int R>
__global__ void __launch_bounds__(256) k_iq3_rows3(const EJob* jobs, const float* __restrict__ x, int ldx, float* y, int ldy, int rows,
                                                   int cols) {
    __shared__ float4 gf[256];
    __shared__ float4 sf[128][2];
    const EJob J = jobs[blockIdx.y];
    {
        const uint32_t g = d_iq3xxs_grid[threadIdx.x];
        gf[threadIdx.x] = make_float4((float) (g & 255), (float) ((g >> 8) & 255), (float) ((g >> 16) & 255), (float) (g >> 24));
        if (threadIdx.x < 128) {
            const uint32_t s = d_ksigns_iq2xs[threadIdx.x];
            sf[threadIdx.x][0] = make_float4(s & 1 ? -1.f : 1.f, s & 2 ? -1.f : 1.f, s & 4 ? -1.f : 1.f, s & 8 ? -1.f : 1.f);
            sf[threadIdx.x][1] = make_float4(s & 16 ? -1.f : 1.f, s & 32 ? -1.f : 1.f, s & 64 ? -1.f : 1.f, s & 128 ? -1.f : 1.f);
        }
    }
    __syncthreads();
    const float* xr = x + (size_t) J.in * ldx;
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    const int r0 = blockIdx.x * (8 * R) + warp * R, nb = cols / 256;
    float acc[R];
#pragma unroll
    for (int a = 0; a < R; ++a) acc[a] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float4 xa = __ldg((const float4*) (xr + b * 256 + l * 8)), xb = __ldg((const float4*) (xr + b * 256 + l * 8 + 4));
        uint16_t dq[R], qq[R], a0[R], a1[R];
#pragma unroll
        for (int a = 0; a < R; ++a) {
            const int r = min(r0 + a, rows - 1);
            const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) r * nb + b) * 98);
            dq[a] = __ldg(blk);
            qq[a] = __ldg(blk + 1 + l);
            a0[a] = __ldg(blk + 33 + 2 * ib);
            a1[a] = __ldg(blk + 34 + 2 * ib);
        }
#pragma unroll
        for (int a = 0; a < R; ++a) {
            const uint32_t aux = (uint32_t) a0[a] | ((uint32_t) a1[a] << 16);
            const float db = __half2float(__ushort_as_half(dq[a])) * (0.5f + (float) (aux >> 28)) * 0.5f;
            const int si = (aux >> (7 * l4)) & 127;
            const float4 s0 = sf[si][0], s1 = sf[si][1], g1 = gf[qq[a] & 255], g2 = gf[qq[a] >> 8];
            float v = g1.x * (s0.x * xa.x);
            v += g1.y * (s0.y * xa.y);
            v += g1.z * (s0.z * xa.z);
            v += g1.w * (s0.w * xa.w);
            v += g2.x * (s1.x * xb.x);
            v += g2.y * (s1.y * xb.y);
            v += g2.z * (s1.z * xb.z);
            v += g2.w * (s1.w * xb.w);
            acc[a] += db * v;
        }
    }
#pragma unroll
    for (int a = 0; a < R; ++a) {
        const float s = warp_sum(acc[a]);
        if (l == 0 && r0 + a < rows) y[(size_t) J.out * ldy + r0 + a] = s;
    }
}

// v1g's third try: no int-to-float conversion and no float tables: a byte b goes into 0x4B0000bb (2^23 + b as a float, one byte
// permute) less 2^23, the sign bit goes onto x with one shift and one and-xor
template <int R>
__global__ void __launch_bounds__(256) k_iq3_rows4(const EJob* jobs, const float* __restrict__ x, int ldx, float* y, int ldy, int rows,
                                                   int cols) {
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    const EJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    __syncthreads();
    const float* xr = x + (size_t) J.in * ldx;
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    const int r0 = blockIdx.x * (8 * R) + warp * R, nb = cols / 256;
    float acc[R];
#pragma unroll
    for (int a = 0; a < R; ++a) acc[a] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float4 xa = __ldg((const float4*) (xr + b * 256 + l * 8)), xb = __ldg((const float4*) (xr + b * 256 + l * 8 + 4));
        const uint32_t xv[8] = {__float_as_uint(xa.x), __float_as_uint(xa.y), __float_as_uint(xa.z), __float_as_uint(xa.w),
                                __float_as_uint(xb.x), __float_as_uint(xb.y), __float_as_uint(xb.z), __float_as_uint(xb.w)};
        uint16_t dq[R], qq[R], a0[R], a1[R];
#pragma unroll
        for (int a = 0; a < R; ++a) {
            const int r = min(r0 + a, rows - 1);
            const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) r * nb + b) * 98);
            dq[a] = __ldg(blk);
            qq[a] = __ldg(blk + 1 + l);
            a0[a] = __ldg(blk + 33 + 2 * ib);
            a1[a] = __ldg(blk + 34 + 2 * ib);
        }
#pragma unroll
        for (int a = 0; a < R; ++a) {
            const uint32_t aux = (uint32_t) a0[a] | ((uint32_t) a1[a] << 16);
            const float db = __half2float(__ushort_as_half(dq[a])) * (0.5f + (float) (aux >> 28)) * 0.5f;
            const uint32_t s = sg[(aux >> (7 * l4)) & 127];
            const uint32_t g[2] = {grid[qq[a] & 255], grid[qq[a] >> 8]};
            float v = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float f = __uint_as_float(__byte_perm(g[j >> 2], 0x4B000000u, 0x7540u + (j & 3))) - 8388608.0f;
                v += f * __uint_as_float(xv[j] ^ ((s << (31 - j)) & 0x80000000u));
            }
            acc[a] += db * v;
        }
    }
#pragma unroll
    for (int a = 0; a < R; ++a) {
        const float s = warp_sum(acc[a]);
        if (l == 0 && r0 + a < rows) y[(size_t) J.out * ldy + r0 + a] = s;
    }
}

// v1g's fourth try: a block's rows are one contiguous piece of the matrix (8R rows of cols / 256 blocks of 98 bytes), so it comes
// into shared memory in 16-byte loads all issued before their stores (the earlier tries gather 2-byte words: ~130 GB/s whatever
// the arithmetic), then v1's sums run from shared memory
template <int R>
__global__ void __launch_bounds__(256) k_iq3_rows5(const EJob* jobs, const float* __restrict__ x, int ldx, float* y, int ldy, int rows,
                                                   int cols) {
    constexpr int MAXB = 8 * R * (IQ_MAXC / 256) * 98 / 16;   // int4 of the widest piece
    constexpr int NU = (MAXB + 255) / 256;
    __shared__ __align__(16) uint8_t wq[MAXB * 16];
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    const EJob J = jobs[blockIdx.y];
    const int nb = cols / 256, rb = blockIdx.x * 8 * R, nrow = min(8 * R, rows - rb);
    const int tot = nrow * nb * 98 / 16;   // 8R rows of 98 nb bytes: a multiple of 16 when R is even
    const int4* src = (const int4*) (J.m + (size_t) rb * nb * 98);
    int4 buf[NU];
#pragma unroll
    for (int u = 0; u < NU; ++u) {
        const int i = threadIdx.x + 256 * u;
        if (i < tot) buf[u] = __ldg(src + i);
    }
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
#pragma unroll
    for (int u = 0; u < NU; ++u) {
        const int i = threadIdx.x + 256 * u;
        if (i < tot) ((int4*) wq)[i] = buf[u];
    }
    __syncthreads();
    const float* xr = x + (size_t) J.in * ldx;
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    float acc[R];
#pragma unroll
    for (int a = 0; a < R; ++a) acc[a] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float4 xa = __ldg((const float4*) (xr + b * 256 + l * 8)), xb = __ldg((const float4*) (xr + b * 256 + l * 8 + 4));
#pragma unroll
        for (int a = 0; a < R; ++a) {
            const int rl = warp * R + a;
            if (rl >= nrow) break;
            const uint16_t* blk = (const uint16_t*) (wq + ((size_t) rl * nb + b) * 98);
            const float d = __half2float(__ushort_as_half(blk[0]));
            const uint16_t qq = blk[1 + l];
            const uint32_t aux = (uint32_t) blk[33 + 2 * ib] | ((uint32_t) blk[34 + 2 * ib] << 16);
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
    for (int a = 0; a < R; ++a) {
        const float s = warp_sum(acc[a]);
        if (l == 0 && warp * R + a < nrow) y[(size_t) J.out * ldy + rb + warp * R + a] = s;
    }
}
// ------------------------------------------------------------------------------------------------------------ indexer
// v1v: the writing path's selection for any context (k_idx_select scored all groups in one block and sorted them with ng
// odd-even rounds: ~4 ms a sparse layer at 11k positions, and its shared memory ran out past ~24k). k_idx_scores: 64 groups a
// block, acc[t][g] = sum_h w[t][h] relu(q[t][h] . pooled[g]) (four heads a thread, then the four sums in order), -inf past the
// visible groups. k_idx_topk2: one block a token, the TOPPOOL-th largest by a radix select on the order-keeping bits (4 x 8
// bits), then the groups above it and the first of its ties in group order through block scans: the same set as k_idx_select
constexpr int ISG = 64;
__global__ void __launch_bounds__(256) k_idx_scores(const float* iq, const float* w, const float* pooled, const int* pos, int ng, float* acc) {
    __shared__ float ps[ISG][IDXD + 1];
    __shared__ float qs[IDXH][IDXD];
    __shared__ float part[4][ISG];
    const int t = blockIdx.y, g0 = blockIdx.x * ISG, vis = pos[t] / KPOOL;
    for (int i = threadIdx.x; i < IDXH * IDXD; i += 256) qs[i / IDXD][i % IDXD] = iq[(size_t) t * IDXH * IDXD + i];
    for (int i = threadIdx.x; i < ISG * IDXD; i += 256) {
        const int j = i / IDXD, c = i % IDXD;
        ps[j][c] = g0 + j < ng ? pooled[(size_t) (g0 + j) * IDXD + c] : 0.0f;
    }
    __syncthreads();
    const int j = threadIdx.x & (ISG - 1), hg = threadIdx.x / ISG;
    float s = 0;
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
        const int h = hg * 4 + hh;
        float d = 0;
        for (int c = 0; c < IDXD; ++c) d += qs[h][c] * ps[j][c];
        s += w[t * IDXH + h] * fmaxf(d, 0.0f);
    }
    part[hg][j] = s;
    __syncthreads();
    if (threadIdx.x < ISG && g0 + j < ng)
        acc[(size_t) t * ng + g0 + j] = g0 + j < vis ? ((part[0][j] + part[1][j]) + part[2][j]) + part[3][j] : -INFINITY;
}
__device__ __forceinline__ unsigned isel_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// exclusive scan of one flag a thread over a 1024-thread block; total in *tot
__device__ __forceinline__ int block_scan1024(int f, int* wsum, int* tot) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const unsigned b = __ballot_sync(0xffffffffu, f);
    const int in_warp = __popc(b & ((1u << l) - 1));
    if (l == 0) wsum[w] = __popc(b);
    __syncthreads();
    if (w == 0) {
        const int v = wsum[l];
        int x = v;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, x, o);
            if (l >= o) x += y;
        }
        wsum[l] = x - v;   // exclusive
        if (l == 31) *tot = x;
    }
    __syncthreads();
    const int r = wsum[w] + in_warp;
    __syncthreads();
    return r;
}
__global__ void __launch_bounds__(1024) k_idx_topk2(const float* acc, int ng, const int* pos, int* sel, int* nsel) {
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, need;
    __shared__ int wsum[32], tot, base, tbase;
    const int t = blockIdx.x, p = pos[t], vis = p / KPOOL;
    if (vis <= TOPPOOL) {
        if (threadIdx.x == 0) nsel[t] = -1;
        return;
    }
    const float* a = acc + (size_t) t * ng;
    if (threadIdx.x == 0) { prefix = 0; need = TOPPOOL; }
    for (int byte = 3; byte >= 0; --byte) {
        for (int i = threadIdx.x; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        const unsigned hm = byte == 3 ? 0u : 0xffffffffu << (8 * (byte + 1)), pf = prefix;
        for (int g = threadIdx.x; g < vis; g += 1024) {
            const unsigned k = isel_key(a[g]);
            if ((k & hm) == (pf & hm)) atomicAdd(&hist[(k >> (8 * byte)) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned c = 0;
            int bin = 255;
            for (; bin > 0; --bin) {
                if (c + hist[bin] >= need) break;
                c += hist[bin];
            }
            prefix |= (unsigned) bin << (8 * byte);
            need -= c;   // still to take inside this bin
        }
        __syncthreads();
    }
    const unsigned thr = prefix;   // the TOPPOOL-th largest key; take every key above it and the first `need` equal to it
    if (threadIdx.x == 0) { base = 0; tbase = 0; }
    __syncthreads();
    for (int g0 = 0; g0 < vis; g0 += 1024) {
        const int g = g0 + threadIdx.x;
        const unsigned k = g < vis ? isel_key(a[g]) : 0u;
        const int eq = g < vis && k == thr;
        const int er = block_scan1024(eq, wsum, &tot);
        const int teq = tot;
        const int take = g < vis && (k > thr || (eq && tbase + er < (int) need));
        const int r = block_scan1024(take, wsum, &tot);
        if (take)
            for (int m = 0; m < KPOOL; ++m) sel[(size_t) t * MAXSEL + (base + r) * KPOOL + m] = g * KPOOL + m;
        __syncthreads();
        if (threadIdx.x == 0) { base += tot; tbase += teq; }
        __syncthreads();
    }
    const int k0 = base * KPOOL;
    for (int q = vis * KPOOL + threadIdx.x; q <= p; q += 1024) sel[(size_t) t * MAXSEL + k0 + (q - vis * KPOOL)] = q;
    if (threadIdx.x == 0) nsel[t] = k0 + (p - vis * KPOOL + 1);
}

// ------------------------------------------------------------------------------------------------------- grouped GEMV
// v1r: up to G1_MAXJ Q8S matrices with the same input at T = 1 in one launch (the attention's q_a, kv_a, gate and the indexer's
// key; the FFN's r_down, shared gate and up): k_gemv8_t1's blocks over the matrices' row blocks one after another, one S for all,
// the last block of a row block adds the parts. Not v1's bits (another S), but the same on both cards
constexpr int G1_MAXJ = 6;
struct G1Group {
    const int8_t* q[G1_MAXJ];
    const __half* d[G1_MAXJ];
    float* y[G1_MAXJ];
    int rows[G1_MAXJ], rb_end[G1_MAXJ], row0[G1_MAXJ];
    int n, cols, total_rows;
};
__global__ void __launch_bounds__(256) k_gemv8_t1g(const G1Group g, const float* __restrict__ x, int S) {
    int jj = 0;
    while (jj + 1 < g.n && (int) blockIdx.x >= g.rb_end[jj]) ++jj;
    const int lb = blockIdx.x - (jj ? g.rb_end[jj - 1] : 0);
    const int8_t* __restrict__ q = g.q[jj];
    const __half* __restrict__ d = g.d[jj];
    const int rows = g.rows[jj], cols = g.cols;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = lb * G1_BR, r0 = rb + warp * G1_R;
    const int nch = (cols + 511) / 512, s = blockIdx.y, ch0 = s * nch / S, ch1 = (s + 1) * nch / S;
    float acc[G1_R] = {};
    for (int ch = ch0; ch < ch1; ++ch) {
        const int c = ch * 512 + lane * 16;
        if (c >= cols) continue;
        const float4 x0 = __ldg((const float4*) (x + c)), x1 = __ldg((const float4*) (x + c + 4)), x2 = __ldg((const float4*) (x + c + 8)),
                     x3 = __ldg((const float4*) (x + c + 12));
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
            if (S == 1) g.y[jj][r] = v;
            else g_part[(size_t) s * g.total_rows + g.row0[jj] + r] = v;
        }
    }
    if (S == 1) return;
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
        for (int k = 0; k < S; ++k) v += ((volatile float*) g_part)[(size_t) k * g.total_rows + g.row0[jj] + r];
        g.y[jj][r] = v;
    }
    if (threadIdx.x == 0) g_g1_cnt[blockIdx.x] = 0;
}
static void gemv_group(std::initializer_list<std::pair<const Mat*, float*>> mats, const float* x, cudaStream_t st) {
    G1Group g{};
    int rb = 0, rows = 0;
    for (const auto& m : mats) {
        if (g.n == G1_MAXJ || m.first->type != 8 || m.first->cols % 16 || (g.n && m.first->cols != g.cols))
            throw std::runtime_error("gemv_group: Q8S matrices of one width only");
        g.q[g.n] = m.first->q;
        g.d[g.n] = (const __half*) m.first->d;
        g.y[g.n] = m.second;
        g.rows[g.n] = m.first->rows;
        g.row0[g.n] = rows;
        rows += m.first->rows;
        rb += (m.first->rows + G1_BR - 1) / G1_BR;
        g.rb_end[g.n] = rb;
        g.cols = m.first->cols;
        ++g.n;
    }
    g.total_rows = rows;
    if (rb > 1024) throw std::runtime_error("gemv_group: more than 1024 row blocks");
    int S = std::max(1, std::min((g.cols + 511) / 512, 448 / rb));
    while (S > 1 && (size_t) S * rows > G1_PART) --S;
    k_gemv8_t1g<<<dim3(rb, S), 256, 0, st>>>(g, x, S);
}

// ------------------------------------------------------------------------------------------------------------ iHC + rms
// k_hc_fin and the rms that always follows it in the window path (xn = x / rms(x) * w), one block of 256 a token with x in
// registers: one launch less, the same bits as k_hc_fin then k_rms (each thread adds its squares in k_rms' order)
__global__ void __launch_bounds__(256) k_hc_fin_rms(const float* R, const float* scale, const float* base, int nmix, float eps_rms,
                                                    float eps_hc, float mag, float* x, float* post, const float* w, float* xn, float eps2) {
    __shared__ float tot[9], pre[HC], red[32];
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
    float v[N / 256], ss = 0;
#pragma unroll
    for (int j = 0; j < N / 256; ++j) {
        const int i = threadIdx.x + 256 * j;
        float a = 0;
#pragma unroll
        for (int c = 0; c < HC; ++c) a += pre[c] * R[(size_t) t * D + c * N + i];
        v[j] = a;
        x[(size_t) t * N + i] = a;
        ss += a * a;
    }
    ss = block_sum(ss, red);
    const float kk = rsqrtf(ss / N + eps2);
#pragma unroll
    for (int j = 0; j < N / 256; ++j) {
        const int i = threadIdx.x + 256 * j;
        xn[(size_t) t * N + i] = v[j] * kk * w[i];
    }
}
static void hc_pre_rms(const float* R, const float* fn, const float* scale, const float* base, int nmix, float eps_rms, float eps_hc,
                       float mag, float* x, float* post, const float* w, float* xn, float eps2, int T, cudaStream_t st) {
    k_hc_mix<<<dim3(T, HCB), 256, 0, st>>>(R, fn, nmix);
    k_hc_fin_rms<<<T, 256, 0, st>>>(R, scale, base, nmix, eps_rms, eps_hc, mag, x, post, w, xn, eps2);
}

// v1r: IQ3 rows in half2. The f32 kernels are held by arithmetic (k_iq3_x in bench_dec: ~35 us of a call's ~50 with no weight
// loads at all). x is scaled to |x| <= 1 per 256-column block (the scale goes back with the block's d, so 4 products of <= 62
// stay far from half's 65504), a pair of grid bytes becomes a half2 with one byte permute (0x64bb = 1024 + b) and one subtract,
// the signs come as four half2 xor masks from a shared table, 4 HFMA2 per 8 values, summed in f32 per block
template <int R>
__global__ void __launch_bounds__(256) k_iq3_rows_h2(const EJob* jobs, const float* __restrict__ x, int ldx, float* y, int ldy, int rows,
                                                     int cols) {
    __shared__ uint32_t grid[256];
    __shared__ uint4 sgm[128];
    __shared__ float bmax[IQ_MAXC / 256];
    const EJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) {
        const uint32_t s = d_ksigns_iq2xs[threadIdx.x];
        uint32_t m[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) m[k] = ((s >> (2 * k)) & 1u) << 15 | ((s >> (2 * k + 1)) & 1u) << 31;
        sgm[threadIdx.x] = make_uint4(m[0], m[1], m[2], m[3]);
    }
    const float* xr = x + (size_t) J.in * ldx;
    const int nb = cols / 256, warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    if (warp < nb) {
        float mx = 0;
        for (int i = l; i < 256; i += 32) mx = fmaxf(mx, fabsf(xr[warp * 256 + i]));
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (l == 0) bmax[warp] = mx > 0 ? mx : 1.0f;
    }
    __syncthreads();
    const int r0 = blockIdx.x * (8 * R) + warp * R;
    const __half2 k1024 = __float2half2_rn(1024.0f);
    float acc[R];
#pragma unroll
    for (int a = 0; a < R; ++a) acc[a] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float sc = bmax[b], inv = 1.0f / sc;
        const float4 xa = __ldg((const float4*) (xr + b * 256 + l * 8)), xb = __ldg((const float4*) (xr + b * 256 + l * 8 + 4));
        const __half2 x0 = __floats2half2_rn(xa.x * inv, xa.y * inv), x1 = __floats2half2_rn(xa.z * inv, xa.w * inv);
        const __half2 x2 = __floats2half2_rn(xb.x * inv, xb.y * inv), x3 = __floats2half2_rn(xb.z * inv, xb.w * inv);
        uint16_t dq[R], qq[R], a0[R], a1[R];
#pragma unroll
        for (int a = 0; a < R; ++a) {
            const int r = min(r0 + a, rows - 1);
            const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) r * nb + b) * 98);
            dq[a] = __ldg(blk);
            qq[a] = __ldg(blk + 1 + l);
            a0[a] = __ldg(blk + 33 + 2 * ib);
            a1[a] = __ldg(blk + 34 + 2 * ib);
        }
#pragma unroll
        for (int a = 0; a < R; ++a) {
            const uint32_t aux = (uint32_t) a0[a] | ((uint32_t) a1[a] << 16);
            const float db = __half2float(__ushort_as_half(dq[a])) * (0.5f + (float) (aux >> 28)) * 0.5f * sc;
            const uint4 m = sgm[(aux >> (7 * l4)) & 127];
            const uint32_t g1 = grid[qq[a] & 255], g2 = grid[qq[a] >> 8];
            const uint32_t h0 = __byte_perm(g1, 0x64646464u, 0x4140), h1 = __byte_perm(g1, 0x64646464u, 0x4342);
            const uint32_t h2 = __byte_perm(g2, 0x64646464u, 0x4140), h3 = __byte_perm(g2, 0x64646464u, 0x4342);
            __half2 w0 = __hsub2(*(const __half2*) &h0, k1024), w1 = __hsub2(*(const __half2*) &h1, k1024);
            __half2 w2 = __hsub2(*(const __half2*) &h2, k1024), w3 = __hsub2(*(const __half2*) &h3, k1024);
            uint32_t u0 = *(uint32_t*) &w0 ^ m.x, u1 = *(uint32_t*) &w1 ^ m.y, u2 = *(uint32_t*) &w2 ^ m.z, u3 = *(uint32_t*) &w3 ^ m.w;
            __half2 p = __hmul2(*(const __half2*) &u0, x0);
            p = __hfma2(*(const __half2*) &u1, x1, p);
            p = __hfma2(*(const __half2*) &u2, x2, p);
            p = __hfma2(*(const __half2*) &u3, x3, p);
            const float2 pf = __half22float2(p);
            acc[a] += db * (pf.x + pf.y);
        }
    }
#pragma unroll
    for (int a = 0; a < R; ++a) {
        const float s = warp_sum(acc[a]);
        if (l == 0 && r0 + a < rows) y[(size_t) J.out * ldy + r0 + a] = s;
    }
}

// v1x: a window's distinct expert once for all its tokens (a 9-token window picks 144 experts a layer, 85 different; 16 tokens:
// 256 and 105). Jobs (matrix, list, n) as the prompt path's, lists (input row, output row); the block decodes its 8R rows' 8
// values a lane once a 256-column block and adds them into up to MTB tokens' sums
constexpr int MTB = 4;
template <int R>
__global__ void __launch_bounds__(256) k_iq3_rows_mt(const PJob* jobs, const int2* lists, const float* __restrict__ x, int ldx, float* y, int ldy,
                                                     int rows, int cols, int in_is_pair) {
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    const PJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    const int r0 = blockIdx.x * (8 * R) + warp * R, nb = cols / 256;
    for (int k0 = 0; k0 < J.n; k0 += MTB) {
        const int nk = min(MTB, J.n - k0);
        const float* xr[MTB];
#pragma unroll
        for (int k = 0; k < MTB; ++k) {
            const int2 e = lists[J.list + k0 + min(k, nk - 1)];
            xr[k] = x + (size_t) (in_is_pair ? e.y : e.x) * ldx;
        }
        float acc[R][MTB];
#pragma unroll
        for (int a = 0; a < R; ++a)
#pragma unroll
            for (int k = 0; k < MTB; ++k) acc[a][k] = 0.0f;
        for (int b = 0; b < nb; ++b) {
            float4 xa[MTB], xb[MTB];
#pragma unroll
            for (int k = 0; k < MTB; ++k)
                if (k < nk) {   // nk is the same for the whole block: no divergence
                    xa[k] = __ldg((const float4*) (xr[k] + b * 256 + l * 8));
                    xb[k] = __ldg((const float4*) (xr[k] + b * 256 + l * 8 + 4));
                }
#pragma unroll
            for (int a = 0; a < R; ++a) {
                const int r = min(r0 + a, rows - 1);
                const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) r * nb + b) * 98);
                const float d = __half2float(__ushort_as_half(__ldg(blk)));
                const uint16_t qq = __ldg(blk + 1 + l);
                const uint32_t aux = (uint32_t) __ldg(blk + 33 + 2 * ib) | ((uint32_t) __ldg(blk + 34 + 2 * ib) << 16);
                const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
                const uint32_t s = sg[(aux >> (7 * l4)) & 127];
                const uint32_t g1 = grid[qq & 255], g2 = grid[qq >> 8];
                float w[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    w[j] = (float) ((g1 >> (8 * j)) & 255) * ((s >> j) & 1 ? -db : db);
                    w[4 + j] = (float) ((g2 >> (8 * j)) & 255) * ((s >> (4 + j)) & 1 ? -db : db);
                }
#pragma unroll
                for (int k = 0; k < MTB; ++k)
                    if (k < nk) acc[a][k] += w[0] * xa[k].x + w[1] * xa[k].y + w[2] * xa[k].z + w[3] * xa[k].w + w[4] * xb[k].x + w[5] * xb[k].y +
                                 w[6] * xb[k].z + w[7] * xb[k].w;
            }
        }
#pragma unroll
        for (int a = 0; a < R; ++a)
#pragma unroll
            for (int k = 0; k < MTB; ++k)
                if (k < nk) {
                    const float v = warp_sum(acc[a][k]);
                    if (l == 0 && r0 + a < rows) y[(size_t) lists[J.list + k0 + k].y * ldy + r0 + a] = v;
                }
    }
}

// k_swiglu over the rows the jobs name (their out field): v1i runs the hits' rows before the misses' copies are in
__global__ void k_swiglu_jobs(const float* g, const float* u, float* h, int n, float clamp, const EJob* jobs) {
    const int t = jobs[blockIdx.y].out;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float a = g[(size_t) t * n + i], b = u[(size_t) t * n + i];
        if (clamp > 0) {
            a = fminf(a, clamp);
            b = fminf(fmaxf(b, -clamp), clamp);
        }
        h[(size_t) t * n + i] = a * sigm(a) * b;
    }
}
// HY4_GUESS_STAT: how many of a layer's picks its router finds already on its attention input (a prefetch could start ~0.25 ms
// earlier from there); g_guess_ov = (found, picks) of card 0
__device__ unsigned long long g_guess_ov[2];
__global__ void k_guess_overlap(const int* real, const int* guess) {
    const int l = threadIdx.x;
    int hit = 0;
    if (l < TOPK)
        for (int k = 0; k < TOPK; ++k) hit |= guess[k] == real[l];
    const int n = __popc(__ballot_sync(0xffffffffu, hit && l < TOPK));
    if (l == 0) {
        atomicAdd(&g_guess_ov[0], (unsigned long long) n);
        atomicAdd(&g_guess_ov[1], (unsigned long long) TOPK);
    }
}
// the floor for the same bytes: 16-byte loads, nothing else
__global__ void k_ld_floor(const int4* p, size_t n, int* out) {
    int a = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int4 v = __ldg(p + i);
        a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a == 0x7fffffff) *out = a;
}

// ---------------------------------------------------------------------------------------------------------------- router
// k_route as a bitonic sort of the 512 scores (descending; ties: the lower index first) by 256 threads, 45 steps: the first TOPK are
// k_route's picks in its order, thread 0 adds their weights in that order
__global__ void __launch_bounds__(256) k_route4(const float* logits, const float* bias, float wscale, int* topi, float* topw) {
    __shared__ float key[NEXP], prs[NEXP];
    __shared__ int idx[NEXP];
    const int t = blockIdx.x, i = threadIdx.x;
    for (int e = i; e < NEXP; e += 256) {
        const float pr = sigm(logits[(size_t) t * NEXP + e]);
        prs[e] = pr;
        key[e] = pr + bias[e];
        idx[e] = e;
    }
    __syncthreads();
    for (int k = 2; k <= NEXP; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int a = 2 * i - (i & (j - 1)), b = a + j;
            const float ka = key[a], kb = key[b];
            const int ia = idx[a], ib = idx[b];
            const bool first = ka > kb || (ka == kb && ia < ib);
            if (first != ((a & k) == 0)) {
                key[a] = kb; key[b] = ka;
                idx[a] = ib; idx[b] = ia;
            }
            __syncthreads();
        }
    if (i == 0) {
        float wsum = 0;
        for (int j = 0; j < TOPK; ++j) wsum += prs[idx[j]];
        const float k = wscale / fmaxf(wsum, 6.103515625e-5f);
        for (int j = 0; j < TOPK; ++j) {
            topi[t * TOPK + j] = idx[j];
            topw[t * TOPK + j] = prs[idx[j]] * k;
        }
    }
}
// k_route with the per-lane arrays in registers: every run-time index replaced by an unrolled compare. The same picks, weights
// and order as k_route
__global__ void k_route2(const float* logits, const float* bias, float wscale, int* topi, float* topw) {
    const int t = blockIdx.x, l = threadIdx.x;
    float pr[NEXP / 32], sc[NEXP / 32];
#pragma unroll
    for (int k = 0; k < NEXP / 32; ++k) {
        pr[k] = sigm(logits[(size_t) t * NEXP + k * 32 + l]);
        sc[k] = pr[k] + bias[k * 32 + l];
    }
    float wsum = 0, wl = 0;   // lane j < TOPK keeps pick j's weight
    int il = 0;
    for (int j = 0; j < TOPK; ++j) {
        float best = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int k = 0; k < NEXP / 32; ++k)
            if (sc[k] > best) { best = sc[k]; bi = k * 32 + l; }
        for (int o = 16; o; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
        }
        float mp = 0;
#pragma unroll
        for (int k = 0; k < NEXP / 32; ++k)
            if ((bi & 31) == l && (bi >> 5) == k) {
                mp = pr[k];
                sc[k] = -INFINITY;
            }
        const float wj = __shfl_sync(0xffffffffu, mp, bi & 31);
        wsum += wj;
        if (l == j) { wl = wj; il = bi; }
    }
    const float k = wscale / fmaxf(wsum, 6.103515625e-5f);
    if (l < TOPK) {
        topi[t * TOPK + l] = il;
        topw[t * TOPK + l] = wl * k;
    }
}

// a block of NEXP threads a token: each expert's rank among the scores (ties: the lower index first, as in k_route); ranks below
// TOPK are k_route's picks in its order, and thread 0 adds their weights in that order: the same bits as k_route
__global__ void __launch_bounds__(NEXP) k_route3(const float* logits, const float* bias, float wscale, int* topi, float* topw) {
    __shared__ __align__(16) float sc[NEXP];
    __shared__ float pw[TOPK];
    __shared__ int pi[TOPK];
    const int t = blockIdx.x, e = threadIdx.x;
    const float pr = sigm(logits[(size_t) t * NEXP + e]), me = pr + bias[e];
    sc[e] = me;
    __syncthreads();
    int rank = 0;
    for (int k = 0; k < NEXP; k += 4) {
        const float4 o = *(const float4*) &sc[k];
        rank += (o.x > me || (o.x == me && k < e)) + (o.y > me || (o.y == me && k + 1 < e)) + (o.z > me || (o.z == me && k + 2 < e)) +
                (o.w > me || (o.w == me && k + 3 < e));
    }
    if (rank < TOPK) {
        pi[rank] = e;
        pw[rank] = pr;
    }
    __syncthreads();
    if (e == 0) {
        float wsum = 0;
        for (int j = 0; j < TOPK; ++j) wsum += pw[j];
        const float k = wscale / fmaxf(wsum, 6.103515625e-5f);
        for (int j = 0; j < TOPK; ++j) {
            topi[t * TOPK + j] = pi[j];
            topw[t * TOPK + j] = pw[j] * k;
        }
    }
}

// F32 rows at T = 1 (the router, 512 x 3072): k_gemv8_t1's shape, 4 rows a warp, the columns in S parts of 128-column chunks
// summed by k_sum_parts in a fixed order
__global__ void __launch_bounds__(256) k_gemv32_t1(const float* __restrict__ W, int rows, int cols, const float* __restrict__ x, float* y,
                                                    int S) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = blockIdx.x * G1_BR + warp * G1_R;
    const int nch = cols / 128, s = blockIdx.y, ch0 = s * nch / S, ch1 = (s + 1) * nch / S;
    float acc[G1_R] = {};
    for (int ch = ch0; ch < ch1; ++ch) {
        const int c = ch * 128 + lane * 4;
        const float4 xv = __ldg((const float4*) (x + c));
#pragma unroll
        for (int a = 0; a < G1_R; ++a) {
            const int r = min(r0 + a, rows - 1);
            const float4 w = __ldg((const float4*) (W + (size_t) r * cols + c));
            acc[a] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
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
}
static void gemv32_t1(const Mat& W, const float* x, float* y, cudaStream_t st) {
    const int rb = (W.rows + G1_BR - 1) / G1_BR, nch = W.cols / 128;
    int S = std::max(1, std::min(nch, 448 / rb));
    while (S > 1 && (size_t) S * W.rows > G1_PART) --S;
    k_gemv32_t1<<<dim3(rb, S), 256, 0, st>>>((const float*) W.d, W.rows, W.cols, x, y, S);
    if (S > 1) k_sum_parts<<<(W.rows + 255) / 256, 256, 0, st>>>(y, W.rows, S);
}

}  // namespace hy4
